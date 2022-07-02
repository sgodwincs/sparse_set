//! A sparsely populated set, written `SparseSet<I, T>`, where `I` is the index type and `T` is the value type.
//!
//! `I` must implement `SparseSetIndex`, in particular, it must be able to be converted to a `usize`.
//!
//! See [this article](https://research.swtch.com/sparse) on more details behind the data structure.
//!
//! Sparse sets ensure they never allocate more than `isize::MAX` bytes.
//!
//! Note that this data structure does not implement some common traits such as [`Eq`], [`PartialEq`], and
//! [`std::hash::Hash`]. This is because it would not be efficient to do so as this data structure does not store the
//! actual indices.

#![allow(unsafe_code)]

use std::{
  alloc::{self, Allocator, Global, Layout, LayoutError},
  cmp,
  collections::{TryReserveError, TryReserveErrorKind},
  fmt,
  marker::PhantomData,
  mem::{self, ManuallyDrop},
  num::NonZeroUsize,
  ops::{Deref, Index, IndexMut},
  ptr::{self, NonNull},
  slice,
};

/// A sparsely populated set, written `SparseSet<I, T>`, where `I` is the index type and `T` is the value type.
pub struct SparseSet<I, T, A: Allocator = Global> {
  /// Memory allocator for allocating a single buffer for dense and sparse buffers.
  alloc: A,

  /// The amount of elements the dense buffer, or similarly the sparse buffer, can hold.
  capacity: usize,

  /// Dummy marker for ensuring the sparse set is specific to a given index type.
  _marker: (PhantomData<I>, PhantomData<T>),

  /// A pointer to the beginning of the dense buffer.
  ///
  /// This can be dangling, but it must always be aligned.
  dense_ptr: NonNull<u8>,

  /// The length of the dense buffer.
  dense_len: usize,

  /// A pointer to the beginning of the sparse buffer.
  ///
  /// This will always point to the same allocated memory block as the dense pointer, just offset by `capacity` number
  /// of elements.
  ///
  /// This can be dangling, but it must always be aligned.
  sparse_ptr: NonNull<u8>,

  /// The length of the sparse buffer.
  ///
  /// This may be different from the dense buffer as elements can be removed from the dense buffer.
  sparse_len: usize,
}

impl<I, T> SparseSet<I, T> {
  /// Constructs a new, empty `SparseSet<I, T>`.
  ///
  /// The sparse set will not allocate until elements are inserted into it.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// # #[allow(unused_mut)]
  /// let mut set: SparseSet<usize, u32> = SparseSet::new();
  /// ```
  #[must_use]
  pub fn new() -> Self {
    SparseSet::new_in(Global)
  }

  /// Constructs a new, empty `SparseSet<I, T>` with the specified capacity.
  ///
  /// The sparse set will be able to hold exactly `capacity` elements without reallocating. If `capacity` is 0, the
  /// sparse set will not allocate.
  ///
  /// It is important to note that although the returned sparse set has the *capacity* specified, the sparse set will
  /// have a zero *len*.
  ///
  /// # Panics
  ///
  /// Panics if the new capacity exceeds `isize::MAX` bytes.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::with_capacity(10);
  ///
  /// // The sparse set contains no items, even though it has capacity for more.
  /// assert_eq!(set.len(), 0);
  /// assert_eq!(set.capacity(), 10);
  ///
  /// // These are all done without reallocating...
  /// for i in 0..10 {
  ///   set.insert(i, i);
  /// }
  ///
  /// assert_eq!(set.len(), 10);
  /// assert_eq!(set.capacity(), 10);
  ///
  /// // ...but this will make the sparse set reallocate.
  /// set.insert(11, 11);
  /// assert_eq!(set.len(), 11);
  /// assert!(set.capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  #[must_use]
  pub fn with_capacity(capacity: usize) -> Self {
    SparseSet::with_capacity_in(capacity, Global)
  }
}

impl<I, T, A: Allocator> SparseSet<I, T, A> {
  /// Constructs a new, empty `SparseSet<I, T, A>`.
  ///
  /// The sparse set will not allocate until elements are pushed onto it.
  ///
  /// # Examples
  ///
  /// ```
  /// #![feature(allocator_api)]
  ///
  /// use std::alloc::System;
  /// #
  /// # use sparse_set::SparseSet;
  ///
  /// # #[allow(unused_mut)]
  /// let mut set: SparseSet<usize, u32, _> = SparseSet::new_in(System);
  /// ```
  #[must_use]
  pub fn new_in(alloc: A) -> Self {
    Self {
      alloc,
      capacity: 0,
      _marker: (PhantomData, PhantomData),
      dense_ptr: NonNull::<T>::dangling().cast(),
      dense_len: 0,
      sparse_ptr: NonNull::<Option<NonZeroUsize>>::dangling().cast(),
      sparse_len: 0,
    }
  }

  /// Constructs a new, empty `SparseSet<I, T, A>` with the specified capacity with the provided allocator.
  ///
  /// The sparse set will be able to hold exactly `capacity` elements without reallocating. If `capacity` is 0, the
  /// sparse set will not allocate.
  ///
  /// It is important to note that although the returned sparse set has the *capacity* specified, the sparse set will
  /// have a zero *len*.
  ///
  /// # Panics
  ///
  /// Panics if the new capacity exceeds `isize::MAX` bytes.
  ///
  /// # Examples
  ///
  /// ```
  /// #![feature(allocator_api)]
  ///
  /// use std::alloc::System;
  /// #
  /// # use sparse_set::SparseSet;
  ///
  /// let mut set = SparseSet::with_capacity_in(10, System);
  ///
  /// // The sparse set contains no items, even though it has capacity for more
  /// assert_eq!(set.len(), 0);
  /// assert_eq!(set.capacity(), 10);
  ///
  /// // These are all done without reallocating...
  /// for i in 0..10 {
  ///   set.insert(i, i);
  /// }
  /// assert_eq!(set.len(), 10);
  /// assert_eq!(set.capacity(), 10);
  ///
  /// // ...but this will make the sparse set reallocate
  /// set.insert(11, 11);
  /// assert_eq!(set.len(), 11);
  /// assert!(set.capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
    Self::allocate_in(capacity, alloc)
  }
}

impl<I: SparseSetIndex, T, A: Allocator> SparseSet<I, T, A> {
  /// Returns a reference to the underlying allocator.
  #[must_use]
  pub fn allocator(&self) -> &A {
    &self.alloc
  }

  /// Extracts a slice containing the entire sparse set's buffer.
  #[must_use]
  pub fn as_slice(&self) -> &[T] {
    self
  }

  /// Returns a raw pointer to the sparse set's buffer, or a dangling raw pointer valid for zero sized reads if the
  /// sparse set didn't allocate.
  ///
  /// The caller must ensure that the sparse set outlives the pointer this function returns, or else it will end up
  /// pointing to garbage. Modifying the sparse set may cause its buffer to be reallocated, which would also make any
  /// pointers to it invalid.
  ///
  /// The caller must also ensure that the memory the pointer (non-transitively) points to is never written to (except
  /// inside an `UnsafeCell`) using this pointer or any pointer derived from it. If you need to mutate the contents of
  /// the slice, use [`as_mut_ptr`].
  #[must_use]
  pub fn as_ptr(&self) -> *const T {
    // We shadow the slice method of the same name to avoid going through `deref`, which creates an intermediate
    // reference.
    self.dense_ptr.as_ptr() as *const T
  }

  /// Returns the number of elements the sparse set can hold without reallocating.
  ///
  /// Note that even for ZSTs, this will be based on the actual allocated memory. This is because memory still needs to
  /// be allocated for the sparse buffer even if the value type itself is a ZST.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let set: SparseSet<usize, i32> = SparseSet::with_capacity(10);
  /// assert_eq!(set.capacity(), 10);
  /// ```
  #[must_use]
  pub fn capacity(&self) -> usize {
    self.capacity
  }

  /// Clears the sparse set, removing all values.
  ///
  /// Note that this method has no effect on the allocated capacity of the sparse set.
  ///
  /// This operation is *O*(*n*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  ///
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// set.insert(2, 3);
  ///
  /// set.clear();
  ///
  /// assert!(set.is_empty());
  /// ```
  pub fn clear(&mut self) {
    let values: *mut [T] = self.as_mut_slice();

    // SAFETY:
    // - `values` comes directly from `as_mut_slice` and is therefore valid.
    // - Setting `self.len` before calling `drop_in_place` means that, if an element's `Drop` impl panics, the sparse
    //   set's `Drop` impl will do nothing (leaking the rest of the elements) instead of dropping some twice.
    unsafe {
      self.dense_len = 0;
      ptr::drop_in_place(values);
    }
  }

  /// Returns `true` if the sparse set contains an element at the given index.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  ///
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// set.insert(2, 3);
  ///
  /// assert!(set.contains(0));
  /// assert!(!set.contains(100));
  /// ```
  #[must_use]
  pub fn contains(&self, index: I) -> bool {
    self.get(index).is_some()
  }

  /// Returns a reference to an element pointed to by the index, if it exists.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  ///
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// set.insert(2, 3);
  /// assert_eq!(Some(&2), set.get(1));
  /// assert_eq!(None, set.get(3));
  ///
  /// set.remove(1);
  /// assert_eq!(None, set.get(1));
  /// ```
  #[must_use]
  pub fn get(&self, index: I) -> Option<&T> {
    self
      .as_sparse_slice()
      .get(index.into())
      .cloned()
      .flatten()
      .map(|dense_index| unsafe { self.as_slice().get_unchecked(dense_index.get() - 1) })
  }

  /// Returns a mutable reference to an element pointed to by the index, if it exists.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  ///
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// set.insert(2, 3);
  ///
  /// if let Some(elem) = set.get_mut(1) {
  ///   *elem = 42;
  /// }
  ///
  /// assert!(set.values().eq(&[1, 42, 3]));
  /// ```
  #[must_use]
  pub fn get_mut(&mut self, index: I) -> Option<&mut T> {
    self
      .as_sparse_slice()
      .get(index.into())
      .cloned()
      .flatten()
      .map(|dense_index| unsafe { self.as_mut_slice().get_unchecked_mut(dense_index.get() - 1) })
  }

  /// Inserts an element at position `index` within the sparse set.
  ///
  /// If a value already existed at `index`, it will be overwritten.
  ///
  /// If `index` is greater than `capacity`, then an allocation will take place.
  ///
  /// This operation is amortized *O*(*1*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  ///
  /// set.insert(0, 1);
  /// set.insert(1, 4);
  /// set.insert(2, 2);
  /// set.insert(3, 3);
  ///
  /// assert!(set.values().eq(&[1, 4, 2, 3]));
  /// set.insert(20, 5);
  /// assert!(set.values().eq(&[1, 4, 2, 3, 5]));
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn insert(&mut self, index: I, value: T) {
    self.insert_raw(index.into(), value);
  }

  /// Returns `true` if the sparse set contains no elements.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  /// assert!(set.is_empty());
  ///
  /// set.insert(0, 1);
  /// assert!(!set.is_empty());
  /// ```
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Returns the number of elements in the sparse set, also referred to as its 'len'.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// set.insert(2, 3);
  ///
  /// assert_eq!(set.len(), 3);
  /// ```
  #[must_use]
  pub fn len(&self) -> usize {
    self.dense_len
  }

  /// Removes and returns the element at position `index` within the sparse set, if it exists.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// set.insert(2, 3);
  ///
  /// assert_eq!(set.remove(1), Some(2));
  /// assert!(set.values().eq(&[1, 3]));
  /// ```
  pub fn remove(&mut self, index: I) -> Option<T> {
    let index = index.into();

    match self.as_sparse_mut_slice().get_mut(index) {
      Some(opt) => {
        if let Some(dense_index) = opt.take() {
          let mut_ptr = self.as_mut_ptr();
          let dense_index = dense_index.get() - 1;

          unsafe {
            let value = ptr::read(mut_ptr.add(dense_index));
            ptr::copy(mut_ptr.add(self.dense_len - 1), mut_ptr.add(dense_index), 1);

            if index == self.sparse_len - 1 {
              self.sparse_len -= 1;
            }

            self.dense_len -= 1;
            return Some(value);
          }
        }

        None
      }
      _ => None,
    }
  }

  /// Reserves capacity for at least `additional` more elements to be inserted in the given `SparseSet<I, T>`.
  ///
  /// The collection may reserve more space to avoid frequent reallocations. After calling `reserve`, capacity will be
  /// greater than or equal to `self.len() + additional`. Does nothing if capacity is already sufficient.
  ///
  /// # Panics
  ///
  /// Panics if the new capacity exceeds `isize::MAX` bytes.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  /// set.insert(0, 1);
  /// set.reserve(10);
  /// assert!(set.capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve(&mut self, additional: usize) {
    // Callers expect this function to be very cheap when there is already sufficient capacity. Therefore, we move all
    // the resizing and error-handling logic from grow_amortized and handle_reserve behind a call, while making sure
    // that this function is likely to be inlined as just a comparison and a call if the comparison fails.
    #[cold]
    fn do_reserve_and_handle<I, T, A: Allocator>(
      slf: &mut SparseSet<I, T, A>,
      len: usize,
      additional: usize,
    ) {
      handle_reserve(slf.grow_amortized(len, additional));
    }

    if self.needs_to_grow(self.dense_len, additional) {
      do_reserve_and_handle(self, self.dense_len, additional);
    }
  }

  /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the given
  /// `SparseSet<I, T>`.
  ///
  /// After calling `reserve_exact`, capacity will be greater than or equal to `self.len() + additional`. Does nothing
  /// if the capacity is already sufficient.
  ///
  /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not be relied
  /// upon to be precisely minimal. Prefer [`reserve`] if future insertions are expected.
  ///
  /// [`reserve`]: SparseSet::reserve
  ///
  /// # Panics
  ///
  /// Panics if the new capacity exceeds `isize::MAX` bytes.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  /// set.insert(0, 1);
  /// set.reserve(10);
  /// set.reserve_exact(10);
  /// assert!(set.capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_exact(&mut self, additional: usize) {
    handle_reserve(self.try_reserve_exact(additional));
  }

  /// Shrinks the capacity of the sparse set as much as possible.
  ///
  /// It will drop down as close as possible to the length but the allocator may still inform the sparse vector that
  /// there is space for a few more elements.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::with_capacity(10);
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// assert_eq!(set.capacity(), 10);
  ///
  /// set.shrink_to_fit();
  /// assert!(set.capacity() >= 2);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to_fit(&mut self) {
    // The capacity is never less than the sparse buffer length, and there's nothing to do when they are equal.
    if self.capacity() > self.sparse_len {
      handle_reserve(self.shrink(self.sparse_len));
    }
  }

  /// Shrinks the capacity of the sparse set with a lower bound.
  ///
  /// The capacity will remain at least as large as both the length and the supplied value.
  ///
  /// If the current capacity is less than the lower limit, this is a no-op.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::with_capacity(10);
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// assert_eq!(set.capacity(), 10);
  /// set.shrink_to(4);
  /// assert!(set.capacity() >= 4);
  /// set.shrink_to(0);
  /// assert!(set.capacity() >= 2);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to(&mut self, min_capacity: usize) {
    if self.capacity() > min_capacity {
      handle_reserve(self.shrink(cmp::max(self.sparse_len, min_capacity)));
    }
  }

  /// Tries to reserve capacity for at least `additional` more elements to be inserted in the given `SparseSet<I, T>`.
  ///
  /// The collection may reserve more space to avoid frequent reallocations. After calling `try_reserve`, capacity will
  /// be greater than or equal to `self.len() + additional`. Does nothing if capacity is already sufficient.
  ///
  /// # Errors
  ///
  /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
  ///
  /// # Examples
  ///
  /// ```
  /// use std::collections::TryReserveError;
  ///
  /// # use sparse_set::SparseSet;
  ///
  /// fn process_data(data: &[u32]) -> Result<SparseSet<usize, u32>, TryReserveError> {
  ///   let mut output = SparseSet::new();
  ///
  ///   // Pre-reserve the memory, exiting if we can't.
  ///   output.try_reserve(data.len())?;
  ///
  ///   // Now we know this can't OOM in the middle of our complex work.
  ///   for (index, value) in data.iter().cloned().enumerate() {
  ///     output.insert(index, value);
  ///   }  
  ///
  ///   Ok(output)
  /// }
  /// # process_data(&[1, 2, 3]).expect("why is the test harness OOMing on 12 bytes?");
  /// ```
  pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
    if self.needs_to_grow(self.dense_len, additional) {
      self.grow_amortized(self.dense_len, additional)
    } else {
      Ok(())
    }
  }

  /// Tries to reserve the minimum capacity for exactly `additional` elements to be inserted in the given
  /// `SparseSet<T>`.
  ///
  /// After calling `try_reserve_exact`, capacity will be greater than or equal to `self.len() + additional` if it
  /// returns `Ok(())`. Does nothing if the capacity is already sufficient.
  ///
  /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not be relied
  /// upon to be precisely minimal. Prefer [`try_reserve`] if future insertions are expected.
  ///
  /// [`try_reserve`]: SparseSet::try_reserve
  ///
  /// # Errors
  ///
  /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
  ///
  /// # Examples
  ///
  /// ```
  /// use std::collections::TryReserveError;
  ///
  /// # use sparse_set::SparseSet;
  ///
  /// fn process_data(data: &[u32]) -> Result<SparseSet<usize, u32>, TryReserveError> {
  ///   let mut output = SparseSet::new();
  ///
  ///   // Pre-reserve the memory, exiting if we can't.
  ///   output.try_reserve_exact(data.len())?;
  ///
  ///   // Now we know this can't OOM in the middle of our complex work.
  ///   for (index, value) in data.iter().cloned().enumerate() {
  ///     output.insert(index, value);
  ///   }  
  ///
  ///   Ok(output)
  /// }
  /// # process_data(&[1, 2, 3]).expect("why is the test harness OOMing on 12 bytes?");
  /// ```
  pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
    if self.needs_to_grow(self.dense_len, additional) {
      self.grow_exact(self.dense_len, additional)
    } else {
      Ok(())
    }
  }

  /// Returns an iterator over the sparse set.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// set.insert(2, 3);
  ///
  /// let mut iterator = set.values();
  ///
  /// assert_eq!(iterator.next(), Some(&1));
  /// assert_eq!(iterator.next(), Some(&2));
  /// assert_eq!(iterator.next(), Some(&3));
  /// assert_eq!(iterator.next(), None);
  /// ```
  pub fn values(&self) -> impl Iterator<Item = &T> {
    self.iter()
  }

  /// Returns an iterator that allows modifying each value.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// set.insert(2, 3);
  ///
  /// for elem in set.values_mut() {
  ///     *elem += 2;
  /// }
  ///
  /// assert!(set.values().eq(&[3, 4, 5]));
  /// ```
  pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
    self.as_mut_slice().iter_mut()
  }
}

impl<I, T, A: Allocator> SparseSet<I, T, A> {
  /// Tiny SparseSets are dumb. Skip to:
  /// - 8 if the element size is 1, because any heap allocators is likely
  ///   to round up a request of less than 8 bytes to at least 8 bytes.
  /// - 4 if elements are moderate-sized (<= 1 KiB).
  /// - 1 otherwise, to avoid wasting too much space for very short SparseSets.
  const MIN_NON_ZERO_CAP: usize = if mem::size_of::<T>() == 1 {
    8
  } else if mem::size_of::<T>() <= 1024 {
    4
  } else {
    1
  };

  /// Allocates a new block of memory enough to hold `capacity` number of elements and copies all existing elements from
  /// the old allocated memory.
  ///
  /// # Safety
  ///
  /// The caller must ensure that the capacity requested is larger than the current length.
  unsafe fn allocate_and_copy(
    &self,
    capacity: usize,
  ) -> Result<(NonNull<u8>, NonNull<u8>), TryReserveError> {
    debug_assert!(capacity >= self.sparse_len);
    let (layout, sparse_offset) =
      Self::layout(capacity).map_err(|_| TryReserveErrorKind::CapacityOverflow)?;

    alloc_guard(layout.size())?;

    let ptr = self
      .alloc
      .allocate(layout)
      .map_err(|_| {
        let error: TryReserveError = TryReserveErrorKind::AllocError {
          layout,
          non_exhaustive: (),
        }
        .into();
        error
      })?
      .as_non_null_ptr();

    if let Some((_, old_layout)) = self.current_memory() {
      unsafe {
        ptr::copy_nonoverlapping(self.as_ptr(), ptr.as_ptr().cast::<T>(), self.dense_len);
        ptr::copy_nonoverlapping(
          self.as_sparse_ptr(),
          ptr
            .as_ptr()
            .add(sparse_offset)
            .cast::<Option<NonZeroUsize>>(),
          self.sparse_len,
        );

        self.alloc.deallocate(self.dense_ptr, old_layout);
      }
    }

    Ok(unsafe { Self::retrieve_ptrs(ptr, sparse_offset) })
  }

  /// Allocates memory to in order for the set to hold `capacity` elements.
  #[cfg(not(no_global_oom_handling))]
  fn allocate_in(capacity: usize, alloc: A) -> Self {
    // Don't allocate here because `Drop` will not deallocate when `capacity` is 0.
    if capacity == 0 {
      Self::new_in(alloc)
    } else {
      // We avoid `unwrap_or_else` here because it bloats the amount of LLVM IR generated.
      let (layout, sparse_offset) = match Self::layout(capacity) {
        Ok(res) => res,
        Err(_) => capacity_overflow(),
      };

      match alloc_guard(layout.size()) {
        Ok(_) => {}
        Err(_) => capacity_overflow(),
      }

      let ptr = match alloc.allocate(layout) {
        Ok(ptr) => ptr.as_non_null_ptr(),
        Err(_) => alloc::handle_alloc_error(layout),
      };
      let (dense_ptr, sparse_ptr) = unsafe { Self::retrieve_ptrs(ptr, sparse_offset) };

      // Allocators currently return a `NonNull<[u8]>` whose len matches the size requested. If that ever
      // changes, the capacity here should change.
      Self {
        alloc,
        capacity,
        _marker: (PhantomData, PhantomData),
        dense_ptr,
        dense_len: 0,
        sparse_ptr,
        sparse_len: 0,
      }
    }
  }

  /// Extracts a mutable slice of the entire sparse set's buffer.
  ///
  /// The caller must ensure they do not cause the dense and sparse buffers to become out of sync.
  #[must_use]
  fn as_mut_slice(&mut self) -> &mut [T] {
    unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.dense_len) }
  }

  /// Returns an unsafe mutable pointer to the sparse set's buffer, or a dangling raw pointer valid for zero sized reads
  /// if the sparse set didn't allocate.
  ///
  /// The caller must ensure that the sparse set outlives the pointer this function returns, or else it will end up
  /// pointing to garbage. Modifying the sparse set may cause its buffer to be reallocated, which would also make any
  /// pointers to it invalid.
  ///
  /// The caller must also ensure they do not cause the dense and sparse buffers to become out of sync.
  #[must_use]
  fn as_mut_ptr(&mut self) -> *mut T {
    // We shadow the slice method of the same name to avoid going through `deref_mut`, which creates an intermediate
    // reference.
    self.dense_ptr.as_ptr() as *mut T
  }

  /// Returns a raw pointer to the sparse set's sparse buffer, or a dangling raw pointer valid for zero sized reads if
  /// the sparse set didn't allocate.
  ///
  /// The caller must ensure that the sparse set outlives the pointer this function returns, or else it will end up
  /// pointing to garbage. Modifying the sparse set may cause its buffer to be reallocated, which would also make any
  /// pointers to it invalid.
  ///
  /// The caller must also ensure that the memory the pointer (non-transitively) points to is never written to (except
  /// inside an `UnsafeCell`) using this pointer or any pointer derived from it. If you need to mutate the contents of
  /// the slice, use [`as_mut_ptr`].
  #[must_use]
  fn as_sparse_ptr(&self) -> *const Option<NonZeroUsize> {
    // We shadow the slice method of the same name to avoid going through `deref`, which creates an intermediate
    // reference.
    self.sparse_ptr.as_ptr() as *const Option<NonZeroUsize>
  }

  /// Returns an unsafe mutable pointer to the sparse set's sparse buffer, or a dangling raw pointer valid for zero
  /// sized reads if the sparse set didn't allocate.
  ///
  /// The caller must ensure that the sparse set outlives the pointer this function returns, or else it will end up
  /// pointing to garbage. Modifying the sparse set may cause its buffer to be reallocated, which would also make any
  /// pointers to it invalid.
  ///
  /// The caller must also ensure they do not cause the dense and sparse buffers to become out of sync.
  #[must_use]
  fn as_sparse_mut_ptr(&mut self) -> *mut Option<NonZeroUsize> {
    // We shadow the slice method of the same name to avoid going through `deref_mut`, which creates an intermediate
    // reference.
    self.sparse_ptr.as_ptr() as *mut Option<NonZeroUsize>
  }

  /// Extracts a slice containing the entire sparse set's sparse buffer.
  #[must_use]
  fn as_sparse_slice(&self) -> &[Option<NonZeroUsize>] {
    unsafe { slice::from_raw_parts(self.as_sparse_ptr(), self.sparse_len) }
  }

  /// Extracts a mutable slice of the entire sparse set's sparse buffer.
  ///
  /// The caller must ensure they do not cause the dense and sparse buffers to become out of sync.
  #[must_use]
  fn as_sparse_mut_slice(&mut self) -> &mut [Option<NonZeroUsize>] {
    unsafe { slice::from_raw_parts_mut(self.as_sparse_mut_ptr(), self.sparse_len) }
  }

  /// Returns a pointer to the currently allocated memory, if any, and its layout.
  #[must_use]
  fn current_memory(&self) -> Option<(NonNull<u8>, Layout)> {
    if self.capacity == 0 {
      None
    } else {
      // We have an allocated chunk of memory, so we can bypass runtime checks to get our current layout.
      unsafe {
        let layout = Self::layout(self.capacity).unwrap_unchecked().0;
        Some((self.dense_ptr, layout))
      }
    }
  }

  /// This method is usually instantiated many times. So we want it to be as small as possible, to improve compile times.
  /// But we also want as much of its contents to be statically computable as possible, to make the generated code run
  /// faster. Therefore, this method is carefully written so that all of the code that depends on `T` is within it,
  /// while as much of the code that doesn't depend on `T` as possible is in functions that are non-generic over `T`.
  fn grow_amortized(&mut self, len: usize, additional: usize) -> Result<(), TryReserveError> {
    // This is ensured by the calling contexts.
    debug_assert!(additional > 0);

    // Nothing we can really do about these checks, sadly.
    let required_capacity = len
      .checked_add(additional)
      .ok_or(TryReserveErrorKind::CapacityOverflow)?;

    // This guarantees exponential growth. The doubling cannot overflow because `cap <= isize::MAX` and the type of
    // `cap` is `usize`.
    let capacity = cmp::max(self.capacity * 2, required_capacity);
    let capacity = cmp::max(Self::MIN_NON_ZERO_CAP, capacity);

    let (dense_ptr, sparse_ptr) = unsafe { self.allocate_and_copy(capacity)? };
    self.set_ptrs_and_capacity(dense_ptr, sparse_ptr, capacity);
    Ok(())
  }

  /// The constraints on this method are much the same as those on `grow_amortized`, but this method is usually
  /// instantiated less often so it's less critical.
  fn grow_exact(&mut self, len: usize, additional: usize) -> Result<(), TryReserveError> {
    let capacity = len
      .checked_add(additional)
      .ok_or(TryReserveErrorKind::CapacityOverflow)?;

    let (dense_ptr, sparse_ptr) = unsafe { self.allocate_and_copy(capacity)? };
    self.set_ptrs_and_capacity(dense_ptr, sparse_ptr, capacity);
    Ok(())
  }

  /// Returns a layout necessary to allocate memory for the dense and sparse arrays.
  fn layout(capacity: usize) -> Result<(Layout, usize), LayoutError> {
    let dense_layout = Layout::array::<T>(capacity)?;
    let sparse_layout = Layout::array::<Option<NonZeroUsize>>(capacity)?;
    dense_layout.extend(sparse_layout)
  }

  /// Returns if the buffer needs to grow to fulfill the needed extra capacity.
  ///
  /// Mainly used to make inlining reserve-calls possible without inlining `grow`.
  #[must_use]
  fn needs_to_grow(&self, len: usize, additional: usize) -> bool {
    additional > self.capacity.wrapping_sub(len)
  }

  /// From a pointer to allocated memory using the correct layout, this returns the corresponding pointers to the dense
  /// and sparse inner layouts.
  #[must_use]
  unsafe fn retrieve_ptrs(ptr: NonNull<u8>, sparse_offset: usize) -> (NonNull<u8>, NonNull<u8>) {
    let sparse_ptr = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(sparse_offset)) };
    (ptr, sparse_ptr)
  }

  /// Updates internal pointers and capacity.
  fn set_ptrs_and_capacity(
    &mut self,
    dense_ptr: NonNull<u8>,
    sparse_ptr: NonNull<u8>,
    capacity: usize,
  ) {
    // Allocators currently return a `NonNull<[u8]>` whose len matches the size requested. If that ever changes,
    // the capacity here should change.
    self.capacity = capacity;
    self.dense_ptr = dense_ptr;
    self.sparse_ptr = sparse_ptr;
  }

  fn shrink(&mut self, capacity: usize) -> Result<(), TryReserveError> {
    debug_assert!(
      capacity <= self.capacity,
      "Tried to shrink to a larger capacity"
    );

    let (dense_ptr, sparse_ptr) = unsafe { self.allocate_and_copy(capacity)? };
    self.set_ptrs_and_capacity(dense_ptr, sparse_ptr, capacity);
    Ok(())
  }
}

impl<I: SparseSetIndex, T, A: Allocator> SparseSet<I, T, A> {
  /// Inserts the value at the given index, presented as a `usize` rather than `I`.
  pub(crate) fn insert_raw(&mut self, index: usize, value: T) {
    let len = self.len();

    match self.as_sparse_slice().get(index) {
      Some(Some(dense_index)) => {
        let dense_index: NonZeroUsize = *dense_index;
        unsafe { ptr::write(self.as_mut_ptr().add(dense_index.get() - 1), value) };
      }
      opt => {
        if opt.is_none() {
          self.resize_sparse(index + 1);
        }

        self.dense_len += 1;
        self.as_sparse_mut_slice()[index] = Some(NonZeroUsize::new(len + 1).unwrap());
        unsafe { ptr::write(self.as_mut_ptr().add(len), value) };
      }
    }
  }

  /// Resizes the sparse buffer in-place so that `len` is equal to `new_len`.
  ///
  /// `new_len` must be greater than or equal to `len`.
  #[cfg(not(no_global_oom_handling))]
  fn resize_sparse(&mut self, new_len: usize) {
    let len = self.sparse_len;
    debug_assert!(new_len >= len);

    let n = new_len - len;
    self.reserve(n);

    unsafe {
      let mut ptr = self.as_sparse_mut_ptr().add(self.sparse_len);

      // Write all elements.
      for _ in 0..n {
        ptr::write(ptr, None);
        ptr = ptr.add(1);
      }
    }

    self.sparse_len = new_len;
  }
}

impl<I: SparseSetIndex, T, A: Allocator> AsRef<SparseSet<I, T, A>> for SparseSet<I, T, A> {
  fn as_ref(&self) -> &SparseSet<I, T, A> {
    self
  }
}

impl<I: SparseSetIndex, T, A: Allocator> AsRef<[T]> for SparseSet<I, T, A> {
  fn as_ref(&self) -> &[T] {
    self
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T: Clone, A: Allocator + Clone> Clone for SparseSet<I, T, A> {
  fn clone(&self) -> Self {
    if self.capacity == 0 {
      return Self::new_in(self.alloc.clone());
    }

    let mut cloned_set = Self::with_capacity_in(self.sparse_len, self.alloc.clone());

    for (index, value) in self.as_slice().iter().enumerate() {
      let clone = ManuallyDrop::new(value.clone());
      unsafe { ptr::copy_nonoverlapping(&*clone, cloned_set.as_mut_ptr().add(index), 1) };
      cloned_set.dense_len += 1;
      cloned_set.sparse_len += 1;
    }

    unsafe {
      ptr::copy_nonoverlapping(
        self.as_sparse_ptr(),
        cloned_set.as_sparse_mut_ptr(),
        self.sparse_len,
      )
    };

    cloned_set
  }
}

impl<I, T> Default for SparseSet<I, T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<I, T, A: Allocator> Deref for SparseSet<I, T, A> {
  type Target = [T];

  fn deref(&self) -> &[T] {
    unsafe { slice::from_raw_parts(self.dense_ptr.as_ptr() as *const T, self.dense_len) }
  }
}

impl<I, T: fmt::Debug, A: Allocator> fmt::Debug for SparseSet<I, T, A> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt::Debug::fmt(&**self, f)
  }
}

unsafe impl<I, #[may_dangle] T, A: Allocator> Drop for SparseSet<I, T, A> {
  fn drop(&mut self) {
    unsafe {
      // Use drop for [T]. Use a raw slice to refer to the elements of the sparse set as weakest necessary type; Could
      // avoid questions of validity in certain cases
      ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
        self.as_mut_ptr(),
        self.dense_len,
      ))
    };

    if let Some((ptr, layout)) = self.current_memory() {
      unsafe { self.alloc.deallocate(ptr, layout) };
    }
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T, A: Allocator> Extend<(I, T)> for SparseSet<I, T, A> {
  fn extend<Iter: IntoIterator<Item = (I, T)>>(&mut self, iter: Iter) {
    for (index, value) in iter {
      self.insert(index, value);
    }
  }

  fn extend_reserve(&mut self, additional: usize) {
    self.reserve(additional);
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T> FromIterator<(I, T)> for SparseSet<I, T> {
  #[inline]
  fn from_iter<Iter: IntoIterator<Item = (I, T)>>(iter: Iter) -> Self {
    let mut set = SparseSet::new();
    set.extend(iter);
    set
  }
}

impl<I: SparseSetIndex, T, A: Allocator> Index<I> for SparseSet<I, T, A> {
  type Output = T;

  fn index(&self, index: I) -> &Self::Output {
    self.get(index).unwrap()
  }
}

impl<I: SparseSetIndex, T, A: Allocator> IndexMut<I> for SparseSet<I, T, A> {
  fn index_mut(&mut self, index: I) -> &mut Self::Output {
    self.get_mut(index).unwrap()
  }
}

impl<'a, I: SparseSetIndex, T, A: Allocator> IntoIterator for &'a SparseSet<I, T, A> {
  type Item = &'a T;
  type IntoIter = impl Iterator<Item = Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.values()
  }
}

impl<'a, I: SparseSetIndex, T, A: Allocator> IntoIterator for &'a mut SparseSet<I, T, A> {
  type Item = &'a mut T;
  type IntoIter = impl Iterator<Item = Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.values_mut()
  }
}

/// A type with this trait indicates it can be used as an index into a `SparseSet`.
pub trait SparseSetIndex: Into<usize> {}

impl SparseSetIndex for usize {}

/// We need to guarantee the following:
/// * We don't ever allocate `> isize::MAX` byte-size objects.
/// * We don't overflow `usize::MAX` and actually allocate too little.
///
/// On 64-bit we just need to check for overflow since trying to allocate `> isize::MAX` bytes will surely fail. On
/// 32-bit and 16-bit we need to add an extra guard for this in case we're running on a platform which can use all 4GB
/// in user-space, e.g., PAE or x32.
fn alloc_guard(alloc_size: usize) -> Result<(), TryReserveError> {
  if usize::BITS < 64 && alloc_size > isize::MAX as usize {
    Err(TryReserveErrorKind::CapacityOverflow.into())
  } else {
    Ok(())
  }
}

/// One central function responsible for reporting capacity overflows. This'll ensure that the code generation related
/// to these panics is minimal as there's only one location which panics rather than a bunch throughout the module.
#[cfg(not(no_global_oom_handling))]
fn capacity_overflow() -> ! {
  panic!("capacity overflow");
}

/// Central function for reserve error handling.
#[cfg(not(no_global_oom_handling))]
fn handle_reserve<T>(result: Result<T, TryReserveError>) -> T {
  match result.map_err(|error| error.kind()) {
    Err(TryReserveErrorKind::CapacityOverflow) => capacity_overflow(),
    Err(TryReserveErrorKind::AllocError { layout, .. }) => alloc::handle_alloc_error(layout),
    Ok(res) => res,
  }
}

#[cfg(test)]
pub mod test {
  use std::{cell::RefCell, rc::Rc};

  use coverage_helper::test;

  use super::*;

  #[derive(Clone)]
  struct Value(Rc<RefCell<u32>>);

  impl Drop for Value {
    fn drop(&mut self) {
      *self.0.borrow_mut() += 1;
    }
  }

  #[derive(Clone, Debug, Eq, PartialEq)]
  struct Zst;

  #[test]
  fn test_zst_is_zero() {
    assert_eq!(mem::size_of::<Zst>(), 0);
  }

  #[test]
  fn test_new() {
    let set: SparseSet<usize, usize> = SparseSet::new();
    assert!(set.is_empty());
    assert_eq!(set.capacity(), 0);
  }

  #[test]
  fn test_new_zst() {
    let set: SparseSet<usize, Zst> = SparseSet::new();
    assert!(set.is_empty());
    assert_eq!(set.capacity(), 0);
  }

  #[test]
  fn test_with_capacity() {
    let set: SparseSet<usize, usize> = SparseSet::with_capacity(10);
    assert_eq!(set.capacity(), 10);
  }

  #[test]
  fn test_with_capacity_zero() {
    let set: SparseSet<usize, usize> = SparseSet::with_capacity(0);
    assert_eq!(set.capacity(), 0);
  }

  #[should_panic]
  #[test]
  fn test_with_capacity_overflow() {
    let _: SparseSet<usize, usize> = SparseSet::with_capacity(usize::MAX);
  }

  #[test]
  fn test_with_capacity_zst() {
    let set: SparseSet<usize, Zst> = SparseSet::with_capacity(10);
    assert_eq!(set.capacity(), 10);
  }

  #[test]
  fn test_allocator() {
    let set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.allocator();
  }

  #[test]
  fn test_as_ptr() {
    let set: SparseSet<usize, usize> = SparseSet::with_capacity(10);
    assert_eq!(set.as_ptr(), set.as_slice().as_ptr());
  }

  #[test]
  fn test_clear() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    set.clear();

    assert!(set.is_empty());
  }

  #[test]
  fn test_clear_zst() {
    let mut set = SparseSet::new();
    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);
    set.clear();

    assert!(set.is_empty());
  }

  #[test]
  fn test_clear_drops() {
    let num_dropped = Rc::new(RefCell::new(0));
    let mut set = SparseSet::new();
    let value = Value(num_dropped.clone());
    set.insert(0, value.clone());
    set.insert(1, value.clone());
    set.insert(2, value.clone());
    set.clear();

    assert_eq!(*num_dropped.borrow(), 3);
  }

  #[test]
  fn test_contains() {
    let mut set = SparseSet::new();
    assert!(!set.contains(0));
    set.insert(0, 1);
    assert!(set.contains(0));
    let _ = set.remove(0);
    assert!(!set.contains(0));
  }

  #[test]
  fn test_contains_zst() {
    let mut set = SparseSet::new();
    assert!(!set.contains(0));
    set.insert(0, Zst);
    assert!(set.contains(0));
    let _ = set.remove(0);
    assert!(!set.contains(0));
  }

  #[test]
  fn test_get() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    assert_eq!(set.get(0), Some(&1));
    assert_eq!(set.get(2), Some(&3));
    assert_eq!(set.get(100), None);
  }

  #[test]
  fn test_get_zst() {
    let mut set = SparseSet::new();
    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);

    assert_eq!(set.get(0), Some(&Zst));
    assert_eq!(set.get(2), Some(&Zst));
    assert_eq!(set.get(100), None);
  }

  #[test]
  fn test_get_mut() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let value = set.get_mut(2);
    assert_eq!(value, Some(&mut 3));
    *value.unwrap() = 10;

    assert_eq!(set.get(2), Some(&10));
  }

  #[test]
  fn test_get_mut_zst() {
    let mut set = SparseSet::new();
    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);

    assert_eq!(set.get_mut(2), Some(&mut Zst));
  }

  #[test]
  fn test_insert_capacity_increases() {
    let mut set = SparseSet::with_capacity(1);
    set.insert(0, 1);
    assert_eq!(set.capacity(), 1);

    set.insert(1, 2);
    assert!(set.capacity() >= 2);

    assert_eq!(set.get(0), Some(&1));
    assert_eq!(set.get(1), Some(&2));
  }

  #[test]
  fn test_insert_capacity_increases_zst() {
    let mut set = SparseSet::with_capacity(1);
    set.insert(0, Zst);
    assert_eq!(set.capacity(), 1);

    set.insert(1, Zst);
    assert!(set.capacity() >= 2);

    assert_eq!(set.get(0), Some(&Zst));
    assert_eq!(set.get(1), Some(&Zst));
  }

  #[test]
  fn test_insert_len_increases() {
    let mut set = SparseSet::with_capacity(1);
    set.insert(0, 1);
    assert_eq!(set.len(), 1);

    set.insert(1, 2);
    assert_eq!(set.len(), 2);

    set.insert(100, 101);
    assert_eq!(set.len(), 3);
  }

  #[test]
  fn test_insert_len_increases_zst() {
    let mut set = SparseSet::with_capacity(1);
    set.insert(0, Zst);
    assert_eq!(set.len(), 1);

    set.insert(1, Zst);
    assert_eq!(set.len(), 2);

    set.insert(100, Zst);
    assert_eq!(set.len(), 3);
  }

  #[test]
  fn test_insert_overwrites() {
    let mut set = SparseSet::with_capacity(1);
    set.insert(0, 1);
    assert_eq!(set.get(0), Some(&1));

    set.insert(0, 2);
    assert_eq!(set.get(0), Some(&2));
  }

  #[test]
  fn test_is_empty() {
    let mut set = SparseSet::new();
    assert!(set.is_empty());

    set.insert(0, 1);
    assert!(!set.is_empty());

    let _ = set.remove(0);
    assert!(set.is_empty());
  }

  #[test]
  fn test_is_empty_zst() {
    let mut set = SparseSet::new();
    assert!(set.is_empty());

    set.insert(0, Zst);
    assert!(!set.is_empty());

    let _ = set.remove(0);
    assert!(set.is_empty());
  }

  #[test]
  fn test_remove_can_return_none() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    assert_eq!(set.remove(1), None);
    assert_eq!(set.remove(100), None);
  }

  #[test]
  fn test_remove_can_return_none_zst() {
    let mut set = SparseSet::new();
    set.insert(0, Zst);
    assert_eq!(set.remove(1), None);
    assert_eq!(set.remove(100), None);
  }

  #[test]
  fn test_remove_can_return_some() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    assert_eq!(set.remove(0), Some(1));
  }

  #[test]
  fn test_remove_can_return_some_zst() {
    let mut set = SparseSet::new();
    set.insert(0, Zst);
    assert_eq!(set.remove(0), Some(Zst));
  }

  #[test]
  fn test_remove_len_decreases() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    assert_eq!(set.len(), 2);
    assert_eq!(set.remove(0), Some(1));
    assert_eq!(set.len(), 1);
    assert_eq!(set.remove(0), None);
    assert_eq!(set.len(), 1);
  }

  #[test]
  fn test_remove_len_decreases_zst() {
    let mut set = SparseSet::new();
    set.insert(0, Zst);
    set.insert(1, Zst);
    assert_eq!(set.len(), 2);
    assert_eq!(set.remove(0), Some(Zst));
    assert_eq!(set.len(), 1);
    assert_eq!(set.remove(0), None);
    assert_eq!(set.len(), 1);
  }

  #[test]
  fn test_remove_swaps_with_last() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    assert!(set.values().eq(&[1, 2, 3]));

    let _ = set.remove(0);
    assert!(set.values().eq(&[3, 2]));
  }

  #[test]
  fn test_reserve() {
    let mut set = SparseSet::new();
    assert_eq!(set.capacity(), 0);

    set.reserve(3);
    let capacity = set.capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    set.reserve(1);
    assert_eq!(set.capacity(), capacity);
  }

  #[test]
  fn test_reserve_zst() {
    let mut set = SparseSet::new();
    assert_eq!(set.capacity(), 0);

    set.reserve(3);
    let capacity = set.capacity();
    assert!(capacity >= 2);

    set.insert(0, Zst);
    set.insert(1, Zst);

    set.reserve(1);
    assert_eq!(set.capacity(), capacity);
  }

  #[test]
  fn test_reserve_exact() {
    let mut set = SparseSet::new();
    assert_eq!(set.capacity(), 0);

    set.reserve_exact(3);
    let capacity = set.capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    set.reserve_exact(1);
    assert_eq!(set.capacity(), capacity);
  }

  #[test]
  fn test_reserve_exact_zst() {
    let mut set = SparseSet::new();
    assert_eq!(set.capacity(), 0);

    set.reserve_exact(3);
    let capacity = set.capacity();
    assert!(capacity >= 2);

    set.insert(0, Zst);
    set.insert(1, Zst);

    set.reserve_exact(1);
    assert_eq!(set.capacity(), capacity);
  }

  #[test]
  fn test_shrink_to_fit() {
    let mut set = SparseSet::with_capacity(3);
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    assert_eq!(set.capacity(), 3);
    let _ = set.remove(2);
    set.shrink_to_fit();
    assert_eq!(set.capacity(), 2);
  }

  #[test]
  fn test_shrink_to_fit_zst() {
    let mut set = SparseSet::with_capacity(3);
    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);

    assert_eq!(set.capacity(), 3);
    let _ = set.remove(2);
    set.shrink_to_fit();
    assert_eq!(set.capacity(), 2);
  }

  #[test]
  fn test_shrink_to_can_reduce() {
    let mut set = SparseSet::with_capacity(3);
    set.insert(0, 1);
    assert_eq!(set.capacity(), 3);
    set.shrink_to(1);
    assert_eq!(set.capacity(), 1);
  }

  #[test]
  fn test_shrink_to_can_reduce_zst() {
    let mut set = SparseSet::with_capacity(3);
    set.insert(0, Zst);
    assert_eq!(set.capacity(), 3);
    set.shrink_to(1);
    assert_eq!(set.capacity(), 1);
  }

  #[test]
  fn test_shrink_to_cannot_reduce() {
    let mut set = SparseSet::with_capacity(3);
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert_eq!(set.capacity(), 3);
    set.shrink_to(0);
    assert_eq!(set.capacity(), 3);
  }

  #[test]
  fn test_shrink_to_cannot_reduce_zst() {
    let mut set = SparseSet::with_capacity(3);
    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);
    assert_eq!(set.capacity(), 3);
    set.shrink_to(0);
    assert_eq!(set.capacity(), 3);
  }

  #[test]
  fn test_try_reserve_succeeds() {
    let mut set = SparseSet::new();
    assert_eq!(set.capacity(), 0);

    assert!(set.try_reserve(3).is_ok());
    let capacity = set.capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    assert!(set.try_reserve(1).is_ok());
    assert_eq!(set.capacity(), capacity);
  }

  #[test]
  fn test_values() {
    let mut set = SparseSet::new();
    assert!(set.values().eq(&[]));

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!(set.values().eq(&[1, 2, 3]));
  }

  #[test]
  fn test_values_zst() {
    let mut set = SparseSet::new();
    assert!(set.values().eq(&[]));

    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);
    assert!(set.values().eq(&[Zst, Zst, Zst]));
  }

  #[test]
  fn test_values_mut() {
    let mut set = SparseSet::new();
    assert!(set.values_mut().eq(&[]));

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!(set.values_mut().eq(&[1, 2, 3]));

    let value = set.values_mut().next().unwrap();
    *value = 100;

    assert_eq!(set.get(0), Some(&100));
  }

  #[test]
  fn test_values_mut_zst() {
    let mut set = SparseSet::new();
    assert!(set.values_mut().eq(&[]));

    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);
    assert!(set.values_mut().eq(&[Zst, Zst, Zst]));
  }

  #[test]
  fn test_as_ref() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let reference: &SparseSet<_, _> = set.as_ref();
    assert_eq!(reference.get(0), Some(&1));

    let reference: &[usize] = set.as_ref();
    assert_eq!(reference.get(0), Some(&1));
  }

  #[test]
  fn test_as_ref_zst() {
    let mut set = SparseSet::new();
    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);

    let reference: &SparseSet<_, _> = set.as_ref();
    assert_eq!(reference.get(0), Some(&Zst));

    let reference: &[Zst] = set.as_ref();
    assert_eq!(reference.get(0), Some(&Zst));
  }

  #[test]
  fn test_clone() {
    let mut set = SparseSet::with_capacity(3);
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let cloned_set = set.clone();
    assert!(set.values().eq(cloned_set.values()));
    assert_eq!(set.len(), cloned_set.len());
  }

  #[test]
  fn test_clone_zst() {
    let mut set = SparseSet::with_capacity(3);
    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);

    let cloned_set = set.clone();
    assert!(set.values().eq(cloned_set.values()));
    assert_eq!(set.len(), cloned_set.len());
  }

  #[test]
  fn test_clone_zero_capacity() {
    let set: SparseSet<usize, usize> = SparseSet::new();
    assert_eq!(set.capacity(), 0);

    let cloned_set = set.clone();
    assert!(set.values().eq(cloned_set.values()));
    assert_eq!(set.len(), cloned_set.len());
  }

  #[test]
  fn test_clone_drops_are_separate() {
    let num_dropped = Rc::new(RefCell::new(0));

    {
      let mut set = SparseSet::new();
      let value = Value(num_dropped.clone());
      set.insert(0, value.clone());
      set.insert(1, value.clone());
      set.insert(2, value);

      let _cloned_set = set.clone();
    }

    assert_eq!(*num_dropped.borrow(), 6);
  }

  #[test]
  fn test_debug() {
    let mut set = SparseSet::new();
    assert_eq!(format!("{:?}", set), "[]");

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert_eq!(format!("{:?}", set), "[1, 2, 3]");
  }

  #[test]
  fn test_debug_zst() {
    let mut set = SparseSet::new();
    assert_eq!(format!("{:?}", set), "[]");

    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);
    assert_eq!(format!("{:?}", set), "[Zst, Zst, Zst]");
  }

  #[test]
  fn test_default() {
    let set: SparseSet<usize, usize> = SparseSet::default();
    assert!(set.is_empty());
    assert_eq!(set.capacity(), 0);
  }

  #[test]
  fn test_default_zst() {
    let set: SparseSet<usize, Zst> = SparseSet::default();
    assert!(set.is_empty());
    assert_eq!(set.capacity(), 0);
  }

  #[test]
  fn test_drop() {
    let num_dropped = Rc::new(RefCell::new(0));

    {
      let mut set = SparseSet::new();
      let value = Value(num_dropped.clone());
      set.insert(0, value.clone());
      set.insert(1, value.clone());
      set.insert(2, value);
    }

    assert_eq!(*num_dropped.borrow(), 3);
  }

  #[test]
  fn test_extend() {
    let mut set = SparseSet::new();
    set.extend([(0, 1), (1, 2), (2, 3)]);
    assert!(set.values().eq(&[1, 2, 3]));
  }

  #[test]
  fn test_extend_zst() {
    let mut set = SparseSet::new();
    set.extend([(0, Zst), (1, Zst), (2, Zst)]);
    assert!(set.values().eq(&[Zst, Zst, Zst]));
  }

  #[test]
  fn test_from_iterator() {
    let set = SparseSet::from_iter([(0, 1), (1, 2), (2, 3)]);
    assert!(set.values().eq(&[1, 2, 3]));
  }

  #[test]
  fn test_from_iterator_zst() {
    let set = SparseSet::from_iter([(0, Zst), (1, Zst), (2, Zst)]);
    assert!(set.values().eq(&[Zst, Zst, Zst]));
  }

  #[test]
  fn test_index() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    assert_eq!(set[0], 1);
    assert_eq!(set[2], 3);
  }

  #[test]
  fn test_index_zst() {
    let mut set = SparseSet::new();
    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);

    assert_eq!(set[0], Zst);
    assert_eq!(set[2], Zst);
  }

  #[should_panic]
  #[test]
  fn test_index_panics() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let _ = &set[100];
  }

  #[test]
  fn test_index_mut() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let value = &mut set[2];
    assert_eq!(value, &mut 3);
    *value = 10;

    assert_eq!(set[2], 10);
  }

  #[test]
  fn test_index_mut_zst() {
    let mut set = SparseSet::new();
    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);

    let value = &mut set[2];
    assert_eq!(value, &mut Zst);
    *value = Zst;

    assert_eq!(set[2], Zst);
  }

  #[should_panic]
  #[test]
  fn test_index_mut_panics() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let _ = &mut set[100];
  }

  #[test]
  fn test_into_iterator() {
    let mut set = SparseSet::new();
    assert!((&set).into_iter().eq(&[]));

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!((&set).into_iter().eq(&[1, 2, 3]));
  }

  #[test]
  fn test_into_iterator_zst() {
    let mut set = SparseSet::new();
    assert!((&set).into_iter().eq(&[]));

    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);
    assert!((&set).into_iter().eq(&[Zst, Zst, Zst]));
  }

  #[test]
  fn test_into_iterator_mut() {
    let mut set = SparseSet::new();
    assert!(set.into_iter().eq(&[]));

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!((&mut set).into_iter().eq(&[1, 2, 3]));

    let value = set.values_mut().next().unwrap();
    *value = 100;

    assert_eq!((&mut set).get(0), Some(&100));
  }

  #[test]
  fn test_into_iterator_mut_zst() {
    let mut set = SparseSet::new();
    assert!((&mut set).into_iter().eq(&[]));

    set.insert(0, Zst);
    set.insert(1, Zst);
    set.insert(2, Zst);
    assert!((&mut set).into_iter().eq(&[Zst, Zst, Zst]));
  }
}
