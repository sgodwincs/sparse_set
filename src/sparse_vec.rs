//! A sparsely populated vector, written `SparseVec<I, T>`, where `I` is the index type and `T` is the value type.
//!
//! `I` must implement `SparseSetIndex`, in particular, it must be able to be converted to a `usize` index.

#![allow(unsafe_code)]

use std::{
  alloc::{Allocator, Global},
  collections::TryReserveError,
  fmt,
  hash::{Hash, Hasher},
  marker::PhantomData,
  ops::{Deref, DerefMut, Index, IndexMut},
};

use crate::SparseSetIndex;

/// A sparsely populated vector, written `SparseVec<I, T>`, where `I` is the index type and `T` is the value type.
///
/// For operation complexity notes, *n* is the number of values in the sparse vec and *m* is the value of the largest
/// index in the sparse vec. Note that *m* will always be at least as large as *n*.
pub struct SparseVec<I, T, A: Allocator = Global> {
  values: Vec<Option<T>, A>,
  _marker: PhantomData<I>,
}

impl<I, T> SparseVec<I, T> {
  /// Constructs a new, empty `SparseVec<I, T>`.
  ///
  /// The sparse vec will not allocate until elements are inserted into it.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// # #[allow(unused_mut)]
  /// let mut vec: SparseVec<usize, u32> = SparseVec::new();
  /// ```
  pub fn new() -> Self {
    SparseVec::new_in(Global)
  }

  /// Constructs a new, empty `SparseVec<I, T>` with the specified capacity.
  ///
  /// The sparse vec will be able to hold exactly `capacity` elements without reallocating. If `capacity` is 0, the
  /// sparse vec will not allocate.
  ///
  /// It is important to note that although the returned sparse vec has the *capacity* specified, the sparse vec will
  /// have a zero *length*.
  ///
  /// # Panics
  ///
  /// Panics if the new capacity exceeds `isize::MAX` bytes.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::with_capacity(10);
  ///
  /// // The sparse vec contains no items, even though it has capacity for more.
  /// assert_eq!(vec.len(), 0);
  /// assert_eq!(vec.capacity(), 10);
  ///
  /// // These are all done without reallocating...
  /// for i in 0..10 {
  ///   vec.insert(i, i);
  /// }
  ///
  /// assert_eq!(vec.len(), 10);
  /// assert_eq!(vec.capacity(), 10);
  ///
  /// // ...but this will make the sparse vec reallocate.
  /// vec.insert(10, 10);
  /// vec.insert(11, 11);
  /// assert_eq!(vec.len(), 12);
  /// assert!(vec.capacity() >= 12);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  #[must_use]
  pub fn with_capacity(capacity: usize) -> Self {
    SparseVec::with_capacity_in(capacity, Global)
  }
}

impl<I, T, A: Allocator> SparseVec<I, T, A> {
  /// Constructs a new, empty `SparseVec<I, T, A>`.
  ///
  /// The sparse vec will not allocate until elements are pushed onto it.
  ///
  /// # Examples
  ///
  /// ```
  /// #![feature(allocator_api)]
  ///
  /// use std::alloc::System;
  /// #
  /// # use sparse_set::SparseVec;
  ///
  /// # #[allow(unused_mut)]
  /// let mut vec: SparseVec<usize, u32, _> = SparseVec::new_in(System);
  /// ```
  #[must_use]
  pub fn new_in(alloc: A) -> Self {
    Self {
      values: Vec::new_in(alloc),
      _marker: PhantomData,
    }
  }

  /// Constructs a new, empty `SparseVec<I, T, A>` with the specified capacity with the provided allocator.
  ///
  /// The sparse vec will be able to hold exactly `capacity` elements without reallocating. If `capacity` is 0, the
  /// sparse vec will not allocate.
  ///
  /// It is important to note that although the returned sparse vec has the *capacity* specified, the sparse vec will
  /// have a zero *length*.
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
  /// # use sparse_set::SparseVec;
  ///
  /// let mut vec = SparseVec::with_capacity_in(10, System);
  ///
  /// // The sparse vec contains no items, even though it has capacity for more
  /// assert_eq!(vec.len(), 0);
  /// assert_eq!(vec.capacity(), 10);
  ///
  /// // These are all done without reallocating...
  /// for i in 0..10 {
  ///   vec.insert(i, i);
  /// }
  ///
  /// assert_eq!(vec.len(), 10);
  /// assert_eq!(vec.capacity(), 10);
  ///
  /// // ...but this will make the sparse vec reallocate.
  /// vec.insert(10, 10);
  /// assert_eq!(vec.len(), 11);
  /// assert!(vec.capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
    Self {
      values: Vec::with_capacity_in(capacity, alloc),
      _marker: PhantomData,
    }
  }
}

impl<I, T, A: Allocator> SparseVec<I, T, A> {
  /// Returns a reference to the underlying allocator.
  #[must_use]
  pub fn allocator(&self) -> &A {
    self.values.allocator()
  }

  /// Extracts a slice containing the entire underlying buffer.
  #[must_use]
  pub fn as_slice(&self) -> &[Option<T>] {
    &self.values
  }

  /// Extracts a mutable slice of the entire underlying buffer.
  #[must_use]
  pub fn as_mut_slice(&mut self) -> &mut [Option<T>] {
    &mut self.values
  }

  /// Returns a raw pointer to the buffer, or a dangling raw pointer valid for zero sized reads if the sparse vec didn't
  /// allocate.
  ///
  /// The caller must ensure that the sparse vec outlives the pointer this function returns, or else it will end up
  /// pointing to garbage. Modifying the sparse vec may cause its buffer to be reallocated, which would also make any
  /// pointers to it invalid.
  ///
  /// The caller must also ensure that the memory the pointer (non-transitively) points to is never written to (except
  /// inside an `UnsafeCell`) using this pointer or any pointer derived from it.
  #[must_use]
  pub fn as_ptr(&self) -> *const Option<T> {
    self.values.as_ptr()
  }

  /// Returns an unsafe mutable pointer to the sparse vec's buffer.
  ///
  /// The caller must ensure that the sparse vec outlives the pointer this function returns, or else it will end up
  /// pointing to garbage. Modifying the sparse vec may cause its buffer to be reallocated, which would also make any
  /// pointers to it invalid.
  ///
  /// Swapping elements in the mutable slice changes which indices point to which values.
  #[must_use]
  pub fn as_mut_ptr(&mut self) -> *mut Option<T> {
    self.values.as_mut_ptr()
  }

  /// Returns the number of elements the sparse vec can hold without reallocating.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let vec: SparseVec<usize, i32> = SparseVec::with_capacity(10);
  /// assert_eq!(vec.capacity(), 10);
  /// ```
  #[must_use]
  pub fn capacity(&self) -> usize {
    self.values.capacity()
  }

  /// Clears the sparse vec, removing all values.
  ///
  /// Note that this method has no effect on the allocated capacity of the sparse vec.
  /// # Examples
  ///
  /// This operation is *O*(*m*).
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  ///
  /// vec.insert(0, 1);
  /// vec.insert(1, 2);
  /// vec.insert(2, 3);
  ///
  /// vec.clear();
  ///
  /// assert!(vec.is_empty());
  /// ```
  pub fn clear(&mut self) {
    self.values.clear();
  }

  /// Returns `true` if the sparse vec contains no elements.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  /// assert!(vec.is_empty());
  ///
  /// vec.insert(0, 1);
  /// assert!(!vec.is_empty());
  /// ```
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Returns an iterator over the sparse vec's values.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// Consuming the iterator is an *O*(*m*) operation.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  /// vec.insert(0, 1);
  /// vec.insert(1, 2);
  /// vec.insert(2, 3);
  ///
  /// let mut iterator = vec.iter();
  ///
  /// assert_eq!(iterator.next(), Some(&Some(1)));
  /// assert_eq!(iterator.next(), Some(&Some(2)));
  /// assert_eq!(iterator.next(), Some(&Some(3)));
  /// ```
  pub fn iter(&self) -> impl Iterator<Item = &Option<T>> {
    self.values.iter()
  }

  /// Returns an iterator that allows modifying each value.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  /// vec.insert(0, 1);
  /// vec.insert(1, 2);
  /// vec.insert(2, 3);
  ///
  /// for elem in vec.iter_mut() {
  ///     *elem = elem.map(|value| value + 2);
  /// }
  ///
  /// assert!(vec.iter().eq(&[Some(3), Some(4), Some(5)]));
  /// ```
  #[must_use]
  pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Option<T>> {
    self.values.iter_mut()
  }

  /// Returns the number of elements in the sparse vec, also referred to as its 'len'.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  /// vec.insert(0, 1);
  /// vec.insert(1, 2);
  /// vec.insert(2, 3);
  ///
  /// assert_eq!(vec.len(), 3);
  /// ```
  #[must_use]
  pub fn len(&self) -> usize {
    self.values.len()
  }

  /// Reserves capacity for at least `additional` more elements to be inserted in the given `SparseVec<I, T>`.
  ///
  /// The collection may reserve more space to avoid frequent reallocations. After calling `reserve`, the capacity will
  /// be greater than or equal to `self.len() + additional`. Does nothing if capacity is already sufficient.
  ///
  /// # Panics
  ///
  /// Panics if the new capacity exceeds `isize::MAX` bytes.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  /// vec.insert(0, 1);
  /// vec.reserve(10);
  /// assert!(vec.capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve(&mut self, additional: usize) {
    self.values.reserve(additional);
  }

  /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the given
  /// `SparseSet<I, T>`'.
  ///
  /// After calling `reserve_exact`, the capacity will be greater than or equal to `self.len() + additional`. Does
  /// nothing if the capacity is already sufficient.
  ///
  /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not be relied
  /// upon to be precisely minimal. Prefer [`reserve`] if future insertions are expected.
  ///
  /// [`reserve`]: SparseVec::reserve
  ///
  /// # Panics
  ///
  /// Panics if the new capacity exceeds `isize::MAX` bytes.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  /// vec.insert(0, 1);
  /// vec.reserve_exact(10);
  /// assert!(vec.capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_exact(&mut self, additional: usize) {
    self.values.reserve_exact(additional);
  }

  /// Tries to reserve capacity for at least `additional` more elements to be inserted in the given `SparseVec<I, T>`.
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
  /// # use sparse_set::SparseVec;
  ///
  /// fn process_data(data: &[u32]) -> Result<SparseVec<usize, u32>, TryReserveError> {
  ///   let mut output = SparseVec::new();
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
    self.values.try_reserve(additional)
  }

  /// Tries to reserve the minimum capacity for exactly `additional` elements to be inserted in the given
  /// `SparseVec<T>`.
  ///
  /// After calling `try_reserve_exact`, capacity will be greater than or equal to `self.len() + additional` if it
  /// returns `Ok(())`. Does nothing if the capacity is already sufficient.
  ///
  /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not be relied
  /// upon to be precisely minimal. Prefer [`try_reserve`] if future insertions are expected.
  ///
  /// [`try_reserve`]: SparseVec::try_reserve
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
  /// # use sparse_set::SparseVec;
  ///
  /// fn process_data(data: &[u32]) -> Result<SparseVec<usize, u32>, TryReserveError> {
  ///   let mut output = SparseVec::new();
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
    self.values.try_reserve_exact(additional)
  }

  /// Shrinks the capacity of the sparse vec as much as possible.
  ///
  /// It will drop down as close as possible to the length but the allocator may still inform the sparse vec that
  /// there is space for a few more elements.
  ///
  /// This operation is *O*(*m*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::with_capacity(10);
  /// vec.insert(0, 1);
  /// vec.insert(1, 2);
  /// assert_eq!(vec.capacity(), 10);
  ///
  /// vec.shrink_to_fit();
  /// assert!(vec.capacity() >= 2);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to_fit(&mut self) {
    self.values.truncate(self.max_index());
    self.values.shrink_to_fit()
  }

  /// Shrinks the capacity of the sparse vec with a lower bound.
  ///
  /// This will also reduce `len` as any empty indices after the maximum index will be removed.
  ///
  /// The capacity will remain at least as large as both the length and the supplied value.
  ///
  /// If the current capacity is less than the lower limit, this is a no-op.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::with_capacity(10);
  /// vec.insert(0, 1);
  /// vec.insert(1, 2);
  /// assert_eq!(vec.capacity(), 10);
  /// vec.shrink_to(4);
  /// assert!(vec.capacity() >= 4);
  /// vec.shrink_to(0);
  /// assert!(vec.capacity() >= 2);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to(&mut self, min_capacity: usize) {
    let len = self.max_index();

    if min_capacity < len {
      self.values.truncate(len);
    }

    self.values.shrink_to(min_capacity);
  }

  /// Returns the largest index in the sparse vec, or None if it is empty.
  ///
  /// This operation is *O*(*m*).
  #[must_use]
  fn max_index(&self) -> usize {
    for (index, value) in self.values.iter().rev().enumerate() {
      if value.is_some() {
        return self.values.len() - index;
      }
    }

    0
  }
}

impl<I: SparseSetIndex, T, A: Allocator> SparseVec<I, T, A> {
  /// Returns `true` if the sparse vec contains an element at the given index.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  ///
  /// vec.insert(0, 1);
  /// vec.insert(1, 2);
  /// vec.insert(2, 3);
  ///
  /// assert!(vec.contains(0));
  /// assert!(!vec.contains(100));
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
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  ///
  /// vec.insert(0, 1);
  /// vec.insert(1, 2);
  /// vec.insert(2, 3);
  /// assert_eq!(Some(&2), vec.get(1));
  /// assert_eq!(None, vec.get(3));
  ///
  /// vec.remove(1);
  /// assert_eq!(None, vec.get(1));
  /// ```
  #[must_use]
  pub fn get(&self, index: I) -> Option<&T> {
    self
      .values
      .get(index.into())
      .and_then(|inner| inner.as_ref())
  }

  /// Returns a mutable reference to an element pointed to by the index, if it exists.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  ///
  /// vec.insert(0, 1);
  /// vec.insert(1, 2);
  /// vec.insert(2, 3);
  ///
  /// if let Some(elem) = vec.get_mut(1) {
  ///   *elem = 42;
  /// }
  ///
  /// assert!(vec.iter().eq(&[Some(1), Some(42), Some(3)]));
  /// ```
  #[must_use]
  pub fn get_mut(&mut self, index: I) -> Option<&mut T> {
    self
      .values
      .get_mut(index.into())
      .and_then(|inner| inner.as_mut())
  }

  /// Inserts an element at position `index` within the sparse vec.
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
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  ///
  /// vec.insert(0, 1);
  /// vec.insert(1, 4);
  /// vec.insert(2, 2);
  /// vec.insert(3, 3);
  ///
  /// assert!(vec.iter().eq(&[Some(1), Some(4), Some(2), Some(3)]));
  /// vec.insert(5, 5);
  /// assert!(vec.iter().eq(&[Some(1), Some(4), Some(2), Some(3), None, Some(5)]));
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn insert(&mut self, index: I, value: T) {
    let index = index.into();

    if index >= self.len() {
      self.values.resize_with(index + 1, || None);
    }

    *unsafe { self.values.get_unchecked_mut(index) } = Some(value);
  }

  /// Removes and returns the element at position `index` within the sparse vec, if it exists.
  ///
  /// This does not change the length of the sparse vec as the value is replaced with `None`.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseVec;
  /// #
  /// let mut vec = SparseVec::new();
  /// vec.insert(0, 1);
  /// vec.insert(1, 2);
  /// vec.insert(2, 3);
  ///
  /// assert_eq!(vec.remove(1), Some(2));
  /// assert!(vec.iter().eq(&[Some(1), None, Some(3)]));
  /// ```
  #[must_use]
  pub fn remove(&mut self, index: I) -> Option<T> {
    let index = index.into();
    self.values.get_mut(index).and_then(|opt| opt.take())
  }
}

impl<I, T, A: Allocator> AsRef<SparseVec<I, T, A>> for SparseVec<I, T, A> {
  fn as_ref(&self) -> &Self {
    self
  }
}

impl<I, T, A: Allocator> AsMut<SparseVec<I, T, A>> for SparseVec<I, T, A> {
  fn as_mut(&mut self) -> &mut Self {
    self
  }
}

impl<I, T, A: Allocator> AsRef<[Option<T>]> for SparseVec<I, T, A> {
  fn as_ref(&self) -> &[Option<T>] {
    &self.values
  }
}

impl<I, T, A: Allocator> AsMut<[Option<T>]> for SparseVec<I, T, A> {
  fn as_mut(&mut self) -> &mut [Option<T>] {
    &mut self.values
  }
}

impl<I, T: Clone, A: Allocator + Clone> Clone for SparseVec<I, T, A> {
  fn clone(&self) -> Self {
    SparseVec {
      values: self.values.clone(),
      _marker: PhantomData,
    }
  }
}
impl<I, T> Default for SparseVec<I, T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<I, T, A: Allocator> Deref for SparseVec<I, T, A> {
  type Target = [Option<T>];

  fn deref(&self) -> &[Option<T>] {
    &*self.values
  }
}

impl<I, T, A: Allocator> DerefMut for SparseVec<I, T, A> {
  fn deref_mut(&mut self) -> &mut [Option<T>] {
    &mut *self.values
  }
}

impl<I, T: fmt::Debug, A: Allocator> fmt::Debug for SparseVec<I, T, A> {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.values.fmt(formatter)
  }
}

#[cfg(not(no_global_oom_handling))]
impl<'a, I: SparseSetIndex, T: Copy + 'a, A: Allocator + 'a> Extend<(I, &'a T)>
  for SparseVec<I, T, A>
{
  fn extend<Iter: IntoIterator<Item = (I, &'a T)>>(&mut self, iter: Iter) {
    for (index, &value) in iter {
      self.insert(index, value);
    }
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T, A: Allocator> Extend<(I, T)> for SparseVec<I, T, A> {
  fn extend<Iter: IntoIterator<Item = (I, T)>>(&mut self, iter: Iter) {
    for (index, value) in iter {
      self.insert(index, value);
    }
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T, const N: usize> From<[(I, T); N]> for SparseVec<I, T> {
  fn from(slice: [(I, T); N]) -> Self {
    let mut vec = SparseVec::with_capacity(slice.len());

    for (index, value) in slice {
      vec.insert(index, value);
    }

    vec
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T> FromIterator<(I, T)> for SparseVec<I, T> {
  fn from_iter<Iter: IntoIterator<Item = (I, T)>>(iter: Iter) -> Self {
    let iter = iter.into_iter();
    let capacity = if let Some(size_hint) = iter.size_hint().1 {
      size_hint
    } else {
      iter.size_hint().0
    };
    let mut vec = SparseVec::with_capacity(capacity);

    for (index, value) in iter {
      vec.insert(index, value);
    }

    vec
  }
}

impl<I, T: Hash, A: Allocator> Hash for SparseVec<I, T, A> {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.values.hash(state);
  }
}

impl<I: SparseSetIndex, T, A: Allocator> Index<I> for SparseVec<I, T, A> {
  type Output = T;

  fn index(&self, index: I) -> &Self::Output {
    self.get(index).unwrap()
  }
}

impl<I: SparseSetIndex, T, A: Allocator> IndexMut<I> for SparseVec<I, T, A> {
  fn index_mut(&mut self, index: I) -> &mut Self::Output {
    self.get_mut(index).unwrap()
  }
}

impl<I, T, A: Allocator> IntoIterator for SparseVec<I, T, A> {
  type Item = Option<T>;
  type IntoIter = impl Iterator<Item = Self::Item>;

  #[inline]
  fn into_iter(self) -> Self::IntoIter {
    self.values.into_iter()
  }
}

impl<'a, I, T, A: Allocator> IntoIterator for &'a SparseVec<I, T, A> {
  type Item = &'a Option<T>;
  type IntoIter = impl Iterator<Item = Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

impl<'a, I, T, A: Allocator> IntoIterator for &'a mut SparseVec<I, T, A> {
  type Item = &'a mut Option<T>;
  type IntoIter = impl Iterator<Item = Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.iter_mut()
  }
}

impl<I, T: PartialEq, A: Allocator> PartialEq for SparseVec<I, T, A> {
  fn eq(&self, other: &Self) -> bool {
    self.values == other.values
  }
}

impl<I, T: Eq, A: Allocator> Eq for SparseVec<I, T, A> {}

#[cfg(test)]
mod test {
  use std::{cell::RefCell, collections::hash_map::DefaultHasher, rc::Rc};

  use coverage_helper::test;

  use super::*;

  #[derive(Clone)]
  struct Value(Rc<RefCell<u32>>);

  impl Drop for Value {
    fn drop(&mut self) {
      *self.0.borrow_mut() += 1;
    }
  }

  #[test]
  fn test_new() {
    let set: SparseVec<usize, usize> = SparseVec::new();
    assert!(set.is_empty());
    assert_eq!(set.capacity(), 0);
  }

  #[test]
  fn test_with_capacity() {
    let set: SparseVec<usize, usize> = SparseVec::with_capacity(10);
    assert_eq!(set.capacity(), 10);
  }

  #[test]
  fn test_with_capacity_zero() {
    let set: SparseVec<usize, usize> = SparseVec::with_capacity(0);
    assert_eq!(set.capacity(), 0);
  }

  #[should_panic]
  #[test]
  fn test_with_capacity_overflow() {
    let _: SparseVec<usize, usize> = SparseVec::with_capacity(usize::MAX);
  }

  #[test]
  fn test_allocator() {
    let set: SparseVec<usize, usize> = SparseVec::new();
    let _ = set.allocator();
  }

  #[test]
  fn test_as_slice() {
    let mut set: SparseVec<usize, usize> = SparseVec::new();
    set.insert(0, 1);
    assert_eq!(set.as_slice(), &[Some(1)]);
  }

  #[test]
  fn test_as_mut_slice() {
    let mut set: SparseVec<usize, usize> = SparseVec::new();
    set.insert(0, 1);
    assert_eq!(set.as_mut_slice(), &mut [Some(1)]);
  }

  #[test]
  fn test_as_ptr() {
    let set: SparseVec<usize, usize> = SparseVec::with_capacity(10);
    assert_eq!(set.as_ptr(), set.as_slice().as_ptr());
  }

  #[test]
  fn test_as_mut_ptr() {
    let mut set: SparseVec<usize, usize> = SparseVec::with_capacity(10);
    assert_eq!(set.as_mut_ptr(), set.as_mut_slice().as_mut_ptr());
  }

  #[test]
  fn test_clear() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    set.clear();

    assert!(set.is_empty());
  }

  #[test]
  fn test_contains() {
    let mut set = SparseVec::new();
    assert!(!set.contains(0));
    set.insert(0, 1);
    assert!(set.contains(0));
    let _ = set.remove(0);
    assert!(!set.contains(0));
  }

  #[test]
  fn test_get() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    assert_eq!(set.get(0), Some(&1));
    assert_eq!(set.get(2), Some(&3));
    assert_eq!(set.get(100), None);
  }

  #[test]
  fn test_get_mut() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let value = set.get_mut(2);
    assert_eq!(value, Some(&mut 3));
    *value.unwrap() = 10;

    assert_eq!(set.get(2), Some(&10));
  }

  #[test]
  fn test_insert_capacity_increases() {
    let mut set = SparseVec::with_capacity(1);
    set.insert(0, 1);
    assert_eq!(set.capacity(), 1);

    set.insert(1, 2);
    assert!(set.capacity() >= 2);

    assert_eq!(set.get(0), Some(&1));
    assert_eq!(set.get(1), Some(&2));
  }

  #[test]
  fn test_insert_len_increases() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    assert_eq!(set.len(), 1);

    set.insert(1, 2);
    assert_eq!(set.len(), 2);

    set.insert(100, 101);
    assert_eq!(set.len(), 101);
  }

  #[test]
  fn test_insert_overwrites() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    assert_eq!(set.get(0), Some(&1));

    set.insert(0, 2);
    assert_eq!(set.get(0), Some(&2));
  }

  #[test]
  fn test_is_empty() {
    let mut set = SparseVec::new();
    assert!(set.is_empty());

    set.insert(0, 1);
    assert!(!set.is_empty());

    let _ = set.remove(0);
    set.shrink_to_fit();
    assert!(set.is_empty());
  }

  #[test]
  fn test_len() {
    let mut set = SparseVec::new();
    assert_eq!(set.len(), 0);

    set.insert(0, 1);
    assert_eq!(set.len(), 1);
  }

  #[test]
  fn test_remove_can_return_none() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    assert_eq!(set.remove(1), None);
    assert_eq!(set.remove(100), None);
  }

  #[test]
  fn test_remove_can_return_some() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    assert_eq!(set.remove(0), Some(1));
  }

  #[test]
  fn test_remove_len_cannot_decreases() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);

    assert_eq!(set.len(), 2);
    assert_eq!(set.remove(0), Some(1));
    assert_eq!(set.len(), 2);
    assert_eq!(set.remove(0), None);
    assert_eq!(set.len(), 2);
  }

  #[test]
  fn test_reserve() {
    let mut set = SparseVec::new();
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
  fn test_reserve_exact() {
    let mut set = SparseVec::new();
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
  fn test_shrink_to_fit() {
    let mut set = SparseVec::with_capacity(3);
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    assert_eq!(set.capacity(), 3);
    let _ = set.remove(2);
    set.shrink_to_fit();
    assert_eq!(set.capacity(), 2);
  }

  #[test]
  fn test_shrink_to_fit_max_index_zero() {
    let mut set: SparseVec<usize, usize> = SparseVec::with_capacity(3);
    assert_eq!(set.capacity(), 3);
    assert_eq!(set.len(), 0);
    set.shrink_to_fit();
    assert_eq!(set.capacity(), 0);
    assert_eq!(set.len(), 0);
  }

  #[test]
  fn test_shrink_to_can_reduce() {
    let mut set = SparseVec::with_capacity(3);
    set.insert(0, 1);
    assert_eq!(set.capacity(), 3);
    set.shrink_to(1);
    assert_eq!(set.capacity(), 1);
  }

  #[test]
  fn test_shrink_to_cannot_reduce() {
    let mut set = SparseVec::with_capacity(3);
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert_eq!(set.capacity(), 3);
    set.shrink_to(0);
    assert_eq!(set.capacity(), 3);
  }

  #[test]
  fn test_shrink_to_max_index_zero() {
    let mut set: SparseVec<usize, usize> = SparseVec::with_capacity(3);
    assert_eq!(set.capacity(), 3);
    assert_eq!(set.len(), 0);
    set.shrink_to(0);
    assert_eq!(set.capacity(), 0);
    assert_eq!(set.len(), 0);
  }

  #[test]
  fn test_try_reserve_succeeds() {
    let mut set = SparseVec::new();
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
  fn test_try_reserve_exact_succeeds() {
    let mut set = SparseVec::new();
    assert_eq!(set.capacity(), 0);

    assert!(set.try_reserve_exact(3).is_ok());
    let capacity = set.capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    assert!(set.try_reserve_exact(1).is_ok());
    assert_eq!(set.capacity(), capacity);
  }

  #[test]
  fn test_values() {
    let mut set = SparseVec::new();
    assert!(set.iter().eq(&[]));

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!(set.iter().eq(&[Some(1), Some(2), Some(3)]));
  }

  #[test]
  fn test_values_mut() {
    let mut set = SparseVec::new();
    assert!(set.iter_mut().eq(&[]));

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!(set.iter_mut().eq(&[Some(1), Some(2), Some(3)]));

    let value = set.iter_mut().next().unwrap();
    *value = Some(100);

    assert_eq!(set.get(0), Some(&100));
  }

  #[test]
  fn test_as_ref() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let reference: &SparseVec<_, _> = set.as_ref();
    assert_eq!(reference.get(0), Some(&1));

    let reference: &[Option<usize>] = set.as_ref();
    assert_eq!(reference.get(0), Some(&Some(1)));
  }

  #[test]
  fn test_as_ref_mut() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let reference: &mut SparseVec<_, _> = set.as_mut();
    assert_eq!(reference.get(0), Some(&1));

    let reference: &mut [Option<usize>] = set.as_mut();
    assert_eq!(reference.get(0), Some(&Some(1)));
  }

  #[test]
  fn test_clone() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let cloned_set = set.clone();
    assert_eq!(set, cloned_set);
  }

  #[test]
  fn test_clone_zero_capacity() {
    let set: SparseVec<usize, usize> = SparseVec::new();
    assert_eq!(set.capacity(), 0);

    let cloned_set = set.clone();
    assert_eq!(set, cloned_set);
  }

  #[test]
  fn test_clone_drops_are_separate() {
    let num_dropped = Rc::new(RefCell::new(0));

    {
      let mut set = SparseVec::new();
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
    let mut set = SparseVec::new();
    assert_eq!(format!("{:?}", set), "[]");

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert_eq!(format!("{:?}", set), "[Some(1), Some(2), Some(3)]");
  }

  #[test]
  fn test_default() {
    let set: SparseVec<usize, usize> = SparseVec::default();
    assert!(set.is_empty());
    assert_eq!(set.capacity(), 0);
  }

  #[test]
  fn test_deref() {
    let mut set: SparseVec<usize, usize> = SparseVec::default();
    set.insert(0, 1);

    assert_eq!(set.deref(), &[Some(1)]);
  }

  #[test]
  fn test_deref_mut() {
    let mut set: SparseVec<usize, usize> = SparseVec::default();
    set.insert(0, 1);

    assert_eq!(set.deref_mut(), &mut [Some(1)]);
  }

  #[test]
  fn test_drop() {
    let num_dropped = Rc::new(RefCell::new(0));

    {
      let mut set = SparseVec::new();
      let value = Value(num_dropped.clone());
      set.insert(0, value.clone());
      set.insert(1, value.clone());
      set.insert(2, value);
    }

    assert_eq!(*num_dropped.borrow(), 3);
  }

  #[test]
  fn test_extend() {
    let mut set = SparseVec::new();
    set.extend([(0, 1), (1, 2), (2, 3)]);
    assert!(set.iter().eq(&[Some(1), Some(2), Some(3)]));
  }

  #[test]
  fn test_extend_ref() {
    let mut set: SparseVec<usize, usize> = SparseVec::new();
    set.extend([(0, &1), (1, &2), (2, &3)]);
    assert!(set.iter().eq(&[Some(1), Some(2), Some(3)]));
  }

  #[test]
  fn test_from_array() {
    let set = SparseVec::from([(0, 1), (1, 2), (2, 3)]);
    assert!(set.iter().eq(&[Some(1), Some(2), Some(3)]));
  }

  #[test]
  fn test_from_iterator() {
    let set = SparseVec::from_iter([(0, 1), (1, 2), (2, 3)]);
    assert!(set.iter().eq(&[Some(1), Some(2), Some(3)]));
  }

  #[test]
  fn test_hash() {
    #[derive(Default)]
    struct TestHasher {
      writes_made: usize,
      delegate: DefaultHasher,
    }

    impl Hasher for TestHasher {
      fn finish(&self) -> u64 {
        self.delegate.finish()
      }

      fn write(&mut self, bytes: &[u8]) {
        self.delegate.write(bytes);
        self.writes_made += 1;
      }
    }

    fn hash(value: &SparseVec<usize, usize>) -> u64 {
      let mut hasher = TestHasher::default();
      value.hash(&mut hasher);
      assert!(hasher.writes_made >= value.len());
      hasher.finish()
    }

    let mut set_1 = SparseVec::new();
    let mut set_2 = SparseVec::new();

    assert_eq!(set_1, set_2);
    assert_eq!(hash(&set_1), hash(&set_2));

    set_1.insert(0, 1);

    assert_ne!(set_1, set_2);

    set_2.insert(0, 2);

    assert_ne!(set_1, set_2);

    let _ = set_2.remove(0);
    set_2.insert(1, 2);

    assert_ne!(set_1, set_2);

    set_1.insert(1, 2);
    set_2.insert(0, 1);

    assert_eq!(set_1, set_2);
    assert_eq!(hash(&set_1), hash(&set_2));

    let _ = set_1.remove(0);
    let _ = set_2.remove(0);

    assert_eq!(set_1, set_2);
    assert_eq!(hash(&set_1), hash(&set_2));
  }

  #[test]
  fn test_index() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    assert_eq!(set[0], 1);
    assert_eq!(set[2], 3);
  }

  #[should_panic]
  #[test]
  fn test_index_panics() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let _ = &set[100];
  }

  #[test]
  fn test_index_mut() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let value = &mut set[2];
    assert_eq!(value, &mut 3);
    *value = 10;

    assert_eq!(set[2], 10);
  }

  #[should_panic]
  #[test]
  fn test_index_mut_panics() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let _ = &mut set[100];
  }

  #[test]
  fn test_into_iterator() {
    let mut set = SparseVec::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!(set.into_iter().eq([Some(1), Some(2), Some(3)]));
  }

  #[test]
  fn test_into_iterator_ref() {
    let mut set = SparseVec::new();
    assert!((&set).into_iter().eq(&[]));

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!((&set).into_iter().eq(&[Some(1), Some(2), Some(3)]));
  }

  #[test]
  fn test_into_iterator_mut() {
    let mut set = SparseVec::new();
    assert!((&mut set).into_iter().eq(&[]));

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!((&mut set).into_iter().eq(&[Some(1), Some(2), Some(3)]));

    let value = set.iter_mut().next().unwrap();
    *value = Some(100);

    assert_eq!(set.get(0), Some(&100));
  }

  #[test]
  fn test_eq() {
    let mut set_1 = SparseVec::new();
    let mut set_2 = SparseVec::new();

    assert_eq!(set_1, set_2);

    set_1.insert(0, 1);

    assert_ne!(set_1, set_2);

    set_2.insert(0, 2);

    assert_ne!(set_1, set_2);

    let _ = set_2.remove(0);
    set_2.insert(1, 2);

    assert_ne!(set_1, set_2);

    set_1.insert(1, 2);

    set_2.insert(0, 1);

    assert_eq!(set_1, set_2);

    let _ = set_1.remove(0);
    let _ = set_2.remove(0);

    assert_eq!(set_1, set_2);
  }
}
