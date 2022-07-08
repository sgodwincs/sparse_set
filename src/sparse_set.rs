//! A sparsely populated set, written `SparseSet<I, T>`, where `I` is the index type and `T` is the value type.
//!
//! `I` must implement `SparseSetIndex`, in particular, it must be able to be converted to a `usize` index.
//!
//! See [this article](https://research.swtch.com/sparse) on more details behind the data structure.

#![allow(unsafe_code)]

use std::{
  alloc::{Allocator, Global},
  collections::TryReserveError,
  fmt,
  hash::{Hash, Hasher},
  num::NonZeroUsize,
  ops::{Deref, DerefMut, Index, IndexMut},
};

use crate::{SparseSetIndex, SparseVec};

/// A sparsely populated set, written `SparseSet<I, T>`, where `I` is the index type and `T` is the value type.
///
/// For operation complexity notes, *n* is the number of values in the sparse set and *m* is the value of the largest
/// index in the sparse set. Note that *m* will always be at least as large as *n*.
#[derive(Clone)]
pub struct SparseSet<I, T, SA: Allocator = Global, DA: Allocator = Global> {
  /// The dense buffer, i.e., the buffer containing the actual data values of type `T`.
  dense: Vec<T, DA>,

  /// The sparse buffer, i.e., the buffer where each index may correspond to an index into `dense`.
  sparse: SparseVec<I, NonZeroUsize, SA>,

  /// All the existing indices in `sparse`.
  ///
  /// The indices here will always be in order based on the `dense` buffer.
  indices: Vec<I, DA>,
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
    SparseSet::new_in(Global, Global, Global)
  }

  /// Constructs a new, empty `SparseSet<I, T>` with the specified capacity.
  ///
  /// The sparse set will be able to hold exactly `capacity` elements without reallocating. If `capacity` is 0, the
  /// sparse set will not allocate.
  ///
  /// It is important to note that although the returned sparse set has the *capacity* specified, the sparse set will
  /// have a zero *length*.
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
  /// let mut set = SparseSet::with_capacity(11, 10);
  ///
  /// // The sparse set contains no items, even though it has capacity for more.
  /// assert_eq!(set.len(), 0);
  /// assert_eq!(set.dense_capacity(), 10);
  /// assert_eq!(set.sparse_capacity(), 11);
  ///
  /// // These are all done without reallocating...
  /// for i in 0..10 {
  ///   set.insert(i, i);
  /// }
  ///
  /// assert_eq!(set.len(), 10);
  /// assert_eq!(set.dense_capacity(), 10);
  /// assert_eq!(set.sparse_capacity(), 11);
  ///
  /// // ...but this will make the sparse set reallocate.
  /// set.insert(10, 10);
  /// set.insert(11, 11);
  /// assert_eq!(set.dense_len(), 12);
  /// assert!(set.dense_capacity() >= 12);
  /// assert!(set.sparse_capacity() >= 12);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  #[must_use]
  pub fn with_capacity(sparse_capacity: usize, dense_capacity: usize) -> Self {
    assert!(
      sparse_capacity >= dense_capacity,
      "Sparse capacity must be at least as large as the dense capacity."
    );
    SparseSet::with_capacity_in(sparse_capacity, Global, dense_capacity, Global, Global)
  }
}

impl<I, T, SA: Allocator, DA: Allocator> SparseSet<I, T, SA, DA> {
  /// Constructs a new, empty `SparseSet<I, T, DA, SA>`.
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
  /// let mut set: SparseSet<usize, u32, _, _> = SparseSet::new_in(System, System, System);
  /// ```
  #[must_use]
  pub fn new_in(sparse_alloc: SA, dense_alloc: DA, indices_alloc: DA) -> Self {
    Self {
      dense: Vec::new_in(dense_alloc),
      sparse: SparseVec::new_in(sparse_alloc),
      indices: Vec::new_in(indices_alloc),
    }
  }

  /// Constructs a new, empty `SparseSet<I, T, DA, SA>` with the specified capacity with the provided allocator.
  ///
  /// The sparse set will be able to hold exactly `capacity` elements without reallocating. If `capacity` is 0, the
  /// sparse set will not allocate.
  ///
  /// It is important to note that although the returned sparse set has the *capacity* specified, the sparse set will
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
  /// # use sparse_set::SparseSet;
  ///
  /// let mut set = SparseSet::with_capacity_in(10, System, 10, System, System);
  ///
  /// // The sparse set contains no items, even though it has capacity for more
  /// assert_eq!(set.dense_len(), 0);
  /// assert_eq!(set.sparse_len(), 0);
  /// assert_eq!(set.dense_capacity(), 10);
  /// assert_eq!(set.sparse_capacity(), 10);
  ///
  /// // These are all done without reallocating...
  /// for i in 0..10 {
  ///   set.insert(i, i);
  /// }
  ///
  /// assert_eq!(set.dense_len(), 10);
  /// assert_eq!(set.sparse_len(), 10);
  /// assert_eq!(set.dense_capacity(), 10);
  /// assert_eq!(set.sparse_capacity(), 10);
  ///
  /// // ...but this will make the sparse set reallocate.
  /// set.insert(10, 10);
  /// assert_eq!(set.dense_len(), 11);
  /// assert!(set.dense_capacity() >= 11);
  /// assert_eq!(set.sparse_len(), 11);
  /// assert!(set.sparse_capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn with_capacity_in(
    sparse_capacity: usize,
    sparse_alloc: SA,
    dense_capacity: usize,
    dense_alloc: DA,
    indices_alloc: DA,
  ) -> Self {
    Self {
      dense: Vec::with_capacity_in(dense_capacity, dense_alloc),
      sparse: SparseVec::with_capacity_in(sparse_capacity, sparse_alloc),
      indices: Vec::with_capacity_in(dense_capacity, indices_alloc),
    }
  }
}

impl<I, T, SA: Allocator, DA: Allocator> SparseSet<I, T, SA, DA> {
  /// Returns a reference to the underlying dense buffer allocator.
  #[must_use]
  pub fn dense_allocator(&self) -> &DA {
    self.dense.allocator()
  }

  /// Returns a reference to the underlying sparse buffer allocator.
  #[must_use]
  pub fn sparse_allocator(&self) -> &SA {
    self.sparse.allocator()
  }

  /// Extracts a slice containing the entire dense buffer.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  #[must_use]
  pub fn as_dense_slice(&self) -> &[T] {
    &self.dense
  }

  /// Extracts a mutable slice of the entire dense buffer.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  #[must_use]
  pub fn as_dense_mut_slice(&mut self) -> &mut [T] {
    &mut self.dense
  }

  /// Returns a raw pointer to the dense buffer, or a dangling raw pointer valid for zero sized reads if the sparse set
  /// didn't allocate.
  ///
  /// The caller must ensure that the sparse set outlives the pointer this function returns, or else it will end up
  /// pointing to garbage. Modifying the sparse set may cause its buffer to be reallocated, which would also make any
  /// pointers to it invalid.
  ///
  /// The caller must also ensure that the memory the pointer (non-transitively) points to is never written to (except
  /// inside an `UnsafeCell`) using this pointer or any pointer derived from it.
  #[must_use]
  pub fn as_dense_ptr(&self) -> *const T {
    self.dense.as_ptr()
  }

  /// Returns an unsafe mutable pointer to the sparse set's dense buffer.
  ///
  /// The caller must ensure that the sparse set outlives the pointer this function returns, or else it will end up
  /// pointing to garbage. Modifying the sparse set may cause its buffer to be reallocated, which would also make any
  /// pointers to it invalid.
  ///
  /// Swapping elements in the mutable slice changes which indices point to which values.
  #[must_use]
  pub fn as_dense_mut_ptr(&mut self) -> *mut T {
    self.dense.as_mut_ptr()
  }

  /// Returns a slice over the sparse set's indices.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
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
  /// assert_eq!(set.as_indices_slice(), &[0, 1, 2]);
  /// ```
  #[must_use]
  pub fn as_indices_slice(&self) -> &[I] {
    &*self.indices
  }

  /// Returns a raw pointer to the index buffer, or a dangling raw pointer valid for zero sized reads if the sparse set
  /// didn't allocate.
  ///
  /// The caller must ensure that the sparse set outlives the pointer this function returns, or else it will end up
  /// pointing to garbage. Modifying the sparse set may cause its buffer to be reallocated, which would also make any
  /// pointers to it invalid.
  ///
  /// The caller must also ensure that the memory the pointer (non-transitively) points to is never written to (except
  /// inside an `UnsafeCell`) using this pointer or any pointer derived from it.
  #[must_use]
  pub fn as_indices_ptr(&self) -> *const I {
    self.indices.as_ptr()
  }

  /// Returns the number of elements the dense buffer can hold without reallocating.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let set: SparseSet<usize, i32> = SparseSet::with_capacity(15, 10);
  /// assert_eq!(set.dense_capacity(), 10);
  /// ```
  #[must_use]
  pub fn dense_capacity(&self) -> usize {
    self.dense.capacity()
  }

  /// Returns the number of elements the sparse buffer can hold without reallocating.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let set: SparseSet<usize, i32> = SparseSet::with_capacity(15, 10);
  /// assert_eq!(set.sparse_capacity(), 15);
  /// ```
  #[must_use]
  pub fn sparse_capacity(&self) -> usize {
    self.sparse.capacity()
  }

  /// Clears the sparse set, removing all values.
  ///
  /// Note that this method has no effect on the allocated capacity of the sparse set.
  ///
  /// This operation is *O*(*m*).
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
    self.dense.clear();
    self.indices.clear();
    self.sparse.clear();
  }

  /// Returns an iterator over the sparse set's indices.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// Consuming the iterator is an *O*(*n*) operation.
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
  /// let mut iterator = set.indices();
  ///
  /// assert_eq!(iterator.next(), Some(&0));
  /// assert_eq!(iterator.next(), Some(&1));
  /// assert_eq!(iterator.next(), Some(&2));
  /// assert_eq!(iterator.next(), None);
  /// ```
  #[must_use]
  pub fn indices(&self) -> impl Iterator<Item = &I> {
    self.indices.iter()
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
    self.dense_len() == 0
  }

  /// Returns the number of elements in the dense set, also referred to as its 'dense_len'.
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
  /// assert_eq!(set.dense_len(), 3);
  /// ```
  #[must_use]
  pub fn dense_len(&self) -> usize {
    self.dense.len()
  }

  /// Returns the number of elements in the sparse set, also referred to as its 'sparse_len'.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// set.insert(200, 3);
  ///
  /// assert_eq!(set.sparse_len(), 201);
  /// ```
  #[must_use]
  pub fn sparse_len(&self) -> usize {
    self.sparse.len()
  }

  /// Reserves capacity for at least `additional` more elements to be inserted in the given `SparseSet<I, T>`'s dense
  /// buffer.
  ///
  /// The collection may reserve more space to avoid frequent reallocations. After calling `reserve`, the dense capacity
  /// will be greater than or equal to `self.dense_len() + additional`. Does nothing if capacity is already sufficient.
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
  /// set.reserve_dense(10);
  /// assert!(set.dense_capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_dense(&mut self, additional: usize) {
    self.dense.reserve(additional);
  }

  /// Reserves capacity for at least `additional` more elements to be inserted in the given `SparseSet<I, T>`'s sparse
  /// buffer.
  ///
  /// The collection may reserve more space to avoid frequent reallocations. After calling `reserve`, the sparse
  /// capacity will be greater than or equal to `self.sparse_len() + additional`. Does nothing if capacity is already
  /// sufficient.
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
  /// set.reserve_sparse(10);
  /// assert!(set.sparse_capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_sparse(&mut self, additional: usize) {
    self.sparse.reserve(additional);
  }

  /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the given
  /// `SparseSet<I, T>`'s dense buffer.
  ///
  /// After calling `reserve_exact`, the dense capacity will be greater than or equal to
  /// `self.dense_len() + additional`. Does nothing if the capacity is already sufficient.
  ///
  /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not be relied
  /// upon to be precisely minimal. Prefer [`reserve_dense`] if future insertions are expected.
  ///
  /// [`reserve_dense`]: SparseSet::reserve_dense
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
  /// set.reserve_exact_dense(10);
  /// assert!(set.dense_capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_exact_dense(&mut self, additional: usize) {
    self.dense.reserve_exact(additional);
  }

  /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the given
  /// `SparseSet<I, T>`'s sparse buffer.
  ///
  /// After calling `reserve_exact`, the sparse capacity will be greater than or equal to
  /// `self.sparse_len() + additional`. Does nothing if the capacity is already sufficient.
  ///
  /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not be relied
  /// upon to be precisely minimal. Prefer [`reserve_sparse`] if future insertions are expected.
  ///
  /// [`reserve_sparse`]: SparseSet::reserve_sparse
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
  /// set.reserve_exact_sparse(10);
  /// assert!(set.sparse_capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_exact_sparse(&mut self, additional: usize) {
    self.sparse.reserve_exact(additional);
  }

  /// Tries to reserve capacity for at least `additional` more elements to be inserted in the given `SparseSet<I, T>`'s
  /// dense buffer.
  ///
  /// The collection may reserve more space to avoid frequent reallocations. After calling `try_reserve_dense`, capacity
  /// will be greater than or equal to `self.dense_len() + additional`. Does nothing if capacity is already sufficient.
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
  ///   output.try_reserve_dense(data.len())?;
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
  pub fn try_reserve_dense(&mut self, additional: usize) -> Result<(), TryReserveError> {
    self.dense.try_reserve(additional)
  }

  /// Tries to reserve capacity for at least `additional` more elements to be inserted in the given `SparseSet<I, T>`'s
  /// sparse buffer.
  ///
  /// The collection may reserve more space to avoid frequent reallocations. After calling `try_reserve_sparse`,
  /// capacity will be greater than or equal to `self.sparse_len() + additional`. Does nothing if capacity is already
  /// sufficient.
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
  ///   output.try_reserve_sparse(data.len())?;
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
  pub fn try_reserve_sparse(&mut self, additional: usize) -> Result<(), TryReserveError> {
    self.sparse.try_reserve(additional)
  }

  /// Tries to reserve the minimum capacity for exactly `additional` elements to be inserted in the given
  /// `SparseSet<T>`'s dense buffer.
  ///
  /// After calling `try_reserve_exact_dense`, capacity will be greater than or equal to `self.dense_len() + additional`
  /// if it returns `Ok(())`. Does nothing if the capacity is already sufficient.
  ///
  /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not be relied
  /// upon to be precisely minimal. Prefer [`try_reserve_dense`] if future insertions are expected.
  ///
  /// [`try_reserve_dense`]: SparseSet::try_reserve_dense
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
  ///   output.try_reserve_exact_dense(data.len())?;
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
  pub fn try_reserve_exact_dense(&mut self, additional: usize) -> Result<(), TryReserveError> {
    self.dense.try_reserve_exact(additional)
  }

  /// Tries to reserve the minimum capacity for exactly `additional` elements to be inserted in the given
  /// `SparseSet<T>`'s sparse buffer.
  ///
  /// After calling `try_reserve_exact_sparse`, capacity will be greater than or equal to
  /// `self.sparse_len() + additional` if it returns `Ok(())`. Does nothing if the capacity is already sufficient.
  ///
  /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not be relied
  /// upon to be precisely minimal. Prefer [`try_reserve_sparse`] if future insertions are expected.
  ///
  /// [`try_reserve_sparse`]: SparseSet::try_reserve_sparse
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
  ///   output.try_reserve_exact_sparse(data.len())?;
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
  pub fn try_reserve_exact_sparse(&mut self, additional: usize) -> Result<(), TryReserveError> {
    self.sparse.try_reserve_exact(additional)
  }

  /// Shrinks the dense capacity of the sparse set as much as possible.
  ///
  /// It will drop down as close as possible to the length but the allocator may still inform the sparse set that
  /// there is space for a few more elements.
  ///
  /// This operation is *O*(*n*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::with_capacity(10, 10);
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// assert_eq!(set.dense_capacity(), 10);
  ///
  /// set.shrink_to_fit_dense();
  /// assert!(set.dense_capacity() >= 2);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to_fit_dense(&mut self) {
    self.dense.shrink_to_fit()
  }

  /// Shrinks the sparse capacity of the sparse set as much as possible.
  ///
  /// Unlike [`shrink_to_fit_dense`], this can also reduce `sparse_len` as any empty indices after the maximum index are
  /// removed.
  ///
  /// It will drop down as close as possible to the length but the allocator may still inform the sparse set that
  /// there is space for a few more elements.
  ///
  /// This operation is *O*(*m*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::with_capacity(10, 10);
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// assert_eq!(set.sparse_capacity(), 10);
  ///
  /// set.shrink_to_fit_dense();
  /// assert!(set.sparse_capacity() >= 2);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to_fit_sparse(&mut self) {
    self.sparse.shrink_to_fit();
  }

  /// Shrinks the dense capacity of the sparse set with a lower bound.
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
  /// let mut set = SparseSet::with_capacity(10, 10);
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// assert_eq!(set.dense_capacity(), 10);
  /// set.shrink_to_dense(4);
  /// assert!(set.dense_capacity() >= 4);
  /// set.shrink_to_dense(0);
  /// assert!(set.dense_capacity() >= 2);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to_dense(&mut self, min_capacity: usize) {
    self.dense.shrink_to(min_capacity);
  }

  /// Shrinks the sparse capacity of the sparse set with a lower bound.
  ///
  /// Unlike [`shrink_to_dense`], this can also reduce `sparse_len` as any empty indices after the maximum index are
  /// removed.
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
  /// let mut set = SparseSet::with_capacity(10, 10);
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// assert_eq!(set.sparse_capacity(), 10);
  /// set.shrink_to_sparse(4);
  /// assert!(set.sparse_capacity() >= 4);
  /// set.shrink_to_sparse(0);
  /// assert!(set.sparse_capacity() >= 2);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to_sparse(&mut self, min_capacity: usize) {
    self.sparse.shrink_to(min_capacity)
  }

  /// Returns an iterator over the sparse set's values.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// Consuming the iterator is an *O*(*n*) operation.
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
  /// assert!(set.values().eq(&[1, 2, 3]));
  /// ```
  #[must_use]
  pub fn values(&self) -> impl Iterator<Item = &T> {
    self.dense.iter()
  }

  /// Returns an iterator that allows modifying each value.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
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
  #[must_use]
  pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
    self.dense.iter_mut()
  }
}

impl<I: SparseSetIndex, T, SA: Allocator, DA: Allocator> SparseSet<I, T, SA, DA> {
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
      .sparse
      .get(index)
      .map(|dense_index| unsafe { self.dense.get_unchecked(dense_index.get() - 1) })
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
      .sparse
      .get(index)
      .map(|dense_index| unsafe { self.dense.get_unchecked_mut(dense_index.get() - 1) })
  }

  /// Inserts an element at position `index` within the sparse set.
  ///
  /// If a value already existed at `index`, it will be overwritten.
  ///
  /// If `index` is greater than `sparse_capacity`, then an allocation will take place.
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
    match self.sparse.get(index) {
      Some(dense_index) => {
        let dense_index = dense_index.get() - 1;
        *unsafe { self.dense.get_unchecked_mut(dense_index) } = value;
      }
      None => {
        self.sparse.insert(index, unsafe {
          NonZeroUsize::new_unchecked(self.dense_len() + 1)
        });
        self.dense.push(value);
        self.indices.push(index);
      }
    }
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
  #[must_use]
  pub fn remove(&mut self, index: I) -> Option<T> {
    match self.sparse.remove(index) {
      Some(dense_index) => {
        let dense_index = dense_index.get() - 1;
        let value = Some(self.dense.swap_remove(dense_index));
        let _ = self.indices.swap_remove(dense_index);

        if dense_index != self.dense.len() {
          let swapped_index: usize = (*unsafe { self.indices.get_unchecked(dense_index) }).into();
          *unsafe { self.sparse.get_unchecked_mut(swapped_index) } =
            Some(unsafe { NonZeroUsize::new_unchecked(dense_index + 1) });
        }

        value
      }
      _ => None,
    }
  }
}

impl<I, T, SA: Allocator, DA: Allocator> AsRef<SparseSet<I, T, SA, DA>>
  for SparseSet<I, T, SA, DA>
{
  fn as_ref(&self) -> &Self {
    self
  }
}

impl<I, T, SA: Allocator, DA: Allocator> AsMut<SparseSet<I, T, SA, DA>>
  for SparseSet<I, T, SA, DA>
{
  fn as_mut(&mut self) -> &mut Self {
    self
  }
}

impl<I, T, SA: Allocator, DA: Allocator> AsRef<[T]> for SparseSet<I, T, SA, DA> {
  fn as_ref(&self) -> &[T] {
    &self.dense
  }
}

impl<I, T, SA: Allocator, DA: Allocator> AsMut<[T]> for SparseSet<I, T, SA, DA> {
  fn as_mut(&mut self) -> &mut [T] {
    &mut self.dense
  }
}

impl<I, T> Default for SparseSet<I, T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<I, T, SA: Allocator, DA: Allocator> Deref for SparseSet<I, T, SA, DA> {
  type Target = [T];

  fn deref(&self) -> &[T] {
    &*self.dense
  }
}

impl<I, T, SA: Allocator, DA: Allocator> DerefMut for SparseSet<I, T, SA, DA> {
  fn deref_mut(&mut self) -> &mut [T] {
    &mut *self.dense
  }
}

impl<I, T: fmt::Debug, SA: Allocator, DA: Allocator> fmt::Debug for SparseSet<I, T, SA, DA> {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.dense.fmt(formatter)
  }
}

#[cfg(not(no_global_oom_handling))]
impl<'a, I: SparseSetIndex, T: Copy + 'a, SA: Allocator + 'a, DA: Allocator + 'a> Extend<(I, &'a T)>
  for SparseSet<I, T, SA, DA>
{
  fn extend<Iter: IntoIterator<Item = (I, &'a T)>>(&mut self, iter: Iter) {
    for (index, &value) in iter {
      self.insert(index, value);
    }
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T, SA: Allocator, DA: Allocator> Extend<(I, T)>
  for SparseSet<I, T, SA, DA>
{
  fn extend<Iter: IntoIterator<Item = (I, T)>>(&mut self, iter: Iter) {
    for (index, value) in iter {
      self.insert(index, value);
    }
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T, const N: usize> From<[(I, T); N]> for SparseSet<I, T> {
  fn from(slice: [(I, T); N]) -> Self {
    let mut set = SparseSet::with_capacity(slice.len(), slice.len());

    for (index, value) in slice {
      set.insert(index, value);
    }

    set
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T> FromIterator<(I, T)> for SparseSet<I, T> {
  fn from_iter<Iter: IntoIterator<Item = (I, T)>>(iter: Iter) -> Self {
    let iter = iter.into_iter();
    let capacity = if let Some(size_hint) = iter.size_hint().1 {
      size_hint
    } else {
      iter.size_hint().0
    };
    let mut set = SparseSet::with_capacity(capacity, capacity);

    for (index, value) in iter {
      set.insert(index, value);
    }

    set
  }
}

impl<I: Hash + SparseSetIndex, T: Hash, SA: Allocator, DA: Allocator> Hash
  for SparseSet<I, T, SA, DA>
{
  fn hash<H: Hasher>(&self, state: &mut H) {
    for index in self.sparse.iter() {
      if let Some(index) = index {
        unsafe { self.sparse.get_unchecked(index.get() - 1) }.hash(state);
        unsafe { self.dense.get_unchecked(index.get() - 1) }.hash(state);
      }
    }
  }
}

impl<I: SparseSetIndex, T, SA: Allocator, DA: Allocator> Index<I> for SparseSet<I, T, SA, DA> {
  type Output = T;

  fn index(&self, index: I) -> &Self::Output {
    self.get(index).unwrap()
  }
}

impl<I: SparseSetIndex, T, SA: Allocator, DA: Allocator> IndexMut<I> for SparseSet<I, T, SA, DA> {
  fn index_mut(&mut self, index: I) -> &mut Self::Output {
    self.get_mut(index).unwrap()
  }
}

impl<I, T, SA: Allocator, DA: Allocator> IntoIterator for SparseSet<I, T, SA, DA> {
  type Item = T;
  type IntoIter = impl Iterator<Item = Self::Item>;

  #[inline]
  fn into_iter(self) -> Self::IntoIter {
    self.dense.into_iter()
  }
}

impl<'a, I, T, SA: Allocator, DA: Allocator> IntoIterator for &'a SparseSet<I, T, SA, DA> {
  type Item = &'a T;
  type IntoIter = impl Iterator<Item = Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.values()
  }
}

impl<'a, I, T, SA: Allocator, DA: Allocator> IntoIterator for &'a mut SparseSet<I, T, SA, DA> {
  type Item = &'a mut T;
  type IntoIter = impl Iterator<Item = Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.values_mut()
  }
}

impl<I: PartialEq + SparseSetIndex, T: PartialEq, SA: Allocator, DA: Allocator> PartialEq
  for SparseSet<I, T, SA, DA>
{
  fn eq(&self, other: &Self) -> bool {
    if self.indices.len() != other.indices.len() {
      return false;
    }

    for index in self.indices.iter() {
      match (self.sparse.get(*index), other.sparse.get(*index)) {
        (Some(index), Some(other_index)) => {
          let index = index.get() - 1;
          let other_index = other_index.get() - 1;

          if unsafe { self.indices.get_unchecked(index) }
            != unsafe { other.indices.get_unchecked(other_index) }
          {
            return false;
          }

          if unsafe { self.dense.get_unchecked(index) }
            != unsafe { other.dense.get_unchecked(other_index) }
          {
            return false;
          }
        }
        (None, None) => {}
        _ => {
          return false;
        }
      }
    }

    true
  }
}

impl<I: Eq + SparseSetIndex, T: Eq, SA: Allocator, DA: Allocator> Eq for SparseSet<I, T, SA, DA> {}

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
    let set: SparseSet<usize, usize> = SparseSet::new();
    assert!(set.is_empty());
    assert_eq!(set.dense_capacity(), 0);
    assert_eq!(set.sparse_capacity(), 0);
  }

  #[test]
  fn test_with_capacity() {
    let set: SparseSet<usize, usize> = SparseSet::with_capacity(15, 10);
    assert_eq!(set.sparse_capacity(), 15);
    assert_eq!(set.dense_capacity(), 10);
  }

  #[test]
  fn test_with_capacity_zero() {
    let set: SparseSet<usize, usize> = SparseSet::with_capacity(0, 0);
    assert_eq!(set.sparse_capacity(), 0);
    assert_eq!(set.dense_capacity(), 0);
  }

  #[should_panic]
  #[test]
  fn test_with_capacity_dense_greater_than_sparse() {
    let _: SparseSet<usize, usize> = SparseSet::with_capacity(0, 1);
  }

  #[should_panic]
  #[test]
  fn test_with_capacity_sparse_overflow() {
    let _: SparseSet<usize, usize> = SparseSet::with_capacity(usize::MAX, 0);
  }

  #[should_panic]
  #[test]
  fn test_with_capacity_dense_overflow() {
    let _: SparseSet<usize, usize> = SparseSet::with_capacity(0, usize::MAX);
  }

  #[test]
  fn test_dense_allocator() {
    let set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.dense_allocator();
  }

  #[test]
  fn test_sparse_allocator() {
    let set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.sparse_allocator();
  }

  #[test]
  fn test_as_dense_slice() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    set.insert(0, 1);
    assert_eq!(set.as_dense_slice(), &[1]);
  }

  #[test]
  fn test_as_dense_mut_slice() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    set.insert(0, 1);
    assert_eq!(set.as_dense_mut_slice(), &mut [1]);
  }

  #[test]
  fn test_as_dense_ptr() {
    let set: SparseSet<usize, usize> = SparseSet::with_capacity(10, 10);
    assert_eq!(set.as_dense_ptr(), set.as_dense_slice().as_ptr());
  }

  #[test]
  fn test_as_dense_mut_ptr() {
    let mut set: SparseSet<usize, usize> = SparseSet::with_capacity(10, 10);
    assert_eq!(
      set.as_dense_mut_ptr(),
      set.as_dense_mut_slice().as_mut_ptr()
    );
  }

  #[test]
  fn test_as_indices_slice() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    set.insert(0, 1);
    assert_eq!(set.as_indices_slice(), &[0]);
  }

  #[test]
  fn test_as_indices_ptr() {
    let set: SparseSet<usize, usize> = SparseSet::with_capacity(10, 10);
    assert_eq!(set.as_indices_ptr(), set.as_indices_slice().as_ptr());
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
  fn test_contains() {
    let mut set = SparseSet::new();
    assert!(!set.contains(0));
    set.insert(0, 1);
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
  fn test_indices() {
    let mut set = SparseSet::new();
    assert!(set.indices().eq(&[]));

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!(set.indices().eq(&[0, 1, 2]));
  }

  #[test]
  fn test_insert_capacity_increases() {
    let mut set = SparseSet::with_capacity(1, 1);
    set.insert(0, 1);
    assert_eq!(set.sparse_capacity(), 1);
    assert_eq!(set.dense_capacity(), 1);

    set.insert(1, 2);
    assert!(set.sparse_capacity() >= 2);
    assert!(set.dense_capacity() >= 2);

    assert_eq!(set.get(0), Some(&1));
    assert_eq!(set.get(1), Some(&2));
  }

  #[test]
  fn test_insert_len_increases() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 1);

    set.insert(1, 2);
    assert_eq!(set.dense_len(), 2);
    assert_eq!(set.sparse_len(), 2);

    set.insert(100, 101);
    assert_eq!(set.dense_len(), 3);
    assert_eq!(set.sparse_len(), 101);
  }

  #[test]
  fn test_insert_overwrites() {
    let mut set = SparseSet::new();
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
  fn test_dense_len() {
    let mut set = SparseSet::new();
    assert_eq!(set.dense_len(), 0);

    set.insert(0, 1);
    assert_eq!(set.dense_len(), 1);
  }

  #[test]
  fn test_sparse_len() {
    let mut set = SparseSet::new();
    assert_eq!(set.sparse_len(), 0);

    set.insert(0, 1);
    assert_eq!(set.sparse_len(), 1);

    set.insert(100, 1);
    assert_eq!(set.sparse_len(), 101);
  }

  #[test]
  fn test_remove_can_return_none() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
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
  fn test_remove_len_decreases() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);

    assert_eq!(set.dense_len(), 2);
    assert_eq!(set.sparse_len(), 2);
    assert_eq!(set.remove(0), Some(1));
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 2);
    assert_eq!(set.remove(0), None);
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 2);
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
  fn test_reserve_dense() {
    let mut set = SparseSet::new();
    assert_eq!(set.dense_capacity(), 0);

    set.reserve_dense(3);
    let capacity = set.dense_capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    set.reserve_dense(1);
    assert_eq!(set.dense_capacity(), capacity);
  }

  #[test]
  fn test_reserve_sparse() {
    let mut set = SparseSet::new();
    assert_eq!(set.sparse_capacity(), 0);

    set.reserve_sparse(3);
    let capacity = set.sparse_capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    set.reserve_sparse(1);
    assert_eq!(set.sparse_capacity(), capacity);
  }

  #[test]
  fn test_reserve_exact_dense() {
    let mut set = SparseSet::new();
    assert_eq!(set.dense_capacity(), 0);

    set.reserve_exact_dense(3);
    let capacity = set.dense_capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    set.reserve_exact_dense(1);
    assert_eq!(set.dense_capacity(), capacity);
  }

  #[test]
  fn test_reserve_exact_sparse() {
    let mut set = SparseSet::new();
    assert_eq!(set.sparse_capacity(), 0);

    set.reserve_exact_sparse(3);
    let capacity = set.sparse_capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    set.reserve_exact_sparse(1);
    assert_eq!(set.sparse_capacity(), capacity);
  }

  #[test]
  fn test_shrink_to_fit_dense() {
    let mut set = SparseSet::with_capacity(3, 3);
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    assert_eq!(set.dense_capacity(), 3);
    let _ = set.remove(2);
    set.shrink_to_fit_dense();
    assert_eq!(set.dense_capacity(), 2);
  }

  #[test]
  fn test_shrink_to_fit_sparse() {
    let mut set = SparseSet::with_capacity(3, 3);
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 3);
    let _ = set.remove(2);
    set.shrink_to_fit_sparse();
    assert_eq!(set.sparse_capacity(), 2);
    assert_eq!(set.sparse_len(), 2);
  }

  #[test]
  fn test_shrink_to_fit_max_index_zero() {
    let mut set: SparseSet<usize, usize> = SparseSet::with_capacity(3, 3);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 0);
    set.shrink_to_fit_sparse();
    assert_eq!(set.sparse_capacity(), 0);
    assert_eq!(set.sparse_len(), 0);
  }

  #[test]
  fn test_shrink_to_dense_can_reduce() {
    let mut set = SparseSet::with_capacity(3, 3);
    set.insert(0, 1);
    assert_eq!(set.dense_capacity(), 3);
    set.shrink_to_dense(1);
    assert_eq!(set.dense_capacity(), 1);
  }

  #[test]
  fn test_shrink_to_dense_cannot_reduce() {
    let mut set = SparseSet::with_capacity(3, 3);
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert_eq!(set.dense_capacity(), 3);
    set.shrink_to_dense(0);
    assert_eq!(set.dense_capacity(), 3);
  }

  #[test]
  fn test_shrink_to_sparse_can_reduce() {
    let mut set = SparseSet::with_capacity(3, 3);
    set.insert(0, 1);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 1);
    set.shrink_to_sparse(1);
    assert_eq!(set.sparse_capacity(), 1);
    assert_eq!(set.sparse_len(), 1);
  }

  #[test]
  fn test_shrink_to_sparse_cannot_reduce() {
    let mut set = SparseSet::with_capacity(3, 3);
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 3);
    set.shrink_to_sparse(0);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 3);
  }

  #[test]
  fn test_shrink_to_max_index_zero() {
    let mut set: SparseSet<usize, usize> = SparseSet::with_capacity(3, 3);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 0);
    set.shrink_to_sparse(0);
    assert_eq!(set.sparse_capacity(), 0);
    assert_eq!(set.sparse_len(), 0);
  }

  #[test]
  fn test_try_reserve_dense_succeeds() {
    let mut set = SparseSet::new();
    assert_eq!(set.dense_capacity(), 0);

    assert!(set.try_reserve_dense(3).is_ok());
    let capacity = set.dense_capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    assert!(set.try_reserve_dense(1).is_ok());
    assert_eq!(set.dense_capacity(), capacity);
  }

  #[test]
  fn test_try_reserve_sparse_succeeds() {
    let mut set = SparseSet::new();
    assert_eq!(set.sparse_capacity(), 0);

    assert!(set.try_reserve_sparse(3).is_ok());
    let capacity = set.sparse_capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    assert!(set.try_reserve_sparse(1).is_ok());
    assert_eq!(set.sparse_capacity(), capacity);
  }

  #[test]
  fn test_try_reserve_exact_succeeds() {
    let mut set = SparseSet::new();
    assert_eq!(set.dense_capacity(), 0);

    assert!(set.try_reserve_exact_dense(3).is_ok());
    let capacity = set.dense_capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    assert!(set.try_reserve_exact_dense(1).is_ok());
    assert_eq!(set.dense_capacity(), capacity);
  }

  #[test]
  fn test_try_reserve_exact_sparse_succeeds() {
    let mut set = SparseSet::new();
    assert_eq!(set.sparse_capacity(), 0);

    assert!(set.try_reserve_exact_sparse(3).is_ok());
    let capacity = set.sparse_capacity();
    assert!(capacity >= 2);

    set.insert(0, 1);
    set.insert(1, 2);

    assert!(set.try_reserve_exact_sparse(1).is_ok());
    assert_eq!(set.sparse_capacity(), capacity);
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
  fn test_as_ref() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let reference: &SparseSet<_, _> = set.as_ref();
    assert_eq!(reference.first(), Some(&1));

    let reference: &[usize] = set.as_ref();
    assert_eq!(reference.first(), Some(&1));
  }

  #[test]
  fn test_as_ref_mut() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let reference: &mut SparseSet<_, _> = set.as_mut();
    assert_eq!(reference.first(), Some(&1));

    let reference: &mut [usize] = set.as_mut();
    assert_eq!(reference.first(), Some(&1));
  }

  #[test]
  fn test_clone() {
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    let cloned_set = set.clone();
    assert_eq!(set, cloned_set);
  }

  #[test]
  fn test_clone_zero_capacity() {
    let set: SparseSet<usize, usize> = SparseSet::new();
    assert_eq!(set.dense_capacity(), 0);

    let cloned_set = set.clone();
    assert_eq!(set, cloned_set);
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
  fn test_default() {
    let set: SparseSet<usize, usize> = SparseSet::default();
    assert!(set.is_empty());
    assert_eq!(set.dense_capacity(), 0);
    assert_eq!(set.sparse_capacity(), 0);
  }

  #[test]
  fn test_deref() {
    let mut set: SparseSet<usize, usize> = SparseSet::default();
    set.insert(0, 1);

    assert_eq!(set.deref(), &[1]);
  }

  #[test]
  fn test_deref_mut() {
    let mut set: SparseSet<usize, usize> = SparseSet::default();
    set.insert(0, 1);

    assert_eq!(set.deref_mut(), &mut [1]);
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
  fn test_extend_ref() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    set.extend([(0, &1), (1, &2), (2, &3)]);
    assert!(set.values().eq(&[1, 2, 3]));
  }

  #[test]
  fn test_from_array() {
    let set = SparseSet::from([(0, 1), (1, 2), (2, 3)]);
    assert!(set.values().eq(&[1, 2, 3]));
  }

  #[test]
  fn test_from_iterator() {
    let set = SparseSet::from_iter([(0, 1), (1, 2), (2, 3)]);
    assert!(set.values().eq(&[1, 2, 3]));
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

    fn hash(value: &SparseSet<usize, usize>) -> u64 {
      let mut hasher = TestHasher::default();
      value.hash(&mut hasher);
      assert!(hasher.writes_made >= value.len());
      hasher.finish()
    }

    let mut set_1 = SparseSet::new();
    let mut set_2 = SparseSet::new();

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
    let mut set = SparseSet::new();
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);

    assert_eq!(set[0], 1);
    assert_eq!(set[2], 3);
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
    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!(set.into_iter().eq([1, 2, 3]));
  }

  #[test]
  fn test_into_iterator_ref() {
    let mut set = SparseSet::new();
    assert!((&set).into_iter().eq(&[]));

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!((&set).into_iter().eq(&[1, 2, 3]));
  }

  #[test]
  fn test_into_iterator_mut() {
    let mut set = SparseSet::new();
    assert!((&mut set).into_iter().eq(&[]));

    set.insert(0, 1);
    set.insert(1, 2);
    set.insert(2, 3);
    assert!((&mut set).into_iter().eq(&[1, 2, 3]));

    let value = set.values_mut().next().unwrap();
    *value = 100;

    assert_eq!(set.first(), Some(&100));
  }

  #[test]
  fn test_eq() {
    let mut set_1 = SparseSet::new();
    let mut set_2 = SparseSet::new();

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
