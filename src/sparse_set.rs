//! A sparsely populated set, written `SparseSet<I, T>`, where `I` is the index type and `T` is the value type.
//!
//! `I` must implement `SparseSetIndex`, in particular, it must be able to be converted to a `usize` index.
//!
//! See [this article](https://research.swtch.com/sparse) on more details behind the data structure.

#![allow(unsafe_code)]

use std::{
  alloc::{Allocator, Global},
  collections::TryReserveError,
  fmt::{self, Debug, Formatter},
  hash::{Hash, Hasher},
  mem,
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
  pub const fn new() -> Self {
    Self::new_in(Global, Global, Global)
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
    Self::with_capacity_in(sparse_capacity, Global, dense_capacity, Global, Global)
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
  pub const fn new_in(sparse_alloc: SA, dense_alloc: DA, indices_alloc: DA) -> Self {
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
  ///
  /// # Safety
  ///
  /// The order of value in the dense buffer must be kept in sync with the order of indices in the index buffer.
  #[must_use]
  pub unsafe fn as_dense_mut_slice(&mut self) -> &mut [T] {
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
  /// # Safety
  ///
  /// The order of value in the dense buffer must be kept in sync with the order of indices in the index buffer.
  #[must_use]
  pub unsafe fn as_dense_mut_ptr(&mut self) -> *mut T {
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
    &self.indices
  }

  /// Extracts a mutable slice of the entire index buffer.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// # Safety
  ///
  /// The order of value in the dense buffer must be kept in sync with the order of indices in the index buffer.
  #[must_use]
  pub unsafe fn as_indices_mut_slice(&mut self) -> &mut [I] {
    &mut self.indices
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

  /// Returns an unsafe mutable pointer to the index buffer, or a dangling raw pointer valid for zero sized reads if the
  /// sparse set didn't allocate.
  ///
  /// The caller must ensure that the sparse set outlives the pointer this function returns, or else it will end up
  /// pointing to garbage. Modifying the sparse set may cause its buffer to be reallocated, which would also make any
  /// pointers to it invalid.
  ///
  /// # Safety
  ///
  /// The order of value in the dense buffer must be kept in sync with the order of indices in the index buffer.
  #[must_use]
  pub unsafe fn as_indices_mut_ptr(&mut self) -> *mut I {
    self.indices.as_mut_ptr()
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

  /// Returns the number of elements in the dense set, also referred to as its '`dense_len`'.
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

  /// Returns the number of elements in the sparse set, also referred to as its '`sparse_len`'.
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

  /// Clears the sparse set, returning all `(index, value)` pairs as an iterator.
  ///
  /// The allocated memory is kept for reuse.
  ///
  /// If the returned iterator is dropped before fully consumed, it drops the remaining `(index, value)` pairs. The
  /// returned iterator keeps a mutable borrow on the sparse set to optimize its implementation.
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
  /// assert!(set.drain().eq([(0, 1), (1, 2), (2, 3)]));
  /// ```
  pub fn drain(
    &mut self,
  ) -> impl Iterator<Item = (I, T)> + DoubleEndedIterator + ExactSizeIterator + '_ {
    self.sparse.clear();
    self.indices.drain(..).zip(self.dense.drain(..))
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
    self.dense.shrink_to_fit();
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
    self.sparse.shrink_to(min_capacity);
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
  pub fn values(&self) -> impl Iterator<Item = &T> + DoubleEndedIterator + ExactSizeIterator {
    self.dense.iter()
  }

  /// Returns an iterator that allows modifying each value.
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
  /// for elem in set.values_mut() {
  ///     *elem += 2;
  /// }
  ///
  /// assert!(set.values().eq(&[3, 4, 5]));
  /// ```
  pub fn values_mut(
    &mut self,
  ) -> impl Iterator<Item = &mut T> + DoubleEndedIterator + ExactSizeIterator {
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

  /// Returns the raw `usize` index into the dense buffer from the given index.
  ///
  /// This is meant to help with usecases of storing additional data outside of the sparse set in the same order.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Examples
  ///
  ///  ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  ///
  /// set.insert(0, 1);
  /// set.insert(1, 2);
  /// set.insert(20, 3);
  /// assert_eq!(Some(1), set.dense_index_of(1));
  /// assert_eq!(Some(2), set.dense_index_of(20));
  /// assert_eq!(None, set.dense_index_of(2));
  /// ```
  #[must_use]
  pub fn dense_index_of(&self, index: I) -> Option<usize> {
    self
      .sparse
      .get(index)
      .map(|dense_index| dense_index.get() - 1)
  }

  /// Gets the given index's corresponding entry in the sparse set for in-place manipulation.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  /// set.entry(1).or_insert(0);
  /// assert_eq!(set.get(1), Some(&0));
  /// ```
  #[must_use]
  pub fn entry(&mut self, index: I) -> Entry<'_, I, T, SA, DA> {
    match self.dense_index_of(index) {
      Some(dense_index) => Entry::Occupied(OccupiedEntry {
        dense_index,
        index,
        sparse_set: self,
      }),
      None => Entry::Vacant(VacantEntry {
        index,
        sparse_set: self,
      }),
    }
  }

  /// Gets the given index's corresponding immutable entry in the sparse set.
  ///
  /// This is primarily useful when you may need to get the full information about an entry, such as its stored index,
  /// dense index, and value.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::SparseSet;
  /// #
  /// let mut set = SparseSet::new();
  /// set.insert(0, 1);
  ///
  /// let entry = set.immutable_entry(0).unwrap();
  /// assert_eq!(entry.get(), &1);
  /// ```
  #[must_use]
  pub fn immutable_entry(&self, index: I) -> Option<ImmutableEntry<'_, I, T, SA, DA>> {
    self
      .dense_index_of(index)
      .map(|dense_index| ImmutableEntry {
        dense_index,
        index,
        sparse_set: self,
      })
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
      .dense_index_of(index)
      .map(|dense_index| unsafe { self.dense.get_unchecked(dense_index) })
  }

  /// Returns a reference to an element pointed to by the index, if it exists along with its index.
  ///
  /// This is useful over [`SparseSet::get`] when the index type has additional information that can be used to
  /// distinguish it from other indices.
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
  /// assert_eq!(Some((1, &2)), set.get_with_index(1));
  /// assert_eq!(None, set.get_with_index(3));
  ///
  /// set.remove(1);
  /// assert_eq!(None, set.get_with_index(1));
  /// ```
  #[must_use]
  pub fn get_with_index(&self, index: I) -> Option<(I, &T)> {
    self.dense_index_of(index).map(|dense_index| {
      (
        *unsafe { self.indices.get_unchecked(dense_index) },
        unsafe { self.dense.get_unchecked(dense_index) },
      )
    })
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
      .dense_index_of(index)
      .map(|dense_index| unsafe { self.dense.get_unchecked_mut(dense_index) })
  }

  /// Returns a mutable reference to an element pointed to by the index,, if it exists along with its index.
  ///
  /// This is useful over [`SparseSet::get_mut`] when the index type has additional information that can be used to
  /// distinguish it from other indices.
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
  /// if let Some((index, elem)) = set.get_mut_with_index(1) {
  ///   *elem = 42;
  /// }
  ///
  /// assert!(set.values().eq(&[1, 42, 3]));
  /// ```
  #[must_use]
  pub fn get_mut_with_index(&mut self, index: I) -> Option<(I, &mut T)> {
    self.dense_index_of(index).map(|dense_index| {
      (
        *unsafe { self.indices.get_unchecked_mut(dense_index) },
        unsafe { self.dense.get_unchecked_mut(dense_index) },
      )
    })
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
  /// assert_eq!(iterator.next(), Some(0));
  /// assert_eq!(iterator.next(), Some(1));
  /// assert_eq!(iterator.next(), Some(2));
  /// assert_eq!(iterator.next(), None);
  /// ```
  pub fn indices(&self) -> impl Iterator<Item = I> + DoubleEndedIterator + ExactSizeIterator + '_ {
    self.indices.iter().cloned()
  }

  /// Returns an iterator over the sparse set's indices and values as pairs.
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
  /// assert!(set.iter().eq([(0, &1), (1, &2), (2, &3)]));
  /// ```
  pub fn iter(&self) -> impl Iterator<Item = (I, &T)> + DoubleEndedIterator + ExactSizeIterator {
    self.indices.iter().cloned().zip(self.dense.iter())
  }

  /// Returns an iterator that allows modifying each value as an `(index, value)` pair.
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
  /// assert!(set.iter_mut().eq([(0, &mut 1), (1, &mut 2), (2, &mut 3)]));
  /// ```
  pub fn iter_mut(
    &mut self,
  ) -> impl Iterator<Item = (I, &mut T)> + DoubleEndedIterator + ExactSizeIterator {
    self.indices.iter().cloned().zip(self.dense.iter_mut())
  }

  /// Inserts an element at position `index` within the sparse set.
  ///
  /// If a value already existed at `index`, it will be replaced and returned. The corresponding index will also be
  /// replaced with the given index allowing indices to store additional information outside of their indexing behavior.
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
  pub fn insert(&mut self, index: I, value: T) -> Option<T> {
    self.insert_with_index(index, value).map(|(_, value)| value)
  }

  /// Inserts an element at position `index` within the sparse set.
  ///
  /// If a value already existed at `index`, it will be replaced and returned. The corresponding index will also be
  /// replaced with the given index allowing indices to store additional information outside of their indexing behavior.
  ///
  /// If `index` is greater than `sparse_capacity`, then an allocation will take place.
  ///
  /// This is useful over [`SparseSet::insert`] when the index type has additional information that can be used to
  /// distinguish it from other indices.
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
  /// set.insert_with_index(0, 1);
  /// set.insert_with_index(1, 4);
  /// set.insert_with_index(2, 2);
  /// set.insert_with_index(3, 3);
  ///
  /// assert!(set.values().eq(&[1, 4, 2, 3]));
  /// set.insert_with_index(20, 5);
  /// assert!(set.values().eq(&[1, 4, 2, 3, 5]));
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn insert_with_index(&mut self, mut index: I, mut value: T) -> Option<(I, T)> {
    match self.dense_index_of(index) {
      Some(dense_index) => {
        mem::swap(&mut index, unsafe {
          self.indices.get_unchecked_mut(dense_index)
        });
        mem::swap(&mut value, unsafe {
          self.dense.get_unchecked_mut(dense_index)
        });
        Some((index, value))
      }
      None => {
        self.dense.push(value);
        self.indices.push(index);
        let _ = self.sparse.insert(index, unsafe {
          NonZeroUsize::new_unchecked(self.dense_len())
        });
        None
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
    self.remove_with_index(index).map(|(_, value)| value)
  }

  /// Removes and returns the element at position `index` within the sparse set, if it exists along with its index.
  ///
  /// This is useful over [`SparseSet::remove`] when the index type has additional information that can be used to
  /// distinguish it from other indices.
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
  /// assert_eq!(set.remove_with_index(1), Some((1, 2)));
  /// assert!(set.values().eq(&[1, 3]));
  /// ```
  #[must_use]
  pub fn remove_with_index(&mut self, index: I) -> Option<(I, T)> {
    self
      .sparse
      .remove(index)
      .map(|dense_index| unsafe { self.remove_at_dense_index(dense_index.get() - 1) })
  }

  unsafe fn remove_at_dense_index(&mut self, dense_index: usize) -> (I, T) {
    let index = self.indices.swap_remove(dense_index);
    let value = self.dense.swap_remove(dense_index);

    if dense_index != self.dense.len() {
      let swapped_index: usize = (*unsafe { self.indices.get_unchecked(dense_index) }).into();
      *unsafe { self.sparse.get_unchecked_mut(swapped_index) } =
        Some(unsafe { NonZeroUsize::new_unchecked(dense_index + 1) });
    }

    (index, value)
  }
}

impl<I, T, SA: Allocator, DA: Allocator> AsRef<Self> for SparseSet<I, T, SA, DA> {
  fn as_ref(&self) -> &Self {
    self
  }
}

impl<I, T, SA: Allocator, DA: Allocator> AsMut<Self> for SparseSet<I, T, SA, DA> {
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
    &self.dense
  }
}

impl<I, T, SA: Allocator, DA: Allocator> DerefMut for SparseSet<I, T, SA, DA> {
  fn deref_mut(&mut self) -> &mut [T] {
    &mut self.dense
  }
}

impl<I: Debug + SparseSetIndex, T: Debug, SA: Allocator, DA: Allocator> Debug
  for SparseSet<I, T, SA, DA>
{
  fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
    formatter.debug_map().entries(self.iter()).finish()
  }
}

#[cfg(not(no_global_oom_handling))]
impl<'a, I: SparseSetIndex, T: Copy + 'a, SA: Allocator + 'a, DA: Allocator + 'a> Extend<(I, &'a T)>
  for SparseSet<I, T, SA, DA>
{
  fn extend<Iter: IntoIterator<Item = (I, &'a T)>>(&mut self, iter: Iter) {
    for (index, &value) in iter {
      let _ = self.insert(index, value);
    }
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T, SA: Allocator, DA: Allocator> Extend<(I, T)>
  for SparseSet<I, T, SA, DA>
{
  fn extend<Iter: IntoIterator<Item = (I, T)>>(&mut self, iter: Iter) {
    for (index, value) in iter {
      mem::drop(self.insert(index, value));
    }
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T, const N: usize> From<[(I, T); N]> for SparseSet<I, T> {
  fn from(slice: [(I, T); N]) -> Self {
    let mut set = Self::with_capacity(slice.len(), slice.len());

    for (index, value) in slice {
      mem::drop(set.insert(index, value));
    }

    set
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T> FromIterator<(I, T)> for SparseSet<I, T> {
  fn from_iter<Iter: IntoIterator<Item = (I, T)>>(iter: Iter) -> Self {
    let iter = iter.into_iter();
    let capacity = iter
      .size_hint()
      .1
      .map_or_else(|| iter.size_hint().0, |size_hint| size_hint);
    let mut set = Self::with_capacity(capacity, capacity);

    for (index, value) in iter {
      mem::drop(set.insert(index, value));
    }

    set
  }
}

impl<I: Hash + SparseSetIndex, T: Hash, SA: Allocator, DA: Allocator> Hash
  for SparseSet<I, T, SA, DA>
{
  fn hash<H: Hasher>(&self, state: &mut H) {
    for index in self.sparse.iter().flatten() {
      unsafe { self.sparse.get_unchecked(index.get() - 1) }.hash(state);
      unsafe { self.indices.get_unchecked(index.get() - 1) }.hash(state);
      unsafe { self.dense.get_unchecked(index.get() - 1) }.hash(state);
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
  type Item = (I, T);
  type IntoIter = impl Iterator<Item = Self::Item> + DoubleEndedIterator + ExactSizeIterator;

  #[inline]
  fn into_iter(self) -> Self::IntoIter {
    self.indices.into_iter().zip(self.dense.into_iter())
  }
}

impl<'a, I: SparseSetIndex, T, SA: Allocator, DA: Allocator> IntoIterator
  for &'a SparseSet<I, T, SA, DA>
{
  type Item = (I, &'a T);
  type IntoIter = impl Iterator<Item = Self::Item> + DoubleEndedIterator + ExactSizeIterator;

  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

impl<'a, I: SparseSetIndex, T, SA: Allocator, DA: Allocator> IntoIterator
  for &'a mut SparseSet<I, T, SA, DA>
{
  type Item = (I, &'a mut T);
  type IntoIter = impl Iterator<Item = Self::Item> + DoubleEndedIterator + ExactSizeIterator;

  fn into_iter(self) -> Self::IntoIter {
    self.iter_mut()
  }
}

impl<I: PartialEq + SparseSetIndex, T: PartialEq, SA: Allocator, DA: Allocator> PartialEq
  for SparseSet<I, T, SA, DA>
{
  fn eq(&self, other: &Self) -> bool {
    if self.indices.len() != other.indices.len() {
      return false;
    }

    for index in &self.indices {
      match (self.dense_index_of(*index), other.dense_index_of(*index)) {
        (Some(index), Some(other_index)) => {
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

/// An immutable view into an entry in a sparse set.
///
/// This differs from the [`Entry`] APIs in that this is purely an immutable view.
pub struct ImmutableEntry<'a, I, T, SA: Allocator = Global, DA: Allocator = Global> {
  /// The raw `usize` index into the dense buffer for this entry.
  dense_index: usize,

  /// The index this entry was created from.
  index: I,

  /// A reference to the sparse set this entry was created for.
  sparse_set: &'a SparseSet<I, T, SA, DA>,
}

impl<I: SparseSetIndex, T, SA: Allocator, DA: Allocator> ImmutableEntry<'_, I, T, SA, DA> {
  /// Returns the raw `usize` index into the dense buffer for this entry.
  #[must_use]
  pub fn dense_index(&self) -> usize {
    self.dense_index
  }

  /// Returns an immutable reference to the value for this entry.
  #[must_use]
  pub fn get(&self) -> &T {
    unsafe { self.sparse_set.dense.get_unchecked(self.dense_index) }
  }

  /// The index used to create this entry.
  ///
  /// This index may be different from the one currently stored (see [`OccupiedEntry::stored_index`]), but both will
  /// have the same behavior with respect to [`SparseSetIndex`].
  #[must_use]
  pub fn entry_index(&self) -> I {
    self.index
  }

  /// The index stored for this index..
  ///
  /// This index may be different from the one used to create this entry (see [`OccupiedEntry::entry_index`]), but both
  /// will have the same behavior with respect to [`SparseSetIndex`].
  #[must_use]
  pub fn stored_index(&self) -> I {
    *unsafe { self.sparse_set.indices.get_unchecked(self.dense_index) }
  }
}

impl<I: Debug + SparseSetIndex, T: Debug, SA: Allocator, DA: Allocator> Debug
  for ImmutableEntry<'_, I, T, SA, DA>
{
  fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("ImmutableEntry")
      .field("index", &self.entry_index())
      .field("value", self.get())
      .finish()
  }
}

/// A view into a single entry in a sparse set, which may be either vacant or occupied.
///
/// This is constructed from the [`SparseSet::entry`] function.
pub enum Entry<'a, I, T, SA: Allocator = Global, DA: Allocator = Global> {
  /// A vacant entry.
  Vacant(VacantEntry<'a, I, T, SA, DA>),

  /// An occupied entry.
  Occupied(OccupiedEntry<'a, I, T, SA, DA>),
}

impl<'a, I: SparseSetIndex, T, SA: Allocator, DA: Allocator> Entry<'a, I, T, SA, DA> {
  /// Provides in-place mutable access to an occupied entry before any potential inserts into the sparse set.
  #[must_use]
  pub fn and_modify<F: FnOnce(&mut T)>(self, function: F) -> Self {
    match self {
      Entry::Vacant(entry) => Entry::Vacant(entry),
      Entry::Occupied(mut entry) => {
        function(entry.get_mut());
        Entry::Occupied(entry)
      }
    }
  }

  /// The index used to create this entry.
  #[must_use]
  pub fn entry_index(&self) -> I {
    match self {
      Entry::Vacant(entry) => entry.entry_index(),
      Entry::Occupied(entry) => entry.entry_index(),
    }
  }

  /// Ensures a value is in the entry by inserting the default if empty, and returns a mutable reference to the value in
  /// the entry.
  pub fn or_insert(self, default: T) -> &'a mut T {
    match self {
      Entry::Vacant(entry) => entry.insert(default),
      Entry::Occupied(entry) => entry.into_mut(),
    }
  }

  /// Ensures a value is in the entry by inserting the result of the default function if empty, and returns a mutable
  /// reference to the value in the entry.
  pub fn or_insert_with<F: FnOnce() -> T>(self, default: F) -> &'a mut T {
    match self {
      Entry::Vacant(entry) => entry.insert(default()),
      Entry::Occupied(entry) => entry.into_mut(),
    }
  }

  /// Sets the value of the entry, and returns an [`OccupiedEntry`].
  pub fn insert_entry(self, value: T) -> OccupiedEntry<'a, I, T, SA, DA> {
    match self {
      Entry::Vacant(entry) => entry.insert_entry(value),
      Entry::Occupied(mut entry) => {
        mem::drop(entry.insert(value));
        entry
      }
    }
  }
}

impl<I: Debug + SparseSetIndex, T: Debug, SA: Allocator, DA: Allocator> Debug
  for Entry<'_, I, T, SA, DA>
{
  fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
    match self {
      Entry::Vacant(entry) => formatter.debug_tuple("Entry").field(entry).finish(),
      Entry::Occupied(entry) => formatter.debug_tuple("Entry").field(entry).finish(),
    }
  }
}

/// A view into a vacant entry in a sparse set.
pub struct VacantEntry<'a, I, T, SA: Allocator = Global, DA: Allocator = Global> {
  /// The index this entry was created from.
  index: I,

  /// A reference to the sparse set this entry was created for.
  sparse_set: &'a mut SparseSet<I, T, SA, DA>,
}

impl<'a, I: SparseSetIndex, T, SA: Allocator, DA: Allocator> VacantEntry<'a, I, T, SA, DA> {
  /// The index used to create this entry.
  #[must_use]
  pub fn entry_index(&self) -> I {
    self.index
  }

  /// Inserts the given value into this entry, returning a mutable reference to it.
  pub fn insert(mut self, value: T) -> &'a mut T {
    let dense_index = self.insert_raw(value);
    unsafe { self.sparse_set.dense.get_unchecked_mut(dense_index) }
  }

  /// Inserts the given value into this entry, returning an occupied entry.
  pub fn insert_entry(mut self, value: T) -> OccupiedEntry<'a, I, T, SA, DA> {
    let dense_index = self.insert_raw(value);
    OccupiedEntry {
      dense_index,
      index: self.index,
      sparse_set: self.sparse_set,
    }
  }

  /// Inserts the given value into this entry without consuming it.
  #[must_use]
  fn insert_raw(&mut self, value: T) -> usize {
    self.sparse_set.dense.push(value);
    self.sparse_set.indices.push(self.index);
    let _ = self.sparse_set.sparse.insert(self.index, unsafe {
      NonZeroUsize::new_unchecked(self.sparse_set.dense_len())
    });
    self.sparse_set.dense_len() - 1
  }
}

impl<I: Debug, T, SA: Allocator, DA: Allocator> Debug for VacantEntry<'_, I, T, SA, DA> {
  fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
    formatter
      .debug_tuple("VacantEntry")
      .field(&self.index)
      .finish()
  }
}

/// A view into an occupied entry in a sparse set.
pub struct OccupiedEntry<'a, I, T, SA: Allocator = Global, DA: Allocator = Global> {
  /// The raw `usize` index into the dense buffer for this entry.
  dense_index: usize,

  /// The index this entry was created from.
  index: I,

  /// A reference to the sparse set this entry was created for.
  sparse_set: &'a mut SparseSet<I, T, SA, DA>,
}

impl<'a, I: SparseSetIndex, T, SA: Allocator, DA: Allocator> OccupiedEntry<'a, I, T, SA, DA> {
  /// Returns the raw `usize` index into the dense buffer for this entry.
  #[must_use]
  pub fn dense_index(&self) -> usize {
    self.dense_index
  }

  /// Returns an immutable reference to the value for this entry.
  #[must_use]
  pub fn get(&self) -> &T {
    unsafe { self.sparse_set.dense.get_unchecked(self.dense_index) }
  }

  /// Returns an mutable reference to the value for this entry.
  #[must_use]
  pub fn get_mut(&mut self) -> &mut T {
    unsafe { self.sparse_set.dense.get_unchecked_mut(self.dense_index) }
  }

  /// Consumes the entry, returning a reference to the entry's value tied to the lifetime of the sparse set.
  #[must_use]
  pub fn into_mut(self) -> &'a mut T {
    unsafe { self.sparse_set.dense.get_unchecked_mut(self.dense_index) }
  }

  /// The index used to create this entry.
  ///
  /// This index may be different from the one currently stored (see [`OccupiedEntry::stored_index`]), but both will
  /// have the same behavior with respect to [`SparseSetIndex`].
  #[must_use]
  pub fn entry_index(&self) -> I {
    self.index
  }

  /// The index stored for this index..
  ///
  /// This index may be different from the one used to create this entry (see [`OccupiedEntry::entry_index`]), but both
  /// will have the same behavior with respect to [`SparseSetIndex`].
  #[must_use]
  pub fn stored_index(&self) -> I {
    *unsafe { self.sparse_set.indices.get_unchecked(self.dense_index) }
  }

  /// Inserts the given value into this entry, returning the existing value.
  pub fn insert(&mut self, value: T) -> T {
    self.insert_with_index(value).1
  }

  /// Inserts the given value into this entry, returning the existing value and its index.
  pub fn insert_with_index(&mut self, mut value: T) -> (I, T) {
    let mut index = self.index;
    mem::swap(&mut index, unsafe {
      self.sparse_set.indices.get_unchecked_mut(self.dense_index)
    });
    mem::swap(&mut value, unsafe {
      self.sparse_set.dense.get_unchecked_mut(self.dense_index)
    });
    (index, value)
  }

  /// Removes and returns the value associated with this entry, consuming it.
  pub fn remove(self) -> T {
    self.remove_with_index().1
  }

  /// Removes and returns the value associated with this entry along with its index, consuming it.
  pub fn remove_with_index(self) -> (I, T) {
    let _ = self.sparse_set.sparse.remove(self.index);
    unsafe { self.sparse_set.remove_at_dense_index(self.dense_index) }
  }
}

impl<I: Debug + SparseSetIndex, T: Debug, SA: Allocator, DA: Allocator> Debug
  for OccupiedEntry<'_, I, T, SA, DA>
{
  fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("OccupiedEntry")
      .field("index", &self.entry_index())
      .field("value", self.get())
      .finish()
  }
}

#[cfg(test)]
mod test {
  use std::{cell::RefCell, collections::hash_map::DefaultHasher, rc::Rc};

  use coverage_helper::test;

  use super::*;

  #[derive(Clone, Copy, Debug, Eq, PartialEq)]
  struct Index {
    index: usize,
    other: u32,
  }

  impl Index {
    pub fn new(index: usize, other: u32) -> Self {
      Index { index, other }
    }
  }

  impl From<Index> for usize {
    fn from(value: Index) -> Self {
      value.index
    }
  }

  impl SparseSetIndex for Index {}

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
    let _set: SparseSet<usize, usize> = SparseSet::with_capacity(0, 1);
  }

  #[should_panic]
  #[test]
  fn test_with_capacity_sparse_overflow() {
    let _set: SparseSet<usize, usize> = SparseSet::with_capacity(usize::MAX, 0);
  }

  #[should_panic]
  #[test]
  fn test_with_capacity_dense_overflow() {
    let _set: SparseSet<usize, usize> = SparseSet::with_capacity(0, usize::MAX);
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
    let _ = set.insert(0, 1);
    assert_eq!(set.as_dense_slice(), &[1]);
  }

  #[test]
  fn test_as_dense_mut_slice() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.insert(0, 1);
    assert_eq!(unsafe { set.as_dense_mut_slice() }, &mut [1]);
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
      unsafe { set.as_dense_mut_ptr() },
      unsafe { set.as_dense_mut_slice() }.as_mut_ptr()
    );
  }

  #[test]
  fn test_as_indices_slice() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.insert(0, 1);
    assert_eq!(set.as_indices_slice(), &[0]);
  }

  #[test]
  fn test_as_indices_ptr() {
    let set: SparseSet<usize, usize> = SparseSet::with_capacity(10, 10);
    assert_eq!(set.as_indices_ptr(), set.as_indices_slice().as_ptr());
  }

  #[test]
  fn test_as_indices_ptr_mut() {
    let mut set: SparseSet<usize, usize> = SparseSet::with_capacity(10, 10);
    assert_eq!(
      unsafe { set.as_indices_mut_ptr() },
      unsafe { set.as_indices_mut_slice() }.as_mut_ptr()
    );
  }

  #[test]
  fn test_clear() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    set.clear();

    assert!(set.is_empty());
  }

  #[test]
  fn test_contains() {
    let mut set = SparseSet::new();
    assert!(!set.contains(0));
    let _ = set.insert(0, 1);
    assert!(set.contains(0));
    let _ = set.remove(0);
    assert!(!set.contains(0));
  }

  #[test]
  fn test_dense_index_of() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(20, 3);

    assert_eq!(set.dense_index_of(0), Some(0));
    assert_eq!(set.dense_index_of(1), Some(1));
    assert_eq!(set.dense_index_of(2), None);
    assert_eq!(set.dense_index_of(20), Some(2));

    let _ = set.remove(1);
    assert_eq!(set.dense_index_of(20), Some(1));
  }

  #[test]
  fn test_drain() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    assert!(set.drain().eq([(0, 1), (1, 2), (2, 3)]));
    assert!(set.is_empty());
  }

  #[test]
  fn test_get() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    assert_eq!(set.get(0), Some(&1));
    assert_eq!(set.get(2), Some(&3));
    assert_eq!(set.get(100), None);
  }

  #[test]
  fn test_get_with_index() {
    let mut set = SparseSet::new();
    let _ = set.insert(Index::new(0, 0), 1);
    let _ = set.insert(Index::new(1, 0), 2);
    let _ = set.insert(Index::new(2, 0), 3);

    assert_eq!(
      set.get_with_index(Index::new(0, 1)),
      Some((Index::new(0, 0), &1))
    );
    assert_eq!(
      set.get_with_index(Index::new(2, 0)),
      Some((Index::new(2, 0), &3))
    );
    assert_eq!(set.get_with_index(Index::new(100, 0)), None);
  }

  #[test]
  fn test_get_mut() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    let value = set.get_mut(2);
    assert_eq!(value, Some(&mut 3));
    *value.unwrap() = 10;

    assert_eq!(set.get(2), Some(&10));
  }

  #[test]
  fn test_get_mut_with_index() {
    let mut set = SparseSet::new();
    let _ = set.insert(Index::new(0, 0), 1);
    let _ = set.insert(Index::new(1, 0), 2);
    let _ = set.insert(Index::new(2, 0), 3);

    let value = set.get_mut_with_index(Index::new(2, 1));
    assert_eq!(value, Some((Index::new(2, 0), &mut 3)));
    *value.unwrap().1 = 10;

    assert_eq!(set.get(Index::new(2, 0)), Some(&10));
  }

  #[test]
  fn test_indices() {
    let mut set = SparseSet::new();
    assert!(set.indices().eq([]));

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    assert!(set.indices().eq([0, 1, 2]));
  }

  #[test]
  fn test_insert_capacity_increases() {
    let mut set = SparseSet::with_capacity(1, 1);
    let _ = set.insert(0, 1);
    assert_eq!(set.sparse_capacity(), 1);
    assert_eq!(set.dense_capacity(), 1);

    let _ = set.insert(1, 2);
    assert!(set.sparse_capacity() >= 2);
    assert!(set.dense_capacity() >= 2);

    assert_eq!(set.get(0), Some(&1));
    assert_eq!(set.get(1), Some(&2));
  }

  #[test]
  fn test_insert_len_increases() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 1);

    let _ = set.insert(1, 2);
    assert_eq!(set.dense_len(), 2);
    assert_eq!(set.sparse_len(), 2);

    let _ = set.insert(100, 101);
    assert_eq!(set.dense_len(), 3);
    assert_eq!(set.sparse_len(), 101);
  }

  #[test]
  fn test_insert_overwrites() {
    let mut set = SparseSet::new();
    assert_eq!(set.insert(0, 1), None);
    assert_eq!(set.get(0), Some(&1));

    assert_eq!(set.insert(0, 2), Some(1));
    assert_eq!(set.get(0), Some(&2));
  }

  #[test]
  fn test_insert_with_index_capacity_increases() {
    let mut set = SparseSet::with_capacity(1, 1);
    let _ = set.insert_with_index(0, 1);
    assert_eq!(set.sparse_capacity(), 1);
    assert_eq!(set.dense_capacity(), 1);

    let _ = set.insert_with_index(1, 2);
    assert!(set.sparse_capacity() >= 2);
    assert!(set.dense_capacity() >= 2);

    assert_eq!(set.get(0), Some(&1));
    assert_eq!(set.get(1), Some(&2));
  }

  #[test]
  fn test_insert_with_index_len_increases() {
    let mut set = SparseSet::new();
    let _ = set.insert_with_index(0, 1);
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 1);

    let _ = set.insert_with_index(1, 2);
    assert_eq!(set.dense_len(), 2);
    assert_eq!(set.sparse_len(), 2);

    let _ = set.insert_with_index(100, 101);
    assert_eq!(set.dense_len(), 3);
    assert_eq!(set.sparse_len(), 101);
  }

  #[test]
  fn test_insert_with_index_overwrites() {
    let mut set = SparseSet::new();
    assert_eq!(set.insert_with_index(Index::new(0, 0), 1), None);
    assert_eq!(set.get(Index::new(0, 0)), Some(&1));

    assert_eq!(
      set.insert_with_index(Index::new(0, 1), 2),
      Some((Index::new(0, 0), 1))
    );
    assert_eq!(set.get(Index::new(0, 0)), Some(&2));
  }

  #[test]
  fn test_iter() {
    let mut set = SparseSet::new();
    assert!(set.iter().eq([]));

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    assert!(set.iter().eq([(0, &1), (1, &2), (2, &3)]));
  }

  #[test]
  fn test_iter_mut() {
    let mut set = SparseSet::new();
    assert!(set.iter_mut().eq([]));

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    assert!(set.iter_mut().eq([(0, &mut 1), (1, &mut 2), (2, &mut 3)]));

    let value = set.iter_mut().next().unwrap();
    *(value.1) = 100;

    assert_eq!(set.first(), Some(&100));
  }

  #[test]
  fn test_is_empty() {
    let mut set = SparseSet::new();
    assert!(set.is_empty());

    let _ = set.insert(0, 1);
    assert!(!set.is_empty());

    let _ = set.remove(0);
    assert!(set.is_empty());
  }

  #[test]
  fn test_dense_len() {
    let mut set = SparseSet::new();
    assert_eq!(set.dense_len(), 0);

    let _ = set.insert(0, 1);
    assert_eq!(set.dense_len(), 1);
  }

  #[test]
  fn test_sparse_len() {
    let mut set = SparseSet::new();
    assert_eq!(set.sparse_len(), 0);

    let _ = set.insert(0, 1);
    assert_eq!(set.sparse_len(), 1);

    let _ = set.insert(100, 1);
    assert_eq!(set.sparse_len(), 101);
  }

  #[test]
  fn test_remove_can_return_none() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    assert_eq!(set.remove(1), None);
    assert_eq!(set.remove(100), None);
  }

  #[test]
  fn test_remove_can_return_some() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    assert_eq!(set.remove(0), Some(1));
  }

  #[test]
  fn test_remove_len_decreases() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);

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
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    assert!(set.values().eq(&[1, 2, 3]));

    let _ = set.remove(0);
    assert!(set.values().eq(&[3, 2]));
  }

  #[test]
  fn test_remove_with_index_can_return_none() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    assert_eq!(set.remove_with_index(1), None);
    assert_eq!(set.remove_with_index(100), None);
  }

  #[test]
  fn test_remove_with_index_can_return_some() {
    let mut set = SparseSet::new();
    let _ = set.insert(Index::new(0, 0), 1);
    assert_eq!(
      set.remove_with_index(Index::new(0, 1)),
      Some((Index::new(0, 0), 1))
    );
  }

  #[test]
  fn test_remove_with_index_len_decreases() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);

    assert_eq!(set.dense_len(), 2);
    assert_eq!(set.sparse_len(), 2);
    assert_eq!(set.remove_with_index(0), Some((0, 1)));
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 2);
    assert_eq!(set.remove_with_index(0), None);
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 2);
  }

  #[test]
  fn test_remove_with_index_swaps_with_last() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    assert!(set.values().eq(&[1, 2, 3]));

    let _ = set.remove_with_index(0);
    assert!(set.values().eq(&[3, 2]));
  }

  #[test]
  fn test_reserve_dense() {
    let mut set = SparseSet::new();
    assert_eq!(set.dense_capacity(), 0);

    set.reserve_dense(3);
    let capacity = set.dense_capacity();
    assert!(capacity >= 2);

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);

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

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);

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

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);

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

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);

    set.reserve_exact_sparse(1);
    assert_eq!(set.sparse_capacity(), capacity);
  }

  #[test]
  fn test_shrink_to_fit_dense() {
    let mut set = SparseSet::with_capacity(3, 3);
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    assert_eq!(set.dense_capacity(), 3);
    let _ = set.remove(2);
    set.shrink_to_fit_dense();
    assert_eq!(set.dense_capacity(), 2);
  }

  #[test]
  fn test_shrink_to_fit_sparse() {
    let mut set = SparseSet::with_capacity(3, 3);
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

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
    let _ = set.insert(0, 1);
    assert_eq!(set.dense_capacity(), 3);
    set.shrink_to_dense(1);
    assert_eq!(set.dense_capacity(), 1);
  }

  #[test]
  fn test_shrink_to_dense_cannot_reduce() {
    let mut set = SparseSet::with_capacity(3, 3);
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    assert_eq!(set.dense_capacity(), 3);
    set.shrink_to_dense(0);
    assert_eq!(set.dense_capacity(), 3);
  }

  #[test]
  fn test_shrink_to_sparse_can_reduce() {
    let mut set = SparseSet::with_capacity(3, 3);
    let _ = set.insert(0, 1);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 1);
    set.shrink_to_sparse(1);
    assert_eq!(set.sparse_capacity(), 1);
    assert_eq!(set.sparse_len(), 1);
  }

  #[test]
  fn test_shrink_to_sparse_cannot_reduce() {
    let mut set = SparseSet::with_capacity(3, 3);
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
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

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);

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

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);

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

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);

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

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);

    assert!(set.try_reserve_exact_sparse(1).is_ok());
    assert_eq!(set.sparse_capacity(), capacity);
  }

  #[test]
  fn test_values() {
    let mut set = SparseSet::new();
    assert!(set.values().eq(&[]));

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    assert!(set.values().eq(&[1, 2, 3]));
  }

  #[test]
  fn test_values_mut() {
    let mut set = SparseSet::new();
    assert!(set.values_mut().eq(&[]));

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    assert!(set.values_mut().eq(&[1, 2, 3]));

    let value = set.values_mut().next().unwrap();
    *value = 100;

    assert_eq!(set.get(0), Some(&100));
  }

  #[test]
  fn test_as_ref() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    let reference: &SparseSet<_, _> = set.as_ref();
    assert_eq!(reference.first(), Some(&1));

    let reference: &[usize] = set.as_ref();
    assert_eq!(reference.first(), Some(&1));
  }

  #[test]
  fn test_as_mut() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    let reference: &mut SparseSet<_, _> = set.as_mut();
    assert_eq!(reference.first(), Some(&1));

    let reference: &mut [usize] = set.as_mut();
    assert_eq!(reference.first(), Some(&1));
  }

  #[test]
  fn test_clone() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

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
      mem::drop(set.insert(0, value.clone()));
      mem::drop(set.insert(1, value.clone()));
      mem::drop(set.insert(2, value));

      let _cloned_set = set.clone();
    }

    assert_eq!(*num_dropped.borrow(), 6);
  }

  #[test]
  fn test_debug() {
    let mut set = SparseSet::new();
    assert_eq!(format!("{:?}", set), "{}");

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    assert_eq!(format!("{:?}", set), "{0: 1, 1: 2, 2: 3}");
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
    let _ = set.insert(0, 1);

    assert_eq!(&*set, &[1]);
  }

  #[test]
  fn test_deref_mut() {
    let mut set: SparseSet<usize, usize> = SparseSet::default();
    let _ = set.insert(0, 1);

    assert_eq!(&mut *set, &mut [1]);
  }

  #[test]
  fn test_drop() {
    let num_dropped = Rc::new(RefCell::new(0));

    {
      let mut set = SparseSet::new();
      let value = Value(num_dropped.clone());
      mem::drop(set.insert(0, value.clone()));
      mem::drop(set.insert(1, value.clone()));
      mem::drop(set.insert(2, value));
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

    let _ = set_1.insert(0, 1);

    assert_ne!(set_1, set_2);

    let _ = set_2.insert(0, 2);

    assert_ne!(set_1, set_2);

    let _ = set_2.remove(0);
    let _ = set_2.insert(1, 2);

    assert_ne!(set_1, set_2);

    let _ = set_1.insert(1, 2);
    let _ = set_2.insert(0, 1);

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
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    assert_eq!(set[0], 1);
    assert_eq!(set[2], 3);
  }

  #[should_panic]
  #[test]
  fn test_index_panics() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    let _ = &set[100];
  }

  #[test]
  fn test_index_mut() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    let value = &mut set[2];
    assert_eq!(value, &mut 3);
    *value = 10;

    assert_eq!(set[2], 10);
  }

  #[should_panic]
  #[test]
  fn test_index_mut_panics() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);

    let _ = &mut set[100];
  }

  #[test]
  fn test_into_iterator() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    assert!(set.into_iter().eq([(0, 1), (1, 2), (2, 3)]));
  }

  #[test]
  fn test_into_iterator_ref() {
    let mut set = SparseSet::new();
    assert!((&set).into_iter().eq([]));

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    assert!((&set).into_iter().eq([(0, &1), (1, &2), (2, &3)]));
  }

  #[test]
  fn test_into_iterator_mut() {
    let mut set = SparseSet::new();
    assert!((&mut set).into_iter().eq([]));

    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    assert!((&mut set)
      .into_iter()
      .eq([(0, &mut 1), (1, &mut 2), (2, &mut 3)]));

    let value = (&mut set).into_iter().next().unwrap();
    *(value.1) = 100;

    assert_eq!(set.first(), Some(&100));
  }

  #[test]
  fn test_eq() {
    let mut set_1 = SparseSet::new();
    let mut set_2 = SparseSet::new();

    assert_eq!(set_1, set_2);

    let _ = set_1.insert(0, 1);

    assert_ne!(set_1, set_2);

    let _ = set_2.insert(0, 2);

    assert_ne!(set_1, set_2);

    let _ = set_2.remove(0);
    let _ = set_2.insert(1, 2);

    assert_ne!(set_1, set_2);

    let _ = set_1.insert(1, 2);
    let _ = set_2.insert(0, 1);

    assert_eq!(set_1, set_2);

    let _ = set_1.remove(0);
    let _ = set_2.remove(0);

    assert_eq!(set_1, set_2);
  }

  #[test]
  fn test_immutable_entry_dense_index() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let entry = match set.entry(1) {
      Entry::Vacant(_) => panic!("expected immutable entry"),
      Entry::Occupied(entry) => entry,
    };

    assert_eq!(entry.dense_index(), 1);
  }

  #[test]
  fn test_immutable_entry_get() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let entry = set.immutable_entry(1).unwrap();

    assert_eq!(entry.get(), &2);
  }

  #[test]
  fn test_immutable_entry_entry_index() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let entry = set.immutable_entry(1).unwrap();

    assert_eq!(entry.entry_index(), 1);
  }

  #[test]
  fn test_immutable_entry_stored_index() {
    let mut set = SparseSet::new();
    let _ = set.insert(Index::new(0, 0), 1);
    let _ = set.insert(Index::new(1, 0), 2);
    let _ = set.insert(Index::new(2, 0), 3);
    let entry = set.immutable_entry(Index::new(1, 1)).unwrap();

    assert_eq!(entry.entry_index(), Index::new(1, 1));
    assert_eq!(entry.stored_index(), Index::new(1, 0));
  }

  #[test]
  fn test_immutable_entry_debug() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let entry = set.immutable_entry(1).unwrap();
    assert_eq!(
      format!("{:?}", entry),
      "ImmutableEntry { index: 1, value: 2 }"
    );
  }

  #[test]
  fn test_entry_and_modify() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.insert(0, 1);

    assert_eq!(set.get(0), Some(&1));
    let _ = set.entry(0).and_modify(|value| *value += 1);
    assert_eq!(set.get(0), Some(&2));

    assert_eq!(set.get(1), None);
    let _ = set.entry(0).and_modify(|value| *value += 1);
    assert_eq!(set.get(1), None);
  }

  #[test]
  fn test_entry_or_insert() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.insert(0, 1);

    assert_eq!(set.get(0), Some(&1));
    let value = set.entry(0).or_insert(2);
    assert_eq!(value, &1);
    assert_eq!(set.get(0), Some(&1));

    assert_eq!(set.get(1), None);
    let value = set.entry(1).or_insert(2);
    assert_eq!(value, &2);
    assert_eq!(set.get(1), Some(&2));
  }

  #[test]
  fn test_entry_or_insert_with() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.insert(0, 1);

    assert_eq!(set.get(0), Some(&1));
    let value = set.entry(0).or_insert_with(|| 2);
    assert_eq!(value, &1);
    assert_eq!(set.get(0), Some(&1));

    assert_eq!(set.get(1), None);
    let value = set.entry(1).or_insert_with(|| 2);
    assert_eq!(value, &2);
    assert_eq!(set.get(1), Some(&2));
  }

  #[test]
  fn test_entry_entry_index() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.insert(0, 1);

    let entry = set.entry(0);
    assert_eq!(entry.entry_index(), 0);

    let entry = set.entry(1);
    assert_eq!(entry.entry_index(), 1);
  }

  #[test]
  fn test_entry_insert_entry() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.insert(0, 1);

    assert_eq!(set.get(0), Some(&1));
    let entry = set.entry(0).insert_entry(2);
    assert_eq!(entry.get(), &2);
    assert_eq!(set.get(0), Some(&2));

    assert_eq!(set.get(1), None);
    let entry = set.entry(0).insert_entry(2);
    assert_eq!(entry.get(), &2);
    assert_eq!(set.get(0), Some(&2));
  }

  #[test]
  fn test_entry_debug() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.insert(0, 1);

    let entry = set.entry(0);
    assert_eq!(
      format!("{:?}", entry),
      "Entry(OccupiedEntry { index: 0, value: 1 })"
    );

    let entry = set.entry(1);
    assert_eq!(format!("{:?}", entry), "Entry(VacantEntry(1))");
  }

  #[test]
  fn test_vacant_entry_entry_index() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let entry = match set.entry(0) {
      Entry::Vacant(entry) => entry,
      Entry::Occupied(_) => panic!("expected vacant entry"),
    };

    assert_eq!(entry.entry_index(), 0);
  }

  #[test]
  fn test_vacant_entry_insert() {
    let mut set = SparseSet::new();
    let entry = match set.entry(0) {
      Entry::Vacant(entry) => entry,
      Entry::Occupied(_) => panic!("expected vacant entry"),
    };

    assert_eq!(entry.insert(1), &mut 1);
    assert_eq!(set.get(0), Some(&1));
  }

  #[test]
  fn test_vacant_entry_insert_entry() {
    let mut set = SparseSet::new();
    let entry = match set.entry(0) {
      Entry::Vacant(entry) => entry,
      Entry::Occupied(_) => panic!("expected vacant entry"),
    };

    assert_eq!(entry.insert_entry(1).get(), &1);
    assert_eq!(set.get(0), Some(&1));
  }

  #[test]
  fn test_vacant_entry_debug() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let entry = match set.entry(0) {
      Entry::Vacant(entry) => entry,
      Entry::Occupied(_) => panic!("expected vacant entry"),
    };
    assert_eq!(format!("{:?}", entry), "VacantEntry(0)");
  }

  #[test]
  fn test_occupied_entry_dense_index() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let entry = match set.entry(1) {
      Entry::Vacant(_) => panic!("expected occupied entry"),
      Entry::Occupied(entry) => entry,
    };

    assert_eq!(entry.dense_index(), 1);
  }

  #[test]
  fn test_occupied_entry_get() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let entry = match set.entry(1) {
      Entry::Vacant(_) => panic!("expected occupied entry"),
      Entry::Occupied(entry) => entry,
    };

    assert_eq!(entry.get(), &2);
  }

  #[test]
  fn test_occupied_entry_get_mut() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let mut entry = match set.entry(1) {
      Entry::Vacant(_) => panic!("expected occupied entry"),
      Entry::Occupied(entry) => entry,
    };

    let value = entry.get_mut();
    assert_eq!(value, &mut 2);
    *value = 3;

    assert_eq!(set.get(1), Some(&3));
  }

  #[test]
  fn test_occupied_entry_into_mut() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let entry = match set.entry(1) {
      Entry::Vacant(_) => panic!("expected occupied entry"),
      Entry::Occupied(entry) => entry,
    };

    let value = entry.into_mut();
    assert_eq!(value, &mut 2);
    *value = 3;

    assert_eq!(set.get(1), Some(&3));
  }

  #[test]
  fn test_occupied_entry_entry_index() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let entry = match set.entry(1) {
      Entry::Vacant(_) => panic!("expected occupied entry"),
      Entry::Occupied(entry) => entry,
    };

    assert_eq!(entry.entry_index(), 1);
  }

  #[test]
  fn test_occupied_entry_stored_index() {
    let mut set = SparseSet::new();
    let _ = set.insert(Index::new(0, 0), 1);
    let _ = set.insert(Index::new(1, 0), 2);
    let _ = set.insert(Index::new(2, 0), 3);
    let entry = match set.entry(Index::new(1, 1)) {
      Entry::Vacant(_) => panic!("expected occupied entry"),
      Entry::Occupied(entry) => entry,
    };

    assert_eq!(entry.entry_index(), Index::new(1, 1));
    assert_eq!(entry.stored_index(), Index::new(1, 0));
  }

  #[test]
  fn test_occupied_entry_insert() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let mut entry = match set.entry(1) {
      Entry::Vacant(_) => panic!("expected occupied entry"),
      Entry::Occupied(entry) => entry,
    };

    assert_eq!(entry.insert(3), 2);
    assert_eq!(set.get(1), Some(&3));
  }

  #[test]
  fn test_occupied_entry_insert_with_index() {
    let mut set = SparseSet::new();
    let _ = set.insert(Index::new(0, 0), 1);
    let _ = set.insert(Index::new(1, 0), 2);
    let _ = set.insert(Index::new(2, 0), 3);
    let mut entry = match set.entry(Index::new(1, 1)) {
      Entry::Vacant(_) => panic!("expected occupied entry"),
      Entry::Occupied(entry) => entry,
    };

    assert_eq!(entry.stored_index(), Index::new(1, 0));
    assert_eq!(entry.insert_with_index(3), (Index::new(1, 0), 2));
    assert_eq!(entry.stored_index(), Index::new(1, 1));
    assert_eq!(
      set.get_with_index(Index::new(1, 0)),
      Some((Index::new(1, 1), &3))
    );
  }

  #[test]
  fn test_occupied_entry_remove() {
    let mut set = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let entry = match set.entry(1) {
      Entry::Vacant(_) => panic!("expected occupied entry"),
      Entry::Occupied(entry) => entry,
    };

    assert_eq!(entry.remove(), 2);
    assert_eq!(set.get(1), None);
  }

  #[test]
  fn test_occupied_entry_remove_with_index() {
    let mut set = SparseSet::new();
    let _ = set.insert(Index::new(0, 0), 1);
    let _ = set.insert(Index::new(1, 0), 2);
    let _ = set.insert(Index::new(2, 0), 3);
    let entry = match set.entry(Index::new(1, 1)) {
      Entry::Vacant(_) => panic!("expected occupied entry"),
      Entry::Occupied(entry) => entry,
    };

    assert_eq!(entry.remove_with_index(), (Index::new(1, 0), 2));
    assert_eq!(set.get_with_index(Index::new(1, 0)), None,);
  }

  #[test]
  fn test_occupied_entry_debug() {
    let mut set: SparseSet<usize, usize> = SparseSet::new();
    let _ = set.insert(0, 1);
    let _ = set.insert(1, 2);
    let _ = set.insert(2, 3);
    let entry = match set.entry(1) {
      Entry::Vacant(_) => panic!("expected occupied entry"),
      Entry::Occupied(entry) => entry,
    };
    assert_eq!(
      format!("{:?}", entry),
      "OccupiedEntry { index: 1, value: 2 }"
    );
  }
}
