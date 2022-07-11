//! A type-erased, sparsely populated set, written `AnySparseSet<I>`, where `I` is the index type.
//!
//! `I` must implement `SparseSetIndex`, in particular, it must be able to be converted to a `usize` index.
//!
//! See [this article](https://research.swtch.com/sparse) on more details behind the data structure.

#![allow(unsafe_code)]

mod any_sparse_set_mut;
mod any_sparse_set_ref;

use std::{
  alloc::{Allocator, Global, Layout},
  any::TypeId,
  fmt::{self, Debug, Formatter},
  num::NonZeroUsize,
};

pub use any_vec::{
  any_value, element, mem, ops, traits, ElementIterator, Iter, IterMut, IterRef, SatisfyTraits,
};
use any_vec::{
  any_value::{AnyValue, AnyValueWrapper},
  element::{ElementMut, ElementRef},
  mem::{Heap, MemBuilder, MemBuilderSizeable, MemResizable},
  ops::SwapRemove,
  traits::{Cloneable, None, Trait},
  AnyVec,
};

use crate::{SparseSetIndex, SparseVec};
pub use any_sparse_set_mut::*;
pub use any_sparse_set_ref::*;

/// A type-erased, sparsely populated set, written `AnySparseSet<I>`, where `I` is the index type.
///
/// For operation complexity notes, *n* is the number of values in the sparse set and *m* is the value of the largest
/// index in the sparse set. Note that *m* will always be at least as large as *n*.
#[allow(missing_debug_implementations)]
pub struct AnySparseSet<
  I,
  Traits: ?Sized + Trait = dyn None,
  SA: Allocator = Global,
  IA: Allocator = Global,
  M: MemBuilder = Heap,
> {
  /// The dense buffer, i.e., the buffer containing the actual data values of some type `T` that is erased.
  dense: AnyVec<Traits, M>,

  /// The sparse buffer, i.e., the buffer where each index may correspond to an index into `dense`.
  sparse: SparseVec<I, NonZeroUsize, SA>,

  /// All the existing indices in `sparse`.
  ///
  /// The indices here will always be in order based on the `dense` buffer.
  indices: Vec<I, IA>,
}

impl<I, Traits: ?Sized + Trait> AnySparseSet<I, Traits, Global, Global, Heap> {
  /// Constructs a new, empty `AnySparseSet<I, Traits>`.
  ///
  /// The sparse set will not allocate until elements are inserted into it.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::AnySparseSet;
  /// #
  /// # #[allow(unused_mut)]
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  /// ```
  #[must_use]
  pub fn new<T: SatisfyTraits<Traits> + 'static>() -> Self {
    AnySparseSet::new_in::<T>(Global, Global, Heap)
  }

  /// Constructs a new, empty `AnySparseSet<I, Traits>` with the specified capacity.
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(11, 10);
  ///
  /// // The sparse set contains no items, even though it has capacity for more.
  /// assert_eq!(set.dense_len(), 0);
  /// assert_eq!(set.sparse_len(), 0);
  /// assert_eq!(set.dense_capacity(), 10);
  /// assert_eq!(set.sparse_capacity(), 11);
  ///
  /// // These are all done without reallocating...
  /// for i in 0..10 {
  ///   set.insert(i, AnyValueWrapper::new(i as usize));
  /// }
  ///
  /// assert_eq!(set.dense_len(), 10);
  /// assert_eq!(set.sparse_len(), 10);
  /// assert_eq!(set.dense_capacity(), 10);
  /// assert_eq!(set.sparse_capacity(), 11);
  ///
  /// // ...but this will make the sparse set reallocate.
  /// set.insert(10, AnyValueWrapper::new(10usize));
  /// set.insert(11, AnyValueWrapper::new(11usize));
  /// assert_eq!(set.dense_len(), 12);
  /// assert_eq!(set.sparse_len(), 12);
  /// assert!(set.dense_capacity() >= 12);
  /// assert!(set.sparse_capacity() >= 12);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  #[must_use]
  pub fn with_capacity<T: SatisfyTraits<Traits> + 'static>(
    sparse_capacity: usize,
    dense_capacity: usize,
  ) -> Self {
    assert!(
      sparse_capacity >= dense_capacity,
      "Sparse capacity must be at least as large as the dense capacity."
    );
    AnySparseSet::with_capacity_in::<T>(sparse_capacity, Global, dense_capacity, Heap, Global)
  }
}

impl<I, Traits: ?Sized + Trait, SA: Allocator, IA: Allocator, M: MemBuilder>
  AnySparseSet<I, Traits, SA, IA, M>
{
  /// Constructs a new, empty `AnySparseSet<I, Traits, SA, IA, M>`.
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
  /// # use sparse_set::{any_sparse_set::{mem::Heap, traits::None}, AnySparseSet};
  ///
  /// # #[allow(unused_mut)]
  /// let mut set: AnySparseSet<usize, dyn None, _, _, _> = AnySparseSet::new_in::<usize>(System, System, Heap);
  /// ```
  #[must_use]
  pub fn new_in<T: SatisfyTraits<Traits> + 'static>(
    sparse_alloc: SA,
    indices_alloc: IA,
    dense_mem: M,
  ) -> Self {
    Self {
      dense: AnyVec::new_in::<T>(dense_mem),
      sparse: SparseVec::new_in(sparse_alloc),
      indices: Vec::new_in(indices_alloc),
    }
  }

  /// Constructs a new, empty `AnySparseSet<I, Traits, SA, IA, M>` with the specified capacity with the provided
  /// allocator.
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
  /// # use sparse_set::{any_sparse_set::{any_value::AnyValueWrapper, mem::Heap, traits::None}, AnySparseSet};
  ///
  /// let mut set: AnySparseSet<usize, dyn None, _, _, _> =
  ///   AnySparseSet::with_capacity_in::<usize>(10, System, 10, Heap, System);
  ///
  /// // The sparse set contains no items, even though it has capacity for more
  /// assert_eq!(set.dense_len(), 0);
  /// assert_eq!(set.sparse_len(), 0);
  /// assert_eq!(set.dense_capacity(), 10);
  /// assert_eq!(set.sparse_capacity(), 10);
  ///
  /// // These are all done without reallocating...
  /// for i in 0..10 {
  ///   set.insert(i, AnyValueWrapper::new(i as usize));
  /// }
  ///
  /// assert_eq!(set.dense_len(), 10);
  /// assert_eq!(set.sparse_len(), 10);
  /// assert_eq!(set.dense_capacity(), 10);
  /// assert_eq!(set.sparse_capacity(), 10);
  ///
  /// // ...but this will make the sparse set reallocate.
  /// set.insert(10, AnyValueWrapper::new(10usize));
  /// assert_eq!(set.dense_len(), 11);
  /// assert!(set.dense_capacity() >= 11);
  /// assert_eq!(set.sparse_len(), 11);
  /// assert!(set.sparse_capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn with_capacity_in<T: SatisfyTraits<Traits> + 'static>(
    sparse_capacity: usize,
    sparse_alloc: SA,
    dense_capacity: usize,
    dense_mem: M,
    indices_alloc: IA,
  ) -> Self
  where
    M: MemBuilderSizeable,
  {
    Self {
      dense: AnyVec::with_capacity_in::<T>(dense_capacity, dense_mem),
      sparse: SparseVec::with_capacity_in(sparse_capacity, sparse_alloc),
      indices: Vec::with_capacity_in(dense_capacity, indices_alloc),
    }
  }
}

impl<I, Traits: ?Sized + Trait, SA: Allocator, IA: Allocator, M: MemBuilder>
  AnySparseSet<I, Traits, SA, IA, M>
{
  /// Returns a reference to the underlying indices buffer allocator.
  #[must_use]
  pub fn indices_allocator(&self) -> &IA {
    self.indices.allocator()
  }

  /// Returns a reference to the underlying sparse buffer allocator.
  #[must_use]
  pub fn sparse_allocator(&self) -> &SA {
    self.sparse.allocator()
  }

  /// Returns a slice over the sparse set's indices.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
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
  /// # use sparse_set::AnySparseSet;
  /// #
  /// let set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(15, 10);
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
  /// # use sparse_set::AnySparseSet;
  /// #
  /// let set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(15, 10);
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
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

  /// Returns a typed view to a const `AnySparseSet`, if it holds elements of type `T` or `None` if not.
  pub fn downcast_ref<T>(&self) -> Option<AnySparseSetRef<'_, I, T, M>> {
    Some(AnySparseSetRef {
      dense: self.dense.downcast_ref::<T>()?,
      indices: &self.indices,
      sparse: &self.sparse,
    })
  }

  /// Returns a typed view to an immutable `AnySparseSet`.
  ///
  /// # Safety
  ///
  /// The elements of the sparse set must be of type `T`.
  pub unsafe fn downcast_ref_unchecked<T>(&self) -> AnySparseSetRef<'_, I, T, M> {
    AnySparseSetRef {
      dense: unsafe { self.dense.downcast_ref_unchecked::<T>() },
      indices: &self.indices,
      sparse: &self.sparse,
    }
  }

  /// Returns a typed view to a mutable `AnySparseSet`, if it holds elements of type `T` or `None` if not.
  pub fn downcast_mut<T>(&mut self) -> Option<AnySparseSetMut<'_, I, T, SA, IA, M>> {
    Some(AnySparseSetMut {
      dense: self.dense.downcast_mut::<T>()?,
      indices: &mut self.indices,
      sparse: &mut self.sparse,
    })
  }

  /// Returns a typed view to an immutable `AnySparseSet`.
  ///
  /// # Safety
  ///
  /// The elements of the sparse set must be of type `T`.
  pub unsafe fn downcast_mut_unchecked<T>(&mut self) -> AnySparseSetMut<'_, I, T, SA, IA, M> {
    AnySparseSetMut {
      dense: unsafe { self.dense.downcast_mut_unchecked::<T>() },
      indices: &mut self.indices,
      sparse: &mut self.sparse,
    }
  }

  /// Returns the memory [`Layout`] of the elements stored in the sparse set.
  pub fn element_layout(&self) -> Layout {
    self.dense.element_layout()
  }

  /// Returns the [`TypeId`] of the elements stored in the sparse set.
  pub fn element_typeid(&self) -> TypeId {
    self.dense.element_typeid()
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
  ///
  /// let mut iterator = set.indices();
  ///
  /// assert_eq!(iterator.next(), Some(&0));
  /// assert_eq!(iterator.next(), Some(&1));
  /// assert_eq!(iterator.next(), Some(&2));
  /// assert_eq!(iterator.next(), None);
  /// ```
  pub fn indices(&self) -> impl Iterator<Item = &I> {
    self.indices.iter()
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
  ///
  /// let mut iterator = set.iter();
  ///
  /// assert!(set.iter().map(|(i, v)| (i, v.downcast_ref::<usize>().unwrap())).eq([(&0, &1), (&1, &2), (&2, &3)]));
  /// ```
  pub fn iter(&self) -> impl Iterator<Item = (&I, ElementRef<'_, Traits, M>)> {
    self.indices.iter().zip(self.dense.iter())
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
  ///
  /// assert!(
  ///   set.iter_mut()
  ///     .map(|(i, mut v)| (i, v.downcast_mut::<usize>().unwrap()))
  ///     .eq([(&0, &mut 1), (&1, &mut 2), (&2, &mut 3)])
  /// );
  /// ```
  pub fn iter_mut(&mut self) -> impl Iterator<Item = (&I, ElementMut<'_, Traits, M>)> {
    self.indices.iter().zip(self.dense.iter_mut())
  }

  /// Returns `true` if the sparse set contains no elements.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  /// assert!(set.is_empty());
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(200, AnyValueWrapper::new(3usize));
  ///
  /// assert_eq!(set.sparse_len(), 201);
  /// ```
  #[must_use]
  pub fn sparse_len(&self) -> usize {
    self.sparse.len()
  }

  /// Reserves capacity for at least `additional` more elements to be inserted in the given `AnySparseSet<I, T>`'s dense
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.reserve_dense(10);
  /// assert!(set.dense_capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_dense(&mut self, additional: usize)
  where
    M::Mem: MemResizable,
  {
    self.dense.reserve(additional);
  }

  /// Reserves capacity for at least `additional` more elements to be inserted in the given `AnySparseSet<I, T>`'s
  /// sparse buffer.
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.reserve_sparse(10);
  /// assert!(set.sparse_capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_sparse(&mut self, additional: usize) {
    self.sparse.reserve(additional);
  }

  /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the given
  /// `AnySparseSet<I>`'s dense buffer.
  ///
  /// After calling `reserve_exact`, the dense capacity will be greater than or equal to
  /// `self.dense_len() + additional`. Does nothing if the capacity is already sufficient.
  ///
  /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not be relied
  /// upon to be precisely minimal. Prefer [`reserve_dense`] if future insertions are expected.
  ///
  /// [`reserve_dense`]: AnySparseSet::reserve_dense
  ///
  /// # Panics
  ///
  /// Panics if the new capacity exceeds `isize::MAX` bytes.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.reserve_exact_dense(10);
  /// assert!(set.dense_capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_exact_dense(&mut self, additional: usize)
  where
    M::Mem: MemResizable,
  {
    self.dense.reserve_exact(additional);
  }

  /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the given
  /// `AnySparseSet<I>`'s sparse buffer.
  ///
  /// After calling `reserve_exact`, the sparse capacity will be greater than or equal to
  /// `self.sparse_len() + additional`. Does nothing if the capacity is already sufficient.
  ///
  /// Note that the allocator may give the collection more space than it requests. Therefore, capacity can not be relied
  /// upon to be precisely minimal. Prefer [`reserve_sparse`] if future insertions are expected.
  ///
  /// [`reserve_sparse`]: AnySparseSet::reserve_sparse
  ///
  /// # Panics
  ///
  /// Panics if the new capacity exceeds `isize::MAX` bytes.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.reserve_exact_sparse(10);
  /// assert!(set.sparse_capacity() >= 11);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_exact_sparse(&mut self, additional: usize) {
    self.sparse.reserve_exact(additional);
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(10, 10);
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// assert_eq!(set.dense_capacity(), 10);
  ///
  /// set.shrink_to_fit_dense();
  /// assert!(set.dense_capacity() >= 2);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to_fit_dense(&mut self)
  where
    M::Mem: MemResizable,
  {
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(10, 10);
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(10, 10);
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  ///
  /// assert_eq!(set.dense_capacity(), 10);
  /// set.shrink_to_dense(4);
  /// assert!(set.dense_capacity() >= 4);
  /// set.shrink_to_dense(0);
  /// assert!(set.dense_capacity() >= 2);
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to_dense(&mut self, min_capacity: usize)
  where
    M::Mem: MemResizable,
  {
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(10, 10);
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  ///
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
  ///
  /// let mut iterator = set.values();
  ///
  /// assert!(set.values().map(|v| v.downcast_ref::<usize>().unwrap()).eq(&[1, 2, 3]));
  /// ```
  #[must_use]
  pub fn values(&self) -> IterRef<'_, Traits, M> {
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
  ///
  /// for mut elem in set.values_mut() {
  ///     *elem.downcast_mut::<usize>().unwrap() += 2;
  /// }
  ///
  /// assert!(set.values().map(|v| v.downcast_ref::<usize>().unwrap()).eq(&[3, 4, 5]));
  /// ```
  #[must_use]
  pub fn values_mut(&mut self) -> IterMut<'_, Traits, M> {
    self.dense.iter_mut()
  }
}

impl<I: SparseSetIndex, Traits: ?Sized + Trait, SA: Allocator, IA: Allocator, M: MemBuilder>
  AnySparseSet<I, Traits, SA, IA, M>
{
  /// Returns a reference to an element pointed to by the index.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Panics
  ///
  /// Panics if `index` does not point to an element.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
  /// assert_eq!(&2, set.at(1).downcast_ref::<usize>().unwrap());
  /// ```
  pub fn at(&self, index: I) -> ElementRef<'_, Traits, M> {
    self.get(index).unwrap()
  }

  /// Returns a mutable reference to an element pointed to by the index.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Panics
  ///
  /// Panics if `index` does not point to an element.
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
  ///
  /// *set.at_mut(1).downcast_mut::<usize>().unwrap() = 42;
  ///
  /// assert!(set.values().map(|v| v.downcast_ref::<usize>().unwrap()).eq(&[1, 42, 3]));
  /// ```
  pub fn at_mut(&mut self, index: I) -> ElementMut<'_, Traits, M> {
    self.get_mut(index).unwrap()
  }

  /// Returns `true` if the sparse set contains an element at the given index.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Examples
  ///
  /// ```
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
  /// assert_eq!(Some(&2), set.get(1).map(|v| v.downcast_ref::<usize>().unwrap()));
  /// assert_eq!(None, set.get(3).map(|v| v.downcast_ref::<usize>().unwrap()));
  ///
  /// set.remove(1);
  /// assert_eq!(None, set.get(1).map(|v| v.downcast_ref::<usize>().unwrap()));
  /// ```
  #[must_use]
  pub fn get(&self, index: I) -> Option<ElementRef<'_, Traits, M>> {
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
  ///
  /// if let Some(mut elem) = set.get_mut(1) {
  ///   *elem.downcast_mut::<usize>().unwrap() = 42;
  /// }
  ///
  /// assert!(set.values().map(|v| v.downcast_ref::<usize>().unwrap()).eq(&[1, 42, 3]));
  /// ```
  #[must_use]
  pub fn get_mut(&mut self, index: I) -> Option<ElementMut<'_, Traits, M>> {
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
  /// # use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(4usize));
  /// set.insert(2, AnyValueWrapper::new(2usize));
  /// set.insert(3, AnyValueWrapper::new(3usize));
  ///
  /// assert!(set.values().map(|v| v.downcast_ref::<usize>().unwrap()).eq(&[1, 4, 2, 3]));
  /// set.insert(20, AnyValueWrapper::new(5usize));
  /// assert!(set.values().map(|v| v.downcast_ref::<usize>().unwrap()).eq(&[1, 4, 2, 3, 5]));
  /// ```
  #[cfg(not(no_global_oom_handling))]
  pub fn insert<T: AnyValue>(&mut self, index: I, value: T) {
    match self.sparse.get(index) {
      Some(dense_index) => {
        let dense_index = dense_index.get() - 1;
        self.dense.insert(dense_index, value);
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
  /// # use sparse_set::{any_sparse_set::any_value::{AnyValue, AnyValueWrapper}, AnySparseSet};
  /// #
  /// let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  ///
  /// set.insert(0, AnyValueWrapper::new(1usize));
  /// set.insert(1, AnyValueWrapper::new(2usize));
  /// set.insert(2, AnyValueWrapper::new(3usize));
  ///
  /// assert_eq!(set.remove(1).map(|v| v.downcast_ref::<usize>().unwrap().clone()), Some(2));
  /// assert!(set.values().map(|v| v.downcast_ref::<usize>().unwrap()).eq(&[1, 3]));
  /// ```
  #[must_use]
  pub fn remove(&mut self, index: I) -> Option<SwapRemove<'_, Traits, M>> {
    match self.sparse.remove(index) {
      Some(dense_index) => {
        let dense_len = self.dense.len();
        let dense_index = dense_index.get() - 1;
        let value = Some(self.dense.swap_remove(dense_index));
        let _ = self.indices.swap_remove(dense_index);

        if dense_index != dense_len - 1 {
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

impl<I, Traits: ?Sized + Trait, SA: Allocator, IA: Allocator, M: MemBuilder>
  AsRef<AnySparseSet<I, Traits, SA, IA, M>> for AnySparseSet<I, Traits, SA, IA, M>
{
  fn as_ref(&self) -> &Self {
    self
  }
}

impl<I, Traits: ?Sized + Trait, SA: Allocator, IA: Allocator, M: MemBuilder>
  AsMut<AnySparseSet<I, Traits, SA, IA, M>> for AnySparseSet<I, Traits, SA, IA, M>
{
  fn as_mut(&mut self) -> &mut Self {
    self
  }
}

impl<
    I: Clone,
    Traits: ?Sized + Cloneable + Trait,
    SA: Allocator + Clone,
    IA: Allocator + Clone,
    M: MemBuilder,
  > Clone for AnySparseSet<I, Traits, SA, IA, M>
{
  fn clone(&self) -> Self {
    Self {
      dense: self.dense.clone(),
      sparse: self.sparse.clone(),
      indices: self.indices.clone(),
    }
  }
}

impl<I: Debug, Traits: ?Sized + Trait, SA: Allocator, IA: Allocator, M: MemBuilder> Debug
  for AnySparseSet<I, Traits, SA, IA, M>
{
  fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
    /// Type used in `Debug` implementation to indicate an erased type.
    #[derive(Debug)]
    struct Erased;

    /// Type used in `Debug` implementation to format the iterator as a map.
    struct Entries<'a, I>(&'a [I]);

    impl<I: Debug> Debug for Entries<'_, I> {
      fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
          .debug_map()
          .entries(self.0.iter().map(|index| (index, Erased)))
          .finish()
      }
    }

    formatter
      .debug_struct("AnySparseSet")
      .field("type_id", &self.element_typeid())
      .field("entries", &Entries(&self.indices))
      .finish()
  }
}

#[cfg(not(no_global_oom_handling))]
impl<
    I: SparseSetIndex,
    T: AnyValue,
    Traits: ?Sized + Trait,
    SA: Allocator,
    IA: Allocator,
    M: MemBuilder,
  > Extend<(I, T)> for AnySparseSet<I, Traits, SA, IA, M>
{
  fn extend<Iter: IntoIterator<Item = (I, T)>>(&mut self, iter: Iter) {
    for (index, value) in iter {
      self.insert(index, value);
    }
  }
}

#[cfg(not(no_global_oom_handling))]
impl<
    I: SparseSetIndex,
    T: SatisfyTraits<Traits> + 'static,
    Traits: ?Sized + Trait,
    const N: usize,
  > From<[(I, T); N]> for AnySparseSet<I, Traits>
{
  fn from(slice: [(I, T); N]) -> Self {
    let mut set = AnySparseSet::with_capacity::<T>(slice.len(), slice.len());

    for (index, value) in slice {
      set.insert(index, AnyValueWrapper::new(value));
    }

    set
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T: SatisfyTraits<Traits> + 'static, Traits: ?Sized + Trait>
  FromIterator<(I, T)> for AnySparseSet<I, Traits>
{
  fn from_iter<Iter: IntoIterator<Item = (I, T)>>(iter: Iter) -> Self {
    let iter = iter.into_iter();
    let capacity = if let Some(size_hint) = iter.size_hint().1 {
      size_hint
    } else {
      iter.size_hint().0
    };
    let mut set = AnySparseSet::with_capacity::<T>(capacity, capacity);

    for (index, value) in iter {
      set.insert(index, AnyValueWrapper::new(value));
    }

    set
  }
}

impl<'a, I, Traits: ?Sized + Trait, SA: Allocator, IA: Allocator, M: MemBuilder> IntoIterator
  for &'a AnySparseSet<I, Traits, SA, IA, M>
{
  type Item = (&'a I, ElementRef<'a, Traits, M>);
  type IntoIter = impl Iterator<Item = Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

impl<'a, I, Traits: ?Sized + Trait, SA: Allocator, IA: Allocator, M: MemBuilder> IntoIterator
  for &'a mut AnySparseSet<I, Traits, SA, IA, M>
{
  type Item = (&'a I, ElementMut<'a, Traits, M>);
  type IntoIter = impl Iterator<Item = Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.iter_mut()
  }
}

#[cfg(test)]
mod test {

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

  #[test]
  fn test_new() {
    let set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.is_empty());
    assert_eq!(set.dense_capacity(), 0);
    assert_eq!(set.sparse_capacity(), 0);
  }

  #[test]
  fn test_with_capacity() {
    let set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(15, 10);
    assert_eq!(set.sparse_capacity(), 15);
    assert_eq!(set.dense_capacity(), 10);
  }

  #[test]
  fn test_with_capacity_zero() {
    let set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(0, 0);
    assert_eq!(set.sparse_capacity(), 0);
    assert_eq!(set.dense_capacity(), 0);
  }

  #[should_panic]
  #[test]
  fn test_with_capacity_dense_greater_than_sparse() {
    let _: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(0, 1);
  }

  #[should_panic]
  #[test]
  fn test_with_capacity_sparse_overflow() {
    let _: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(usize::MAX, 0);
  }

  #[should_panic]
  #[test]
  fn test_with_capacity_dense_overflow() {
    let _: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(0, usize::MAX);
  }

  #[test]
  fn test_at() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    assert_eq!(set.at(0).downcast_ref::<usize>().unwrap(), &1);
    assert_eq!(set.at(2).downcast_ref::<usize>().unwrap(), &3);
  }

  #[should_panic]
  #[test]
  fn test_at_panics() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let _ = set.at(100);
  }

  #[test]
  fn test_at_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let value = set.at_mut(2).downcast_mut::<usize>().unwrap();
    assert_eq!(value, &mut 3);
    *value = 10;

    assert_eq!(set.at(2).downcast_ref::<usize>().unwrap(), &10);
  }

  #[should_panic]
  #[test]
  fn test_at_mut_panics() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let _ = set.at_mut(100);
  }

  #[test]
  fn test_indices_allocator() {
    let set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    let _ = set.indices_allocator();
  }

  #[test]
  fn test_sparse_allocator() {
    let set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    let _ = set.sparse_allocator();
  }

  #[test]
  fn test_as_indices_slice() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.as_indices_slice(), &[0]);
  }

  #[test]
  fn test_as_indices_ptr() {
    let set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(10, 10);
    assert_eq!(set.as_indices_ptr(), set.as_indices_slice().as_ptr());
  }

  #[test]
  fn test_clear() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(0, AnyValueWrapper::new(2usize));
    set.insert(0, AnyValueWrapper::new(3usize));
    set.clear();

    assert!(set.is_empty());
  }

  #[test]
  fn test_contains() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(!set.contains(0));
    set.insert(0, AnyValueWrapper::new(1usize));
    assert!(set.contains(0));
    let _ = set.remove(0);
    assert!(!set.contains(0));
  }

  #[test]
  fn test_downcast_ref() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));

    let reference = set.downcast_ref::<usize>().unwrap();
    assert_eq!(reference.get(0), Some(&1));
  }

  #[test]
  fn test_downcast_ref_unchecked() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));

    let reference = unsafe { set.downcast_ref_unchecked::<usize>() };
    assert_eq!(reference.get(0), Some(&1));
  }

  #[test]
  fn test_downcast_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));

    {
      let mut mutable = set.downcast_mut::<usize>().unwrap();
      mutable.insert(1, 2);
    }

    assert_eq!(
      set.get(1).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&2)
    );
  }

  #[test]
  fn test_downcast_mut_unchecked() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));

    {
      let mut mutable = unsafe { set.downcast_mut_unchecked::<usize>() };
      mutable.insert(1, 2);
    }

    assert_eq!(
      set.get(1).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&2)
    );
  }

  #[test]
  fn test_element_layout() {
    let set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.element_layout(), Layout::new::<usize>());
  }

  #[test]
  fn test_element_typeid() {
    let set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.element_typeid(), TypeId::of::<usize>());
  }

  #[test]
  fn test_get() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    assert_eq!(
      set.get(0).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&1)
    );
    assert_eq!(
      set.get(2).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&3)
    );
    assert_eq!(
      set.get(100).map(|v| v.downcast_ref::<usize>().unwrap()),
      None
    );
  }

  #[test]
  fn test_get_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let value = set
      .get_mut(2)
      .map(|mut v| v.downcast_mut::<usize>().unwrap());
    assert_eq!(value, Some(&mut 3));
    *value.unwrap() = 10;

    assert_eq!(
      set.get(2).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&10)
    );
  }

  #[test]
  fn test_indices() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.indices().eq(&[]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set.indices().eq(&[0, 1, 2]));
  }

  #[test]
  fn test_insert_capacity_increases() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(1, 1);
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.sparse_capacity(), 1);
    assert_eq!(set.dense_capacity(), 1);

    set.insert(1, AnyValueWrapper::new(2usize));
    assert!(set.sparse_capacity() >= 2);
    assert!(set.dense_capacity() >= 2);

    assert_eq!(
      set.get(0).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&1)
    );
    assert_eq!(
      set.get(1).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&2)
    );
  }

  #[test]
  fn test_insert_len_increases() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 1);

    set.insert(1, AnyValueWrapper::new(2usize));
    assert_eq!(set.dense_len(), 2);
    assert_eq!(set.sparse_len(), 2);

    set.insert(100, AnyValueWrapper::new(101usize));
    assert_eq!(set.dense_len(), 3);
    assert_eq!(set.sparse_len(), 101);
  }

  #[test]
  fn test_insert_overwrites() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(
      set.get(0).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&1)
    );

    set.insert(0, AnyValueWrapper::new(2usize));
    assert_eq!(
      set.get(0).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&2)
    );
  }

  #[test]
  fn test_iter() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set
      .iter()
      .map(|(i, v)| (i, v.downcast_ref::<usize>().unwrap()))
      .eq([]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set
      .iter()
      .map(|(i, v)| (i, v.downcast_ref::<usize>().unwrap()))
      .eq([(&0, &1), (&1, &2), (&2, &3)]));
  }

  #[test]
  fn test_iter_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set
      .iter_mut()
      .map(|(i, mut v)| (i, v.downcast_mut::<usize>().unwrap()))
      .eq([]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set
      .iter_mut()
      .map(|(i, mut v)| (i, v.downcast_mut::<usize>().unwrap()))
      .eq([(&0, &mut 1), (&1, &mut 2), (&2, &mut 3)]));

    let mut value = set.iter_mut().next().unwrap();
    *(value.1.downcast_mut::<usize>().unwrap()) = 100;

    assert_eq!(
      (&mut set)
        .iter_mut()
        .map(|(i, mut v)| (i, v.downcast_mut::<usize>().unwrap()))
        .next(),
      Some((&0, &mut 100))
    );
  }

  #[test]
  fn test_is_empty() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.is_empty());

    set.insert(0, AnyValueWrapper::new(1usize));
    assert!(!set.is_empty());

    let _ = set.remove(0);
    assert!(set.is_empty());
  }

  #[test]
  fn test_dense_len() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.dense_len(), 0);

    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.dense_len(), 1);
  }

  #[test]
  fn test_sparse_len() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.sparse_len(), 0);

    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.sparse_len(), 1);

    set.insert(100, AnyValueWrapper::new(1usize));
    assert_eq!(set.sparse_len(), 101);
  }

  #[test]
  fn test_remove_can_return_none() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.remove(1).map(|v| v.downcast::<usize>().unwrap()), None);
    assert_eq!(
      set.remove(100).map(|v| v.downcast::<usize>().unwrap()),
      None
    );
  }

  #[test]
  fn test_remove_can_return_some() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(
      set.remove(0).map(|v| v.downcast::<usize>().unwrap()),
      Some(1)
    );
  }

  #[test]
  fn test_remove_len_decreases() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));

    assert_eq!(set.dense_len(), 2);
    assert_eq!(set.sparse_len(), 2);
    assert_eq!(
      set.remove(0).map(|v| v.downcast::<usize>().unwrap()),
      Some(1)
    );
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 2);
    assert_eq!(set.remove(0).map(|v| v.downcast::<usize>().unwrap()), None);
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 2);
  }

  #[test]
  fn test_remove_swaps_with_last() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    assert!(set
      .values()
      .map(|v| v.downcast_ref::<usize>().unwrap())
      .eq(&[1, 2, 3]));

    let _ = set.remove(0);
    assert!(set
      .values()
      .map(|v| v.downcast_ref::<usize>().unwrap())
      .eq(&[3, 2]));
  }

  #[test]
  fn test_reserve_dense() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.dense_capacity(), 0);

    set.reserve_dense(3);
    let capacity = set.dense_capacity();
    assert!(capacity >= 2);

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));

    set.reserve_dense(1);
    assert_eq!(set.dense_capacity(), capacity);
  }

  #[test]
  fn test_reserve_sparse() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.sparse_capacity(), 0);

    set.reserve_sparse(3);
    let capacity = set.sparse_capacity();
    assert!(capacity >= 2);

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));

    set.reserve_sparse(1);
    assert_eq!(set.sparse_capacity(), capacity);
  }

  #[test]
  fn test_reserve_exact_dense() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.dense_capacity(), 0);

    set.reserve_exact_dense(3);
    let capacity = set.dense_capacity();
    assert!(capacity >= 2);

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));

    set.reserve_exact_dense(1);
    assert_eq!(set.dense_capacity(), capacity);
  }

  #[test]
  fn test_reserve_exact_sparse() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.sparse_capacity(), 0);

    set.reserve_exact_sparse(3);
    let capacity = set.sparse_capacity();
    assert!(capacity >= 2);

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));

    set.reserve_exact_sparse(1);
    assert_eq!(set.sparse_capacity(), capacity);
  }

  #[test]
  fn test_shrink_to_fit_dense() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    assert_eq!(set.dense_capacity(), 3);
    let _ = set.remove(2);
    set.shrink_to_fit_dense();
    assert_eq!(set.dense_capacity(), 2);
  }

  #[test]
  fn test_shrink_to_fit_sparse() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 3);
    let _ = set.remove(2);
    set.shrink_to_fit_sparse();
    assert_eq!(set.sparse_capacity(), 2);
    assert_eq!(set.sparse_len(), 2);
  }

  #[test]
  fn test_shrink_to_fit_max_index_zero() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 0);
    set.shrink_to_fit_sparse();
    assert_eq!(set.sparse_capacity(), 0);
    assert_eq!(set.sparse_len(), 0);
  }

  #[test]
  fn test_shrink_to_dense_can_reduce() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.dense_capacity(), 3);
    set.shrink_to_dense(1);
    assert_eq!(set.dense_capacity(), 1);
  }

  #[test]
  fn test_shrink_to_dense_cannot_reduce() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert_eq!(set.dense_capacity(), 3);
    set.shrink_to_dense(0);
    assert_eq!(set.dense_capacity(), 3);
  }

  #[test]
  fn test_shrink_to_sparse_can_reduce() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 1);
    set.shrink_to_sparse(1);
    assert_eq!(set.sparse_capacity(), 1);
    assert_eq!(set.sparse_len(), 1);
  }

  #[test]
  fn test_shrink_to_sparse_cannot_reduce() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 3);
    set.shrink_to_sparse(0);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 3);
  }

  #[test]
  fn test_shrink_to_max_index_zero() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 0);
    set.shrink_to_sparse(0);
    assert_eq!(set.sparse_capacity(), 0);
    assert_eq!(set.sparse_len(), 0);
  }

  #[test]
  fn test_values() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set
      .values()
      .map(|v| v.downcast_ref::<usize>().unwrap())
      .eq(&[]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set
      .values()
      .map(|v| v.downcast_ref::<usize>().unwrap())
      .eq(&[1, 2, 3]));
  }

  #[test]
  fn test_values_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set
      .values_mut()
      .map(|v| v.downcast_ref::<usize>().unwrap())
      .eq(&[]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set
      .values_mut()
      .map(|v| v.downcast_ref::<usize>().unwrap())
      .eq(&[1, 2, 3]));

    let mut value = set.values_mut().next().unwrap();
    *(value.downcast_mut::<usize>().unwrap()) = 100;

    assert_eq!(
      set.get(0).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&100)
    );
  }

  #[test]
  fn test_as_ref() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let reference: &AnySparseSet<_, _> = set.as_ref();
    assert_eq!(
      reference.get(0).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&1)
    );
  }

  #[test]
  fn test_as_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let reference: &mut AnySparseSet<_, _> = set.as_mut();
    assert_eq!(
      reference.get(0).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&1)
    );
  }

  #[test]
  fn test_clone() {
    let mut set: AnySparseSet<usize, dyn Cloneable> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let cloned_set = set.clone();
    assert!(set
      .iter()
      .map(|(i, v)| (i, v.downcast_ref::<usize>().unwrap()))
      .eq(
        cloned_set
          .iter()
          .map(|(i, v)| (i, v.downcast_ref::<usize>().unwrap()))
      ));
  }

  #[test]
  fn test_clone_zero_capacity() {
    let set: AnySparseSet<usize, dyn Cloneable> = AnySparseSet::new::<usize>();
    assert_eq!(set.dense_capacity(), 0);

    let cloned_set = set.clone();
    assert!(set
      .iter()
      .map(|(i, v)| (i, v.downcast_ref::<usize>().unwrap()))
      .eq(
        cloned_set
          .iter()
          .map(|(i, v)| (i, v.downcast_ref::<usize>().unwrap()))
      ));
  }

  #[test]
  fn test_clone_drops_are_separate() {
    let num_dropped = Rc::new(RefCell::new(0));

    {
      let mut set: AnySparseSet<usize, dyn Cloneable> = AnySparseSet::new::<Value>();
      let value = Value(num_dropped.clone());
      set.insert(0, AnyValueWrapper::new(value.clone()));
      set.insert(1, AnyValueWrapper::new(value.clone()));
      set.insert(2, AnyValueWrapper::new(value));

      let _cloned_set = set.clone();
    }

    assert_eq!(*num_dropped.borrow(), 6);
  }

  #[test]
  fn test_debug() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(
      format!("{:?}", set),
      format!(
        "AnySparseSet {{ type_id: {:?}, entries: {{}} }}",
        TypeId::of::<usize>()
      )
    );

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert_eq!(
      format!("{:?}", set),
      format!(
        "AnySparseSet {{ type_id: {:?}, entries: {{0: Erased, 1: Erased, 2: Erased}} }}",
        TypeId::of::<usize>()
      )
    );
  }

  #[test]
  fn test_drop() {
    let num_dropped = Rc::new(RefCell::new(0));

    {
      let mut set: AnySparseSet<usize, dyn Cloneable> = AnySparseSet::new::<Value>();
      let value = Value(num_dropped.clone());
      set.insert(0, AnyValueWrapper::new(value.clone()));
      set.insert(1, AnyValueWrapper::new(value.clone()));
      set.insert(2, AnyValueWrapper::new(value));
    }

    assert_eq!(*num_dropped.borrow(), 3);
  }

  #[test]
  fn test_extend() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.extend([
      (0, AnyValueWrapper::new(1usize)),
      (1, AnyValueWrapper::new(2usize)),
      (2, AnyValueWrapper::new(3usize)),
    ]);
    assert!(set
      .values()
      .map(|v| v.downcast_ref::<usize>().unwrap())
      .eq(&[1, 2, 3]));
  }

  #[test]
  fn test_from_array() {
    let set: AnySparseSet<usize> = AnySparseSet::from([(0, 1usize), (1, 2usize), (2, 3usize)]);
    assert!(set
      .values()
      .map(|v| v.downcast_ref::<usize>().unwrap())
      .eq(&[1, 2, 3]));
  }

  #[test]
  fn test_from_iterator() {
    let set: AnySparseSet<usize> = AnySparseSet::from_iter([(0, 1usize), (1, 2usize), (2, 3usize)]);
    assert!(set
      .values()
      .map(|v| v.downcast_ref::<usize>().unwrap())
      .eq(&[1, 2, 3]));
  }

  #[test]
  fn test_into_iterator_ref() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!((&set)
      .into_iter()
      .map(|(i, v)| (i, v.downcast_ref::<usize>().unwrap()))
      .eq([]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!((&set)
      .into_iter()
      .map(|(i, v)| (i, v.downcast_ref::<usize>().unwrap()))
      .eq([(&0, &1), (&1, &2), (&2, &3)]));
  }

  #[test]
  fn test_into_iterator_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!((&mut set)
      .into_iter()
      .map(|(i, mut v)| (i, v.downcast_mut::<usize>().unwrap()))
      .eq([]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!((&mut set)
      .into_iter()
      .map(|(i, mut v)| (i, v.downcast_mut::<usize>().unwrap()))
      .eq([(&0, &mut 1), (&1, &mut 2), (&2, &mut 3)]));

    let mut value = (&mut set).into_iter().next().unwrap();
    *(value.1.downcast_mut::<usize>().unwrap()) = 100;

    assert_eq!(
      set.get(0).map(|v| v.downcast_ref::<usize>().unwrap()),
      Some(&100)
    );
  }
}
