#![allow(unsafe_code)]

use std::{
  alloc::{Allocator, Global},
  fmt::{self, Debug, Formatter},
  hash::{Hash, Hasher},
  num::NonZeroUsize,
  ops::{Deref, DerefMut, Index, IndexMut},
};

use any_vec::{
  mem::{Heap, MemBuilder, MemResizable},
  AnyVecMut,
};

use crate::{SparseSetIndex, SparseVec};

/// A reference to a mutable `AnySparseSet<I>` which has been downcasted to some type `T`.
pub struct AnySparseSetMut<
  'a,
  I,
  T: 'static,
  SA: Allocator = Global,
  IA: Allocator = Global,
  M: MemBuilder = Heap,
> {
  /// The a reference dense buffer, i.e., the buffer containing the actual data values.
  pub(in crate::any_sparse_set) dense: AnyVecMut<'a, T, M>,

  /// The sparse buffer, i.e., the buffer where each index may correspond to an index into `dense`.
  pub(in crate::any_sparse_set) sparse: &'a mut SparseVec<I, NonZeroUsize, SA>,

  /// All the existing indices in `sparse`.
  ///
  /// The indices here will always be in order based on the `dense` buffer.
  pub(in crate::any_sparse_set) indices: &'a mut Vec<I, IA>,
}

impl<I, T: 'static, SA: Allocator, IA: Allocator, M: MemBuilder>
  AnySparseSetMut<'_, I, T, SA, IA, M>
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

  /// Extracts a slice containing the entire dense buffer.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  #[must_use]
  pub fn as_dense_slice(&self) -> &[T] {
    self.dense.as_slice()
  }

  /// Extracts a mutable slice of the entire dense buffer.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  #[must_use]
  pub fn as_dense_mut_slice(&mut self) -> &mut [T] {
    self.dense.as_mut_slice()
  }

  /// Returns a slice over the sparse set's indices.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
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

  /// Clears the sparse set, removing all values.
  ///
  /// Note that this method has no effect on the allocated capacity of the sparse set.
  ///
  /// This operation is *O*(*m*).
  pub fn clear(&mut self) {
    self.dense.clear();
    self.indices.clear();
    self.sparse.clear();
  }

  /// Returns `true` if the sparse set contains no elements.
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.dense_len() == 0
  }

  /// Returns the number of elements in the dense set, also referred to as its '`dense_len`'.
  #[must_use]
  pub fn dense_len(&self) -> usize {
    self.dense.len()
  }

  /// Returns the number of elements in the sparse set, also referred to as its '`sparse_len`'.
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
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_sparse(&mut self, additional: usize) {
    self.sparse.reserve(additional);
  }

  /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the given
  /// `AnySparseSet<I, T>`'s dense buffer.
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
  #[cfg(not(no_global_oom_handling))]
  pub fn reserve_exact_dense(&mut self, additional: usize)
  where
    M::Mem: MemResizable,
  {
    self.dense.reserve_exact(additional);
  }

  /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the given
  /// `AnySparseSet<I, T>`'s sparse buffer.
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
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to_fit_sparse(&mut self) {
    self.sparse.shrink_to_fit();
  }

  /// Shrinks the dense capacity of the sparse set with a lower bound.
  ///
  /// The capacity will remain at least as large as both the length and the supplied value.
  ///
  /// If the current capacity is less than the lower limit, this is a no-op.
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
  #[cfg(not(no_global_oom_handling))]
  pub fn shrink_to_sparse(&mut self, min_capacity: usize) {
    self.sparse.shrink_to(min_capacity);
  }

  /// Returns an iterator over the sparse set's values.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// Consuming the iterator is an *O*(*n*) operation.
  pub fn values(&self) -> impl Iterator<Item = &T> {
    self.dense.iter()
  }

  /// Returns an iterator that allows modifying each value.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// Consuming the iterator is an *O*(*n*) operation.
  pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
    self.dense.iter_mut()
  }
}

impl<I: SparseSetIndex, T: 'static, SA: Allocator, IA: Allocator, M: MemBuilder>
  AnySparseSetMut<'_, I, T, SA, IA, M>
{
  /// Returns a reference to an element pointed to by the index.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Panics
  ///
  /// Panics if `index` does not point to an element.
  pub fn at(&self, index: I) -> &T {
    self.get(index).unwrap()
  }

  /// Returns a mutable reference to an element pointed to by the index.
  ///
  /// This operation is *O*(*1*).
  ///
  /// # Panics
  ///
  /// Panics if `index` does not point to an element.
  pub fn at_mut(&mut self, index: I) -> &mut T {
    self.get_mut(index).unwrap()
  }

  /// Returns `true` if the sparse set contains an element at the given index.
  ///
  /// This operation is *O*(*1*).
  #[must_use]
  pub fn contains(&self, index: I) -> bool {
    self.get(index).is_some()
  }

  /// Returns a reference to an element pointed to by the index, if it exists.
  ///
  /// This operation is *O*(*1*).
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
  #[must_use]
  pub fn get_mut(&mut self, index: I) -> Option<&mut T> {
    self
      .sparse
      .get(index)
      .map(|dense_index| unsafe { self.dense.get_unchecked_mut(dense_index.get() - 1) })
  }

  /// Returns an iterator over the sparse set's indices.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// Consuming the iterator is an *O*(*n*) operation.
  pub fn indices(&self) -> impl Iterator<Item = I> + '_ {
    self.indices.iter().cloned()
  }

  /// Returns an iterator over the sparse set's indices and values as pairs.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// Consuming the iterator is an *O*(*n*) operation.
  pub fn iter(&self) -> impl Iterator<Item = (I, &T)> {
    self.indices.iter().cloned().zip(self.dense.iter())
  }

  /// Returns an iterator that allows modifying each value as an `(index, value)` pair.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// Consuming the iterator is an *O*(*n*) operation.
  pub fn iter_mut(&mut self) -> impl Iterator<Item = (I, &mut T)> {
    self.indices.iter().cloned().zip(self.dense.iter_mut())
  }

  /// Inserts an element at position `index` within the sparse set.
  ///
  /// If a value already existed at `index`, it will be overwritten.
  ///
  /// If `index` is greater than `sparse_capacity`, then an allocation will take place.
  ///
  /// This operation is amortized *O*(*1*).
  #[cfg(not(no_global_oom_handling))]
  pub fn insert(&mut self, index: I, value: T) {
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
  #[must_use]
  pub fn remove(&mut self, index: I) -> Option<T> {
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

impl<'a, I, T, SA: Allocator, IA: Allocator, M: MemBuilder>
  AsRef<AnySparseSetMut<'a, I, T, SA, IA, M>> for AnySparseSetMut<'a, I, T, SA, IA, M>
{
  fn as_ref(&self) -> &Self {
    self
  }
}

impl<'a, I, T, SA: Allocator, IA: Allocator, M: MemBuilder>
  AsMut<AnySparseSetMut<'a, I, T, SA, IA, M>> for AnySparseSetMut<'a, I, T, SA, IA, M>
{
  fn as_mut(&mut self) -> &mut Self {
    self
  }
}

impl<I, T, SA: Allocator, IA: Allocator, M: MemBuilder> AsRef<[T]>
  for AnySparseSetMut<'_, I, T, SA, IA, M>
{
  fn as_ref(&self) -> &[T] {
    self.dense.as_slice()
  }
}

impl<I, T, SA: Allocator, IA: Allocator, M: MemBuilder> AsMut<[T]>
  for AnySparseSetMut<'_, I, T, SA, IA, M>
{
  fn as_mut(&mut self) -> &mut [T] {
    self.dense.as_mut_slice()
  }
}

impl<I, T, SA: Allocator, IA: Allocator, M: MemBuilder> Deref
  for AnySparseSetMut<'_, I, T, SA, IA, M>
{
  type Target = [T];

  fn deref(&self) -> &[T] {
    self.dense.as_slice()
  }
}

impl<I, T, SA: Allocator, IA: Allocator, M: MemBuilder> DerefMut
  for AnySparseSetMut<'_, I, T, SA, IA, M>
{
  fn deref_mut(&mut self) -> &mut [T] {
    self.dense.as_mut_slice()
  }
}

impl<I: Debug + SparseSetIndex, T: Debug, SA: Allocator, IA: Allocator, M: MemBuilder> Debug
  for AnySparseSetMut<'_, I, T, SA, IA, M>
{
  fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
    formatter.debug_map().entries(self.iter()).finish()
  }
}

#[cfg(not(no_global_oom_handling))]
impl<'a, I: SparseSetIndex, T: Copy + 'a, SA: Allocator + 'a, IA: Allocator + 'a, M: MemBuilder>
  Extend<(I, &'a T)> for AnySparseSetMut<'a, I, T, SA, IA, M>
{
  fn extend<Iter: IntoIterator<Item = (I, &'a T)>>(&mut self, iter: Iter) {
    for (index, &value) in iter {
      self.insert(index, value);
    }
  }
}

#[cfg(not(no_global_oom_handling))]
impl<I: SparseSetIndex, T, SA: Allocator, IA: Allocator, M: MemBuilder> Extend<(I, T)>
  for AnySparseSetMut<'_, I, T, SA, IA, M>
{
  fn extend<Iter: IntoIterator<Item = (I, T)>>(&mut self, iter: Iter) {
    for (index, value) in iter {
      self.insert(index, value);
    }
  }
}

impl<I: Hash + SparseSetIndex, T: Hash, SA: Allocator, IA: Allocator, M: MemBuilder> Hash
  for AnySparseSetMut<'_, I, T, SA, IA, M>
{
  fn hash<H: Hasher>(&self, state: &mut H) {
    for index in self.sparse.iter().flatten() {
      unsafe { self.sparse.get_unchecked(index.get() - 1) }.hash(state);
      unsafe { self.dense.get_unchecked(index.get() - 1) }.hash(state);
    }
  }
}

impl<I: SparseSetIndex, T, SA: Allocator, IA: Allocator, M: MemBuilder> Index<I>
  for AnySparseSetMut<'_, I, T, SA, IA, M>
{
  type Output = T;

  fn index(&self, index: I) -> &Self::Output {
    self.get(index).unwrap()
  }
}

impl<I: SparseSetIndex, T, SA: Allocator, IA: Allocator, M: MemBuilder> IndexMut<I>
  for AnySparseSetMut<'_, I, T, SA, IA, M>
{
  fn index_mut(&mut self, index: I) -> &mut Self::Output {
    self.get_mut(index).unwrap()
  }
}

impl<'a, I: SparseSetIndex, T, SA: Allocator, IA: Allocator, M: MemBuilder> IntoIterator
  for &'a AnySparseSetMut<'_, I, T, SA, IA, M>
{
  type Item = (I, &'a T);
  type IntoIter = impl Iterator<Item = Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

impl<'a, I: SparseSetIndex, T, SA: Allocator, IA: Allocator, M: MemBuilder> IntoIterator
  for &'a mut AnySparseSetMut<'_, I, T, SA, IA, M>
{
  type Item = (I, &'a mut T);
  type IntoIter = impl Iterator<Item = Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.iter_mut()
  }
}

impl<I: PartialEq + SparseSetIndex, T: PartialEq, SA: Allocator, IA: Allocator, M: MemBuilder>
  PartialEq for AnySparseSetMut<'_, I, T, SA, IA, M>
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

impl<I: Eq + SparseSetIndex, T: Eq, SA: Allocator, IA: Allocator, M: MemBuilder> Eq
  for AnySparseSetMut<'_, I, T, SA, IA, M>
{
}

#[cfg(test)]
mod test {
  use std::collections::hash_map::DefaultHasher;

  use any_vec::any_value::AnyValueWrapper;
  use coverage_helper::test;

  use crate::AnySparseSet;

  use super::*;

  #[test]
  fn test_indices_allocator() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    let _ = set.downcast_mut::<usize>().unwrap().indices_allocator();
  }

  #[test]
  fn test_sparse_allocator() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    let _ = set.downcast_mut::<usize>().unwrap().sparse_allocator();
  }

  #[test]
  fn test_as_dense_slice() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.downcast_mut::<usize>().unwrap().as_dense_slice(), &[1]);
  }

  #[test]
  fn test_as_dense_mut_slice() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(
      set.downcast_mut::<usize>().unwrap().as_dense_mut_slice(),
      &mut [1]
    );
  }

  #[test]
  fn test_as_indices_slice() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(
      set.downcast_mut::<usize>().unwrap().as_indices_slice(),
      &[0]
    );
  }

  #[test]
  fn test_as_indices_ptr() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(
      set.downcast_mut::<usize>().unwrap().as_indices_ptr(),
      set
        .downcast_mut::<usize>()
        .unwrap()
        .as_indices_slice()
        .as_ptr()
    );
  }

  #[test]
  fn test_at() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    assert_eq!(set.downcast_mut::<usize>().unwrap().at(0), &1);
    assert_eq!(set.downcast_mut::<usize>().unwrap().at(2), &3);
  }

  #[should_panic]
  #[test]
  fn test_at_panics() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let _ = set.downcast_mut::<usize>().unwrap().at(100);
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

    assert_eq!(set.downcast_mut::<usize>().unwrap().at(2), &10);
  }

  #[should_panic]
  #[test]
  fn test_at_mut_panics() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let _ = set.downcast_mut::<usize>().unwrap().at_mut(100);
  }

  #[test]
  fn test_clear() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    set.downcast_mut::<usize>().unwrap().clear();

    assert!(set.is_empty());
  }

  #[test]
  fn test_contains() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(!set.downcast_mut::<usize>().unwrap().contains(0));
    set.insert(0, AnyValueWrapper::new(1usize));
    assert!(set.downcast_mut::<usize>().unwrap().contains(0));
    let _ = set.remove(0);
    assert!(!set.downcast_mut::<usize>().unwrap().contains(0));
  }

  #[test]
  fn test_get() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    assert_eq!(set.downcast_mut::<usize>().unwrap().get(0), Some(&1));
    assert_eq!(set.downcast_mut::<usize>().unwrap().get(2), Some(&3));
    assert_eq!(set.downcast_mut::<usize>().unwrap().get(100), None);
  }

  #[test]
  fn test_get_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let mut set_ref = set.downcast_mut::<usize>().unwrap();
    let value = set_ref.get_mut(2);
    assert_eq!(value, Some(&mut 3));
    *value.unwrap() = 10;

    assert_eq!(set.downcast_mut::<usize>().unwrap().get(2), Some(&10));
  }

  #[test]
  fn test_indices() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.downcast_mut::<usize>().unwrap().indices().eq([]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set.downcast_mut::<usize>().unwrap().indices().eq([0, 1, 2]));
  }

  #[test]
  fn test_insert_capacity_increases() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(1, 1);
    set.downcast_mut::<usize>().unwrap().insert(0, 1);
    assert_eq!(set.sparse_capacity(), 1);
    assert_eq!(set.dense_capacity(), 1);

    set.downcast_mut::<usize>().unwrap().insert(1, 2);
    assert!(set.sparse_capacity() >= 2);
    assert!(set.dense_capacity() >= 2);

    assert_eq!(set.downcast_mut::<usize>().unwrap().get(0), Some(&1));
    assert_eq!(set.downcast_mut::<usize>().unwrap().get(1), Some(&2));
  }

  #[test]
  fn test_insert_len_increases() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.downcast_mut::<usize>().unwrap().insert(0, 1);
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 1);

    set.downcast_mut::<usize>().unwrap().insert(1, 2);
    assert_eq!(set.dense_len(), 2);
    assert_eq!(set.sparse_len(), 2);

    set.downcast_mut::<usize>().unwrap().insert(100, 101);
    assert_eq!(set.dense_len(), 3);
    assert_eq!(set.sparse_len(), 101);
  }

  #[test]
  fn test_insert_overwrites() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.downcast_mut::<usize>().unwrap().insert(0, 1);
    assert_eq!(set.downcast_mut::<usize>().unwrap().get(0), Some(&1));

    set.downcast_mut::<usize>().unwrap().insert(0, 2);
    assert_eq!(set.downcast_mut::<usize>().unwrap().get(0), Some(&2));
  }

  #[test]
  fn test_iter() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.downcast_mut::<usize>().unwrap().iter().eq([]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set
      .downcast_mut::<usize>()
      .unwrap()
      .iter()
      .eq([(0, &1), (1, &2), (2, &3)]));
  }

  #[test]
  fn test_iter_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.downcast_mut::<usize>().unwrap().iter_mut().eq([]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set.downcast_mut::<usize>().unwrap().iter_mut().eq([
      (0, &mut 1),
      (1, &mut 2),
      (2, &mut 3)
    ]));

    let mut set_ref = set.downcast_mut::<usize>().unwrap();
    let value = set_ref.iter_mut().next().unwrap();
    *(value.1) = 100;

    assert_eq!(set.downcast_mut::<usize>().unwrap().first(), Some(&100));
  }

  #[test]
  fn test_is_empty() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.downcast_mut::<usize>().unwrap().is_empty());

    set.insert(0, AnyValueWrapper::new(1usize));
    assert!(!set.downcast_mut::<usize>().unwrap().is_empty());

    let _ = set.remove(0);
    assert!(set.downcast_mut::<usize>().unwrap().is_empty());
  }

  #[test]
  fn test_dense_len() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.downcast_mut::<usize>().unwrap().dense_len(), 0);

    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.downcast_mut::<usize>().unwrap().dense_len(), 1);
  }

  #[test]
  fn test_sparse_len() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.downcast_mut::<usize>().unwrap().sparse_len(), 0);

    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.downcast_mut::<usize>().unwrap().sparse_len(), 1);

    set.insert(100, AnyValueWrapper::new(1usize));
    assert_eq!(set.downcast_mut::<usize>().unwrap().sparse_len(), 101);
  }

  #[test]
  fn test_remove_can_return_none() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.downcast_mut::<usize>().unwrap().remove(1), None);
    assert_eq!(set.downcast_mut::<usize>().unwrap().remove(100), None);
  }

  #[test]
  fn test_remove_can_return_some() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.downcast_mut::<usize>().unwrap().remove(0), Some(1));
  }

  #[test]
  fn test_remove_len_decreases() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));

    assert_eq!(set.dense_len(), 2);
    assert_eq!(set.sparse_len(), 2);
    assert_eq!(set.downcast_mut::<usize>().unwrap().remove(0), Some(1));
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 2);
    assert_eq!(set.downcast_mut::<usize>().unwrap().remove(0), None);
    assert_eq!(set.dense_len(), 1);
    assert_eq!(set.sparse_len(), 2);
  }

  #[test]
  fn test_remove_swaps_with_last() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    assert!(set.downcast_mut::<usize>().unwrap().values().eq(&[1, 2, 3]));

    let _ = set.downcast_mut::<usize>().unwrap().remove(0);
    assert!(set.downcast_mut::<usize>().unwrap().values().eq(&[3, 2]));
  }

  #[test]
  fn test_reserve_dense() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.dense_capacity(), 0);

    set.downcast_mut::<usize>().unwrap().reserve_dense(3);
    let capacity = set.dense_capacity();
    assert!(capacity >= 2);

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));

    set.downcast_mut::<usize>().unwrap().reserve_dense(1);
    assert_eq!(set.dense_capacity(), capacity);
  }

  #[test]
  fn test_reserve_sparse() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.sparse_capacity(), 0);

    set.downcast_mut::<usize>().unwrap().reserve_sparse(3);
    let capacity = set.sparse_capacity();
    assert!(capacity >= 2);

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));

    set.downcast_mut::<usize>().unwrap().reserve_sparse(1);
    assert_eq!(set.sparse_capacity(), capacity);
  }

  #[test]
  fn test_reserve_exact_dense() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.dense_capacity(), 0);

    set.downcast_mut::<usize>().unwrap().reserve_exact_dense(3);
    let capacity = set.dense_capacity();
    assert!(capacity >= 2);

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));

    set.downcast_mut::<usize>().unwrap().reserve_exact_dense(1);
    assert_eq!(set.dense_capacity(), capacity);
  }

  #[test]
  fn test_reserve_exact_sparse() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.sparse_capacity(), 0);

    set.downcast_mut::<usize>().unwrap().reserve_exact_sparse(3);
    let capacity = set.sparse_capacity();
    assert!(capacity >= 2);

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));

    set.downcast_mut::<usize>().unwrap().reserve_exact_sparse(1);
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
    set.downcast_mut::<usize>().unwrap().shrink_to_fit_dense();
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
    set.downcast_mut::<usize>().unwrap().shrink_to_fit_sparse();
    assert_eq!(set.sparse_capacity(), 2);
    assert_eq!(set.sparse_len(), 2);
  }

  #[test]
  fn test_shrink_to_fit_max_index_zero() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 0);
    set.downcast_mut::<usize>().unwrap().shrink_to_fit_sparse();
    assert_eq!(set.sparse_capacity(), 0);
    assert_eq!(set.sparse_len(), 0);
  }

  #[test]
  fn test_shrink_to_dense_can_reduce() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.dense_capacity(), 3);
    set.downcast_mut::<usize>().unwrap().shrink_to_dense(1);
    assert_eq!(set.dense_capacity(), 1);
  }

  #[test]
  fn test_shrink_to_dense_cannot_reduce() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert_eq!(set.dense_capacity(), 3);
    set.downcast_mut::<usize>().unwrap().shrink_to_dense(0);
    assert_eq!(set.dense_capacity(), 3);
  }

  #[test]
  fn test_shrink_to_sparse_can_reduce() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 1);
    set.downcast_mut::<usize>().unwrap().shrink_to_sparse(1);
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
    set.downcast_mut::<usize>().unwrap().shrink_to_sparse(0);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 3);
  }

  #[test]
  fn test_shrink_to_max_index_zero() {
    let mut set: AnySparseSet<usize> = AnySparseSet::with_capacity::<usize>(3, 3);
    assert_eq!(set.sparse_capacity(), 3);
    assert_eq!(set.sparse_len(), 0);
    set.downcast_mut::<usize>().unwrap().shrink_to_sparse(0);
    assert_eq!(set.sparse_capacity(), 0);
    assert_eq!(set.sparse_len(), 0);
  }

  #[test]
  fn test_values() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.downcast_mut::<usize>().unwrap().values().eq(&[]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set.downcast_mut::<usize>().unwrap().values().eq(&[1, 2, 3]));
  }

  #[test]
  fn test_values_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.downcast_mut::<usize>().unwrap().values_mut().eq(&[]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set
      .downcast_mut::<usize>()
      .unwrap()
      .values_mut()
      .eq(&[1, 2, 3]));

    let mut set_ref = set.downcast_mut::<usize>().unwrap();
    let value = set_ref.values_mut().next().unwrap();
    *value = 100;

    assert_eq!(set.downcast_mut::<usize>().unwrap().get(0), Some(&100));
  }

  #[test]
  fn test_as_ref() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let set_ref = set.downcast_mut::<usize>().unwrap();
    let reference: &AnySparseSetMut<'_, _, _> = set_ref.as_ref();
    assert_eq!(reference.first(), Some(&1));

    let reference: &[usize] = set_ref.as_ref();
    assert_eq!(reference.first(), Some(&1));
  }

  #[test]
  fn test_as_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let mut set_ref = set.downcast_mut::<usize>().unwrap();
    let reference: &mut AnySparseSetMut<'_, _, _> = set_ref.as_mut();
    assert_eq!(reference.first(), Some(&1));

    let reference: &mut [usize] = set_ref.as_mut();
    assert_eq!(reference.first(), Some(&1));
  }

  #[test]
  fn test_debug() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(format!("{:?}", set.downcast_mut::<usize>().unwrap()), "{}");

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert_eq!(
      format!("{:?}", set.downcast_mut::<usize>().unwrap()),
      "{0: 1, 1: 2, 2: 3}"
    );
  }

  #[test]
  fn test_deref() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));

    assert_eq!(&*set.downcast_mut::<usize>().unwrap(), &[1]);
  }

  #[test]
  fn test_deref_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));

    assert_eq!(&mut *set.downcast_mut::<usize>().unwrap(), &mut [1]);
  }

  #[test]
  fn test_extend() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set
      .downcast_mut::<usize>()
      .unwrap()
      .extend([(0, 1), (1, 2), (2, 3)]);
    assert!(set.downcast_mut::<usize>().unwrap().values().eq(&[1, 2, 3]));
  }

  #[test]
  fn test_extend_ref() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set
      .downcast_mut::<usize>()
      .unwrap()
      .extend([(0, &1), (1, &2), (2, &3)]);
    assert!(set.downcast_mut::<usize>().unwrap().values().eq(&[1, 2, 3]));
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

    fn hash(value: &AnySparseSetMut<'_, usize, usize>) -> u64 {
      let mut hasher = TestHasher::default();
      value.hash(&mut hasher);
      assert!(hasher.writes_made >= value.len());
      hasher.finish()
    }

    let mut set_1: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    let mut set_2: AnySparseSet<usize> = AnySparseSet::new::<usize>();

    assert_eq!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );
    assert_eq!(
      hash(&set_1.downcast_mut::<usize>().unwrap()),
      hash(&set_2.downcast_mut::<usize>().unwrap())
    );

    set_1.insert(0, AnyValueWrapper::new(1usize));

    assert_ne!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );

    set_2.insert(0, AnyValueWrapper::new(2usize));

    assert_ne!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );

    let _ = set_2.remove(0);
    set_2.insert(1, AnyValueWrapper::new(2usize));

    assert_ne!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );

    set_1.insert(1, AnyValueWrapper::new(2usize));
    set_2.insert(0, AnyValueWrapper::new(1usize));

    assert_eq!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );
    assert_eq!(
      hash(&set_1.downcast_mut::<usize>().unwrap()),
      hash(&set_2.downcast_mut::<usize>().unwrap())
    );

    let _ = set_1.remove(0);
    let _ = set_2.remove(0);

    assert_eq!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );
    assert_eq!(
      hash(&set_1.downcast_mut::<usize>().unwrap()),
      hash(&set_2.downcast_mut::<usize>().unwrap())
    );
  }

  #[test]
  fn test_index() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    assert_eq!(set.downcast_mut::<usize>().unwrap()[0], 1);
    assert_eq!(set.downcast_mut::<usize>().unwrap()[2], 3);
  }

  #[should_panic]
  #[test]
  fn test_index_panics() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let _ = &set.downcast_mut::<usize>().unwrap()[100];
  }

  #[test]
  fn test_index_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let value = &mut set.downcast_mut::<usize>().unwrap()[2];
    assert_eq!(value, &mut 3);
    *value = 10;

    assert_eq!(set.downcast_mut::<usize>().unwrap()[2], 10);
  }

  #[should_panic]
  #[test]
  fn test_index_mut_panics() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let _ = &mut set.downcast_mut::<usize>().unwrap()[100];
  }

  #[test]
  fn test_into_iterator_ref() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!((&set.downcast_mut::<usize>().unwrap()).into_iter().eq([]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!((&set.downcast_mut::<usize>().unwrap())
      .into_iter()
      .eq([(0, &1), (1, &2), (2, &3)]));
  }

  #[test]
  fn test_into_iterator_mut() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!((&mut set.downcast_mut::<usize>().unwrap())
      .into_iter()
      .eq([]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!((&mut set.downcast_mut::<usize>().unwrap()).into_iter().eq([
      (0, &mut 1),
      (1, &mut 2),
      (2, &mut 3)
    ]));

    let mut set_ref = set.downcast_mut::<usize>().unwrap();
    let value = (&mut set_ref).into_iter().next().unwrap();
    *(value.1) = 100;

    assert_eq!(set.downcast_mut::<usize>().unwrap().first(), Some(&100));
  }

  #[test]
  fn test_eq() {
    let mut set_1: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    let mut set_2: AnySparseSet<usize> = AnySparseSet::new::<usize>();

    assert_eq!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );

    set_1.insert(0, AnyValueWrapper::new(1usize));

    assert_ne!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );

    set_2.insert(0, AnyValueWrapper::new(2usize));

    assert_ne!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );

    let _ = set_2.remove(0);
    set_2.insert(1, AnyValueWrapper::new(2usize));

    assert_ne!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );

    set_1.insert(1, AnyValueWrapper::new(2usize));
    set_2.insert(0, AnyValueWrapper::new(1usize));

    assert_eq!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );

    let _ = set_1.remove(0);
    let _ = set_2.remove(0);

    assert_eq!(
      set_1.downcast_mut::<usize>().unwrap(),
      set_2.downcast_mut::<usize>().unwrap()
    );
  }
}
