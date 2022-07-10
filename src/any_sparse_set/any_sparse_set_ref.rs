#![allow(unsafe_code)]

use std::{
  fmt::{self, Debug, Formatter},
  hash::{Hash, Hasher},
  num::NonZeroUsize,
  ops::{Deref, Index},
};

use any_vec::{
  mem::{Heap, MemBuilder},
  AnyVecRef,
};

use crate::SparseSetIndex;

/// A reference to an immutable `AnySparseSet<I>` which has been downcasted to some type `T`.
pub struct AnySparseSetRef<'a, I, T: 'static, M: MemBuilder = Heap> {
  /// The a reference dense buffer, i.e., the buffer containing the actual data values.
  pub(in crate::any_sparse_set) dense: AnyVecRef<'a, T, M>,

  /// The sparse buffer, i.e., the buffer where each index may correspond to an index into `dense`.
  pub(in crate::any_sparse_set) sparse: &'a [Option<NonZeroUsize>],

  /// All the existing indices in `sparse`.
  ///
  /// The indices here will always be in order based on the `dense` buffer.
  pub(in crate::any_sparse_set) indices: &'a [I],
}

impl<I, T: 'static, M: MemBuilder> AnySparseSetRef<'_, I, T, M> {
  /// Extracts a slice containing the entire dense buffer.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  #[must_use]
  pub fn as_dense_slice(&self) -> &[T] {
    self.dense.as_slice()
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

  /// Returns an iterator over the sparse set's indices.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// Consuming the iterator is an *O*(*n*) operation.
  #[must_use]
  pub fn indices(&self) -> impl Iterator<Item = &I> {
    self.indices.iter()
  }

  /// Returns an iterator over the sparse set's indices and values as pairs.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// Consuming the iterator is an *O*(*n*) operation.
  pub fn iter(&self) -> impl Iterator<Item = (&I, &T)> {
    self.indices.iter().zip(self.dense.iter())
  }

  /// Returns `true` if the sparse set contains no elements.
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.dense_len() == 0
  }

  /// Returns the number of elements in the dense set, also referred to as its 'dense_len'.
  #[must_use]
  pub fn dense_len(&self) -> usize {
    self.dense.len()
  }

  /// Returns the number of elements in the sparse set, also referred to as its 'sparse_len'.
  #[must_use]
  pub fn sparse_len(&self) -> usize {
    self.sparse.len()
  }

  /// Returns an iterator over the sparse set's values.
  ///
  /// Do not rely on the order being consistent across insertions and removals.
  ///
  /// Consuming the iterator is an *O*(*n*) operation.
  #[must_use]
  pub fn values(&self) -> impl Iterator<Item = &T> {
    self.dense.iter()
  }
}

impl<I: SparseSetIndex, T: 'static, M: MemBuilder> AnySparseSetRef<'_, I, T, M> {
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
      .get(index.into())
      .and_then(|opt: &Option<NonZeroUsize>| opt.as_ref())
      .map(|dense_index| unsafe { self.dense.get_unchecked(dense_index.get() - 1) })
  }
}

impl<'a, I, T, M: MemBuilder> AsRef<AnySparseSetRef<'a, I, T, M>> for AnySparseSetRef<'a, I, T, M> {
  fn as_ref(&self) -> &Self {
    self
  }
}

impl<I, T, M: MemBuilder> AsRef<[T]> for AnySparseSetRef<'_, I, T, M> {
  fn as_ref(&self) -> &[T] {
    self.dense.as_slice()
  }
}

impl<I: Clone, T: Clone, M: MemBuilder> Clone for AnySparseSetRef<'_, I, T, M> {
  fn clone(&self) -> Self {
    AnySparseSetRef {
      dense: self.dense.clone(),
      sparse: self.sparse,
      indices: self.indices,
    }
  }
}

impl<I, T, M: MemBuilder> Deref for AnySparseSetRef<'_, I, T, M> {
  type Target = [T];

  fn deref(&self) -> &[T] {
    self.dense.as_slice()
  }
}

impl<I: Debug, T: Debug, M: MemBuilder> Debug for AnySparseSetRef<'_, I, T, M> {
  fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
    formatter.debug_map().entries(self.iter()).finish()
  }
}

impl<I: SparseSetIndex, T: Hash, M: MemBuilder> Hash for AnySparseSetRef<'_, I, T, M> {
  fn hash<H: Hasher>(&self, state: &mut H) {
    for index in self.sparse.iter() {
      if let Some(index) = index {
        unsafe { self.sparse.get_unchecked(index.get() - 1) }.hash(state);
        unsafe { self.dense.get_unchecked(index.get() - 1) }.hash(state);
      }
    }
  }
}

impl<I: SparseSetIndex, T, M: MemBuilder> Index<I> for AnySparseSetRef<'_, I, T, M> {
  type Output = T;

  fn index(&self, index: I) -> &Self::Output {
    self.get(index).unwrap()
  }
}

impl<'a, I: SparseSetIndex, T, M: MemBuilder> IntoIterator for &'a AnySparseSetRef<'_, I, T, M> {
  type Item = (&'a I, &'a T);
  type IntoIter = impl Iterator<Item = Self::Item>;

  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

impl<I: PartialEq + SparseSetIndex, T: PartialEq, M: MemBuilder> PartialEq
  for AnySparseSetRef<'_, I, T, M>
{
  fn eq(&self, other: &Self) -> bool {
    if self.indices.len() != other.indices.len() {
      return false;
    }

    for index in self.indices.iter() {
      let index: usize = (*index).into();

      match (self.sparse.get(index), other.sparse.get(index)) {
        (Some(&Some(index)), Some(&Some(other_index))) => {
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
        (Some(None), Some(None)) | (None, None) => {}
        _ => {
          return false;
        }
      }
    }

    true
  }
}

impl<I: Eq + SparseSetIndex, T: Eq, M: MemBuilder> Eq for AnySparseSetRef<'_, I, T, M> {}

#[cfg(test)]
mod test {
  use std::collections::hash_map::DefaultHasher;

  use any_vec::{any_value::AnyValueWrapper, traits::Cloneable};
  use coverage_helper::test;

  use crate::AnySparseSet;

  use super::*;

  #[test]
  fn test_as_dense_slice() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.downcast_ref::<usize>().unwrap().as_dense_slice(), &[1]);
  }

  #[test]
  fn test_as_indices_slice() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(
      set.downcast_ref::<usize>().unwrap().as_indices_slice(),
      &[0]
    );
  }

  #[test]
  fn test_as_indices_ptr() {
    let set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(
      set.downcast_ref::<usize>().unwrap().as_indices_ptr(),
      set
        .downcast_ref::<usize>()
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

    assert_eq!(set.downcast_ref::<usize>().unwrap().at(0), &1);
    assert_eq!(set.downcast_ref::<usize>().unwrap().at(2), &3);
  }

  #[test]
  fn test_contains() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(!set.downcast_ref::<usize>().unwrap().contains(0));
    set.insert(0, AnyValueWrapper::new(1usize));
    assert!(set.downcast_ref::<usize>().unwrap().contains(0));
    let _ = set.remove(0);
    assert!(!set.downcast_ref::<usize>().unwrap().contains(0));
  }

  #[test]
  fn test_get() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    assert_eq!(set.downcast_ref::<usize>().unwrap().get(0), Some(&1));
    assert_eq!(set.downcast_ref::<usize>().unwrap().get(2), Some(&3));
    assert_eq!(set.downcast_ref::<usize>().unwrap().get(100), None);
  }

  #[test]
  fn test_indices() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.downcast_ref::<usize>().unwrap().indices().eq(&[]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set
      .downcast_ref::<usize>()
      .unwrap()
      .indices()
      .eq(&[0, 1, 2]));
  }

  #[test]
  fn test_iter() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.downcast_ref::<usize>().unwrap().indices().eq(&[]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set
      .downcast_ref::<usize>()
      .unwrap()
      .iter()
      .eq([(&0, &1), (&1, &2), (&2, &3)]));
  }

  #[test]
  fn test_is_empty() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.downcast_ref::<usize>().unwrap().is_empty());

    set.insert(0, AnyValueWrapper::new(1usize));
    assert!(!set.downcast_ref::<usize>().unwrap().is_empty());

    let _ = set.remove(0);
    assert!(set.downcast_ref::<usize>().unwrap().is_empty());
  }

  #[test]
  fn test_dense_len() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.downcast_ref::<usize>().unwrap().dense_len(), 0);

    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.downcast_ref::<usize>().unwrap().dense_len(), 1);
  }

  #[test]
  fn test_sparse_len() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(set.downcast_ref::<usize>().unwrap().sparse_len(), 0);

    set.insert(0, AnyValueWrapper::new(1usize));
    assert_eq!(set.downcast_ref::<usize>().unwrap().sparse_len(), 1);

    set.insert(100, AnyValueWrapper::new(1usize));
    assert_eq!(set.downcast_ref::<usize>().unwrap().sparse_len(), 101);
  }

  #[test]
  fn test_values() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!(set.downcast_ref::<usize>().unwrap().values().eq(&[]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!(set.downcast_ref::<usize>().unwrap().values().eq(&[1, 2, 3]));
  }

  #[test]
  fn test_as_ref() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let set_ref = set.downcast_ref::<usize>().unwrap();
    let reference: &AnySparseSetRef<'_, _, _> = set_ref.as_ref();
    assert_eq!(reference.first(), Some(&1));

    let reference: &[usize] = set_ref.as_ref();
    assert_eq!(reference.first(), Some(&1));
  }

  #[test]
  fn test_clone() {
    let mut set: AnySparseSet<usize, dyn Cloneable> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let cloned_set = set.clone();
    assert_eq!(
      set.downcast_ref::<usize>().unwrap(),
      cloned_set.downcast_ref::<usize>().unwrap()
    );
  }

  #[test]
  fn test_clone_zero_capacity() {
    let set: AnySparseSet<usize, dyn Cloneable> = AnySparseSet::new::<usize>();
    assert_eq!(set.dense_capacity(), 0);

    let cloned_set = set.clone();
    assert_eq!(
      set.downcast_ref::<usize>().unwrap(),
      cloned_set.downcast_ref::<usize>().unwrap()
    );
  }

  #[test]
  fn test_debug() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert_eq!(format!("{:?}", set.downcast_ref::<usize>().unwrap()), "{}");

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert_eq!(
      format!("{:?}", set.downcast_ref::<usize>().unwrap()),
      "{0: 1, 1: 2, 2: 3}"
    );
  }

  #[test]
  fn test_deref() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    set.insert(0, AnyValueWrapper::new(1usize));

    assert_eq!(set.downcast_ref::<usize>().unwrap().deref(), &[1]);
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

    fn hash(value: &AnySparseSetRef<'_, usize, usize>) -> u64 {
      let mut hasher = TestHasher::default();
      value.hash(&mut hasher);
      assert!(hasher.writes_made >= value.len());
      hasher.finish()
    }

    let mut set_1: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    let mut set_2: AnySparseSet<usize> = AnySparseSet::new::<usize>();

    assert_eq!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );
    assert_eq!(
      hash(&set_1.downcast_ref::<usize>().unwrap()),
      hash(&set_2.downcast_ref::<usize>().unwrap())
    );

    set_1.insert(0, AnyValueWrapper::new(1usize));

    assert_ne!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );

    set_2.insert(0, AnyValueWrapper::new(2usize));

    assert_ne!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );

    let _ = set_2.remove(0);
    set_2.insert(1, AnyValueWrapper::new(2usize));

    assert_ne!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );

    set_1.insert(1, AnyValueWrapper::new(2usize));
    set_2.insert(0, AnyValueWrapper::new(1usize));

    assert_eq!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );
    assert_eq!(
      hash(&set_1.downcast_ref::<usize>().unwrap()),
      hash(&set_2.downcast_ref::<usize>().unwrap())
    );

    let _ = set_1.remove(0);
    let _ = set_2.remove(0);

    assert_eq!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );
    assert_eq!(
      hash(&set_1.downcast_ref::<usize>().unwrap()),
      hash(&set_2.downcast_ref::<usize>().unwrap())
    );
  }

  #[test]
  fn test_index() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    assert_eq!(set.downcast_ref::<usize>().unwrap()[0], 1);
    assert_eq!(set.downcast_ref::<usize>().unwrap()[2], 3);
  }

  #[should_panic]
  #[test]
  fn test_index_panics() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));

    let _ = &set.downcast_ref::<usize>().unwrap()[100];
  }

  #[test]
  fn test_into_iterator_ref() {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    assert!((&set.downcast_ref::<usize>().unwrap()).into_iter().eq([]));

    set.insert(0, AnyValueWrapper::new(1usize));
    set.insert(1, AnyValueWrapper::new(2usize));
    set.insert(2, AnyValueWrapper::new(3usize));
    assert!((&set.downcast_ref::<usize>().unwrap())
      .into_iter()
      .eq([(&0, &1), (&1, &2), (&2, &3)]));
  }

  #[test]
  fn test_eq() {
    let mut set_1: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    let mut set_2: AnySparseSet<usize> = AnySparseSet::new::<usize>();

    assert_eq!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );

    set_1.insert(0, AnyValueWrapper::new(1usize));

    assert_ne!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );

    set_2.insert(0, AnyValueWrapper::new(2usize));

    assert_ne!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );

    let _ = set_2.remove(0);
    set_2.insert(1, AnyValueWrapper::new(2usize));

    assert_ne!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );

    set_1.insert(1, AnyValueWrapper::new(2usize));
    set_2.insert(0, AnyValueWrapper::new(1usize));

    assert_eq!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );

    let _ = set_1.remove(0);
    let _ = set_2.remove(0);

    assert_eq!(
      set_1.downcast_ref::<usize>().unwrap(),
      set_2.downcast_ref::<usize>().unwrap()
    );
  }
}
