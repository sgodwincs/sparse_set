//! `Arbitrary` implementations for the types in this crate.

use std::mem;

use arbitrary::{Arbitrary, Unstructured};

use crate::{SparseSet, SparseSetIndex, SparseVec};

impl<'a, I: From<usize> + SparseSetIndex, T: Arbitrary<'a>> Arbitrary<'a> for SparseSet<I, T> {
  fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
    // Get the number of `T`s we should insert into our collection.
    let len = u.arbitrary_len::<T>()?;
    let mut set = Self::with_capacity(len, len);

    for _ in 0..len {
      if *u.choose(&[false, true])? {
        let index = u.int_in_range(0..=(len - 1))?.into();
        let value = T::arbitrary(u)?;
        mem::drop(set.insert(index, value));
      }
    }

    Ok(set)
  }
}

impl<'a, I: From<usize> + SparseSetIndex, T: Arbitrary<'a>> Arbitrary<'a> for SparseVec<I, T> {
  fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
    // Get the number of `T`s we should insert into our collection.
    let len = u.arbitrary_len::<T>()?;
    let mut set = Self::with_capacity(len);

    for _ in 0..len {
      if *u.choose(&[false, true])? {
        let index = u.int_in_range(0..=(len - 1))?.into();
        let value = T::arbitrary(u)?;
        mem::drop(set.insert(index, value));
      }
    }

    Ok(set)
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_sparse_set_arbitrary() {
    let bytes = (0..255).into_iter().collect::<Vec<u8>>().repeat(100);
    let mut u = Unstructured::new(&bytes);
    let set: SparseSet<usize, usize> = SparseSet::arbitrary(&mut u).unwrap();
    assert!(!set.is_empty());
  }

  #[test]
  fn test_sparse_vec_arbitrary() {
    let bytes = (0..255).into_iter().collect::<Vec<u8>>().repeat(100);
    let mut u = Unstructured::new(&bytes);
    let set: SparseVec<usize, usize> = SparseVec::arbitrary(&mut u).unwrap();
    assert!(!set.is_empty());
  }
}
