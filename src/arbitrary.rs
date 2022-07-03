//! `Arbitrary` implementations for the types in this crate.

use arbitrary::{Arbitrary, Unstructured};

use crate::{SparseSet, SparseSetIndex};

impl<'a, I: From<usize> + SparseSetIndex, T: Arbitrary<'a>> Arbitrary<'a> for SparseSet<I, T> {
  fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
    // Get the number of `T`s we should insert into our collection.
    let len = u.arbitrary_len::<T>()?;
    let mut set = SparseSet::with_capacity(len, len);

    for _ in 0..len {
      if *u.choose(&[false, true])? {
        let index = u.int_in_range(0..=(len - 1))?.into();
        let value = T::arbitrary(u)?;
        set.insert(index, value);
      }
    }

    Ok(set)
  }
}
