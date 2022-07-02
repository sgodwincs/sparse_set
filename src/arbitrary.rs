//! `Arbitrary` implementations for the types in this crate.

use arbitrary::{Arbitrary, Unstructured};

use crate::{SparseSet, SparseSetIndex};

impl<'a, I: Arbitrary<'a> + SparseSetIndex, T: Arbitrary<'a> + std::fmt::Debug> Arbitrary<'a>
  for SparseSet<I, T>
{
  fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
    // Get the number of `T`s we should insert into our collection.
    let len = u.arbitrary_len::<T>()?;
    let mut set = SparseSet::with_capacity(len);

    for index in 0..len {
      if *u.choose(&[false, true])? {
        let value = T::arbitrary(u)?;
        set.insert_raw(index, value);
      }
    }

    Ok(set)
  }
}
