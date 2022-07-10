//! `Arbitrary` implementations for the types in this crate.

use std::{
  marker::PhantomData,
  ops::{Deref, DerefMut},
};

use arbitrary::{Arbitrary, Unstructured};

use crate::{
  any_sparse_set::{
    any_value::AnyValueWrapper,
    traits::{None, Trait},
    AnySparseSet, SatisfyTraits,
  },
  SparseSetIndex,
};

/// Wrapper around an `AnySparseSet` to enable an `Arbitrary implementation.
#[derive(Debug)]
pub struct AnySparseSetWrapper<I, T, Traits: ?Sized + Trait = dyn None> {
  set: AnySparseSet<I, Traits>,
  _marker: PhantomData<T>,
}

impl<I, T, Traits: ?Sized + Trait> Deref for AnySparseSetWrapper<I, T, Traits> {
  type Target = AnySparseSet<I, Traits>;

  fn deref(&self) -> &Self::Target {
    &self.set
  }
}

impl<I, T, Traits: ?Sized + Trait> DerefMut for AnySparseSetWrapper<I, T, Traits> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.set
  }
}

impl<'a, I: From<usize> + SparseSetIndex, T: Arbitrary<'a> + 'static, Traits: ?Sized + Trait>
  Arbitrary<'a> for AnySparseSetWrapper<I, T, Traits>
where
  T: SatisfyTraits<Traits>,
{
  fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
    // Get the number of `T`s we should insert into our collection.
    let len = u.arbitrary_len::<T>()?;
    let mut set = AnySparseSet::<I, Traits>::with_capacity::<T>(len, len);

    for _ in 0..len {
      if *u.choose(&[false, true])? {
        let index = u.int_in_range(0..=(len - 1))?.into();
        let value = T::arbitrary(u)?;
        set.insert(index, AnyValueWrapper::new(value));
      }
    }

    Ok(AnySparseSetWrapper {
      set,
      _marker: PhantomData,
    })
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_any_sparse_set_arbitrary() {
    let bytes = (0..255).into_iter().collect::<Vec<u8>>().repeat(100);
    let mut u = Unstructured::new(&*bytes);
    let set: AnySparseSetWrapper<usize, usize> = AnySparseSetWrapper::arbitrary(&mut u).unwrap();
    assert!(!set.is_empty());
  }
}
