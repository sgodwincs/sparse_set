//! A type-erased, sparsely populated set, written `AnySparseSet<I>`, where `I` is the index type.
//!
//! `I` must implement `SparseSetIndex`, in particular, it must be able to be converted to a `usize` index.
//!
//! See [this article](https://research.swtch.com/sparse) on more details behind the data structure.

mod any_sparse_set;
mod any_sparse_set_mut;
mod any_sparse_set_ref;

pub use any_vec::{
  any_value, element, mem, ops, traits, ElementIterator, Iter, IterMut, IterRef, SatisfyTraits,
};

pub use any_sparse_set::*;
pub use any_sparse_set_mut::*;
pub use any_sparse_set_ref::*;
