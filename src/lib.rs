//! A crate that implements the sparse set data structure.
//!
//! See [this article](https://research.swtch.com/sparse) on more details behind the data structure.

#![cfg_attr(coverage_nightly, feature(no_coverage))]
#![feature(allocator_api)]
#![feature(type_alias_impl_trait)]

#[cfg(feature = "arbitrary")]
pub mod arbitrary;

#[cfg(feature = "any_vec")]
pub use any_vec;
#[cfg(feature = "any_vec")]
pub mod any_sparse_set;

pub mod index;
pub mod sparse_set;
pub mod sparse_vec;

pub use crate::{index::SparseSetIndex, sparse_set::SparseSet, sparse_vec::SparseVec};

#[cfg(feature = "any_vec")]
pub use crate::any_sparse_set::{AnySparseSet, AnySparseSetMut, AnySparseSetRef};
