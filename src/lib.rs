//! A crate that implements the sparse set data structure.
//!
//! See [this article](https://research.swtch.com/sparse) on more details behind the data structure.

#![cfg_attr(coverage_nightly, feature(no_coverage))]
#![feature(allocator_api)]
#![feature(dropck_eyepatch)]
#![feature(extend_one)]
#![feature(slice_ptr_get)]
#![feature(type_alias_impl_trait)]
#![feature(try_reserve_kind)]

#[cfg(feature = "arbitrary")]
pub mod arbitrary;

pub mod sparse_set;

pub use crate::sparse_set::{SparseSet, SparseSetIndex};
