//! Defines types and implementations for indexing the sparse set data structure.

/// A type with this trait indicates it can be used as an index into a `SparseSet`.
///
/// Two indices must convert to the same `usize` if and only if they are equal.
pub trait SparseSetIndex: Copy + Into<usize> {}

impl SparseSetIndex for usize {}
