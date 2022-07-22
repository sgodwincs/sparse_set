//! Defines types and implementations for indexing the sparse set data structure.

/// A type with this trait indicates it can be used as an index into a `SparseSet`.
///
/// Two indices may the same index if they are unequal, but if equal they must return the same index.
pub trait SparseSetIndex: Copy + Into<usize> {}

impl SparseSetIndex for usize {}
