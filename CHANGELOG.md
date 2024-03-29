# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.2] - 2023-07-27

### Added

- Added `SparseSet::retain` function.

## [0.8.1] - 2023-07-13

### Changed

- Fixed breakage from unstable feature.

## [0.8.0] - 2023-03-31

### Added

- Added `ImmutableEntry` and its corresponding API `SparseSet::immutable_entry`.

### Removed

- Removed the experimental `AnySparseSet` implementation along with the `unstable` feature.

## [0.7.1] - 2022-07-29

### Added

- Added additional trait bounds on returned iterator types.

## [0.7.0] - 2022-07-24

### Added

- Added an entry API.

### Changed

- All functions that give direct mutable access to the dense or indices buffer are now considered unsafe as its the
  caller's responsibility to ensure the relative ordering between the two is maintained.

## [0.6.1] - 2022-07-22

### Added

- Added `dense_index_of` to retrieve the raw `usize` index into the dense buffer from the higher-level index type.

## [0.6.0] - 2022-07-22

### Added

- Added `get_with_index`, `get_mut_with_index`, `insert_with_index`, and `remove_with_index` variants that return the
  corresponding stored index. This index is guaranteed to have mapped to the same `usize` as the given one, but it may
  not be equivalent, that's up to how the user defined the index type.

### Changed

- `SparseSet` no longer assumes that two indices mapping to the same `usize` imply they're equivalent.

## [0.5.0] - 2022-07-20

### Changed

- Changed `insert` calls to return the previous value if it existed.

## [0.4.1] - 2022-07-20

### Added

- Added `drain` functions.

## [0.4.0] - 2022-07-16

### Changed

- Changed indices iterators to copy the index instead of returning a reference.

## [0.3.0] - 2022-07-16

### Changed

- Fixed a safety issue when the collection reaches maximum capacity.
- Changed `(Index, Value)` iterators to copy the index instead of returning a reference. This required changes on some
  trait bounds.

## [0.2.0] - 2022-07-10

### Added

- Added support for a type-erased sparse set, `AnySparseSet`, behind an `unstable` feature.
- Added support for a sparse vector, `SparseVec`.
- Added additional functions: `iter`, `iter_mut

### Changed

- Loosened some trait bounds.
- Fixed incorrect `SparseSet::clear` implementation.
- Changed `SparseSet`'s `Debug` implementation to format as an `(index, value)` map.
- Changed `SparseSet`'s `IntoIterator` implementations to return pairs `(index, value)` instead of just values.

## [0.1.0] - 2022-07-04

### Added

- Initial release.
