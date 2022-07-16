# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
