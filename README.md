# Sparse Set

[![Crates.io](https://img.shields.io/crates/v/sparse_set.svg)](https://crates.io/crates/sparse_set)
[![Docs.rs](https://docs.rs/sparse_set/badge.svg)](https://docs.rs/sparse_set)
[![CI](https://github.com/sgodwincs/sparse_set/workflows/CI/badge.svg)](https://github.com/{{gh-username}}/sparse_set/actions)

This crate is a sparse set implementation.

This won't go into detail on what it is, but instead I'll recommend reading https://research.swtch.com/sparse for a nice,
succinct description.

One thing to note is that a separate buffer is used to store the actual indices. This makes a tradeoff between
value insertion/removal and iteration speed.

Nightly is required and I have no motivation to change it as I use this crate for other projects on nightly.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as
defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

See [CONTRIBUTING.md](CONTRIBUTING.md).
