# Sparse Set

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
