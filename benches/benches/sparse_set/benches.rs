#![allow(missing_docs)]
#![allow(unsafe_code)]
#![allow(unused_results)]

use criterion::criterion_main;

mod insert_with_capacity;
mod insert_without_capacity;
mod iter_dense;
mod iter_sparse;
mod remove;

criterion_main!(
  insert_with_capacity::benches,
  insert_without_capacity::benches,
  iter_dense::benches,
  iter_sparse::benches,
  remove::benches
);
