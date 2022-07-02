#![no_main]
use libfuzzer_sys::fuzz_target;
use sparse_set::SparseSet;

fuzz_target!(|set: SparseSet<usize, usize>| {});
