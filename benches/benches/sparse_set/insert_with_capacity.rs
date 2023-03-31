use criterion::{criterion_group, Bencher, Criterion};
use sparse_set::SparseSet;
use sparseset::SparseSet as CrateSparseSet;
use std::collections::HashMap;

const ELEMENT_COUNT: usize = 100000;

fn input_iter() -> impl Iterator<Item = (usize, usize)> {
  (0..ELEMENT_COUNT).into_iter().enumerate()
}

fn sparse_set(b: &mut Bencher<'_>) {
  b.iter(|| {
    let mut set = SparseSet::with_capacity(ELEMENT_COUNT, ELEMENT_COUNT);

    for (i, v) in input_iter() {
      set.insert(i, v);
    }
  });
}

fn hash_map(b: &mut Bencher<'_>) {
  b.iter(|| {
    let mut map = HashMap::with_capacity(ELEMENT_COUNT);

    for (i, v) in input_iter() {
      map.insert(i, v);
    }
  });
}

fn crate_sparse_set(b: &mut Bencher<'_>) {
  b.iter(|| {
    let mut set = CrateSparseSet::with_capacity(ELEMENT_COUNT);

    for (i, v) in input_iter() {
      set.insert(i, v);
    }
  });
}

fn benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("insert with capacity");

  group.bench_function("SparseSet", |b| sparse_set(b));
  group.bench_function("HashMap", |b| hash_map(b));
  group.bench_function("CrateSparseSet", |b| crate_sparse_set(b));

  group.finish();
}

criterion_group!(benches, benchmark);
