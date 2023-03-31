use criterion::{criterion_group, Bencher, Criterion};
use sparse_set::SparseSet;
use std::collections::HashMap;

const ELEMENT_COUNT: usize = 100000;

fn input_iter() -> impl Iterator<Item = (usize, usize)> {
  (0..ELEMENT_COUNT).into_iter().enumerate()
}

fn sparse_set(b: &mut Bencher<'_>) {
  b.iter(|| {
    let mut set = SparseSet::new();

    for (i, v) in input_iter() {
      set.insert(i, v);
    }
  });
}

fn hash_map(b: &mut Bencher<'_>) {
  b.iter(|| {
    let mut map = HashMap::new();

    for (i, v) in input_iter() {
      map.insert(i, v);
    }
  });
}

fn benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("insert without capacity");

  group.bench_function("SparseSet", |b| sparse_set(b));
  group.bench_function("HashMap", |b| hash_map(b));

  group.finish();
}

criterion_group!(benches, benchmark);
