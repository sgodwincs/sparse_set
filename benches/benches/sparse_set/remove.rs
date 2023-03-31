use criterion::{criterion_group, Bencher, Criterion};
use sparse_set::SparseSet;
use sparseset::SparseSet as CrateSparseSet;
use std::collections::HashMap;

const ELEMENT_COUNT: usize = 100000;
const REMOVE_COUNT: usize = 20000;

fn input_iter() -> impl Iterator<Item = (usize, usize)> {
  (0..ELEMENT_COUNT).into_iter().enumerate()
}

fn sparse_set(b: &mut Bencher<'_>) {
  let mut set = SparseSet::new();
  set.extend(input_iter());

  b.iter(|| {
    for i in 0..REMOVE_COUNT {
      let _ = set.remove(i * 5);
    }
  });
}

fn hash_map(b: &mut Bencher<'_>) {
  let mut map = HashMap::new();
  map.extend(input_iter());

  b.iter(|| {
    for i in 0..REMOVE_COUNT {
      map.remove(&(i * 5));
    }
  });
}

fn crate_sparse_set(b: &mut Bencher<'_>) {
  let mut set = CrateSparseSet::with_capacity(ELEMENT_COUNT);

  for (i, v) in input_iter() {
    set.insert(i, v);
  }

  b.iter(|| {
    for i in 0..REMOVE_COUNT {
      set.remove(i * 5);
    }
  });
}

fn benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("remove");

  group.bench_function("SparseSet", |b| sparse_set(b));
  group.bench_function("HashMap", |b| hash_map(b));
  group.bench_function("CrateSparseSet", |b| crate_sparse_set(b));

  group.finish();
}

criterion_group!(benches, benchmark);
