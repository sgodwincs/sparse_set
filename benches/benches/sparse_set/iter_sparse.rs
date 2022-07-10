use criterion::{criterion_group, Bencher, Criterion};
use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet, SparseSet};
use sparseset::SparseSet as CrateSparseSet;
use std::collections::HashMap;

const ELEMENT_COUNT: usize = 100000;

fn input_iter() -> impl Iterator<Item = (usize, usize)> {
  (0..ELEMENT_COUNT).into_iter().map(|i| i * 5).enumerate()
}

fn sparse_set(b: &mut Bencher<'_>) {
  let mut set = SparseSet::new();
  set.extend(input_iter());
  let mut sum = 0;

  b.iter(|| {
    for i in set.values() {
      sum += i;
    }
  });
}

fn any_sparse_set_pre_downcast(b: &mut Bencher<'_>) {
  let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  set.extend(input_iter().map(|(i, v)| (i, AnyValueWrapper::new(v))));
  let downcast_set = set.downcast_ref::<usize>().unwrap();
  let mut sum = 0;

  b.iter(|| {
    for i in downcast_set.values() {
      sum += i;
    }
  });
}

fn any_sparse_set_post_downcast(b: &mut Bencher<'_>) {
  let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  set.extend(input_iter().map(|(i, v)| (i, AnyValueWrapper::new(v))));
  let mut sum = 0;

  b.iter(|| {
    for i in set.values() {
      sum += unsafe { i.downcast_ref_unchecked::<usize>() };
    }
  });
}

fn hash_map(b: &mut Bencher<'_>) {
  let mut map = HashMap::new();
  map.extend(input_iter());
  let mut sum = 0;

  b.iter(|| {
    for i in map.values() {
      sum += i;
    }
  });
}

fn crate_sparse_set(b: &mut Bencher<'_>) {
  let mut set = CrateSparseSet::with_capacity(ELEMENT_COUNT * 5);

  for (i, v) in input_iter() {
    set.insert(i, v);
  }

  let mut sum = 0;

  b.iter(|| {
    for i in set.iter() {
      sum += i.value;
    }
  });
}

fn benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("iterate sparse");

  group.bench_function("SparseSet", |b| sparse_set(b));
  group.bench_function("AnySparseSet pre-downcast", |b| {
    any_sparse_set_pre_downcast(b)
  });
  group.bench_function("AnySparseSet post-downcast", |b| {
    any_sparse_set_post_downcast(b)
  });
  group.bench_function("HashMap", |b| hash_map(b));
  group.bench_function("CrateSparseSet", |b| crate_sparse_set(b));

  group.finish();
}

criterion_group!(benches, benchmark);
