use criterion::{criterion_group, Bencher, Criterion};
use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet, SparseSet};
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

fn any_sparse_set_pre_downcast(b: &mut Bencher<'_>) {
  b.iter(|| {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
    let mut downcast_set = unsafe { set.downcast_mut_unchecked::<usize>() };

    for (i, v) in input_iter() {
      downcast_set.insert(i, v);
    }
  });
}

fn any_sparse_set_post_downcast(b: &mut Bencher<'_>) {
  b.iter(|| {
    let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();

    for (i, v) in input_iter().map(|(i, v)| (i, AnyValueWrapper::new(v))) {
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
  group.bench_function("AnySparseSet pre-downcast", |b| {
    any_sparse_set_pre_downcast(b)
  });
  group.bench_function("AnySparseSet post-downcast", |b| {
    any_sparse_set_post_downcast(b)
  });
  group.bench_function("HashMap", |b| hash_map(b));

  group.finish();
}

criterion_group!(benches, benchmark);
