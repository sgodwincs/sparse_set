use criterion::{criterion_group, Bencher, Criterion};
use sparse_set::{any_sparse_set::any_value::AnyValueWrapper, AnySparseSet, SparseSet};
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

fn any_sparse_set_pre_downcast(b: &mut Bencher<'_>) {
  let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  set.extend(input_iter().map(|(i, v)| (i, AnyValueWrapper::new(v))));
  let mut downcast_set = set.downcast_mut::<usize>().unwrap();

  b.iter(|| {
    for i in 0..REMOVE_COUNT {
      let _ = downcast_set.remove(i * 5);
    }
  });
}

fn any_sparse_set_post_downcast(b: &mut Bencher<'_>) {
  let mut set: AnySparseSet<usize> = AnySparseSet::new::<usize>();
  set.extend(input_iter().map(|(i, v)| (i, AnyValueWrapper::new(v))));

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
