#![allow(missing_docs)]
#![allow(unused_results)]

use criterion::{criterion_group, criterion_main, Criterion};
use sparse_set::SparseSet;
use sparseset::SparseSet as OtherSparseSet;
use std::collections::HashMap;

pub fn insert_with_capacity_benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("insert with capacity");

  group.bench_function("SparseSet", |b| {
    b.iter(|| {
      let mut set = SparseSet::with_capacity(100000);

      for i in 0..100000usize {
        set.insert(i, i);
      }
    });
  });
  group.bench_function("HashMap", |b| {
    b.iter(|| {
      let mut map = HashMap::with_capacity(100000);

      for i in 0..100000usize {
        map.insert(i, i);
      }
    });
  });
  group.bench_function("OtherSparseSet", |b| {
    b.iter(|| {
      let mut set = OtherSparseSet::with_capacity(100000);

      for i in 0..100000usize {
        set.insert(i, i);
      }
    });
  });

  group.finish();
}

pub fn insert_without_capacity_benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("insert without capacity");

  // OtherSparseSet doesn't let you grow capacity with inserts.
  group.bench_function("SparseSet", |b| {
    b.iter(|| {
      let mut set = SparseSet::new();

      for i in 0..100000usize {
        set.insert(i, i);
      }
    });
  });
  group.bench_function("HashMap", |b| {
    b.iter(|| {
      let mut map = HashMap::new();

      for i in 0..100000usize {
        map.insert(i, i);
      }
    });
  });

  group.finish();
}

pub fn iterate_dense_benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("iterate dense");

  group.bench_function("SparseSet", |b| {
    let mut set = SparseSet::new();
    set.extend((0..100000usize).into_iter().enumerate());
    let mut sum = 0;

    b.iter(|| {
      criterion::black_box(for i in set.values() {
        sum += i;
      })
    });
  });
  group.bench_function("HashMap", |b| {
    let mut map = HashMap::with_capacity(100000);
    map.extend((0..100000usize).into_iter().enumerate());
    let mut sum = 0;

    b.iter(|| {
      criterion::black_box(for i in map.values() {
        sum += i;
      })
    });
  });
  group.bench_function("OtherSparseSet", |b| {
    let mut set = OtherSparseSet::with_capacity(100000);

    for i in 0..100000usize {
      set.insert(i, i);
    }

    let mut sum = 0;

    b.iter(|| {
      criterion::black_box(for i in set.iter() {
        sum += i.value;
      })
    });
  });

  group.finish();
}

pub fn iterate_sparse_benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("iterate sparse");
  let iter = (0..20000usize)
    .into_iter()
    .enumerate()
    .map(|(i, _)| (i * 5, i));

  group.bench_function("SparseSet", |b| {
    let mut set = SparseSet::new();
    set.extend(iter.clone());
    let mut sum = 0;

    b.iter(|| {
      criterion::black_box(for i in set.values() {
        sum += i;
      })
    });
  });
  group.bench_function("HashMap", |b| {
    let mut map = HashMap::with_capacity(100000);
    map.extend(iter.clone());
    let mut sum = 0;

    b.iter(|| {
      criterion::black_box(for i in map.values() {
        sum += i;
      })
    });
  });
  group.bench_function("OtherSparseSet", |b| {
    let mut set = OtherSparseSet::with_capacity(100000);

    for i in iter.clone() {
      set.insert(i.0, i.1);
    }

    let mut sum = 0;

    b.iter(|| {
      criterion::black_box(for i in set.iter() {
        sum += i.value;
      })
    });
  });

  group.finish();
}

pub fn remove_benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("remove");

  group.bench_function("SparseSet", |b| {
    let mut set = SparseSet::new();
    set.extend((0..100000usize).into_iter().enumerate());

    b.iter(|| {
      for i in 0..20000usize {
        set.remove(i * 5);
      }
    });
  });
  group.bench_function("HashMap", |b| {
    let mut map = HashMap::with_capacity(100000);
    map.extend((0..100000usize).into_iter().enumerate());

    b.iter(|| {
      for i in 0..20000usize {
        map.remove(&(i * 5));
      }
    });
  });
  group.bench_function("OtherSparseSet", |b| {
    let mut set = OtherSparseSet::with_capacity(100000);

    for i in 0..100000usize {
      set.insert(i, i);
    }

    b.iter(|| {
      for i in 0..20000usize {
        set.remove(i * 5);
      }
    });
  });

  group.finish();
}

criterion_group!(
  benches,
  insert_with_capacity_benchmark,
  insert_without_capacity_benchmark,
  iterate_dense_benchmark,
  iterate_sparse_benchmark,
  remove_benchmark,
);
criterion_main!(benches);
