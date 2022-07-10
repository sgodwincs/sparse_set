#![no_main]
use libfuzzer_sys::{
  arbitrary::{Arbitrary, Unstructured},
  fuzz_target,
};
use sparse_set::arbitrary::AnySparseSetWrapper;

fuzz_target!(|bytes: &[u8]| {
  let u = Unstructured::new(bytes);
  let mut set = match AnySparseSetWrapper::<usize, usize>::arbitrary_take_rest(u) {
    Ok(set) => set,
    _ => return,
  };

  assert_eq!(
    set.downcast_ref::<usize>().unwrap(),
    set.downcast_ref::<usize>().unwrap()
  );
  set.clear();
});
