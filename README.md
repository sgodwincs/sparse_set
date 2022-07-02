# Sparse Set

This crate is a sparse set implementation.

This won't go into detail on what it is, but instead I'd recommend reading https://research.swtch.com/sparse for a nice,
succinct description.

## Implementation details

So the way this crate is implemented is rather complex, and a simpler implementation is definitely possible (see
https://github.com/bombela/sparseset). So where is this complexity coming from and is it paying off in anyway?

The straightforward implementation is just having two `Vec`s, one for the dense buffer and one for the sparse buffer. I
was curious what it would look like to implement it with a single backing buffer. This would hopefully result in faster
allocations (i.e. inserts should be faster on average). My, admittedly probably naive, benchmark showed that inserting
100000 values showed this approach was 17% faster. Wow, 17% you say, but the numbers are already really small so the
absolute difference is minimal. The only other benefit is that the structure is slightly smaller.
