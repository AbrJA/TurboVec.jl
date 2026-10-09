# Validation & development

## Fidelity vs the Rust implementation

This port was validated against `turbovec` 1.0 on the same deterministic corpus:

* the ChaCha8 sign/permutation stream is **bit-identical**;
* the Lloyd–Max codebooks are **bit-identical** for every tested `(bits, dim)`;
* the packed code bytes are **bit-identical** (calibrated and uncalibrated, 2/3/4-bit);
* top-k result sets match 100% and aligned scores agree to ~1e-6 (the residual is the Rust
  SIMD kernels' integer accumulation vs this port's, not a coding difference).

Deliberate differences:

* the on-disk format is Julia-native rather than the Rust v6/v7 format;
* search quantizes the lookup tables to u8 exactly like Rust, but combines the two nibble
  lookups into one 256-entry integer table per byte group (mathematically the same integer
  sum);
* errors are Julia exceptions, so the `try_*` `Result` forms are not needed.

## Performance vs Rust turbovec

Identical deterministic corpora (uniform `[-0.5, 0.5]`), `nq = 100`, `k = 64`, same 16-core
Skylake-AVX512 host, best of 3 search rounds, encode warmed. `add` is a single bulk insert of
`n` vectors.

| shape | Rust 1t add | Julia 1t add | Rust 16t add | Julia 16t add |
| --- | --- | --- | --- | --- |
| 768 / 4-bit / 100k | 0.84 s | 1.38 s | 0.41 s | 0.35 s |
| 768 / 2-bit / 100k | 0.68 s | 1.26 s | 0.24 s | 0.28 s |
| 1536 / 4-bit / 50k | 0.82 s | 1.58 s | 0.36 s | 0.46 s |
| 1536 / 2-bit / 50k | 0.58 s | 1.20 s | 0.22 s | 0.40 s |

| shape | Rust 1t search | Julia 1t search | Rust 16t search | Julia 16t search |
| --- | --- | --- | --- | --- |
| 768 / 4-bit / 100k | 1.30 ms | 2.21 ms | 0.17 ms | 0.34 ms |
| 768 / 2-bit / 100k | 0.70 ms | 1.14 ms | 0.11 ms | 0.21 ms |
| 1536 / 4-bit / 50k | 1.35 ms | 2.93 ms | 0.17 ms | 0.43 ms |
| 1536 / 2-bit / 50k | 0.68 ms | 1.49 ms | 0.11 ms | 0.24 ms |

Search numbers are a 100-query batch; a single query (`nq = 1`) pays ~3.5 ms (768/4-bit)
because it cannot amortize code reads across queries. Batches are scored **two queries per
code pass** — the same 64 code bytes are shuffled against both queries' tables, cutting batch
cost to ~1.5 single-query scans per pair and halving code traffic — which is what puts batch
search within ~1.6–2.2x of Rust. Results are bit-identical to per-query searches, masked or
not.

Run-to-run variance on this shared host is ±20%, so treat the ratios as approximate: encode
is ~1.6–2x slower single-threaded and comparable at 16 threads; batch search is ~1.6–2.4x
slower. The Rust kernels are hand-tuned AVX-512/AVX2 kernels, while this port runs a 64-lane
AVX-512BW pair kernel and a two-query variant (`src/simd.jl`) with an AVX2 single-block
kernel and a portable bit-identical scalar fallback, selected at runtime via `Base.llvmcall`.
`llvmcall` is the same lowering Rust's `std::arch` intrinsics use; the gather-based
alternatives measured far worse (LoopVectorization's `vindex` scan: 45.8 ms vs 19.5 ms scalar
vs 3.4 ms kernel), and SIMD.jl only exposes static shuffles and hardware gathers, which
cannot express a runtime `vpshufb` table lookup. Encode fuses rotation and quantization per
row so the `dim × n` rotated matrix never exists. The first version of this port was 10–20x
slower on search; the remaining gap is kernel micro-optimization and cache behavior, not
algorithm.

## Optimization log

Accepted: the 32-vector blocked code layout with u8 nibble LUTs and integer accumulation;
AVX2 `vpshufb` and AVX-512BW pair kernels via `Base.llvmcall` with a bit-identical scalar
fallback; radix-8 block-Hadamard rotation; fused rotate+quantize per row (no `dim × n`
intermediate); and two queries per code pass (`scan_pair2`), which cut batch search ~1.6x.

Measured and rejected, with the numbers that killed them (100k × 768 4-bit scan, 1 thread):

| idea | result |
| --- | --- |
| u16 low/high-byte accumulator trick (Rust-style, no widening) | exact, but 3.42 ms vs 3.35 ms — no gain |
| software prefetch 256 B / 512 B / 1 KiB ahead | 3.28 / 4.40 / 3.62 ms — ≤2% at best |
| Q=4 query blocking | 3.03 ms/query vs 1.89 for Q=2 — register spills |
| full-matrix transpose before encode | `permutedims` 3.8–5.2 µs/row vs 1.3–1.8 µs/row saved |
| two-level 4-bit boundary scan | 4.6 vs 5.0 ns/coord ≈ 2% of encode |

SMT helps this latency-bound scan: `-t 16` (8 physical cores × 2 threads) beats `-t 8` on the
batch cells (768/4-bit: 0.34 vs 0.46 ms/query), so leave thread counts above the physical
core count.

## Development tooling

`dev/` holds the cross-validation and benchmark harnesses used to build this port (not part
of the package):

* `dev/tvref/` — Rust reference dumper; `cargo run --release` regenerates `dev/out/` from
  the published `turbovec = "=1.0.0"` crate.
* `dev/validate.jl` — compares this implementation with that reference: rotation streams,
  codebooks and packed codes bit-identical; 100% top-k set overlap; aligned scores within
  ~1e-6.
* `dev/tvbench/` + `dev/bench2.jl` — identical-corpus Rust and Julia benchmarks
  (`RAYON_NUM_THREADS=N` / `julia -t N`).
* `dev/profiling/` — the kernel and encode experiments from the optimization passes.

```bash
(cd dev/tvref && cargo run --release)      # regenerate reference data
julia --project=. dev/validate.jl          # bit-exactness check
```

## Testing

```bash
julia --project=. -t auto -e 'using Pkg; Pkg.test()'
```

The suite ports the applicable parts of turbovec's own suite — rotation golden bits,
codebook determinism, kernel correctness, query-scale invariance, concurrent search,
swap-remove, lazy init, filtering/allowlists, id-map semantics, state sequences, calibration
and its bounds, `from_parts`, bytes I/O, the full public surface (accessors, IO-generic entry
points, `serialized_len`, `is_addable`, the Base protocol) — plus recall against brute force.
Rust-generated golden fixtures pin the full encode pipeline (rotation → codebook →
quantization → bit packing) for 2/3/4-bit shapes, and the rotation/codebook goldens pin the
deterministic primitives. `Aqua` checks pass (no method ambiguities, undefined exports,
unbound type parameters, piracy, stale deps, or missing compat entries). Tests that pin SIMD
byte-layouts, v7 crash consistency, or the Rust on-disk format are not applicable to this
port.
