# TurboVec.jl

A Julia port of [`turbovec`](https://github.com/RyanCodrai/turbovec), a
vector index built on Google Research's
[TurboQuant](https://arxiv.org/abs/2504.19874) algorithm. Vectors are
compressed to 2–4 bits per coordinate after a deterministic random
rotation and scored directly against a per-query lookup table — no
training phase, no decompression, online ingest.

```julia
using TurboVec

index = TurboQuantIndex(1536, 4)      # dim, bits per coordinate
add!(index, X)                        # X::Matrix{Float32}, n × 1536
add!(index, more_vectors)

scores, indices = search(index, Q, 10)   # Q::Matrix{Float32}, nq × 1536

write_index("my_index.tv", index)
loaded = load_index("my_index.tv")
```

Stable external ids:

```julia
index = IdMapIndex(1536, 4)
add_with_ids!(index, X, ids)             # ids::AbstractVector{<:Integer}
scores, ids = search(index, Q, 10)
remove!(index, 1002)                     # O(1) swap-and-remove
write_idmap("my_index.tvim", index)
loaded = load_idmap("my_index.tvim")
```

## Algorithm

1. **Normalize.** Each vector is stored as its length plus a unit
   direction.
2. **Random rotation.** A deterministic globally-permuted block-Hadamard
   transform (two rounds of ChaCha8-seeded permutation, ±1 sign flip, and
   a normalized Walsh–Hadamard butterfly per block) makes each coordinate
   follow a known near-Gaussian marginal.
3. **Lloyd–Max codebook.** Optimal scalar-quantizer boundaries and
   centroids are solved once from Beta((d−1)/2, (d−1)/2) — no data
   training.
4. **TQ+ calibration (optional).** `calibrate!(index, sample)` fits two
   scalars per coordinate (shift and scale) so anisotropic data matches
   the codebook's target distribution; every stored row is re-encoded.
5. **Bit-pack.** Codes are packed into a 32-vector blocked layout, one
   code byte per byte group per lane.
6. **Length renormalization.** Each vector stores `||v|| / ⟨u, x̂⟩`, which
   removes the quantization bias from the inner-product estimator.

Search rotates the query once, builds u8 nibble lookup tables quantized
exactly as the Rust kernels do, scans the blocked codes with integer
accumulation, applies the per-query scale/bias, and multiplies each
candidate by its stored renormalization scale.

## API

| Function | Description |
| --- | --- |
| `TurboQuantIndex(dim, bits)` | Eager index (`bits ∈ {2,3,4}`, `dim` a positive multiple of 8) |
| `TurboQuantIndex(bits)` | Lazy index; dim inferred on first non-empty `add!` |
| `add!(index, X)` | Append `n × dim` `Float32` vectors |
| `search(index, Q, k; mask = nothing)` | `k` nearest slots: `(scores, indices)` `nq × k_eff` |
| `calibrate!(index, sample)` | Fit TQ+ from a `rows × dim` sample; re-encodes stored rows |
| `calibration_state(index)` | `:uncalibrated` or `:calibrated` |
| `swap_remove!(index, slot)` | O(1) positional removal (returns moved-from slot) |
| `IdMapIndex(dim, bits)` | Stable-id wrapper |
| `add_with_ids!(index, X, ids)` | Add with `UInt64` external ids |
| `search(index, Q, k; allowlist = ids)` | Search restricted to external ids |
| `remove!(index, id)` | Remove by id, returns `Bool` |
| `contains_id(index, id)` | Id membership |
| `write_index` / `load_index` | Persist / load `TurboQuantIndex` |
| `write_idmap` / `load_idmap` | Persist / load `IdMapIndex` |
| `to_bytes` / `from_bytes` | In-memory serialization, same layout as files |
| `from_parts(dim, bits, n, packed, scales, shift, scale)` | Rebuild from validated raw parts |
| `packed_codes(index)` | Canonical bit-plane codes |
| `dim`, `dim_opt`, `bit_width`, `scales`, `tqplus_shift`, `tqplus_scale` | Accessors |
| `prepare(index)` | No-op; kept for API parity |

Filtering happens inside the scan. A slot `Bool` mask (or an id
allowlist) short-circuits whole 32-vector blocks with no allowed slots,
and non-allowed individual slots are dropped before heap insertion, so
`k_eff = min(k, n, n_allowed)` and no over-fetch happens.

Invalid input (NaN, ±Inf, `|x| ≥ 1e16`) is rejected with typed errors;
the zero vector is stored with score 0 and ranks last. Both index types
are safe for concurrent `search` calls.

## Layout and precision notes

* The public API takes `n × dim` `Float32` matrices (the numpy/FAISS
  convention). Julia arrays are column-major, so an encode row is
  strided. The encode path copies each row once into a contiguous
  worker buffer; transposing the whole matrix first measured *slower*
  than the strided copy (3.8 µs/row for `permutedims` vs 1.5 µs/row
  saved on 100k × 768), so it is deliberately not done.
* Stored codes use a 32-vector blocked layout so every scan loop walks
  contiguous bytes (one 32-byte group per block per byte group).
* Hot paths are `Float32` end to end. `Float64` appears only where it
  mirrors the Rust arithmetic bit-for-bit: the per-vector reconstruction
  inner product, the calibration fit, and the off-line Lloyd-Max solve.

## Threading

Search and encode use `Threads.@spawn` / `Threads.@threads`. Start Julia
with threads to use them:

```bash
julia -t auto
```

Inside a multi-threaded run, a single query splits its block range across
workers; a batch of queries runs one query per worker. No thread-count
configuration leaks into results: every row's score is computed by a
fixed accumulation order, so results are bit-identical regardless of the
number of threads.

## Persistence

Files are Julia-native, versioned, little-endian, and written to a
temporary file that is fsynced and atomically renamed. They are **not**
byte-compatible with the Rust `.tv` format (see below); `.tv` / `.tvim`
are used only by convention.

## Fidelity vs the Rust implementation

This port was validated against `turbovec` 1.0 on the same deterministic
corpus:

* the ChaCha8 sign/permutation stream is **bit-identical**;
* the Lloyd–Max codebooks are **bit-identical** for every tested
  `(bits, dim)`;
* the packed code bytes are **bit-identical** (calibrated and
  uncalibrated, 2/3/4-bit);
* top-k result sets match 100% and aligned scores agree to ~1e-6
  (the residual is the Rust SIMD kernels' integer accumulation vs this
  port's, not a coding difference).

Deliberate differences:

* the on-disk format is Julia-native rather than the Rust v6/v7 format;
* search quantizes the lookup tables to u8 exactly like Rust, but
  combines the two nibble lookups into one 256-entry integer table per
  byte group (mathematically the same integer sum);
* errors are Julia exceptions, so the `try_*` `Result` forms are not
  needed.

### Not ported

* v7 incremental `sync()` (append-only delta journaling) and its
  crash-consistency machinery; this port writes whole snapshots
  atomically;
* the `.tv`/`.tvim` v2-v7 readers and `convert` tooling (this port has a
  single native format version);
* the AVX-512 VNNI/`vpermb` and NEON SDOT/SMMLA kernel families and the
  two-stage "planes" shortlist. AVX-512BW and AVX2 `vpshufb` kernels are
  ported (`src/simd.jl`) with a bit-identical portable scalar fallback;
* warning hooks and allocation-count guarantees;
* the Python framework integrations (LangChain, LlamaIndex, Haystack,
  Agno).

## Performance vs Rust turbovec

Identical deterministic corpora (uniform `[-0.5, 0.5]`), `nq = 100`,
`k = 64`, same 16-core Skylake-AVX512 host, best of 3 search rounds,
encode warmed. `add` is a single bulk insert of `n` vectors.

| shape | Rust 1t add | Julia 1t add | Rust 16t add | Julia 16t add |
| --- | --- | --- | --- | --- |
| 768 / 4-bit / 100k | 0.84 s | 1.38 s | 0.41 s | 0.35 s |
| 768 / 2-bit / 100k | 0.68 s | 1.26 s | 0.24 s | 0.28 s |
| 1536 / 4-bit / 50k | 0.82 s | 1.58 s | 0.36 s | 0.46 s |
| 1536 / 2-bit / 50k | 0.58 s | 1.20 s | 0.22 s | 0.40 s |

| shape | Rust 1t search | Julia 1t search | Rust 16t search | Julia 16t search |
| --- | --- | --- | --- | --- |
| 768 / 4-bit / 100k | 1.30 ms | 3.55 ms | 0.17 ms | 0.54 ms |
| 768 / 2-bit / 100k | 0.70 ms | 1.68 ms | 0.11 ms | 0.23 ms |
| 1536 / 4-bit / 50k | 1.35 ms | 3.83 ms | 0.17 ms | 0.48 ms |
| 1536 / 2-bit / 50k | 0.68 ms | 1.55 ms | 0.11 ms | 0.22 ms |

Run-to-run variance on this shared host is ±20%, so treat the ratios as
approximate: encode is ~1.6–2x slower single-threaded and comparable at
16 threads; search is ~2–3x slower. The Rust kernels are hand-tuned
AVX-512/AVX2 kernels, while this port runs a 64-lane AVX-512BW pair
kernel (`src/simd.jl`) with an AVX2 single-block kernel and a portable
bit-identical scalar fallback, selected at runtime via `Base.llvmcall`.
`llvmcall` is the same lowering Rust's `std::arch` intrinsics use; the
gather-based alternatives measured far worse (LoopVectorization's
`vindex` scan: 45.8 ms vs 19.5 ms scalar vs 3.4 ms kernel), and SIMD.jl
only exposes static shuffles and hardware gathers, which cannot express
a runtime `vpshufb` table lookup.
Encode fuses rotation and quantization per row so the `dim × n` rotated
matrix never exists. The first version of this port was 10–20x slower on
search; the remaining gap is kernel micro-optimization and cache
behavior, not algorithm.

## Testing

```bash
julia --project=. -t auto -e 'using Pkg; Pkg.test()'
```

1605 assertions. The suite ports the applicable parts of turbovec's own
suite — rotation golden bits, codebook determinism, kernel correctness,
query-scale invariance, concurrent search, swap-remove, lazy init,
filtering/allowlists, id-map semantics, state sequences, calibration
and its bounds, `from_parts`, bytes I/O, error surface — plus recall
against brute force. Rust-generated golden fixtures pin the full encode
pipeline (rotation → codebook → quantization → bit packing) for 2/3/4-bit
shapes, and the rotation/codebook goldens pin the deterministic
primitives. Tests that pin SIMD byte-layouts, v7 crash
consistency, or the Rust on-disk format are not applicable to this
port.
