# Guide

## How the compression works

1. **Normalize.** Each vector is stored as its length plus a unit direction.
2. **Random rotation.** A deterministic globally-permuted block-Hadamard transform (two
   rounds of ChaCha8-seeded permutation, ±1 sign flip, and a normalized Walsh–Hadamard
   butterfly per block) makes each coordinate follow a known near-Gaussian marginal.
3. **Lloyd–Max codebook.** Optimal scalar-quantizer boundaries and centroids are solved once
   from Beta((d−1)/2, (d−1)/2) — no data training.
4. **TQ+ calibration (optional).** `calibrate!(index, sample)` fits two scalars per coordinate
   (shift and scale) so anisotropic data matches the codebook's target distribution; every
   stored row is re-encoded.
5. **Bit-pack.** Codes are packed into a 32-vector blocked layout, one code byte per byte
   group per lane.
6. **Length renormalization.** Each vector stores `||v|| / ⟨u, x̂⟩`, which removes the
   quantization bias from the inner-product estimator.

Search rotates the query once, builds u8 nibble lookup tables quantized exactly as the Rust
kernels do, scans the blocked codes with integer accumulation, applies the per-query
scale/bias, and multiplies each candidate by its stored renormalization scale.

## Filtering

Filtering happens inside the scan. A slot `Bool` mask (or an id allowlist) short-circuits
whole 32-vector blocks with no allowed slots, and non-allowed individual slots are dropped
before heap insertion, so `k_eff = min(k, n, n_allowed)` and no over-fetch happens.

```julia
# positional index: one Bool per slot
mask = falses(length(index)); mask[allowed_slots] .= true
scores, slots = search(index, Q, 10; mask = mask)

# id map: external ids, deduplicated; an empty or unknown id errors
scores, ids = search(idmap, Q, 10; allowlist = allowed_ids)
```

## Input validation

Invalid input (NaN, ±Inf, `|x| ≥ 1e16`) is rejected with typed errors; the zero vector is
stored with score 0 and ranks last. Both index types are safe for concurrent `search` calls.
Use `first_invalid_coord(values, dim)` to locate the first offending coordinate (1-based)
before calling `add!`.

## Layout and precision notes

* The public API takes `n × dim` `Float32` matrices (the numpy/FAISS convention). Julia
  arrays are column-major, so an encode row is strided. The encode path copies each row once
  into a contiguous worker buffer; transposing the whole matrix first measured *slower* than
  the strided copy (3.8 µs/row for `permutedims` vs 1.5 µs/row saved on 100k × 768), so it is
  deliberately not done.
* Stored codes use a 32-vector blocked layout so every scan loop walks contiguous bytes (one
  32-byte group per block per byte group).
* Hot paths are `Float32` end to end. `Float64` appears only where it mirrors the Rust
  arithmetic bit-for-bit: the per-vector reconstruction inner product, the calibration fit,
  and the off-line Lloyd-Max solve.

## Threading

Search and encode use `Threads.@spawn` / `Threads.@threads`. Start Julia with threads to use
them:

```bash
julia -t auto
```

Inside a multi-threaded run, a single query splits its block range across workers; a batch of
queries is scored two queries per code pass. No thread-count configuration leaks into
results: every row's score is computed by a fixed accumulation order, so results are
bit-identical regardless of the number of threads.

## Persistence

Files are Julia-native, versioned, little-endian, and written to a temporary file that is
fsynced and atomically renamed. They are **not** byte-compatible with the Rust `.tv` format
(see [Validation](validation.md)); `.tv` / `.tvim` are used only by convention.

```julia
write_index("index.tv", index);    loaded = load_index("index.tv")
write_idmap("index.tvim", idmap);  loaded = load_idmap("index.tvim")

buf = to_bytes(index);             copy  = from_bytes(TurboQuantIndex, buf)
serialized_len(index)              # exact on-disk length, without serializing
```
