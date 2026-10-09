# API reference

## Functions

| Function | Description |
| --- | --- |
| `TurboQuantIndex(dim, bits)` | Eager index (`bits ∈ {2,3,4}`, `dim` a positive multiple of 8) |
| `TurboQuantIndex(bits)` | Lazy index; dim inferred on first non-empty `add!` |
| `add!(index, X)` | Append `n × dim` `Float32` vectors |
| `search(index, Q, k; mask = nothing)` | `k` nearest slots: `(scores, indices)` `nq × k_eff` |
| `calibrate!(index, sample)` | Fit TQ+ from a `rows × dim` sample; re-encodes stored rows |
| `calibration_state(index)` | `:uncalibrated` or `:calibrated` |
| `is_calibrated(index)` / `calibration(index)` | Predicate / `(; shift, scale)` or `nothing` |
| `codebook(bits, dim)` | Canonical Lloyd-Max codebook `(; boundaries, centroids)` |
| `swap_remove!(index, slot)` | O(1) positional removal (returns moved-from slot) |
| `IdMapIndex(dim, bits)` | Stable-id wrapper |
| `add_with_ids!(index, X, ids)` | Add with `UInt64` external ids |
| `search(index, Q, k; allowlist = ids)` | Search restricted to external ids |
| `remove!(index, id)` | Remove by id, returns `Bool` |
| `contains_id(index, id)` | Id membership (`id in index`) |
| `write_index` / `load_index` | Persist / load `TurboQuantIndex` (path or `IO`) |
| `write_idmap` / `load_idmap` | Persist / load `IdMapIndex` (path or `IO`) |
| `to_bytes` / `from_bytes` | In-memory serialization, same layout as files |
| `serialized_len(index)` | Exact on-disk length, without serializing |
| `from_parts(dim, bits, n, packed, scales, shift, scale)` | Rebuild from validated raw parts |
| `packed_codes(index)` | Canonical bit-plane codes |
| `blocked_codes(index)` | Sequential blocked code bytes (the file payload) |
| `codebook_for_write(index)` | Codebook arrays a file embeds |
| `size(index)` / `size(index, d)`, `is_lazy`, `dim_opt`, `bit_width`, `scales`, `tqplus_shift`, `tqplus_scale` | Accessors (`size` is `(n, dim)`) |
| `is_packed_ready` / `is_slots_ready` | Layout-state probes (always `true` here) |
| `is_addable(index, ids)` | Whether an id batch is addable |
| `first_invalid_coord(values, dim)` | First invalid coordinate (1-based) |
| `MIN_INPUT_NORM`, `MIN_CALIBRATION_ROWS`, `RECOMMENDED_CALIBRATION_ROWS` | Constants |
| `prepare(index)` | No-op; kept for API parity |

## Julia interface conventions

The port leans on Julia's own protocol rather than a bespoke API:

* **Collections**: `length`, `isempty`, `size(idx) == (n, dim)`, `copy`, `empty!`, and a
  compact `show`. `IdMapIndex` also supports `id in index`, iteration
  (`for id in index`) and `keys(index)`.
* **Booleans**: predicates are named `is*` — `is_calibrated`, `is_lazy`, `is_addable`,
  `is_packed_ready`, `is_slots_ready` — and multi-value returns are named tuples
  (`calibration(idx)`, `codebook(bits, dim)`).
* **Single vectors**: `add!(idx, x)`, `search(idx, q, k)` (returns `1 × k_eff` matrices) and
  `add_with_ids!(idx, x, id)` are overloads of the matrix forms.
* **Predicates and accessors**: `is_calibrated(idx)` and `calibration(idx)` (a `NamedTuple`
  or `nothing`) are the idiomatic spellings; `calibration_state(idx)` (a `Symbol`) is kept
  for parity. `dim(idx)` is available but **not exported** — `size(idx, 2)` is the Julia
  spelling, and `is_lazy(idx)` answers the uncommitted-dim question.
* **Ownership**: accessors that hand back stored arrays (`scales`, `tqplus_shift`,
  `tqplus_scale`) return live internal storage — treat them as read-only; the serializers
  and `blocked_codes`/`external_ids` return copies.
* **Errors**: Julia exceptions, all subtypes of `TurboVecError`, instead of `Result` values;
  validator messages distinguish malformed parts (`InvalidParts`) from malformed files
  (`InvalidFileFormat`).
* **Load time**: the only runtime dependency is PrecompileTools, used for a precompile
  workload over the common paths. A cached `using` is ~0.1 s and the first `add!` / `search`
  land in ~0.2 s / ~0 ms, against ~1.8 s / ~1.6 s without it (the `llvmcall` kernel code is
  cached in the package image too).

## Not ported from the Rust crate

* v7 incremental `sync()` (append-only delta journaling) and its crash-consistency
  machinery; this port writes whole snapshots atomically (`write`/`load`/`to_bytes`/
  `from_bytes` are all present);
* the `.tv`/`.tvim` v2-v7 readers and `convert` tooling (this port has a single native format
  version);
* the AVX-512 VNNI/`vpermb` and NEON SDOT/SMMLA kernel families and the two-stage "planes"
  shortlist. AVX-512BW and AVX2 `vpshufb` kernels are ported (`src/simd.jl`) with a
  bit-identical portable scalar fallback;
* warning hooks and the mask-skip telemetry counter;
* the `try_*` `Result` forms — Julia raises typed exceptions instead (the error surface is
  otherwise mirrored);
* the Python framework integrations (LangChain, LlamaIndex, Haystack, Agno);
* `add_2d`/`calibrate_2d`: the matrix API carries the dim, so the separate-dim forms are
  unnecessary.

## Docstrings

```@autodocs
Modules = [TurboVec]
```
