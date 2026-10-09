# TurboVec.jl — Production-Readiness Proposals

A gap analysis of the current port (`TurboVec.jl` 0.1.0) against the Rust
reference (`turbovec` 1.0.0) plus Julia-idiomatic design improvements,
ranked for impact vs. effort. Prepared 2026-10-09 from commit `55c7b77`.

**Priority legend**

| Tag | Meaning |
| --- | --- |
| **P0** | Do before the first public release (registration / v1.0). Correctness, data safety, or hot-path waste. |
| **P1** | Do before claiming production parity (large corpus + multi-platform). |
| **P2** | Nice-to-have; do opportunistically or after 1.0. |

Each proposal lists: location → problem → proposal → impact/effort.

---

## Implementation status

**Phase A (all P0 items) has landed** on top of `55c7b77`:

| Item | Status | Commit |
| --- | --- | --- |
| P0-1 typed out-of-domain ids, P0-2 two-query tail | ✅ done | `d56eef3` |
| P0-3/4/5 format v2 + CRC-32C + loader/atomic-write hardening | ✅ done | `13c7dad` |
| P0-6/7/8 LUT/scan allocation removal | ✅ done | `1d4a623` |
| P0-9 validation scan, P0-10 calibration buffers | ✅ done | `255c590` |
| P0-11 precompile coverage | ✅ done | `ec77d7a` |
| P0-12 changelog (registration + SemVer policy deferred by choice) | ✅ changelog done | `4466b5d` |

Verified with the full `Pkg.test()` suite, the `--check-bounds=yes
--depwarn=error` hardening run, and `dev/validate.jl` (all green).

**Phase B (P1-3, P1-7…P1-11, P1-13…P1-15, P1-16) has landed:**

| Item | Status | Commit |
| --- | --- | --- |
| P1-7…P1-11 API/polish, P1-13…P1-15 cleanup | ✅ done | `72b306b` |
| P1-3 AVX2 two-query batch kernel (1.54x measured) | ✅ done | `f1e457a` |
| P1-16 CI: aarch64, validate, hardening, JET, formatter | ✅ done | `8dd442f` |
| P0-12(3) SemVer policy in the guide (registration still yours) | ✅ docs done | `72b306b` |

The CI jobs added in Phase B can only be verified on your next push
(they never run locally).

**Quick-wins pass (P1-6, P1-12, P1-17, P2-3, P2-7) has landed:**

| Item | Status | Commit |
| --- | --- | --- |
| P1-6 `single_query_parallelizes`, P1-12 `EncodeCtx`/`ScanCtx` | ✅ done | `c7a9123` |
| P2-3 CPUID probe hardened (no public replacement exists) | ✅ done | `87474c9` |
| P1-17 harness docs + manual bench job, P2-7 docs compat | ✅ done | `a25fd4e` |

**Worthwhile Phase C items have landed:**

| Item | Status | Commit |
| --- | --- | --- |
| P1-5 aarch64 NEON kernels (single + two-query, bit-exact) | ✅ done | `e5a4bbe` |
| P1-2 / P2-5 `fast = true` writes with a warning | ✅ done | `80e2878` |

Verified on aarch64 under Docker/QEMU: the full suite (23 files) and
`dev/validate.jl` both pass, and NEON parity is pinned by
`test/test_simd.jl` alongside the AVX2/AVX-512 checks.

**Assessed and deliberately not done:** P1-1 (demand-driven), P1-4
(rejected: the Rust planes pass is approximate, see the note below),
P2-1 (no VNNI/VBMI hardware to test on), P2-2 (cosmetic, frozen IR),
P2-4 (the calibration matrix is required for exact quantiles), P2-6
(JET is clean; no measured dispatch overhead). See §5 for the non-goals
rationale.

---

## 1. Where the port stands today

The port is already unusually strong for a "first version":

- **Bit-exact encode** against the Rust crate, pinned by golden bytes
  (`test/test_rust_golden.jl`, `dev/validate.jl`), kernel parity
  (`test/test_simd.jl`), rotation and codebook determinism tests.
- **Single runtime dependency** (PrecompileTools), Julia 1.13 floor, opaque-pointer
  `llvmcall` kernels, threaded encode/search, TTFX handled by a precompile workload.
- **Typed errors**, Aqua-clean, 23 test files (~1750 assertions), Documenter docs with a
  validation methodology page, and a frozen Rust cross-validation corpus committed in `dev/out/`.

The main gaps versus a "production ready" bar are: file-format hardening, a handful of
per-query/per-coordinate allocations in hot paths, an ARM (scalar-only) story, no incremental
persistence, no release process, and a few latent code smells. All are enumerated below.

---

## 2. Feature comparison: `turbovec` 1.0.0 vs `TurboVec.jl` 0.1.0

| Area | turbovec 1.0.0 (Rust) | TurboVec.jl 0.1.0 | Gap |
| --- | --- | --- | --- |
| Encode / `add` | `&[f32]` + `add_2d` | matrices / vectors | none (Julia idiom) |
| TQ+ calibration + stored-row re-encode | yes | yes | none |
| Lazy init (`new_lazy`) | yes | yes (`dim == 0` sentinel) | none |
| Search + slot mask | yes | yes | none |
| IdMap: stable ids, remove, allowlist, iter | yes | yes (`in`, `iterate`, `keys`) | none |
| Two-queries-per-code-pass | all x86 tiers | AVX-512BW only | AVX2 batches run per-query (P1-3) |
| Single-query block parallelism | yes (≥1024 blocks) | yes (≥1024 blocks) | none |
| AVX-512 VNNI/`vpermb` 4-bit kernels | yes | no | P2-2 |
| NEON/SVE aarch64 kernels | yes | no (scalar) | P1-2 |
| Planes two-phase search (n ≥ 32 768) | yes | no | P1-4 |
| `from_parts` / `to_bytes` / readers | yes | yes | none |
| Atomic snapshot write | `Durable` + `Fast` modes | always-durable | dir-fsync missing, fixed tmp name (P0-4) |
| File integrity | CRC-32C + nonce, v7 protocol | none (trusted bytes) | P0-3 |
| Incremental `sync()` (crash-safe) | yes (A/B commit protocol) | no | P1-1 |
| Legacy v5/v6 `convert` | yes | n/a (fresh format) | none — skip |
| Warning hook | global `fn(&str)` hook | none | use `@warn` instead (P2-5) |
| `try_*` Result forms | yes | no (typed exceptions) | intentional — keep |
| Python/framework bindings | yes | n/a | skip |
| Release/registration | crates.io, 1.0.0 | unregistered, 0.1.0 | P0-1 |

---

## 3. Proposals

### P0 — Correctness & robustness

**P0-1. Typed error for negative external ids.**
`src/id_map.jl:124` (`UInt64(ids[i])`) and `:193` leak a bare `InexactError`
for negative ids; every other input problem is a typed `TurboVecError`. Wrap the conversion
and throw a new `InvalidIdValue(id)` (or reuse `IdsCountMismatch`-style message). Small change;
users get a consistent, catchable error surface.

**P0-2. Fix the two-query kernel tail artifact.**
`src/search.jl:322-324`: the odd-tail branch of `_scan_two_avx512!` writes query B's block
result into `poa` (A's buffer) and then reads it back from `outA`. It happens to work because
`outA` aliases the same 64-float buffer, but it is a copy/paste trap. Pass `pob` and read
`outB` — behavior-identical, obviously correct.

**P0-3. Loader hardening (before anyone has files to keep compatible).**
Today `_read_index_body` (`src/io.jl:113-125`) accepts file-supplied centroids/boundaries
without verification and mutates every struct field directly; truncated files escape as raw
`EOFError` (tests literally assert `@test_throws Exception`, `test/test_io.jl:54`). Because
nothing is registered yet, this is the one chance to finalize the format cheaply:

1. Verify the embedded codebook **bitwise** against the canonical
   `codebook(bits, dim)` memo, and reject mismatches with `InvalidFileFormat`.
2. Add a footer **CRC-32C** (the Rust crate uses the same; `crc32c` is hardware-accelerated
   where available) and bump the version byte in `TV_MAGIC` (`src/io.jl:8`) from `\1` to `\2`.
   Reader: accept only v2; no legacy reader (AGENTS.md policy: single-version format).
3. Wrap all `read!` calls so `EOFError` becomes `InvalidFileFormat("truncated file")`.
4. Replace the direct field mutation with an internal `_from_loaded(...)` constructor
   that asserts the post-load invariants (lengths, `n_blocks`, codebook identity).

Impact: hostile/corrupt files can no longer inject a codebook or crash with an unrelated
error type. Effort: small; format change is painless pre-registration.

**P0-4. Atomic-write hardening.**
`src/io.jl:28` writes to a fixed `path * ".tmp"` — two concurrent writers (or a crashed
writer) collide and can clobber each other or a partially-written file; `mv(force=true)`
masks the race. Rust uses `O_EXCL` unique temp names (`<dest>.tmp.<pid>.<seq>.<rand>`).
Proposal: `tmp = mktemp(dirname(abspath(path)))` (guaranteed same-filesystem, exclusive), then
rename; afterwards **fsync the directory** on Unix so the rename itself is durable (currently
only the file is fsynced, `src/io.jl:18-25`). Impact: real crash/durability correctness.
Effort: small.

**P0-5. Tighten truncated-file test.**
After P0-3(3), change `test/test_io.jl:54` and `test/test_bytes_io.jl` from
`@test_throws Exception` to `@test_throws InvalidFileFormat`. Keeps regressions honest.

### P0 — Hot-path performance

These are cheap, measurable wins in the search/encode paths (the port currently pays
~1.6–2.4× Rust on search; these close part of the constant-factor gap):

**P0-6. Kill the per-group `prods` allocations in `build_query_lut`.**
`src/lut.jl:73` and `:94` allocate a fresh `cpn × 16` matrix **twice per byte group** inside
the `g` loop for 3/4-bit queries. At dim 768 / 4-bit (ng = 384) that is 768 tiny allocations
per query. Preallocate one scratch `cpn × 16` matrix (plus keep the 32×`ng` `fv`/`mins`
matrices already allocated) and reuse it across groups, mirroring how the 2-bit `_sub2!` path
already writes directly into `fv`. Bit-exactness is unaffected (same values, same stores).
Impact: removes ~768 allocs/query; effort: small.

**P0-7. Don't build the 256-entry `comb` table on AVX2+ machines.**
`_prepare_lut` (`src/search.jl:221-231`) always builds `comb` (`256 * ng` bytes — 98 KB at
dim 768/4-bit) but it is only read by the scalar kernel; the AVX-512 path and its AVX2 tail
fallback never touch it. Build `comb` only when `!HAS_AVX2` (constant-folded branch). Impact:
saves a 100 KB alloc+fill per query on the common path; effort: trivial.

**P0-8. Reuse the `out` buffer in the AVX2 scan.**
`_scan_blocks_avx2!` (`src/search.jl:94`) allocates a 32-float `out` per query. Hoist it into
the caller (`_search_one!`/`_search_batch!`) and pass it down, or fold it into `PreparedLut`.
Impact: minor but free; effort: trivial.

**P0-9. Cache-friendly input validation with identical error precedence.**
`_validate_input` (`src/index.jl:90-100`) and the query validation in
`src/search.jl:396-403` walk `X[i, d]` row-outer over column-major data — the inner loop
strides by `n` (hundreds of KB at 100k rows), thrashing cache. Scan column-outer (d outer),
tracking per column the first invalid row and the global minimum (row, then column), which
preserves the exact `InvalidInputValue(i, d, x)` reported today (first row-major invalid).
`first_invalid_coord` (`src/validation.jl:62`) is already flat-order and correct. Impact:
validation stops being a memory-bandwidth tax on large adds; effort: small-medium.

**P0-10. Remove the per-coordinate `keys` allocation in calibration.**
`compute_tqplus_calibration` (`src/encode.jl:110-123`) allocates an `n`-element `UInt32`
sort buffer per coordinate inside `Threads.@threads` — dim allocations per `calibrate!`.
Allocate one buffer per thread (`Vector{Vector{UInt32}}` indexed by `Threads.threadid()`)
once per call. Impact: fewer GC pauses during calibration; effort: small.

**P0-11. Extend the precompile workload.**
`src/precompile.jl` covers 2/4-bit at dim 32, plain+masked search, swap_remove, idmap
round-trips — but not: 3-bit, the lazy first-`add!` commit path, `from_parts`,
`packed_codes`/`blocked_codes`, error paths, or any other dim (other WHT radix shapes
JIT on first use). Add one 3-bit case, one dim-768 case, the lazy path, and `from_parts`.
Impact: fewer first-call stalls in production; effort: small.

### P0 — Release engineering

**P0-12. Register in General + changelog.**
Unregistered (`README.md:30-35`), version 0.1.0, no `CHANGELOG.md`/`NEWS.md`, TagBot
already configured. Before/with registration:
1. Add `CHANGELOG.md` (keep-a-changelog style) seeded with the 0.1.0 changes; from then on
   every PR updates it.
2. Register the package; switch README install instructions to `Pkg.add("TurboVec")`.
3. Document the versioning/SemVer policy (bit-exactness contracts are semver-relevant:
   any change to stored scales/codes/rotation is a breaking change).
Impact: discoverability + a real release cadence; effort: small (registration process).

### P1 — Persistence & durability

**P1-1. Incremental persistence (`sync`).**
Rust's v7 `sync()` (append units + A/B commit headers + redo ops) is the crate's largest
single feature (~1.5 kLoC + adversarial crash tests) and exists so huge corpora don't rewrite
the whole file on every add. **Recommendation: defer until there is measured demand, and if
built, build a simpler Julia-idiomatic version, not a port of v7**: an append-only journal
file (`TVJ` magic) recording `(add batch | swap_remove slot)` frames + a CRC per frame, with
periodic full-snapshot compaction (rewrite as v2 when the journal exceeds X% of the snapshot)
and crash recovery = validate + replay frames. This keeps the 80% value (no full rewrite on
add, crash-safe) at ~20% of the Rust complexity. Effort: high; timeline: post-1.0.

**P1-2. `Durability` modes. ✅ done.**
`write_index(path, idx; fast = true)` (and `write_idmap`) skips the fsyncs for cache-style
files; the rename stays atomic and a `@warn` explains the power-loss trade-off. Julia's
logging replaces the Rust warning hook (P2-5), so no hook machinery was added.

### P1 — Remaining Rust features worth porting (ranked by value)

**P1-3. Multi-query AVX2 batch kernel (two queries per code pass on AVX2).**
Batch search on non-AVX-512 machines currently runs per-query in parallel
(`src/search.jl:366-371`), while the AVX-512 path scores two queries per pass
(`_scan_two_avx512!`). An AVX2 twin of `scan_pair2` (same structure, `vpshufb` on two LUTs)
would roughly halve batch-search time on the large installed base of AVX2-only hosts, and the
existing `test/test_simd.jl` parity harness makes it low-risk. Effort: medium.

**P1-4. Two-phase "planes" search for n ≥ 32 768 — ❌ rejected (approximate).**
Rust's `planes` pass is **not lossless**: `planes_shortlist_len(k) = 12.8·k` (floor 128) is
tuned to a ~99.9% sign-plane miss target, and only `planes_rescore_len(k) = max(2k, 32)`
shortlist candidates get the exact rescore (`search.rs:4291`, `:4355`). Rust trades
result-identity for speed at ≥32 768 vectors; porting it would break this package's "same
results as the exhaustive scan" contract. Only worth revisiting as an explicit opt-in
approximate mode (e.g. `search(...; mode = :fast)`) with documented recall — a product
decision, not a port.

**P1-5. aarch64 NEON kernels.**
macOS ARM (already in CI) and future ARM servers run the scalar path today; Rust has
permute-dot (`TBL`+`SDOT`/SMMLA/i8mm) kernels. Port the permute-dot LUT kernel against the
scalar reference (the same parity harness applies). Effort: high; worthwhile once ARM hosts
are a supported deployment target.

**P1-6. Public threading/telemetry helpers.**
Rust exports `single_query_parallelizes(n)` and block-skip counters. Expose
`single_query_parallelizes(n_vectors)` (the ≥1024-block rule in `src/search.jl:244`)
and, if masking is a hot use case, a cheap `mask_skip_counter()` toggle. Small, useful for
tuning and for bindings.

### P1 — API design & polish

**P1-7. Unified 1-based indices in error messages.**
`_scale_error`/`_calibration_error` (`src/validation.jl:25,38`) report 0-based "slot/coord"
strings while `first_invalid_coord` and `InvalidInputValue` are 1-based. Error *payloads*
are messages, not a wire format, so pick the Julia convention (1-based) everywhere and
document the intentional divergence from Rust's 0-based `vector_index` payloads.

**P1-8. IdMap add-path deduplication.**
`is_addable` (`src/id_map.jl:90-98`) and `add_with_ids!` (`:122-129`) implement the same
duplicate/already-present check twice (two `Set`s per call). Factor one helper
(`_checked_uids(index, ids)` returning the converted `Vector{UInt64}` or throwing) used by
both, and have `is_addable` call it inside `try`/`catch`. Removes duplicated logic and the
per-call Set in the hot add path. Effort: small.

**P1-9. `==` for both index types.**
No `==`/`isapprox` exists; equality on `(dim, bits, n, codes, scales, calibration, ids)`
with geometry-by-construction (not field identity) is genuinely useful for tests, caches and
debugging. For `IdMapIndex` compare `slot_to_id` (slot order is observable). Effort: small.

**P1-10. `show` for `MIME"text/plain"`.**
Only compact `show` exists (`src/index.jl:353`, `src/id_map.jl:61`) — fine for REPL one-liners,
but `display` falls back to the compact form. Add the two-line plain-text form for parity with
other collection types. Cosmetic; effort: trivial.

**P1-11. Document the concurrency contract.**
`search`/`prepare` are safe to call concurrently; `add!`/`calibrate!`/`swap_remove!`/`remove!`
require exclusive access (same as Rust's `&self`/`&mut self` split). This is undocumented
in the README/guide — add a short "Thread safety" section.

### P1 — Maintainability & code structure

**P1-12. Reduce kernel argument count via a context struct.**
`quantize_scale_pack!` (12 positional args + `Val{CAL}`), `_encode_rows!` (11), and
`_scan_blocks_*!` (10) are working but hard to audit. Bundle the invariant arrays into an
immutable `EncodeCtx` / `ScanCtx` (codes, scales, centroids, boundaries, shift/scale pairs,
bits, dim, ng) constructed once per operation. Type-stable (all concrete fields), no perf
cost, and future kernels get one argument instead of a dozen. Effort: medium (mechanical).

**P1-13. Remove dead code.**
`NORM_CHAINS` (`src/encode.jl:16`, never read) and `scale_lut` (`src/lut.jl:155`, unused
anywhere). Trivial. (The `ng` argument of `byte_offset` flagged in the first draft turned
out to be used by the offset arithmetic and stays.)

**P1-14. Simplify the codebook memo lock discipline.**
`codebook` (`src/codebook.jl:174`, memo at `:165-166`) does lock→get→unlock, compute, then
lock→store (benignly racy recomputation). Use a single `lock(CODEBOOK_LOCK) do; get!(...) end`
with the computation inside — same results, obviously correct, no recompute. Trivial.

**P1-15. Small cleanliness items.**
- `_grow_codes!` manual zero loop (`src/index.jl:79-81`) → `fill!(view(...), 0x00)`.
- `serialized_len` hardcoded `27` (`src/io.jl:129`) → derive from a `TV_HEADER_LEN` constant.
- `to_bytes` (`src/io.jl:197`) → `IOBuffer(sizehint = serialized_len(index))` to avoid
  geometric doubling of large serializations.
- Parallelize `packed_codes` (`src/index.jl:406`) and `from_parts`' rebuild (`:486-499`)
  with `Threads.@threads` per vector; both are embarrassingly parallel and used by embedders.

### P1 — CI, testing, observability

**P1-16. CI additions (all currently missing from `.github/workflows/CI.yml`).**
- **linux-aarch64** job (QEMU or ARM runner) to keep the scalar fallback honest.
- **`dev/validate.jl`** as a CI job — it needs no Rust/network (frozen bytes in `dev/out/`)
  and is the strongest guard against breaking the bit-exactness contract.
- **JET** type-stability check on the test target (complements the existing Aqua run) —
  catches dynamic dispatch in the hot paths.
- **JuliaFormatter** (`.JuliaFormatter.toml` + check job) — no formatter config exists.
- Formalize the manual hardening run as a CI job:
  `Pkg.test(; julia_args=["--check-bounds=yes", "--depwarn=error"])`.

**P1-17. Benchmark harness upkeep.**
`dev/bench2.jl`/`dev/tvbench` are manual, and `dev/tvbench/target/` (build artifacts) sits
committed/ignored in the tree. Either gitignore-and-document it cleanly or add an
optional/manual benchmark CI job; do not gate merges on noisy ±20% benchmarks.

### P2 — Later / opportunistic

- **P2-1. ❌ Skipped (for now): AVX-512 VNNI/`vpermb` vector-major 4-bit kernels.** Rust's
  biggest single-query win, but the dev box has no VNNI/VBMI (verified via CPUID), so the
  kernels could be neither tested nor benchmarked locally. Revisit with access to Ice
  Lake+ hardware.
- **P2-2. ❌ Skipped: deduplicate the LLVM IR kernels** (`src/simd.jl`) via a shared template.
  Cosmetic — the IR strings are frozen and parity-tested; a generator adds risk without
  payoff.
- **P2-3. ✅ CPU feature probe hardened** (`src/simd.jl`): no public feature-detection API
  exists in Base, so the probe now falls back to the scalar path on any failure instead of
  being replaced (commit `87474c9`).
- **P2-4. ❌ Won't fix: chunked `calibrate!` rotation.** The `dim × nr` rotated matrix is
  required to compute exact per-coordinate quantiles; chunking the rotation does not lower
  the peak (the matrix is the peak), and approximate quantile sketches would break
  bit-exactness. The recommended 1000-row samples are fine as-is.
- **P2-5. ✅ Done with P1-2** via `@warn` — no hook machinery.
- **P2-6. ❌ Skipped: `@assume_effects`/`@constprop` annotations.** The JET gate is clean and
  no profile shows dynamic dispatch in the hot paths; speculative micro-tuning.
- **P2-7. ✅ docs/Project.toml** gained a `julia = "1.13"` compat entry (commit `a25fd4e`).

---

## 4. Suggested roadmap

| Phase | Contents | Exit criteria |
| --- | --- | --- |
| **A — v0.2 (pre-registration hardening)** | P0-1…P0-11, P0-12(1) | `Pkg.test()` + bounds/depwarn run green, `dev/validate.jl` green, format finalized (v2 + CRC) |
| **B — v0.3 / v1.0 release** | P0-12(2,3), P1-3, P1-7…P1-11, P1-13…P1-15, P1-16 | registered, changelog, JET/formatter/validate CI green, AVX2 two-query kernel bit-exact |
| **C — post-1.0 scale features** | P1-5 NEON (done, bit-exact); P1-1 `sync` (deferred until demand); P1-4 planes (rejected — approximate) | NEON verified on aarch64 (full suite + `dev/validate.jl`); P1-4 would need an explicit opt-in approximate mode |
| **D — opportunistic** | all P2 | — |

Phases A and B are deliberately sized so the library can ship 1.0 within a few weeks of
focused work; phase C items are the "large corpus / multi-platform production" story.

---

## 5. Deliberate non-goals (keep skipping)

These Rust features should stay unported; the reasons are design decisions, not gaps:

- **v5/v6/v7 wire-format compatibility and `convert`** — the port owns its own single-version
  Julia-native format; no legacy corpus exists (AGENTS.md policy).
- **`try_*` Result forms** — typed exceptions are the Julia idiom; the error surface is
  otherwise mirrored.
- **`add_2d`/`calibrate_2d` flat-buffer forms** — matrices carry the dim.
- **Python/framework bindings (LangChain, LlamaIndex, …)** — out of scope for a Julia package.
- **A global warning-hook mechanism** — Julia's logging (`@warn`) replaces it (P2-5).
- **Fork-safety machinery** — no equivalent threat model for Julia threads/tasks.
- **`packed_ready`/`slots_ready` duality** — the port materializes one layout; the predicates
  stay as documented parity stubs (`src/index.jl:332`, `src/id_map.jl:106`).
- **The Rust `planes` two-phase search** — approximate by construction (12.8×k shortlist,
  2×k rescore, ~99.9% miss target); would break result-identity. An opt-in `mode = :fast`
  is a possible future feature, not a port.
- **VNNI/`vpermb` kernels** — skipped until VNNI/VBMI hardware is available for testing and
  benchmarking; the dev box has none (CPUID-verified).
- **IR-template code generation and speculative `@assume_effects` tuning** — the kernels are
  frozen and parity-tested, and JET is clean; neither is worth the risk today.

---

## 6. Appendix: file map for the proposals

| File | Proposals touching it |
| --- | --- |
| `src/id_map.jl` | P0-1, P1-8 |
| `src/search.jl` | P0-2, P0-7, P0-8, P0-9, P1-3, P1-6, P1-12 |
| `src/lut.jl` | P0-6, P1-13 |
| `src/encode.jl` | P0-10, P1-13 |
| `src/index.jl` | P0-9, P1-15 |
| `src/io.jl` | P0-3, P0-4, P1-2, P1-15 |
| `src/validation.jl` | P1-7 |
| `src/codebook.jl` | P1-14 |
| `src/pack.jl` | P1-13 |
| `src/precompile.jl` | P0-11 |
| `src/simd.jl` | P2-2, P2-3 |
| `test/test_io.jl`, `test/test_bytes_io.jl` | P0-5 |
| `.github/workflows/CI.yml` | P1-16 |
| `README.md`, `docs/` | P0-12, P1-11, P2-7 |
