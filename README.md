# ⚡ TurboVec.jl

**Search millions of embeddings in pure Julia — 8–16× smaller, no training step, no rebuilds.**

[![CI](https://github.com/AbrJA/TurboVec.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/AbrJA/TurboVec.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Julia 1.13+](https://img.shields.io/badge/Julia-1.13%2B-9558B2.svg)](https://julialang.org)

TurboVec.jl is a Julia port of [`turbovec`](https://github.com/RyanCodrai/turbovec), built on
Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm. It compresses
float32 embeddings to **2–4 bits per coordinate** and searches the compressed form directly —
no k-means training, no index rebuilds as your corpus grows.

## ✨ Why you might want this

| | |
| --- | --- |
| 🗜️ **8–16× smaller** | 10M vectors at 768 dims: ~31 GB as float32 → **~4 GB at 4-bit**, ~2 GB at 2-bit |
| ⚡ **Fast** | SIMD AVX2/AVX-512 scan with multi-threaded search: 100k × 768 in ~0.2–0.35 ms per query |
| 🧠 **No training** | `add!` and the vectors are immediately searchable; ingest grows online |
| 🎯 **Optional calibration** | `calibrate!` fits *your* data for a recall bump on skewed embeddings |
| 🔎 **Filtered search** | id allowlists / slot masks are honoured inside the SIMD scan |
| 🆔 **Stable ids** | `IdMapIndex` maps results back to your own `UInt64` ids, with O(1) removal |
| 💾 **Durable** | atomic, fsynced snapshots — reload and get identical results |
| 🔒 **Local** | pure Julia, no service, nothing leaves your machine |
| ✅ **Trustworthy** | bit-identical to the Rust reference for encode; 1751 tests; Aqua-clean |

## 🚀 Quick start

Not in the General registry yet — install straight from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/AbrJA/TurboVec.jl")
```

```julia
using TurboVec

X = randn(Float32, 10_000, 768)       # one row per item
Q = randn(Float32, 3, 768)            # queries

index = TurboQuantIndex(768, 4)       # dim, bits per coordinate (2, 3 or 4)
add!(index, X)                        # online: call add! as new vectors arrive

scores, slots = search(index, Q, 5)   # 3×5 scores and matching row numbers
```

Use **4 bits** for best recall, **2 bits** when memory matters most, **3** in between.

## 🔍 A realistic example: RAG over your notes

```julia
using TurboVec

# embeddings is n×1536 Float32; doc_ids identifies each document.
index = IdMapIndex(1536, 4)
add_with_ids!(index, embeddings, doc_ids)

scores, ids = search(index, query_embeddings, 5)   # your ids, not row numbers

remove!(index, 1002)        # O(1); every other id stays valid
1002 in index               # false
```

## 🎛️ Filter at search time

Narrow candidates with SQL / BM25 / ACLs, then rerank densely — the filter is applied inside
the kernel, so you get exactly `min(k, n_allowed)` results with no over-fetching:

```julia
allowed = UInt64[1005, 1011, 1042, 1099]
scores, ids = search(index, query_embeddings, 10; allowlist = allowed)
```

## 💾 Save and load

```julia
write_idmap("notes.tvim", index)         # fsynced, atomically renamed
restored = load_idmap("notes.tvim")      # scores identical to the in-memory index
```

Need bytes instead of files (a database column, a cache)? Use `to_bytes` / `from_bytes`.

## 📊 Performance at a glance

100k vectors, 768 dims, `k = 64`, 16 vCPU host; run with `julia -t auto` to use threads.

| | 4-bit | 2-bit |
| --- | --- | --- |
| index size | 38 MB | 19 MB |
| build (`add!`) | ~0.35 s | ~0.29 s |
| search (batch) | ~0.34 ms/query | ~0.21 ms/query |

That's within ~1.6–2.4× of the hand-tuned Rust kernels, with encode on par at 16 threads;
single queries cost ~3.5 ms since they cannot share code reads across queries.
Full tables, methodology and the bit-exactness story: [Validation & development](docs/src/validation.md).

## 🧠 How it works (30 seconds)

1. **Strip the length** from each vector and rotate the unit direction with a fixed
   block-Hadamard transform — after rotation every coordinate follows the *same* known
   distribution, for any data. That's why **no training is needed**.
2. **Quantize** each coordinate with a precomputed Lloyd–Max codebook (4 or 16 levels) and
   bit-pack the codes into a search-friendly layout.
3. **Search** rotates the query once and scores the packed codes against a small lookup table,
   reading each stored byte only once per query pair.

Optionally, `calibrate!(index, sample)` fits a tiny per-coordinate correction to *your* data.

## 📚 Learn more

- [Guide](docs/src/guide.md) — algorithm, filtering, threading, persistence, precision notes.
- [API reference](docs/src/reference.md) — every exported function.
- [Validation & development](docs/src/validation.md) — fidelity vs Rust, benchmarks, dev tools.

Build the docs locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path="."))'
julia --project=docs docs/make.jl
```

## 🧪 Testing

```bash
julia --project=. -t auto -e 'using Pkg; Pkg.test()'
```

## 📄 License

MIT. A derived work of [`turbovec`](https://github.com/RyanCodrai/turbovec) by Ryan Codrai —
see [`LICENSE`](LICENSE).
