# ⚡ TurboVec.jl

A Julia vector index built on Google Research's
[TurboQuant](https://arxiv.org/abs/2504.19874) algorithm: compress embeddings to **2–4 bits
per coordinate** and search the compressed form directly — no training phase, no index
rebuilds, online ingest.

```julia
using TurboVec

X = randn(Float32, 10_000, 768)     # one row per item
Q = randn(Float32, 3, 768)          # queries

index = TurboQuantIndex(768, 4)     # dim, bits per coordinate
add!(index, X)
scores, slots = search(index, Q, 5)
```

Stable ids and filtered search:

```julia
index = IdMapIndex(768, 4)
add_with_ids!(index, X, ids)
scores, ids = search(index, Q, 10; allowlist = allowed_ids)
remove!(index, 1002)                # O(1); other ids stay valid
```

## Where to go next

* [Guide](guide.md) — how the compression works, filtering, threading, persistence,
  precision notes.
* [API reference](reference.md) — every exported function and the Julia interface
  conventions.
* [Validation & development](validation.md) — bit-exactness vs the Rust crate, benchmarks,
  the optimization log and the dev harnesses.
