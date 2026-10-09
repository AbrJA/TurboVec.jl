# TurboVec.jl

A Julia port of [`turbovec`](https://github.com/RyanCodrai/turbovec), a
vector index built on Google Research's
[TurboQuant](https://arxiv.org/abs/2504.19874) algorithm: vectors are
compressed to 2–4 bits per coordinate by a deterministic random rotation
and scored directly against a per-query lookup table — no training
phase, no decompression, online ingest, multithreaded search.

```julia
using TurboVec

index = TurboQuantIndex(1536, 4)     # dim, bits per coordinate
add!(index, X)                       # X::Matrix{Float32}, n × 1536
scores, ids = search(index, Q, 10)   # Q::Matrix{Float32}, nq × 1536
```

The full reference — algorithm, API, persistence, filtering, thread and
precision notes, performance vs the Rust implementation, and the
development tooling — lives in the
[README](https://github.com/AbrJA/TurboVec.jl#readme).

## API

```@autodocs
Modules = [TurboVec]
```
