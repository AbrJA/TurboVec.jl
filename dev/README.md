# Developer harnesses

These scripts are development tools, not part of the package. They are
how the fidelity and performance numbers in the docs are produced.

## Rust cross-validation

* `tvref/` — small Rust program pinned to crates.io `turbovec = "=1.0.0"`.
  `(cd tvref && cargo run --release)` regenerates `out/` from the
  published crate. Change the pin only deliberately.
* `validate.jl` — `julia --project=. dev/validate.jl` checks encode
  bytes bit-exactly and search results to ~1e-6 against the frozen
  `out/` corpus. Needs no Rust or network; CI runs it on every push.
* `out/` — committed reference bytes and search expectations (~444 KB).

## Benchmarks

* `bench2.jl` — Julia harness: `julia --project=. -t N dev/bench2.jl`
  runs the standard 100k × 768 and 50k × 1536 configurations, or
  `julia --project=. -t N dev/bench2.jl <dim> <bits> <n>` for one.
* `tvbench/` — the Rust counterpart with identical corpora. Run it with
  `RAYON_NUM_THREADS=N` to compare against `julia -t N`. Build artifacts
  under `dev/*/target/` are gitignored.
* Benchmarks are not a CI gate (run variance is ±20%); use the manual
  `benchmark` job in the CI workflow (`workflow_dispatch`) or run the
  scripts locally.

## Profiling

`profiling/` holds the kernel and encode experiments behind the
optimization log. Rejected ideas are documented in the README's
"Optimization log"; do not retry them without new evidence.
