# Changelog

All notable changes to TurboVec.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Because stored codes, scales,
calibration and rotation are part of the file format, any change to them is a
breaking change.

## [Unreleased]

### Added
- `InvalidIdValue`: typed error for external ids that are negative or not
  representable as `UInt64`. Membership (`in`, `contains_id`), `is_addable` and
  `remove!` treat such ids as absent (`false`) instead of leaking
  `InexactError`.
- File-format integrity: a trailing CRC-32C checksum over the whole image and a
  bitwise check of the embedded codebook against the canonical codebook.
- Tests for truncated, corrupted, duplicate-id and non-canonical-codebook
  files, for validation error precedence, and for out-of-domain ids.

### Changed
- **Breaking:** the persistence format is now v2 (`TVECJL\2\0` with a CRC-32C
  footer). v1 files are not readable (single-version format policy). The loader
  refuses `dim > MAX_DIM`, implausible vector counts, and oversized implied
  payloads before allocating.
- Atomic writes create an exclusive temporary file in the destination
  directory (no fixed `<path>.tmp` collisions), fsync the directory after the
  rename, and remove the temporary file on failure. New files are created with
  owner-only permissions (`0600`); writing into a missing directory raises
  `ArgumentError`.
- Search hot path: `build_query_lut` no longer allocates per byte group, the
  scalar fallback's 256-entry table is only built on CPUs without AVX2, and
  scan kernels take caller-provided output buffers.
- Input/query validation scans column-major while reporting the same first
  invalid coordinate; `calibrate!` reuses one sort buffer per worker task.
- `to_bytes` preallocates the exact serialized size.
- Precompile workload covers 3-bit, lazy, `from_parts`, and 768-dim paths.

### Fixed
- Truncated files now raise `InvalidFileFormat` instead of a bare `EOFError`.
- The AVX-512 two-query kernel's odd-tail branch wrote query B's block through
  query A's output buffer (results were correct, but the aliasing was a
  copy/paste trap).

## [0.1.0]

Initial port of the Rust `turbovec` crate: bit-exact encode, TQ+ calibration,
blocked SIMD scan with AVX2/AVX-512 kernels, `IdMapIndex`, and a Julia-native
persistence format.
