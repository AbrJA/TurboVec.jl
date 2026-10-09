# Julia-native persistence for TurboVec indices.
#
# Version 2 single-file format: a 27-byte header (magic, kind, bit
# width, calibration flag, dim, n), the committed codebook, optional TQ+
# calibration, per-vector scales, blocked codes, an optional id table
# (IdMapIndex), and a trailing CRC-32C over every preceding byte. Writes
# go to an exclusive temporary file in the target directory, fsynced and
# atomically renamed into place, and the directory is fsynced afterwards.

const TV_MAGIC = UInt8[0x54, 0x56, 0x45, 0x43, 0x4a, 0x4c, 0x02, 0x00]  # "TVECJL\2\0"
const TV_KIND_INDEX = 0x00
const TV_KIND_IDMAP = 0x01
const TV_HEADER_LEN = 27                     # magic 8 + kind/bits/cal 3 + dim 8 + n 8
const TV_FOOTER_LEN = 4                      # u32 CRC-32C
const TV_MAX_IMPLIED_BYTES = UInt64(1) << 40 # refuse absurd header claims

function _check_endian()
    ENDIAN_BOM == 0x04030201 ||
        throw(InvalidFileFormat("big-endian hosts are not supported"))
    nothing
end

function _fsync(io::IO)
    Sys.isunix() || return nothing
    try
        ccall(:fsync, Cint, (Cint,), fd(io))
    catch
    end
    nothing
end

function _fsync_dir(dir::AbstractString)
    Sys.isunix() || return nothing
    fd = ccall(:open, Cint, (Cstring, Cint), dir, 0)  # O_RDONLY
    fd < 0 && return nothing
    try
        ccall(:fsync, Cint, (Cint,), fd)
    catch
    finally
        ccall(:close, Cint, (Cint,), fd)
    end
    nothing
end

function _write_atomic(f::F, path::AbstractString; fast::Bool = false) where {F}
    dir = dirname(abspath(path))
    isdir(dir) || throw(ArgumentError("directory does not exist: $dir"))
    tmp, io = mktemp(dir)
    try
        f(io)
        flush(io)
        fast || _fsync(io)
        close(io)
        mv(tmp, path; force = true)
    catch
        close(io)
        rm(tmp; force = true)
        rethrow()
    end
    fast || _fsync_dir(dir)
    nothing
end

@inline function _store_u64_le!(b::AbstractVector{UInt8}, off::Int, x::UInt64)
    @inbounds for k in 0:7
        b[off + k] = UInt8((x >> (8k)) & 0xff)
    end
    nothing
end

@inline function _load_u64_le(b::AbstractVector{UInt8}, off::Int)
    x = zero(UInt64)
    @inbounds for k in 0:7
        x |= UInt64(b[off + k]) << (8k)
    end
    x
end

@inline _load_u32_le(b::AbstractVector{UInt8}, off::Int) = UInt32(b[off]) |
                                                           (UInt32(b[off + 1]) << 8) |
                                                           (UInt32(b[off + 2]) << 16) |
                                                           (UInt32(b[off + 3]) << 24)

@inline _byte_view(x::AbstractVector{UInt8}) = x
@inline _byte_view(x::AbstractArray) = reinterpret(UInt8, vec(x))

function _read_checked!(io::IO, buf)
    try
        read!(io, buf)
    catch e
        e isa EOFError && throw(InvalidFileFormat("truncated file"))
        rethrow()
    end
    buf
end

function _read_crc!(io::IO, crc::Ref{UInt32}, buf)
    _read_checked!(io, buf)
    crc[] = _crc32c_update(crc[], _byte_view(buf))
    buf
end

function _write_crc(io::IO, crc::Ref{UInt32}, x)
    write(io, x)
    crc[] = _crc32c_update(crc[], _byte_view(x))
    nothing
end

function _write_header(io::IO, crc::Ref{UInt32}, kind::UInt8, dim::Int, bits::Int,
                       n::Int, calibrated::Bool)
    hdr = Vector{UInt8}(undef, TV_HEADER_LEN)
    copyto!(hdr, 1, TV_MAGIC, 1, length(TV_MAGIC))
    hdr[9] = kind
    hdr[10] = UInt8(bits)
    hdr[11] = calibrated ? 0x01 : 0x00
    _store_u64_le!(hdr, 12, UInt64(dim))
    _store_u64_le!(hdr, 20, UInt64(n))
    _write_crc(io, crc, hdr)
    nothing
end

function _check_implied_size(kind::UInt8, bits::Int, dim::Int, n::Int,
                             calibrated::Bool)
    n == 0 && return nothing
    rows = UInt64(n)
    blocks = (rows + UInt64(BLOCK) - 1) ÷ UInt64(BLOCK)
    code_bytes = blocks * UInt64(n_byte_groups(dim, bits)) * UInt64(BLOCK)
    total = rows * 4 + code_bytes
    kind == TV_KIND_IDMAP && (total += rows * 8)
    calibrated && (total += 8 * UInt64(dim))
    total <= TV_MAX_IMPLIED_BYTES ||
        throw(InvalidFileFormat("file claims $total bytes of payload, above the $TV_MAX_IMPLIED_BYTES safety cap"))
    nothing
end

function _read_header(io::IO, crc::Ref{UInt32})
    hdr = _read_crc!(io, crc, Vector{UInt8}(undef, TV_HEADER_LEN))
    view(hdr, 1:8) == TV_MAGIC ||
        throw(InvalidFileFormat("bad magic (expected a v2 TurboVec file)"))
    kind = hdr[9]
    bits = Int(hdr[10])
    cal = hdr[11]
    dim_u = _load_u64_le(hdr, 12)
    n_u = _load_u64_le(hdr, 20)
    (kind == TV_KIND_INDEX || kind == TV_KIND_IDMAP) ||
        throw(InvalidFileFormat("unknown kind $kind"))
    (2 <= bits <= 4) || throw(InvalidFileFormat("bad bit width $bits"))
    (cal == 0x00 || cal == 0x01) || throw(InvalidFileFormat("bad calibration flag"))
    dim_u <= UInt64(MAX_DIM) || throw(InvalidFileFormat("dim $dim_u exceeds MAX_DIM"))
    dim = Int(dim_u)
    (dim == 0 || dim % 8 == 0) || throw(InvalidFileFormat("bad dim $dim"))
    n_u <= (UInt64(1) << 40) ||
        throw(InvalidFileFormat("implausible vector count $n_u"))
    n = Int(n_u)
    calibrated = cal == 0x01
    (dim == 0 && (n != 0 || calibrated)) &&
        throw(InvalidFileFormat("a lazy index cannot carry rows or a calibration"))
    _check_implied_size(kind, bits, dim, n, calibrated)
    (kind, bits, dim, n, calibrated)
end

function _write_index_body(io::IO, crc::Ref{UInt32}, index::TurboQuantIndex)
    _write_crc(io, crc, index.centroids)
    _write_crc(io, crc, index.boundaries)
    if !isempty(index.tqplus_shift)
        _write_crc(io, crc, index.tqplus_shift)
        _write_crc(io, crc, index.tqplus_scale)
    end
    _write_crc(io, crc, index.scales)
    _write_crc(io, crc, index.codes)
    nothing
end

# Internal constructor for the loader: lengths, calibration bounds and
# the canonical codebook are all established before this point.
function _loaded_index(bits::Int, dim::Int, n::Int, codes::Vector{UInt8},
                       scales::Vector{Float32}, shift::Vector{Float32},
                       scale::Vector{Float32}, centroids::Vector{Float32},
                       boundaries::Vector{Float32})
    @assert length(codes) == blocked_len(n, bits, dim)
    @assert length(scales) == n
    @assert isempty(shift) == isempty(scale)
    TurboQuantIndex(dim, bits, n, n_blocks(n), codes, scales, shift, scale,
                    Rotation(dim), boundaries, centroids)
end

function _read_index_body(io::IO, crc::Ref{UInt32}, bits::Int, dim::Int, n::Int,
                          calibrated::Bool)
    dim == 0 && return TurboQuantIndex(bits)
    n_levels = 1 << bits
    centroids = _read_crc!(io, crc, Vector{Float32}(undef, n_levels))
    boundaries = _read_crc!(io, crc, Vector{Float32}(undef, n_levels - 1))
    ref_boundaries, ref_centroids = codebook(bits, dim)
    centroids == ref_centroids ||
        throw(InvalidFileFormat("embedded centroids do not match the canonical codebook"))
    boundaries == ref_boundaries ||
        throw(InvalidFileFormat("embedded boundaries do not match the canonical codebook"))
    shift = Float32[]
    scale = Float32[]
    if calibrated
        shift = _read_crc!(io, crc, Vector{Float32}(undef, dim))
        scale = _read_crc!(io, crc, Vector{Float32}(undef, dim))
        msg = _calibration_error(shift, scale)
        msg === nothing || throw(InvalidFileFormat(msg))
    end
    scales = _read_crc!(io, crc, Vector{Float32}(undef, n))
    msg = _scale_error(scales)
    msg === nothing || throw(InvalidFileFormat(msg))
    codes = _read_crc!(io, crc, Vector{UInt8}(undef, blocked_len(n, bits, dim)))
    _loaded_index(bits, dim, n, codes, scales, shift, scale, centroids, boundaries)
end

function _write_footer(io::IO, crc::Ref{UInt32})
    write(io, htol(crc[] ⊻ _CRC32C_INIT))
    nothing
end

function _read_footer(io::IO, crc::Ref{UInt32})
    buf = _read_checked!(io, Vector{UInt8}(undef, TV_FOOTER_LEN))
    stored = _load_u32_le(buf, 1)
    stored == (crc[] ⊻ _CRC32C_INIT) ||
        throw(InvalidFileFormat("checksum mismatch (file is corrupt)"))
    nothing
end

"""Exact on-disk length of a positional index (header + body + footer)."""
function serialized_len(index::TurboQuantIndex)
    total = TV_HEADER_LEN + TV_FOOTER_LEN
    index.dim == 0 && return total
    n_levels = 1 << index.bit_width
    total += 4 * n_levels                      # centroids
    total += 4 * (n_levels - 1)                # boundaries
    isempty(index.tqplus_shift) || (total += 8 * index.dim)  # shift + scale
    total += 4 * index.n                       # per-vector scales
    total += blocked_len(index.n, index.bit_width, index.dim)
    total
end

"""Exact on-disk length of an id-map index (positional image + id table)."""
serialized_len(index::IdMapIndex) = serialized_len(index.inner) + 8 * index.inner.n

"""Serialize a [`TurboQuantIndex`](@ref) to an IO sink in the file layout."""
function write_index(io::IO, index::TurboQuantIndex)
    _check_endian()
    crc = Ref(_CRC32C_INIT)
    _write_header(io, crc, TV_KIND_INDEX, index.dim, index.bit_width, index.n,
                  !isempty(index.tqplus_shift))
    _write_index_body(io, crc, index)
    _write_footer(io, crc)
    nothing
end

"""Serialize an [`IdMapIndex`](@ref) to an IO sink in the file layout."""
function write_idmap(io::IO, index::IdMapIndex)
    _check_endian()
    crc = Ref(_CRC32C_INIT)
    _write_header(io, crc, TV_KIND_IDMAP, index.inner.dim, index.inner.bit_width,
                  index.inner.n, !isempty(index.inner.tqplus_shift))
    _write_index_body(io, crc, index.inner)
    _write_crc(io, crc, index.slot_to_id)
    _write_footer(io, crc)
    nothing
end

"""Read a [`TurboQuantIndex`](@ref) from an IO source."""
function load_index(io::IO)
    _check_endian()
    crc = Ref(_CRC32C_INIT)
    kind, bits, dim, n, calibrated = _read_header(io, crc)
    kind == TV_KIND_INDEX ||
        throw(InvalidFileFormat("stream holds an IdMapIndex, use load_idmap"))
    index = _read_index_body(io, crc, bits, dim, n, calibrated)
    _read_footer(io, crc)
    index
end

"""Read an [`IdMapIndex`](@ref) from an IO source."""
function load_idmap(io::IO)
    _check_endian()
    crc = Ref(_CRC32C_INIT)
    kind, bits, dim, n, calibrated = _read_header(io, crc)
    kind == TV_KIND_IDMAP ||
        throw(InvalidFileFormat("stream holds a positional index, use load_index"))
    index = _read_index_body(io, crc, bits, dim, n, calibrated)
    slot_to_id = _read_crc!(io, crc, Vector{UInt64}(undef, n))
    _read_footer(io, crc)
    id_to_slot = Dict{UInt64,Int}()
    sizehint!(id_to_slot, n)
    @inbounds for (slot, id) in enumerate(slot_to_id)
        haskey(id_to_slot, id) &&
            throw(InvalidFileFormat("duplicate id $id in file"))
        id_to_slot[id] = slot
    end
    IdMapIndex(index, slot_to_id, id_to_slot)
end

"""
    to_bytes(index) -> Vector{UInt8}

Serialize an index to an in-memory buffer using the same versioned
layout as [`write_index`](@ref).
"""
function to_bytes(index::TurboQuantIndex)
    (io = IOBuffer(; sizehint = serialized_len(index)); write_index(io, index); take!(io))
end
function to_bytes(index::IdMapIndex)
    (io = IOBuffer(; sizehint = serialized_len(index)); write_idmap(io, index); take!(io))
end

"""Deserialize an index produced by [`to_bytes`](@ref)."""
function from_bytes(::Type{TurboQuantIndex}, bytes::AbstractVector{UInt8})
    load_index(IOBuffer(bytes))
end
from_bytes(::Type{IdMapIndex}, bytes::AbstractVector{UInt8}) = load_idmap(IOBuffer(bytes))

"""
    write_index(path, index; fast = false)

Persist a [`TurboQuantIndex`](@ref) to a single versioned file
(fsynced, atomically renamed into place). `fast = true` skips the
fsyncs for cache-style files that can be regenerated: the rename stays
atomic, but a completed write may not survive a power loss (a warning
is emitted).
"""
function write_index(path::AbstractString, index::TurboQuantIndex;
                     fast::Bool = false)
    fast && @warn "fast write: skipping fsync; the file may not survive a power loss" path
    _write_atomic(path; fast = fast) do io
        write_index(io, index)
    end
    nothing
end

"""Load a [`TurboQuantIndex`](@ref) written by [`write_index`](@ref)."""
load_index(path::AbstractString) = open(load_index, path, "r")

"""
    write_idmap(path, index; fast = false)

Persist an [`IdMapIndex`](@ref), external ids included. See
[`write_index`](@ref) for the `fast` durability trade-off.
"""
function write_idmap(path::AbstractString, index::IdMapIndex; fast::Bool = false)
    fast && @warn "fast write: skipping fsync; the file may not survive a power loss" path
    _write_atomic(path; fast = fast) do io
        write_idmap(io, index)
    end
    nothing
end

"""Load an [`IdMapIndex`](@ref) written by [`write_idmap`](@ref)."""
load_idmap(path::AbstractString) = open(load_idmap, path, "r")
