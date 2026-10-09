# Julia-native persistence for Turbovec indices.
#
# A single versioned binary file: magic, kind, shape, committed codebook,
# optional TQ+ calibration, per-vector scales, blocked codes, and (for
# IdMapIndex) the external ids. Writes go to a temporary file that is
# fsynced and atomically renamed over the target.

const TV_MAGIC = UInt8[0x54, 0x56, 0x45, 0x43, 0x4a, 0x4c, 0x01, 0x00]  # "TVECJL\1\0"
const TV_KIND_INDEX = 0x00
const TV_KIND_IDMAP = 0x01

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

function _write_atomic(f::F, path::AbstractString) where {F}
    tmp = path * ".tmp"
    open(tmp, "w") do io
        f(io)
        flush(io)
        _fsync(io)
    end
    mv(tmp, path; force = true)
    nothing
end

function _write_header(io::IO, kind::UInt8, dim::Int, bits::Int, n::Int, calibrated::Bool)
    write(io, TV_MAGIC)
    write(io, kind)
    write(io, UInt8(bits))
    write(io, calibrated ? 0x01 : 0x00)
    write(io, htol(UInt64(dim)))
    write(io, htol(UInt64(n)))
    nothing
end

function _read_header(io::IO)
    magic = read(io, 8)
    length(magic) == 8 && magic == TV_MAGIC ||
        throw(InvalidFileFormat("bad magic"))
    kind = read(io, UInt8)
    bits = read(io, UInt8)
    cal = read(io, UInt8)
    dim = Int(ltoh(read(io, UInt64)))
    n = Int(ltoh(read(io, UInt64)))
    (kind == TV_KIND_INDEX || kind == TV_KIND_IDMAP) ||
        throw(InvalidFileFormat("unknown kind $kind"))
    (2 <= bits <= 4) || throw(InvalidFileFormat("bad bit width $bits"))
    (cal == 0x00 || cal == 0x01) || throw(InvalidFileFormat("bad calibration flag"))
    (dim == 0 || (dim % 8 == 0 && dim <= MAX_DIM)) ||
        throw(InvalidFileFormat("bad dim $dim"))
    n >= 0 || throw(InvalidFileFormat("bad vector count $n"))
    (kind, Int(bits), dim, n, cal == 0x01)
end

"""Smallest TQ+ scale at `dim` that cannot drive a divided query to overflow."""
min_tqplus_scale(dim::Int) =
    Float32(max(dim, 1)) * MAX_INPUT_MAGNITUDE / floatmax(Float32) * 10.0f0

"""Largest TQ+ shift magnitude at `dim` whose bias dot product cannot overflow."""
max_tqplus_shift(dim::Int) =
    floatmax(Float32) / (Float32(max(dim, 1)) * MAX_INPUT_MAGNITUDE) / 10.0f0

"""Largest per-vector renormalization scale that cannot by itself overflow."""
const MAX_VECTOR_SCALE = 1.0f22

function _validate_calibration(shift::AbstractVector{Float32},
                               scale::AbstractVector{Float32})
    cap = max_tqplus_shift(length(shift))
    @inbounds for (i, v) in enumerate(shift)
        if !isfinite(v) || abs(v) > cap
            throw(InvalidFileFormat(
                "invalid TQ+ shift at coord $(i - 1): $v (must be finite and |shift| <= $cap)"))
        end
    end
    floor = min_tqplus_scale(length(scale))
    @inbounds for (i, v) in enumerate(scale)
        if !isfinite(v) || v < floor
            throw(InvalidFileFormat(
                "invalid TQ+ scale at coord $(i - 1): $v (must be finite and >= $floor)"))
        end
    end
    nothing
end

function _validate_scales(scales::AbstractVector{Float32})
    @inbounds for (i, s) in enumerate(scales)
        if !isfinite(s) || s < 0.0f0 || s > MAX_VECTOR_SCALE
            throw(InvalidFileFormat(
                "invalid per-vector scale at slot $(i - 1): $s (must be finite and in [0, $MAX_VECTOR_SCALE])"))
        end
    end
    nothing
end

function _write_index_body(io::IO, index::TurboQuantIndex)
    n_levels = 1 << index.bit_width
    write(io, index.centroids)
    write(io, index.boundaries)
    if !isempty(index.tqplus_shift)
        write(io, index.tqplus_shift)
        write(io, index.tqplus_scale)
    end
    write(io, index.scales)
    write(io, index.codes)
    nothing
end

function _read_index_body(io::IO, kind::UInt8, bits::Int, dim::Int, n::Int,
                          calibrated::Bool)
    if dim == 0
        (n == 0 && !calibrated) ||
            throw(InvalidFileFormat("a lazy index cannot carry rows or a calibration"))
        index = TurboQuantIndex(bits)
        return index
    end
    n_levels = 1 << bits
    centroids = Vector{Float32}(undef, n_levels)
    read!(io, centroids)
    boundaries = Vector{Float32}(undef, n_levels - 1)
    read!(io, boundaries)
    shift = Float32[]
    scale = Float32[]
    if calibrated
        shift = Vector{Float32}(undef, dim)
        read!(io, shift)
        scale = Vector{Float32}(undef, dim)
        read!(io, scale)
    end
    scales = Vector{Float32}(undef, n)
    read!(io, scales)
    _validate_scales(scales)
    if calibrated
        _validate_calibration(shift, scale)
    end
    ncodes = dim == 0 ? 0 : blocked_len(n, bits, dim)
    codes = Vector{UInt8}(undef, ncodes)
    read!(io, codes)

    index = TurboQuantIndex(bits)
    index.dim = dim
    index.bit_width = bits
    index.n = n
    index.n_blocks = n_blocks(n)
    index.codes = codes
    index.scales = scales
    index.tqplus_shift = shift
    index.tqplus_scale = scale
    index.centroids = centroids
    index.boundaries = boundaries
    index.rotation = dim == 0 ? nothing : Rotation(dim)
    index
end

"""Exact on-disk length of a positional index (header + body)."""
function serialized_len(index::TurboQuantIndex)
    total = 27  # magic 8 + kind/bits/cal 3 + dim 8 + n 8
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
    _write_header(io, TV_KIND_INDEX, index.dim, index.bit_width, index.n,
                  !isempty(index.tqplus_shift))
    _write_index_body(io, index)
    nothing
end

"""Serialize an [`IdMapIndex`](@ref) to an IO sink in the file layout."""
function write_idmap(io::IO, index::IdMapIndex)
    _check_endian()
    _write_header(io, TV_KIND_IDMAP, index.inner.dim, index.inner.bit_width,
                  index.inner.n, !isempty(index.inner.tqplus_shift))
    _write_index_body(io, index.inner)
    write(io, index.slot_to_id)
    nothing
end

"""Read a [`TurboQuantIndex`](@ref) from an IO source."""
function load_index(io::IO)
    _check_endian()
    kind, bits, dim, n, calibrated = _read_header(io)
    kind == TV_KIND_INDEX ||
        throw(InvalidFileFormat("stream holds an IdMapIndex, use load_idmap"))
    _read_index_body(io, kind, bits, dim, n, calibrated)
end

"""Read an [`IdMapIndex`](@ref) from an IO source."""
function load_idmap(io::IO)
    _check_endian()
    kind, bits, dim, n, calibrated = _read_header(io)
    kind == TV_KIND_IDMAP ||
        throw(InvalidFileFormat("stream holds a positional index, use load_index"))
    index = _read_index_body(io, kind, bits, dim, n, calibrated)
    slot_to_id = Vector{UInt64}(undef, n)
    read!(io, slot_to_id)
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
to_bytes(index::TurboQuantIndex) = (io = IOBuffer(); write_index(io, index); take!(io))
to_bytes(index::IdMapIndex) = (io = IOBuffer(); write_idmap(io, index); take!(io))

"""Deserialize an index produced by [`to_bytes`](@ref)."""
from_bytes(::Type{TurboQuantIndex}, bytes::AbstractVector{UInt8}) =
    load_index(IOBuffer(bytes))
from_bytes(::Type{IdMapIndex}, bytes::AbstractVector{UInt8}) =
    load_idmap(IOBuffer(bytes))

"""
    write_index(path, index)

Persist a [`TurboQuantIndex`](@ref) to a single versioned file
(fsynced, atomically renamed into place).
"""
function write_index(path::AbstractString, index::TurboQuantIndex)
    _write_atomic(path) do io
        write_index(io, index)
    end
    nothing
end

"""Load a [`TurboQuantIndex`](@ref) written by [`write_index`](@ref)."""
load_index(path::AbstractString) = open(load_index, path, "r")

"""
    write_idmap(path, index)

Persist an [`IdMapIndex`](@ref), external ids included.
"""
function write_idmap(path::AbstractString, index::IdMapIndex)
    _write_atomic(path) do io
        write_idmap(io, index)
    end
    nothing
end

"""Load an [`IdMapIndex`](@ref) written by [`write_idmap`](@ref)."""
load_idmap(path::AbstractString) = open(load_idmap, path, "r")
