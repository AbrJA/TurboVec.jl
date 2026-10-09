# TurboQuantIndex: positional vector index, slots `1..n` not stable
# across `swap_remove!`. Port of `turbovec::TurboQuantIndex`.

mutable struct TurboQuantIndex
    dim::Int                    # 0 = lazy (uncommitted)
    bit_width::Int
    n::Int
    n_blocks::Int
    codes::Vector{UInt8}        # blocked layout, length == blocked_len(n, bits, dim)
    scales::Vector{Float32}
    tqplus_shift::Vector{Float32}
    tqplus_scale::Vector{Float32}
    rotation::Union{Nothing,Rotation}
    boundaries::Vector{Float32}
    centroids::Vector{Float32}
end

function _check_bit_width(bit_width::Integer)
    (2 <= bit_width <= 4) || throw(BitWidthOutOfRange(Int(bit_width)))
    Int(bit_width)
end

function _check_dim(dim::Integer)
    dim == 0 && throw(DimNotPositiveMultipleOf8(0))
    dim % 8 != 0 && throw(DimNotPositiveMultipleOf8(Int(dim)))
    dim > MAX_DIM && throw(DimTooLarge(Int(dim), MAX_DIM))
    Int(dim)
end

function _empty_index(dim::Int, bit_width::Int, boundaries::Vector{Float32},
                      centroids::Vector{Float32}, rotation::Union{Nothing,Rotation})
    TurboQuantIndex(dim, bit_width, 0, 0, UInt8[], Float32[], Float32[], Float32[],
                    rotation, boundaries, centroids)
end

"""
    TurboQuantIndex(dim, bit_width)

Construct an empty index with a known dimensionality. `bit_width` must
be 2, 3 or 4; `dim` must be a positive multiple of 8 (and <= 16384).
"""
function TurboQuantIndex(dim::Integer, bit_width::Integer)
    bw = _check_bit_width(bit_width)
    d = _check_dim(dim)
    boundaries, centroids = codebook(bw, d)
    _empty_index(d, bw, boundaries, centroids, Rotation(d))
end

"""
    TurboQuantIndex(bit_width)

Construct an empty index without committing to a dimensionality; the
dim is inferred and locked on the first `add!`.
"""
function TurboQuantIndex(bit_width::Integer)
    bw = _check_bit_width(bit_width)
    _empty_index(0, bw, Float32[], Float32[], nothing)
end

Base.length(index::TurboQuantIndex) = index.n
Base.isempty(index::TurboQuantIndex) = index.n == 0

"""The committed dimensionality, or `nothing` for a lazy index."""
dim_opt(index::TurboQuantIndex) = index.dim == 0 ? nothing : index.dim

"""`true` when a TQ+ per-coordinate calibration is committed."""
calibration_state(index::TurboQuantIndex) =
    isempty(index.tqplus_shift) ? :uncalibrated : :calibrated

"""Grow the blocked code buffer to cover `n` vectors, zeroing new bytes."""
function _grow_codes!(index::TurboQuantIndex, n::Int)
    need = blocked_len(n, index.bit_width, index.dim)
    old = length(index.codes)
    if need > old
        resize!(index.codes, need)
        @inbounds for i in (old + 1):need
            index.codes[i] = 0x00
        end
    elseif need < old
        resize!(index.codes, need)
    end
    index.n_blocks = n_blocks(n)
    nothing
end

"""Scan for the first non-finite or over-magnitude coordinate."""
function _validate_input(X::AbstractMatrix{Float32}, dim::Int)
    @inbounds for i in 1:size(X, 1)
        for d in 1:dim
            x = X[i, d]
            if !(abs(x) < MAX_INPUT_MAGNITUDE)
                throw(InvalidInputValue(i, d, x))
            end
        end
    end
    nothing
end

function _commit_geometry!(index::TurboQuantIndex, dim::Int)
    if index.dim == 0
        d = _check_dim(dim)
        boundaries, centroids = codebook(index.bit_width, d)
        index.dim = d
        index.boundaries = boundaries
        index.centroids = centroids
        index.rotation = Rotation(d)
    elseif index.dim != dim
        throw(DimMismatch(index.dim, dim))
    end
    nothing
end

"""
    add!(index, vectors)

Add a batch of `n × dim` `Float32` vectors. An empty batch is a no-op.
"""
function add!(index::TurboQuantIndex, X::AbstractMatrix{Float32})
    dim = size(X, 2)
    dim == 0 && throw(ZeroDim())
    n = size(X, 1)
    if n == 0
        # An empty batch is a no-op: it neither commits a lazy dim nor
        # changes a committed one, but a mismatched committed dim is
        # still an error.
        index.dim != 0 && index.dim != dim && throw(DimMismatch(index.dim, dim))
        return index
    end
    _commit_geometry!(index, dim)
    _validate_input(X, dim)

    old_n = index.n
    _grow_codes!(index, old_n + n)

    calibrated = !isempty(index.tqplus_shift)
    shift, scale_tq = if calibrated
        index.tqplus_shift, index.tqplus_scale
    else
        zeros(Float32, dim), ones(Float32, dim)
    end
    inv_scale_tq = 1.0f0 ./ scale_tq
    resize!(index.scales, old_n + n)

    # Rotate and quantize per row so the dim x n rotated matrix never
    # exists: each worker keeps its own row-sized buffers and feeds the
    # rotated row straight into the quantizer. Same per-row op order, so
    # the codes and scales are bit-identical to the two-pass version.
    rot = index.rotation
    bits = index.bit_width
    if calibrated
        _encode_rows!(index, X, old_n, n, dim, rot, shift, scale_tq,
                      inv_scale_tq, bits, Val(true))
    else
        _encode_rows!(index, X, old_n, n, dim, rot, shift, scale_tq,
                      inv_scale_tq, bits, Val(false))
    end
    index.n = old_n + n
    index
end

function _encode_rows!(index::TurboQuantIndex, X::AbstractMatrix{Float32},
                       old_n::Int, n::Int, dim::Int, rot::Rotation,
                       shift::Vector{Float32}, scale_tq::Vector{Float32},
                       inv_scale_tq::Vector{Float32}, bits::Int, ::Val{CAL}) where {CAL}
    @sync for (lo, hi) in _parallel_ranges(n)
        Threads.@spawn begin
            src = Vector{Float32}(undef, dim)
            scratch = Vector{Float32}(undef, dim)
            dst = Vector{Float32}(undef, dim)
            for i in lo:hi
                @inbounds for d in 1:dim
                    src[d] = X[i, d]
                end
                nrm = simd_norm(src, dim)
                inv = nrm > MIN_INPUT_NORM ? 1.0f0 / nrm : 0.0f0
                apply_scaled_into!(rot, src, inv, dst, scratch)
                index.scales[old_n + i] = quantize_scale_pack!(
                    index.codes, old_n + i - 1, dst, shift, scale_tq,
                    inv_scale_tq, index.centroids, index.boundaries, bits,
                    dim, nrm, Val(CAL))
            end
        end
    end
    nothing
end

add!(index::TurboQuantIndex, X::AbstractMatrix{<:Real}) = add!(index, Float32.(X))

"""
    calibrate!(index, sample)

Fit a TQ+ per-coordinate calibration from a `rows × dim` sample. On a
populated index every stored row is re-encoded under the new pair.
"""
function calibrate!(index::TurboQuantIndex, sample::AbstractMatrix{Float32})
    dim = size(sample, 2)
    if index.dim == 0
        dim == 0 && throw(ZeroDim())
        _check_dim(dim)
    else
        index.dim == dim || throw(DimMismatch(index.dim, dim))
    end
    nr = size(sample, 1)
    nr >= MIN_CALIBRATION_ROWS || throw(EmptyCalibrationSample(nr, MIN_CALIBRATION_ROWS))
    _validate_input(sample, dim)

    had_dim = index.dim != 0
    if !had_dim
        _commit_geometry!(index, dim)
    end
    rotation = index.rotation
    rotated = Matrix{Float32}(undef, dim, nr)
    norms = Vector{Float32}(undef, nr)
    rotate_batch!(rotated, norms, sample, rotation)
    shift, scale_tq = compute_tqplus_calibration(rotated, nr, dim, index.centroids)

    if all(iszero, shift) && all(==(1.0f0), scale_tq)
        if !had_dim
            index.dim = 0
            index.boundaries = Float32[]
            index.centroids = Float32[]
            index.rotation = nothing
        end
        throw(DegenerateSample())
    end

    if index.n > 0
        _reencode_stored_rows!(index, shift, scale_tq)
    end
    index.tqplus_shift = shift
    index.tqplus_scale = scale_tq
    index
end

calibrate!(index::TurboQuantIndex, sample::AbstractMatrix{<:Real}) =
    calibrate!(index, Float32.(sample))

"""Re-encode every stored row under `new_shift`/`new_scale`."""
function _reencode_stored_rows!(index::TurboQuantIndex, new_shift::Vector{Float32},
                                new_scale::Vector{Float32})
    dim = index.dim
    bits = index.bit_width
    n = index.n
    calibrated_old = !isempty(index.tqplus_shift)
    old_shift = calibrated_old ? index.tqplus_shift : nothing
    old_inv = calibrated_old ? (1.0f0 ./ index.tqplus_scale) : nothing
    inv_new = 1.0f0 ./ new_scale
    @sync for (lo, hi) in _parallel_ranges(n)
        Threads.@spawn begin
            codes = Vector{UInt8}(undef, dim)
            recon = Vector{Float32}(undef, dim)
            for i in lo:hi
                extract_codes_lane!(codes, index.codes, i - 1, dim, bits)
                sumsq = 0.0
                @inbounds for d in 1:dim
                    code = Int(codes[d])
                    x = calibrated_old ?
                        (index.centroids[code + 1] * old_inv[d] - old_shift[d]) :
                        (index.centroids[code + 1] * 1.0f0 - 0.0f0)
                    recon[d] = x
                    xf = Float64(x)
                    sumsq += xf * xf
                end
                norm = Float32(Float64(index.scales[i]) * sumsq)
                index.scales[i] = quantize_scale_pack!(index.codes, i - 1, recon,
                                                       new_shift, new_scale, inv_new,
                                                       index.centroids, index.boundaries,
                                                       bits, dim, norm, Val(true))
            end
        end
    end
    nothing
end

"""Committed dimensionality; `0` for a lazy, uncommitted index."""
dim(index::TurboQuantIndex) = index.dim

"""Bits per coordinate (2, 3 or 4)."""
bit_width(index::TurboQuantIndex) = index.bit_width

"""Per-vector length-renormalization scales, one per slot."""
scales(index::TurboQuantIndex) = index.scales

"""Committed TQ+ shift, empty when uncalibrated."""
tqplus_shift(index::TurboQuantIndex) = index.tqplus_shift

"""Committed TQ+ scale, empty when uncalibrated."""
tqplus_scale(index::TurboQuantIndex) = index.tqplus_scale

"""No-op kept for API parity: this port has no search caches to warm."""
prepare(index::TurboQuantIndex) = index

"""
    packed_codes(index) -> Vector{UInt8}

The canonical bit-plane encoding of every row: `bits * (dim ÷ 8)` bytes
per vector, plane `p`'s byte `c` holding bit `p` of the eight codes in
coordinate chunk `c` (most significant coordinate in bit 7). This is
the layout [`from_parts`](@ref) accepts.
"""
function packed_codes(index::TurboQuantIndex)
    dim = index.dim
    bits = index.bit_width
    (dim == 0 || index.n == 0) && return UInt8[]
    bpp = dim ÷ 8
    bytes_per_row = bits * bpp
    out = zeros(UInt8, bytes_per_row * index.n)
    codes = Vector{UInt8}(undef, dim)
    for v in 1:index.n
        extract_codes_lane!(codes, index.codes, v - 1, dim, bits)
        base = (v - 1) * bytes_per_row
        @inbounds for c in 0:(bpp - 1)
            for k in 0:7
                code = codes[8c + k + 1]
                for p in 0:(bits - 1)
                    out[base + p * bpp + c + 1] |= ((code >> p) & 0x1) << (7 - k)
                end
            end
        end
    end
    out
end

"""
    from_parts(dim, bit_width, n, packed_codes, scales, tqplus_shift, tqplus_scale)
        -> TurboQuantIndex

Rebuild an index from its raw parts, validating everything before
constructing: bit width, dim, part lengths, per-vector scale bounds, and
the TQ+ calibration bounds. An identity `(0, 1)` calibration pair is
canonicalized to the empty (uncalibrated) representation.
"""
function from_parts(dim::Integer, bit_width::Integer, n::Integer,
                    packed::AbstractVector{UInt8},
                    scales::AbstractVector{Float32},
                    tqplus_shift::AbstractVector{Float32} = Float32[],
                    tqplus_scale::AbstractVector{Float32} = Float32[])
    bits = _check_bit_width(bit_width)
    d = Int(dim)
    nn = Int(n)
    nn < 0 && throw(InvalidParts("n_vectors must be nonnegative, got $nn"))
    if d == 0
        if nn != 0 || !isempty(packed) || !isempty(scales) ||
           !isempty(tqplus_shift) || !isempty(tqplus_scale)
            throw(InvalidParts("a lazy (dim = 0) index must carry no rows, codes or calibration"))
        end
        return TurboQuantIndex(bits)
    end
    _check_dim(d)
    bytes_per_row = bits * (d ÷ 8)
    if length(packed) != nn * bytes_per_row
        throw(InvalidParts(
            "packed codes length $(length(packed)) does not match n * bytes_per_row = $(nn * bytes_per_row)"))
    end
    length(scales) == nn ||
        throw(InvalidParts("scales length $(length(scales)) does not match n_vectors $nn"))
    has_shift = !isempty(tqplus_shift)
    has_scale = !isempty(tqplus_scale)
    has_shift != has_scale &&
        throw(InvalidParts("tqplus_shift and tqplus_scale must both be empty or both be length dim"))
    if has_shift
        length(tqplus_shift) == d || throw(InvalidParts("tqplus_shift length must equal dim"))
        length(tqplus_scale) == d || throw(InvalidParts("tqplus_scale length must equal dim"))
        _validate_calibration(Vector{Float32}(tqplus_shift), Vector{Float32}(tqplus_scale))
    end
    _validate_scales(scales)

    shift = has_shift ? Vector{Float32}(tqplus_shift) : Float32[]
    sc = has_scale ? Vector{Float32}(tqplus_scale) : Float32[]
    if has_shift && all(iszero, shift) && all(==(1.0f0), sc)
        shift = Float32[]
        sc = Float32[]
    end

    boundaries, centroids = codebook(bits, d)
    codes = zeros(UInt8, blocked_len(nn, bits, d))
    buf = Vector{UInt8}(undef, d)
    bpp = d ÷ 8
    for v in 1:nn
        base = (v - 1) * bytes_per_row
        @inbounds for c in 0:(bpp - 1)
            for k in 0:7
                code = 0x00
                for p in 0:(bits - 1)
                    bit = (packed[base + p * bpp + c + 1] >> (7 - k)) & 0x1
                    code |= bit << p
                end
                buf[8c + k + 1] = code
            end
        end
        write_codes_lane!(codes, v - 1, buf, d, bits)
    end

    TurboQuantIndex(d, bits, nn, n_blocks(nn), codes, Vector{Float32}(scales),
                    shift, sc, Rotation(d), boundaries, centroids)
end

"""
    swap_remove!(index, slot) -> Int

Remove the vector at 1-based `slot` in O(1) by moving the last vector
into it. Returns the old index of the moved vector. Order is not
preserved.
"""
function swap_remove!(index::TurboQuantIndex, slot::Integer)
    (1 <= slot <= index.n) ||
        throw(BoundsError(index, slot))
    last = index.n
    if slot != last
        move_lane!(index.codes, n_byte_groups(index.dim, index.bit_width),
                   last - 1, slot - 1)
        index.scales[slot] = index.scales[last]
    end
    pop!(index.scales)
    index.n = last - 1
    index.n_blocks = n_blocks(index.n)
    resize!(index.codes, blocked_len(index.n, index.bit_width, index.dim))
    last
end
