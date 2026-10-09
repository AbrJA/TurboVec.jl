# Encode vectors: normalize, rotate, calibrate, quantize, bit-pack,
# scale. Port of `turbovec::encode`.
#
# For each vector `v` with rotated unit form `u` and reconstructed
# centroid vector `x_hat`, the stored scale is `||v|| / <u, x_hat>` --
# the RaBitQ-style length-renormalization correction. Applying it at the
# end of scoring yields an unbiased estimator of `<v, q>`.
#
# # TQ+ per-coordinate calibration
#
# `u_calibrated[d] = (u_rot[d] + shift[d]) * scale_tq[d]`; quantization
# runs on the calibrated values, and search applies the inverse on the
# query side plus a `-<q_rot, shift>` bias correction. An uncalibrated
# index is arithmetically the identity pair.

const MAX_INPUT_MAGNITUDE = 1.0f16

"""L2 norm at or below which a vector has no representable direction.

Such a vector is stored with scale 0 (it scores 0 against every query),
not rejected.
"""
const MIN_INPUT_NORM = 1.0f-10

const DEGENERATE_INNER_EPS = 0.1

"""Calibration sample size to aim for (advisory, not enforced)."""
const RECOMMENDED_CALIBRATION_ROWS = 1000

"""Fewest rows a calibration fit can structurally use."""
const MIN_CALIBRATION_ROWS = 2

"""Fixed-order 8-chain Euclidean norm of one row (matches Rust `simd_norm`)."""
@inline function simd_norm(row::AbstractVector{Float32}, n::Int)
    c0 = 0.0f0; c1 = 0.0f0; c2 = 0.0f0; c3 = 0.0f0
    c4 = 0.0f0; c5 = 0.0f0; c6 = 0.0f0; c7 = 0.0f0
    i = 1
    @inbounds while i + 7 <= n
        x0 = row[i]; x1 = row[i + 1]; x2 = row[i + 2]; x3 = row[i + 3]
        x4 = row[i + 4]; x5 = row[i + 5]; x6 = row[i + 6]; x7 = row[i + 7]
        c0 += x0 * x0; c1 += x1 * x1; c2 += x2 * x2; c3 += x3 * x3
        c4 += x4 * x4; c5 += x5 * x5; c6 += x6 * x6; c7 += x7 * x7
        i += 8
    end
    @inbounds while i <= n
        j = (i - 1) % 8
        x = row[i]
        if j == 0
            c0 += x * x
        elseif j == 1
            c1 += x * x
        elseif j == 2
            c2 += x * x
        elseif j == 3
            c3 += x * x
        elseif j == 4
            c4 += x * x
        elseif j == 5
            c5 += x * x
        elseif j == 6
            c6 += x * x
        else
            c7 += x * x
        end
        i += 1
    end
    sqrt(((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7)))
end

@inline function f32_sort_key(x::Float32)
    b = reinterpret(UInt32, x)
    b ⊻ (reinterpret(UInt32, reinterpret(Int32, b) >> 31) | 0x80000000)
end

@inline function f32_from_sort_key(k::UInt32)
    mask = (k & 0x80000000) != 0 ? 0x80000000 : 0xffffffff
    reinterpret(Float32, k ⊻ mask)
end

"""
    compute_tqplus_calibration(rotated, n, dim, centroids) -> (shift, scale)

Fit the per-coordinate `(shift, scale)` pair from `n` rotated rows
(`rotated` is `dim × n`). Maps each coordinate's empirical quantiles at
the codebook's anchor levels onto the canonical Beta marginal's edges.
"""
function compute_tqplus_calibration(rotated::AbstractMatrix{Float32}, n::Int,
                                    dim::Int, centroids::Vector{Float32})
    n >= MIN_CALIBRATION_ROWS ||
        throw(ArgumentError("calibration fit needs at least $MIN_CALIBRATION_ROWS rows"))
    a = (Float64(dim) - 1.0) / 2.0
    c_outer = 0.0f0
    @inbounds for c in centroids
        ac = abs(c)
        ac > c_outer && (c_outer = ac)
    end
    p_hi = beta_inc(a, a, (Float64(c_outer) + 1.0) / 2.0)
    p_lo = 1.0 - p_hi
    qc_lo = -c_outer
    qc_hi = c_outer
    qc_span = qc_hi - qc_lo

    lo_idx = Int(floor(Float64(n) * p_lo))
    hi_idx = min(Int(floor(Float64(n) * p_hi)), n - 1)
    hi_idx = max(hi_idx, lo_idx + 1)

    shift = zeros(Float32, dim)
    scale = ones(Float32, dim)
    # One sort buffer per worker task, reused across the coordinates in
    # that task's range: the old code allocated an n-element buffer per
    # coordinate, which made `calibrate!` allocation-bound on the GC.
    @sync for (lo, hi) in _parallel_ranges(dim)
        Threads.@spawn begin
            keys = Vector{UInt32}(undef, n)
            for d in lo:hi
                @inbounds for i in 1:n
                    keys[i] = f32_sort_key(rotated[d, i])
                end
                qe_lo = f32_from_sort_key(partialsort!(keys, lo_idx + 1))
                qe_hi = f32_from_sort_key(partialsort!(keys, hi_idx + 1))
                span = qe_hi - qe_lo
                if span > 1.0f-6
                    sc = qc_span / span
                    shift[d] = qc_lo / sc - qe_lo
                    scale[d] = sc
                end
            end
        end
    end
    (shift, scale)
end

"""Contiguous index ranges, one per worker, for row-parallel kernels."""
function _parallel_ranges(n::Int)
    n <= 0 && return Tuple{Int,Int}[]
    nt = Threads.nthreads()
    nt <= 1 && return [(1, n)]
    chunk = max(cld(n, nt), 1)
    ranges = Tuple{Int,Int}[]
    lo = 1
    while lo <= n
        hi = min(lo + chunk - 1, n)
        push!(ranges, (lo, hi))
        lo = hi + 1
    end
    ranges
end

"""
    rotate_batch!(rotated, norms, X, rotation)

Rotate the rows of the `n × dim` matrix `X` (normalizing each by
`1/||row||`) into the `dim × n` output `rotated`, returning per-row
norms. `rotated[:, i]` is row `i`'s rotated unit vector.

Each worker owns its scratch buffers, so the kernel is race-free.
"""
function rotate_batch!(rotated::AbstractMatrix{Float32}, norms::Vector{Float32},
                       X::AbstractMatrix{Float32}, rotation::Rotation)
    n, dim = size(X)
    @sync for (lo, hi) in _parallel_ranges(n)
        Threads.@spawn begin
            src = Vector{Float32}(undef, dim)
            scratch = Vector{Float32}(undef, dim)
            for i in lo:hi
                @inbounds for d in 1:dim
                    src[d] = X[i, d]
                end
                nrm = simd_norm(src, dim)
                norms[i] = nrm
                inv = nrm > MIN_INPUT_NORM ? 1.0f0 / nrm : 0.0f0
                apply_scaled_into!(rotation, src, inv, view(rotated, :, i), scratch)
            end
        end
    end
    nothing
end

"""
    quantize_scale_pack!(blocked, lane, rot_orig, shift, scale_tq,
                         inv_scale_tq, centroids, boundaries, bits, dim,
                         norm) -> Float32

Quantize one rotated unit row under the calibration `(shift, scale_tq)`,
write its code bytes into the blocked layout at `lane`, and return the
stored length-renormalization scale. An uncalibrated index passes the
identity pair (`shift == 0`, `scale_tq == 1`), which is arithmetically
exact.
"""
function quantize_scale_pack!(blocked::AbstractVector{UInt8}, lane::Int,
                              rot_orig::AbstractVector{Float32},
                              shift::Vector{Float32},
                              scale_tq::Vector{Float32},
                              inv_scale_tq::Vector{Float32},
                              centroids::Vector{Float32}, boundaries::Vector{Float32},
                              bits::Int, dim::Int, norm::Float32,
                              ::Val{CAL}) where {CAL}
    ng = n_byte_groups(dim, bits)
    b = lane ÷ BLOCK
    l = lane % BLOCK
    chunks = dim ÷ 8
    limits = (1 << bits) - 1

    a0 = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0
    @inbounds for c in 0:(chunks - 1)
        offset = 8c
        if bits == 2
            b0 = 0x00; b1 = 0x00
            for k in 0:7
                j = offset + k + 1
                local calib::Float32
                if CAL
                    calib = (rot_orig[j] + shift[j]) * scale_tq[j]
                else
                    calib = rot_orig[j]
                end
                v = 0
                for bi in 1:limits
                    calib > boundaries[bi] && (v += 1)
                end
                co = CAL ?
                     (Float64(centroids[v + 1]) * Float64(inv_scale_tq[j]) -
                      Float64(shift[j])) :
                     Float64(centroids[v + 1])
                if k < 4
                    b0 |= UInt8(v) << (6 - 2k)
                else
                    b1 |= UInt8(v) << (6 - 2(k - 4))
                end
                kk = k & 3
                if kk == 0
                    a0 += Float64(rot_orig[j]) * co
                elseif kk == 1
                    a1 += Float64(rot_orig[j]) * co
                elseif kk == 2
                    a2 += Float64(rot_orig[j]) * co
                else
                    a3 += Float64(rot_orig[j]) * co
                end
            end
            blocked[byte_offset(b, 2c, l, ng)] = b0
            blocked[byte_offset(b, 2c + 1, l, ng)] = b1
        else
            for k in 0:7
                j = offset + k + 1
                local calib::Float32
                if CAL
                    calib = (rot_orig[j] + shift[j]) * scale_tq[j]
                else
                    calib = rot_orig[j]
                end
                v = 0
                for bi in 1:limits
                    calib > boundaries[bi] && (v += 1)
                end
                co = CAL ?
                     (Float64(centroids[v + 1]) * Float64(inv_scale_tq[j]) -
                      Float64(shift[j])) :
                     Float64(centroids[v + 1])
                local boff = byte_offset(b, 4c + (k >> 1), l, ng)
                if (k & 1) == 0
                    blocked[boff] = UInt8(v) << 4
                else
                    blocked[boff] |= UInt8(v)
                end
                kk = k & 3
                if kk == 0
                    a0 += Float64(rot_orig[j]) * co
                elseif kk == 1
                    a1 += Float64(rot_orig[j]) * co
                elseif kk == 2
                    a2 += Float64(rot_orig[j]) * co
                else
                    a3 += Float64(rot_orig[j]) * co
                end
            end
        end
    end

    inner = (a0 + a1) + (a2 + a3)
    if inner > DEGENERATE_INNER_EPS
        norm / Float32(inner)
    else
        0.0f0
    end
end
