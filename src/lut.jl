# Per-query nibble lookup tables, a direct port of `build_query_lut`.
#
# Each byte group has two 16-entry sub-tables (high nibble first, low
# nibble second). Values are quantised to u8 with a shared per-sub-table
# minimum and a per-query scale, exactly as the Rust kernel does, so the
# integer sums reconstructed at score time match.

struct QueryLut
    table::Vector{UInt8}   # 32 entries per byte group: sub1 at +0, sub2 at +16
    scale::Float32
    bias::Float32
end

@inline function round_half_away_f32(x::Float32)
    t = trunc(x)
    f = x - t
    t + Float32(Int(f >= 0.5f0) - Int(f <= -0.5f0))
end

# The 2-bit sub-table: entry (a,b) is `q[d]*c[a] + q[d+1]*c[b]` formed as
# `(0.0 + q[d]*c[a]) + q[d+1]*c[b]`. Min/max come from the addends'
# minima/maxima (f32 addition is monotone in each operand).
@inline function _sub2!(fv::AbstractMatrix{Float32}, base::Int, g::Int,
                        q::AbstractVector{Float32}, d0::Int,
                        c4::NTuple{4,Float32})
    @inbounds begin
        pd = q[d0]
        pe = q[d0 + 1]
        pa0 = (0.0f0 + pd * c4[1], 0.0f0 + pd * c4[2],
               0.0f0 + pd * c4[3], 0.0f0 + pd * c4[4])
        pb = (pe * c4[1], pe * c4[2], pe * c4[3], pe * c4[4])
        for a in 1:4
            va = pa0[a]
            for b in 1:4
                fv[base + (a - 1) * 4 + (b - 1), g] = va + pb[b]
            end
        end
        amn = min(pa0[1], pa0[2], pa0[3], pa0[4])
        amx = max(pa0[1], pa0[2], pa0[3], pa0[4])
        bmn = min(pb[1], pb[2], pb[3], pb[4])
        bmx = max(pb[1], pb[2], pb[3], pb[4])
        (amn + bmn, amx + bmx)
    end
end

"""
    build_query_lut(q_rot_row, centroids, bits, dim) -> QueryLut

Build the per-query nibble tables in rotated (and TQ+-inverse-calibrated)
query space.
"""
function build_query_lut(q_rot_row::AbstractVector{Float32},
                         centroids::Vector{Float32}, bits::Int, dim::Int)
    cpb = 8 ÷ bits
    cpn = cpb ÷ 2
    ng = dim ÷ cpb
    code_mask = (1 << bits) - 1

    fv = Matrix{Float32}(undef, 32, ng)
    mins = Matrix{Float32}(undef, 2, ng)
    max_span = 0.0f0
    bias = 0.0f0

    c4 = (centroids[1], centroids[2], centroids[3], centroids[4])
    for g in 0:(ng - 1)
        ds = g * cpb
        local lo_min::Float32, lo_max::Float32, hi_min::Float32, hi_max::Float32
        if bits == 2
            lo_min, lo_max = _sub2!(fv, 1, g + 1, q_rot_row, ds + 1, c4)
            hi_min, hi_max = _sub2!(fv, 17, g + 1, q_rot_row, ds + 3, c4)
        else
            # first sub-table: coords ds .. ds+cpn-1
            prods = Matrix{Float32}(undef, cpn, 16)
            for c in 0:(cpn - 1)
                qq = q_rot_row[ds + c + 1]
                for code in 0:(1 << bits)-1
                    prods[c + 1, code + 1] = qq * centroids[code + 1]
                end
            end
            lo_min = typemax(Float32)
            lo_max = -typemax(Float32)
            for nib in 0:15
                s = 0.0f0
                for c in 0:(cpn - 1)
                    sh = (cpn - 1 - c) * bits
                    code = (nib >> sh) & code_mask
                    s += prods[c + 1, code + 1]
                end
                fv[1 + nib, g + 1] = s
                s < lo_min && (lo_min = s)
                s > lo_max && (lo_max = s)
            end
            # second sub-table: coords ds+cpn ..
            for c in 0:(cpn - 1)
                qq = q_rot_row[ds + cpn + c + 1]
                for code in 0:(1 << bits)-1
                    prods[c + 1, code + 1] = qq * centroids[code + 1]
                end
            end
            hi_min = typemax(Float32)
            hi_max = -typemax(Float32)
            for nib in 0:15
                s = 0.0f0
                for c in 0:(cpn - 1)
                    sh = (cpn - 1 - c) * bits
                    code = (nib >> sh) & code_mask
                    s += prods[c + 1, code + 1]
                end
                fv[17 + nib, g + 1] = s
                s < hi_min && (hi_min = s)
                s > hi_max && (hi_max = s)
            end
        end
        mins[1, g + 1] = lo_min
        mins[2, g + 1] = hi_min
        bias += lo_min + hi_min
        ls = lo_max - lo_min
        hs = hi_max - hi_min
        ls > max_span && (max_span = ls)
        hs > max_span && (max_span = hs)
    end

    max_lut = 127.0f0
    scale = max_span > 0.0f0 ? max_span / max_lut : 1.0f0
    if scale >= floatmin(Float32)
        inv_scale = 1.0f0 / scale
    else
        scale = 1.0f0
        inv_scale = 1.0f0
    end

    table = Vector{UInt8}(undef, 32 * ng)
    for g in 0:(ng - 1)
        for sub in 0:1
            m = mins[sub + 1, g + 1]
            for e in 0:15
                v = fv[sub * 16 + e + 1, g + 1]
                r = round_half_away_f32((v - m) * inv_scale)
                isnan(r) && (r = 0.0f0)
                r = clamp(r, 0.0f0, max_lut)
                table[g * 32 + sub * 16 + e + 1] = UInt8(trunc(r))
            end
        end
    end

    QueryLut(table, scale, bias)
end

"""
    scale_lut(lut) -> Vector{Float32}

`scale * Float32(entry)` for every table entry. Bit-identical to the
per-lookup multiply the Rust kernel performs, hoisted out of the scan.
"""
function scale_lut(lut::QueryLut)
    t = lut.table
    out = Vector{Float32}(undef, length(t))
    s = lut.scale
    @inbounds for i in eachindex(t)
        out[i] = s * Float32(t[i])
    end
    out
end
