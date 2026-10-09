# Deterministic orthogonal rotation via a globally-permuted
# block-Hadamard transform. Port of `turbovec::rotation`.
#
# One round is: (1) a global Fisher-Yates permutation, (2) a ±1 sign
# flip, (3) a normalized Walsh-Hadamard transform (× 1/√B) applied to
# each contiguous B-coordinate block, where B is the largest power-of-two
# divisor of `dim`. The rotation is K = 2 rounds. The sign flips and
# permutations come from a ChaCha8 stream seeded with the frozen
# turbovec seed, so the transform matches the Rust implementation
# bit-for-bit.

const ROTATION_K = 2

const ROTATION_SEED = UInt8[164, 143, 161, 123, 88, 50, 61, 10, 234, 184, 161, 204, 105, 1,
                            20, 184,
                            43, 140, 200, 117, 24, 180, 247, 84, 141, 68, 110, 161, 228,
                            223, 32, 242]

"""Largest power-of-two divisor of `dim` (always >= 8)."""
block_size(dim::Int) = dim & -dim

struct Rotation
    dim::Int
    block::Int
    inv_sqrt_block::Float32
    signs::Vector{Vector{Float32}}
    perms::Vector{Vector{Int32}}
    signs1_pre::Vector{Float32}
end

function fisher_yates(dim::Int, rng::ChaCha8)
    perm = Int32.(0:(dim - 1))
    @inbounds for i in (dim - 1):-1:1
        j = Int(next_u64!(rng) % UInt64(i + 1))
        perm[i + 1], perm[j + 1] = perm[j + 1], perm[i + 1]
    end
    perm
end

function Rotation(dim::Int)
    (dim > 0 && dim % 8 == 0) ||
        throw(ArgumentError("rotation dim must be a positive multiple of 8, got $dim"))
    dim <= MAX_DIM || throw(ArgumentError("rotation dim $dim exceeds MAX_DIM ($MAX_DIM)"))
    block = block_size(dim)
    inv_sqrt_block = 1.0f0 / sqrt(Float32(block))

    rng = ChaCha8(ROTATION_SEED)
    signs = Vector{Vector{Float32}}(undef, ROTATION_K)
    perms = Vector{Vector{Int32}}(undef, ROTATION_K)
    for r in 1:ROTATION_K
        signs[r] = Float32[(next_u32!(rng) & 0x1) == 0x1 ? -1.0f0 : 1.0f0 for _ in 1:dim]
        perms[r] = fisher_yates(dim, rng)
    end

    signs1_pre = ones(Float32, dim)
    @inbounds for i in 1:dim
        signs1_pre[Int(perms[2][i]) + 1] = signs[2][i]
    end

    Rotation(dim, block, inv_sqrt_block, signs, perms, signs1_pre)
end

# Unnormalized Walsh-Hadamard butterfly over `buf[off:off+block-1]`,
# then the 1/√B orthonormalization scale. Radix-8 passes resolve three
# stages per memory pass (`(a±b)±(c±d)` trees) with radix-4 / radix-2
# tails; every output is the identical expression tree with the same f32
# roundings as three sequential radix-2 stages, so the result is
# bit-identical (pinned by `test_rotation_determinism.jl`).
@inline function wht_block!(blk::AbstractVector{Float32}, off::Int, block::Int,
                            inv_sqrt_block::Float32)
    len = 1
    @inbounds while 4 * len < block
        iend = off + block
        i = off
        while i < iend
            j = i
            jend = i + len - 1
            while j <= jend
                a = blk[j]
                b = blk[j + len]
                c = blk[j + 2len]
                d = blk[j + 3len]
                e = blk[j + 4len]
                f = blk[j + 5len]
                g = blk[j + 6len]
                h = blk[j + 7len]
                apb = a + b
                amb = a - b
                cpd = c + d
                cmd = c - d
                epf = e + f
                emf = e - f
                gph = g + h
                gmh = g - h
                s0 = apb + cpd
                s1 = amb + cmd
                s2 = apb - cpd
                s3 = amb - cmd
                s4 = epf + gph
                s5 = emf + gmh
                s6 = epf - gph
                s7 = emf - gmh
                blk[j] = s0 + s4
                blk[j + len] = s1 + s5
                blk[j + 2len] = s2 + s6
                blk[j + 3len] = s3 + s7
                blk[j + 4len] = s0 - s4
                blk[j + 5len] = s1 - s5
                blk[j + 6len] = s2 - s6
                blk[j + 7len] = s3 - s7
                j += 1
            end
            i += 8 * len
        end
        len <<= 3
    end
    if 2 * len < block
        i = off
        while i < off + block
            j = i
            jend = i + len - 1
            while j <= jend
                a = blk[j]
                b = blk[j + len]
                c = blk[j + 2len]
                d = blk[j + 3len]
                apb = a + b
                amb = a - b
                cpd = c + d
                cmd = c - d
                blk[j] = apb + cpd
                blk[j + len] = amb + cmd
                blk[j + 2len] = apb - cpd
                blk[j + 3len] = amb - cmd
                j += 1
            end
            i += 4 * len
        end
        len <<= 2
    end
    if len < block
        i = off
        while i < off + block
            j = i
            jend = i + len - 1
            while j <= jend
                a = blk[j]
                b = blk[j + len]
                blk[j] = a + b
                blk[j + len] = a - b
                j += 1
            end
            i += 2 * len
        end
    end
    i = off
    while i < off + block
        blk[i] *= inv_sqrt_block
        i += 1
    end
    nothing
end

@inline function wht!(buf::AbstractVector{Float32}, offset::Int, dim::Int,
                      block::Int, inv_sqrt_block::Float32)
    o = 0
    while o < dim
        wht_block!(buf, offset + o + 1, block, inv_sqrt_block)
        o += block
    end
    nothing
end

"""
    apply_scaled_into!(rot, src, inv, dst, scratch)

Rotate `src` scaled by `inv` into `dst`, leaving `src` untouched. Exact
op order of the Rust `Rotation::apply_scaled_into`.
"""
function apply_scaled_into!(rot::Rotation, src::AbstractVector{Float32}, inv::Float32,
                            dst::AbstractVector{Float32}, scratch::AbstractVector{Float32})
    dim = rot.dim
    @inbounds begin
        p0 = rot.perms[1]
        s0 = rot.signs[1]
        for i in 1:dim
            scratch[i] = (src[Int(p0[i]) + 1] * inv) * s0[i]
        end
    end
    wht!(scratch, 0, dim, rot.block, rot.inv_sqrt_block)
    @inbounds begin
        s1p = rot.signs1_pre
        for i in 1:dim
            scratch[i] *= s1p[i]
        end
        p1 = rot.perms[2]
        for i in 1:dim
            dst[i] = scratch[Int(p1[i]) + 1]
        end
    end
    wht!(dst, 0, dim, rot.block, rot.inv_sqrt_block)
    nothing
end

"""Rotate one row in place."""
function apply_rotation!(rot::Rotation, row::AbstractVector{Float32})
    scratch = Vector{Float32}(undef, rot.dim)
    apply_scaled_into!(rot, row, 1.0f0, row, scratch)
    nothing
end
