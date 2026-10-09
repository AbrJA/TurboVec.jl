using TurboVec, Random
const TV = TurboVec

# Radix-8 unnormalized butterfly + 1/sqrt(block) scale, same expression
# trees as the sequential radix-2 stages, then radix-4 / radix-2 tails.
@inline function wht_block_fast!(blk::AbstractVector{Float32}, block::Int,
                                 off::Int, inv_sqrt_block::Float32)
    p = off
    len = 1
    while 4 * len < block
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
                apb = a + b; amb = a - b
                cpd = c + d; cmd = c - d
                epf = e + f; emf = e - f
                gph = g + h; gmh = g - h
                s0 = apb + cpd; s1 = amb + cmd; s2 = apb - cpd; s3 = amb - cmd
                s4 = epf + gph; s5 = emf + gmh; s6 = epf - gph; s7 = emf - gmh
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
                a = blk[j]; b = blk[j + len]
                c = blk[j + 2len]; d = blk[j + 3len]
                apb = a + b; amb = a - b
                cpd = c + d; cmd = c - d
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
                a = blk[j]; b = blk[j + len]
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

function rotate_fast!(rotated, norms, X, rot; tile_rows = 512)
    n, dim = size(X)
    block = rot.block
    p0 = rot.perms[1]; p1 = rot.perms[2]
    s0 = rot.signs[1]; s1p = rot.signs1_pre
    @sync for (lo, hi) in TV._parallel_ranges(n)
        Threads.@spawn begin
            src = Vector{Float32}(undef, dim)
            scratch = Vector{Float32}(undef, dim)
            for i in lo:hi
                @inbounds for d in 1:dim
                    src[d] = X[i, d]
                end
                nrm = TV.simd_norm(src, dim)
                norms[i] = nrm
                inv = nrm > TV.MIN_INPUT_NORM ? 1.0f0 / nrm : 0.0f0
                dst = view(rotated, :, i)
                @inbounds for k in 1:dim
                    scratch[k] = (src[Int(p0[k]) + 1] * inv) * s0[k]
                end
                o = 1
                while o <= dim
                    wht_block_fast!(scratch, block, o, rot.inv_sqrt_block)
                    o += block
                end
                @inbounds for k in 1:dim
                    scratch[k] *= s1p[k]
                end
                @inbounds for k in 1:dim
                    dst[k] = scratch[Int(p1[k]) + 1]
                end
                o = 1
                while o <= dim
                    wht_block_fast!(dst, block, o, rot.inv_sqrt_block)
                    o += block
                end
            end
        end
    end
    nothing
end

function main()
    Random.seed!(1)
    for (n, dim) in ((2000, 256), (2000, 768), (2000, 1536), (500, 200), (500, 1000))
        X = randn(MersenneTwister(2), Float32, n, dim)
        rot = TV.Rotation(dim)
        r1 = Matrix{Float32}(undef, dim, n); n1 = Vector{Float32}(undef, n)
        r2 = Matrix{Float32}(undef, dim, n); n2 = Vector{Float32}(undef, n)
        TV.rotate_batch!(r1, n1, X, rot)
        rotate_fast!(r2, n2, X, rot)
        exact = reinterpret.(UInt32, r1) == reinterpret.(UInt32, r2) &&
                reinterpret.(UInt32, n1) == reinterpret.(UInt32, n2)
        t1 = @elapsed TV.rotate_batch!(r1, n1, X, rot)
        t2 = @elapsed rotate_fast!(r2, n2, X, rot)
        println("n=$n dim=$dim block=$(rot.block) exact=$exact old=",
                round(t1/n*1e6, digits=2), "us/row new=",
                round(t2/n*1e6, digits=2), "us/row")
        flush(stdout)
    end
end

main()
