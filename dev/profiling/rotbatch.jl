using TurboVec
using Random
const TV = TurboVec

"""
Batch rotation: transform T rows at once so the per-stage inner loop
runs across rows and vectorizes. Per-row op order is unchanged.
"""
function rotate_batch_tiled!(rotated, norms, X, rot; tile = 1024)
    n, dim = size(X)
    block = rot.block
    inv_sqrt_block = rot.inv_sqrt_block
    p0 = rot.perms[1]
    p1 = rot.perms[2]
    s0 = rot.signs[1]
    s1p = rot.signs1_pre
    src = Matrix{Float32}(undef, dim, tile)
    scratch = Matrix{Float32}(undef, dim, tile)
    chains = Matrix{Float32}(undef, 8, tile)

    for lo in 1:tile:n
        hi = min(lo + tile - 1, n)
        T = hi - lo + 1
        @inbounds for r in 1:T
            row = lo + r - 1
            @simd for d in 1:dim
                src[d, r] = X[row, d]
            end
        end
        fill!(chains, 0)
        @inbounds for j in 1:dim
            c = ((j - 1) & 7) + 1
            @simd for r in 1:T
                v = src[j, r]
                chains[c, r] += v * v
            end
        end
        @inbounds @simd for r in 1:T
            s = ((chains[1, r] + chains[2, r]) + (chains[3, r] + chains[4, r])) +
                ((chains[5, r] + chains[6, r]) + (chains[7, r] + chains[8, r]))
            norms[lo + r - 1] = sqrt(s)
        end
        invs = Vector{Float32}(undef, T)
        @inbounds @simd for r in 1:T
            nrm = norms[lo + r - 1]
            invs[r] = nrm > TV.MIN_INPUT_NORM ? 1.0f0 / nrm : 0.0f0
        end
        @inbounds for i in 1:dim
            p = Int(p0[i]) + 1
            si = s0[i]
            @simd for r in 1:T
                scratch[i, r] = (src[p, r] * invs[r]) * si
            end
        end
        for o in 0:block:(dim - block)
            len = 1
            while len < block
                ii = o
                while ii < o + block
                    @inbounds for j in (ii + 1):(ii + len)
                        @simd for r in 1:T
                            a = scratch[j, r]
                            b = scratch[j + len, r]
                            scratch[j, r] = a + b
                            scratch[j + len, r] = a - b
                        end
                    end
                    ii += 2 * len
                end
                len <<= 1
            end
            @inbounds for j in (o + 1):(o + block)
                @simd for r in 1:T
                    scratch[j, r] *= inv_sqrt_block
                end
            end
        end
        @inbounds for i in 1:dim
            si = s1p[i]
            @simd for r in 1:T
                scratch[i, r] *= si
            end
        end
        @inbounds for i in 1:dim
            p = Int(p1[i]) + 1
            @simd for r in 1:T
                rotated[i, lo + r - 1] = scratch[p, r]
            end
        end
        for o in 0:block:(dim - block)
            len = 1
            while len < block
                ii = o
                while ii < o + block
                    @inbounds for j in (ii + 1):(ii + len)
                        @simd for r in 1:T
                            a = rotated[j, lo + r - 1]
                            b = rotated[j + len, lo + r - 1]
                            rotated[j, lo + r - 1] = a + b
                            rotated[j + len, lo + r - 1] = a - b
                        end
                    end
                    ii += 2 * len
                end
                len <<= 1
            end
            @inbounds for j in (o + 1):(o + block)
                @simd for r in 1:T
                    rotated[j, lo + r - 1] *= inv_sqrt_block
                end
            end
        end
    end
    nothing
end

function main()
    Random.seed!(1)
    for (n, dim) in ((2000, 256), (2000, 768), (500, 200), (500, 100))
        X = randn(MersenneTwister(2), Float32, n, dim)
        rot = TV.Rotation(dim)
        r1 = Matrix{Float32}(undef, dim, n)
        n1 = Vector{Float32}(undef, n)
        r2 = Matrix{Float32}(undef, dim, n)
        n2 = Vector{Float32}(undef, n)
        TV.rotate_batch!(r1, n1, X, rot)
        rotate_batch_tiled!(r2, n2, X, rot)
        exact = reinterpret.(UInt32, r1) == reinterpret.(UInt32, r2) &&
                reinterpret.(UInt32, n1) == reinterpret.(UInt32, n2)
        t1 = @elapsed TV.rotate_batch!(r1, n1, X, rot)
        t2 = @elapsed rotate_batch_tiled!(r2, n2, X, rot)
        println("n=$n dim=$dim exact=$exact old=", round(t1 / n * 1e6, digits = 2),
                "us/row new=", round(t2 / n * 1e6, digits = 2), "us/row")
        flush(stdout)
    end
end

main()
