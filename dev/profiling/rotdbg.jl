using TurboVec, Random
using Printf
const TV = TurboVec

function dbg(n, dim)
    X = randn(MersenneTwister(2), Float32, n, dim)
    rot = TV.Rotation(dim)
    block = rot.block
    r1 = Matrix{Float32}(undef, dim, n)
    n1 = Vector{Float32}(undef, n)
    TV.rotate_batch!(r1, n1, X, rot)
    @printf("n=%d dim=%d block=%d\n", n, dim, block)

    # tiled round 0 only
    src = Matrix{Float32}(undef, dim, n)
    for r in 1:n, d in 1:dim
        src[d, r] = X[r, d]
    end
    chains = zeros(Float32, 8, n)
    for j in 1:dim
        c = ((j - 1) & 7) + 1
        for r in 1:n
            v = src[j, r]
            chains[c, r] += v * v
        end
    end
    n2 = Vector{Float32}(undef, n)
    for r in 1:n
        s = ((chains[1, r] + chains[2, r]) + (chains[3, r] + chains[4, r])) +
            ((chains[5, r] + chains[6, r]) + (chains[7, r] + chains[8, r]))
        n2[r] = sqrt(s)
    end
    println("norms exact: ", reinterpret.(UInt32, n1) == reinterpret.(UInt32, n2),
            " maxdiff=", maximum(abs.(n1 .- n2)))

    invs = [nrm > TV.MIN_INPUT_NORM ? 1.0f0 / nrm : 0.0f0 for nrm in n1]
    scratch = Matrix{Float32}(undef, dim, n)
    for i in 1:dim
        p = Int(rot.perms[1][i]) + 1
        si = rot.signs[1][i]
        for r in 1:n
            scratch[i, r] = (src[p, r] * invs[r]) * si
        end
    end
    for o in 0:block:(dim - block)
        len = 1
        while len < block
            for j in (o + 1):(o + len)
                for r in 1:n
                    a = scratch[j, r]
                    b = scratch[j + len, r]
                    scratch[j, r] = a + b
                    scratch[j + len, r] = a - b
                end
            end
            len <<= 1
        end
        for j in (o + 1):(o + block), r in 1:n
            scratch[j, r] *= rot.inv_sqrt_block
        end
    end
    for i in 1:dim, r in 1:n
        scratch[i, r] *= rot.signs1_pre[i]
    end
    r2 = Matrix{Float32}(undef, dim, n)
    for i in 1:dim
        p = Int(rot.perms[2][i]) + 1
        for r in 1:n
            r2[i, r] = scratch[p, r]
        end
    end
    for o in 0:block:(dim - block)
        len = 1
        while len < block
            for j in (o + 1):(o + len)
                for r in 1:n
                    a = r2[j, r]
                    b = r2[j + len, r]
                    r2[j, r] = a + b
                    r2[j + len, r] = a - b
                end
            end
            len <<= 1
        end
        for j in (o + 1):(o + block), r in 1:n
            r2[j, r] *= rot.inv_sqrt_block
        end
    end
    same = reinterpret.(UInt32, r1) == reinterpret.(UInt32, r2)
    println("rotated exact: ", same)
    if !same
        d = abs.(r1 .- r2)
        idx = argmax(d)
        println("maxdiff=", maximum(d), " at ", idx, " r1=", r1[idx], " r2=", r2[idx])
        println("n mismatch bits: ", count(reinterpret.(UInt32, r1) .!=
                                         reinterpret.(UInt32, r2)), " of ", length(r1))
    end
end

dbg(64, 256)
dbg(64, 200)
dbg(64, 768)
