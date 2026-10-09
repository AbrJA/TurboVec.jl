using TurboVec

mutable struct Xs
    x::UInt64
end
function next_f32!(r::Xs)
    x = r.x
    x ⊻= x << 13
    x ⊻= x >> 7
    x ⊻= x << 17
    r.x = x
    Float32(Float64(x) / Float64(typemax(UInt64))) - 0.5f0
end

function bench(dim, bits, n, nq, k)
    seed = 0x9E3779B97F4A7C15 ⊻ UInt64(dim) ⊻ (UInt64(bits) << 32)
    rng = Xs(seed)
    db = Matrix{Float32}(undef, n, dim)
    for i in 1:n, d in 1:dim
        db[i, d] = next_f32!(rng)
    end
    queries = Matrix{Float32}(undef, nq, dim)
    for i in 1:nq, d in 1:dim
        queries[i, d] = next_f32!(rng)
    end

    # Warm the encode path so JIT compilation is not timed.
    warm = TurboQuantIndex(dim, bits)
    add!(warm, db[1:min(64, n), :])
    idx = TurboQuantIndex(dim, bits)
    t_add = @elapsed add!(idx, db)
    search(idx, queries[1:1, :], k)
    best = Inf
    for _ in 1:3
        t = @elapsed search(idx, queries, k)
        best = min(best, t)
    end
    println("julia dim=$dim bits=$bits n=$n threads=$(Threads.nthreads()) " *
            "add=$(round(t_add, digits=3))s search=$(round(best / nq * 1000, digits=3))ms/q")
end

if length(ARGS) == 3
    bench(parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]), 100, 64)
else
    for (dim, bits, n) in ((768, 4, 100_000), (768, 2, 100_000),
                           (1536, 4, 50_000), (1536, 2, 50_000))
        bench(dim, bits, n, 100, 64)
    end
end
