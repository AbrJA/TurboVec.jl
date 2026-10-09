using Random
using TurboVec
const TV = TurboVec

function main()
    dim, bits, n = 768, 4, 20_000
    rng = MersenneTwister(1)
    X = randn(rng, Float32, n, dim)

    rot = TV.Rotation(dim)
    rotated = Matrix{Float32}(undef, dim, n)
    norms = Vector{Float32}(undef, n)
    TV.rotate_batch!(rotated, norms, X, rot)

    t = @elapsed TV.rotate_batch!(rotated, norms, X, rot)
    println("rotate+norms: ", round(t, digits = 3), "s  (",
            round(t / n * 1e6, digits = 2), " us/row)")

    idx = TurboQuantIndex(dim, bits)
    add!(idx, X)
    shift = zeros(Float32, dim)
    scale = ones(Float32, dim)
    inv = ones(Float32, dim)
    codes = zeros(UInt8, TV.blocked_len(n, bits, dim))
    scales = Vector{Float32}(undef, n)
    TV.quantize_scale_pack!(codes, 0, view(rotated, :, 1), shift, scale, inv,
                            idx.centroids, idx.boundaries, bits, dim, norms[1],
                            Val(false))
    t = @elapsed for i in 1:n
        scales[i] = TV.quantize_scale_pack!(codes, i - 1, view(rotated, :, i),
                                            shift, scale, inv, idx.centroids,
                                            idx.boundaries, bits, dim, norms[i],
                                            Val(false))
    end
    println("quantize:     ", round(t, digits = 3), "s  (",
            round(t / n * 1e6, digits = 2), " us/row)")

    println("--- bits=2")
    TV.quantize_scale_pack!(codes, 0, view(rotated, :, 1), shift, scale, inv,
                            idx.centroids, idx.boundaries, 2, dim, norms[1],
                            Val(false))
    t = @elapsed for i in 1:n
        scales[i] = TV.quantize_scale_pack!(codes, i - 1, view(rotated, :, i),
                                            shift, scale, inv, idx.centroids,
                                            idx.boundaries, 2, dim, norms[i],
                                            Val(false))
    end
    println("quantize 2b:  ", round(t, digits = 3), "s  (",
            round(t / n * 1e6, digits = 2), " us/row)")

    # warm add (whole encode) single-threaded timing
    t = @elapsed add!(idx, X)
    println("add! total:   ", round(t, digits = 3), "s  (",
            round(t / n * 1e6, digits = 2), " us/row)")
end

main()
