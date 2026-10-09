using TurboVec, Random
const TV = TurboVec

function main()
    dim, bits, n = 768, 4, 100_000
    X = randn(MersenneTwister(1), Float32, n, dim)
    rot = TV.Rotation(dim)
    rotated = Matrix{Float32}(undef, dim, n)
    norms = Vector{Float32}(undef, n)

    t = @elapsed TV._validate_input(X, dim)
    println("validate:    ", round(t, digits = 3), "s  (",
            round(t / n * 1e6, digits = 2), " us/row)")
    t = @elapsed TV.rotate_batch!(rotated, norms, X, rot)
    println("rotate:      ", round(t, digits = 3), "s  (",
            round(t / n * 1e6, digits = 2), " us/row)")

    idx = TurboQuantIndex(dim, bits)
    add!(idx, X)
    shift = zeros(Float32, dim)
    scale = ones(Float32, dim)
    inv = ones(Float32, dim)
    codes = zeros(UInt8, TV.blocked_len(n, bits, dim))
    scales = Vector{Float32}(undef, n)
    quant = () -> begin
        for i in 1:n
            scales[i] = TV.quantize_scale_pack!(
                codes, i - 1, view(rotated, :, i), shift, scale, inv,
                idx.centroids, idx.boundaries, bits, dim, norms[i], Val(false))
        end
    end
    quant()
    t = @elapsed quant()
    println("quantize:    ", round(t, digits = 3), "s  (",
            round(t / n * 1e6, digits = 2), " us/row)")

    t = @elapsed add!(idx, X)
    println("add! total:  ", round(t, digits = 3), "s  (",
            round(t / n * 1e6, digits = 2), " us/row)")
    t = @elapsed add!(idx, X)
    println("add! again:  ", round(t, digits = 3), "s  (",
            round(t / n * 1e6, digits = 2), " us/row)")
end

main()
