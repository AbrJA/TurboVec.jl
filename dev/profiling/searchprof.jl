using TurboVec, Random
const TV = TurboVec
function main()
    dim, bits, n = 768, 4, 100_000
    X = randn(MersenneTwister(1), Float32, n, dim)
    idx = TurboQuantIndex(dim, bits)
    add!(idx, X)
    q = X[1, :]
    qb = Vector{Float32}(undef, dim)
    sc = Vector{Float32}(undef, dim)
    prep = TV._prepare_lut(idx, q, qb, sc)
    tprep = @elapsed for _ in 1:20
        TV._prepare_lut(idx, q, qb, sc)
    end
    h = TV.TopK(64)
    out = Vector{Float32}(undef, 64)
    tscan = @elapsed for _ in 1:5
        h2 = TV.TopK(64)
        TV._scan_blocks!(h2, prep, idx.codes, idx.scales, idx.n, 0, idx.n_blocks,
                         nothing, out)
    end
    tfull = @elapsed for _ in 1:5
        search(idx, view(X, 1:1, :), 64)
    end
    println("prepare_lut: ", round(tprep / 20 * 1000, digits = 3), " ms/query")
    println("scan:        ", round(tscan / 5 * 1000, digits = 3), " ms/query")
    println("full search: ", round(tfull / 5 * 1000, digits = 3), " ms/query")
end
main()
