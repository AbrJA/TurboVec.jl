@testset "recall" begin
    rng = MersenneTwister(21)
    dim, n, nq, k = 128, 2000, 50, 10
    X = Matrix{Float32}(randn(rng, Float32, n, dim))
    Q = Matrix{Float32}(randn(rng, Float32, nq, dim))
    truth = brute_topk(Q, X, k)

    for (bits, floor) in ((2, 0.55), (3, 0.70), (4, 0.85))
        idx = TurboQuantIndex(dim, bits)
        add!(idx, X)
        _, indices = search(idx, Q, k)
        rec = 0.0
        for i in 1:nq
            rec += length(intersect(indices[i, :], truth[i])) / k
        end
        rec /= nq
        @info "recall" bits rec
        @test rec >= floor

        cidx = TurboQuantIndex(dim, bits)
        calibrate!(cidx, X)
        add!(cidx, X)
        _, cindices = search(cidx, Q, k)
        crec = 0.0
        for i in 1:nq
            crec += length(intersect(cindices[i, :], truth[i])) / k
        end
        crec /= nq
        @info "recall calibrated" bits crec
        @test crec >= floor
    end
end
