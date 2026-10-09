# Port of turbovec's tests/query_scale_invariance.rs: ranking must be
# invariant to positive uniform query scaling, scores must track the
# query magnitude linearly, and an exact power of two must be bit-exact.

@testset "query scale invariance" begin
    rng = MersenneTwister(0x5EED)
    dim, n, nq, k = 256, 1000, 16, 10
    X = unit_rows(rng, n, dim)
    Q = unit_rows(rng, nq, dim)

    idx = TurboQuantIndex(dim, 4)
    add!(idx, X)
    base_s, base_i = search(idx, Q, k)

    @testset "power of two scaling is bit exact" begin
        for c in (1024.0f0, 65536.0f0, 1.0f0 / 1024.0f0, 2.0f0^-60)
            Qs = Q .* c
            s, i = search(idx, Qs, k)
            @test i == base_i
            @test s == base_s .* c
        end
    end

    @testset "scores scale linearly for non-power-of-two factors" begin
        for c in (1.0f6, 1.0f3, 1.0f-3, 1.0f-6)
            Qs = Q .* c
            s, _ = search(idx, Qs, k)
            want = base_s .* c
            @test maximum(abs.(s .- want) ./ max.(abs.(want), floatmin(Float32))) < 2.0f-2
        end
    end

    @testset "ids are invariant across many scales" begin
        for c in (1.0f8, 1.0f4, 1.0f0, 1.0f-4, 1.0f-8, 1.0f-12)
            Qs = Q .* c
            _, i = search(idx, Qs, k)
            @test i == base_i
        end
        # very small queries must not collapse the LUT or produce NaN
        for c in (1.0f-20, 1.0f-30)
            Qs = Q .* c
            s, i = search(idx, Qs, k)
            @test size(s) == (nq, k)
            @test all(isfinite, s)
            @test all(1 .<= i .<= n)
            overlap = sum(length(intersect(i[r, :], base_i[r, :])) for r in 1:nq)
            @test overlap >= nq * k ÷ 2
        end
    end

    @testset "zero query is well formed" begin
        s, i = search(idx, zeros(Float32, 1, dim), k)
        @test size(i) == (1, k)
        @test all(isfinite, s)
    end
end
