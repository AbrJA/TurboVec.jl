# Port of turbovec's tests/concurrent_search.rs: concurrent readers are
# safe, deterministic, and agree with serial and post-mutation searches.

@testset "concurrent search" begin
    rng = MersenneTwister(0x5EED)
    dim, n, nq, k = 128, 1500, 24, 10
    X = rand_rows(rng, n, dim)
    Q = rand_rows(rng, nq, dim)

    idx = TurboQuantIndex(dim, 4)
    add!(idx, X)
    base_s, base_i = search(idx, Q, k)

    @testset "search is deterministic across threads" begin
        for _ in 1:4
            s, i = search(idx, Q, k)
            @test s == base_s
            @test i == base_i
        end
    end

    @testset "concurrent self-consistency" begin
        results = Vector{Any}(undef, 8)
        Threads.@threads for t in 1:8
            results[t] = search(idx, Q, k)
        end
        for t in 1:8
            @test results[t][1] == base_s
            @test results[t][2] == base_i
        end
    end

    @testset "write/load preserves results" begin
        mktempdir() do dir
            path = joinpath(dir, "idx.tv")
            write_index(path, idx)
            loaded = load_index(path)
            s, i = search(loaded, Q, k)
            @test s == base_s
            @test i == base_i
        end
    end

    @testset "mutation then search reflects the new layout" begin
        idx2 = TurboQuantIndex(dim, 4)
        add!(idx2, X)
        s0, i0 = search(idx2, X[1:1, :], 1)
        @test i0[1, 1] == 1
        # remove slot 1: the last vector moves in
        swap_remove!(idx2, 1)
        s1, i1 = search(idx2, X[n:n, :], 1)
        @test i1[1, 1] == 1
        add!(idx2, X[1:1, :])
        s2, i2 = search(idx2, X[1:1, :], 1)
        @test i2[1, 1] == n
    end

    @testset "id-map concurrent search is deterministic" begin
        ids = UInt64.(1:n)
        m = IdMapIndex(dim, 4)
        add_with_ids!(m, X, ids)
        s, g = search(m, Q, k)
        for _ in 1:3
            s2, g2 = search(m, Q, k)
            @test s2 == s && g2 == g
        end
    end
end
