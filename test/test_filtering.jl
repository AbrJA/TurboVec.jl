# Port of turbovec's tests/filtering.rs: slot masks and id allowlists.

@testset "filtering" begin
    rng = MersenneTwister(0xF117)
    dim, n, nq, k = 64, 200, 6, 10
    X = rand_rows(rng, n, dim)
    Q = rand_rows(rng, nq, dim)

    idx = TurboQuantIndex(dim, 4)
    add!(idx, X)
    full_s, full_i = search(idx, Q, n)

    # A deterministic mix of allowed slots.
    mask = falses(n)
    mask[1:3:n] .= true
    n_allowed = count(mask)

    @testset "mask matches post-hoc filtering" begin
        s, i = search(idx, Q, k; mask = mask)
        @test size(s) == (nq, k)
        for r in 1:nq
            allowed = [(full_s[r, j], full_i[r, j]) for j in 1:n if mask[full_i[r, j]]]
            expected = sort(allowed; by = x -> (-x[1], x[2]))[1:k]
            @test i[r, :] == [e[2] for e in expected]
            @test s[r, :] == Float32[e[1] for e in expected]
        end
    end

    @testset "mask nothing equals all true" begin
        s1, i1 = search(idx, Q, k)
        s2, i2 = search(idx, Q, k; mask = trues(n))
        @test s1 == s2 && i1 == i2
    end

    @testset "all-false mask returns empty" begin
        s, i = search(idx, Q, k; mask = falses(n))
        @test size(s) == (nq, 0)
        @test size(i) == (nq, 0)
    end

    @testset "effective k shrinks to the allowed count" begin
        small = falses(n)
        small[[4, 9, 15]] .= true
        s, i = search(idx, Q, k; mask = small)
        @test size(s) == (nq, 3)
        @test size(i) == (nq, 3)
        @test all(v -> v in (4, 9, 15), i)
    end

    @testset "multi-query batch respects the mask" begin
        s, i = search(idx, Q, k; mask = mask)
        for r in 1:nq
            sr, ir = search(idx, Q[r:r, :], k; mask = mask)
            @test s[r, :] == sr[1, :]
            @test i[r, :] == ir[1, :]
        end
    end

    @testset "mask length mismatch errors" begin
        @test_throws MaskLengthMismatch search(idx, Q, k; mask = trues(n - 1))
        @test_throws MaskLengthMismatch search(idx, Q, k; mask = trues(n + 64))
    end

    @testset "sparse masks return only allowed slots" begin
        sparse_mask = falses(n)
        sparse_mask[1:50:end] .= true
        s, i = search(idx, Q, k; mask = sparse_mask)
        @test all(v -> sparse_mask[v], i)
        @test size(s, 2) == min(k, count(sparse_mask))
    end

    @testset "allowlist returns only listed ids" begin
        ids = UInt64.(10_000:(10_000 + n - 1))
        m = IdMapIndex(dim, 4)
        add_with_ids!(m, X, ids)
        allow = [ids[3], ids[3], ids[7], ids[190]]   # duplicate is deduplicated
        s, got = search(m, Q, k; allowlist = allow)
        @test size(s) == (nq, 3)
        @test all(v -> v in (ids[3], ids[7], ids[190]), got)
        for r in 1:nq
            allowed = [(full_s[r, j], full_i[r, j]) for j in 1:n
                       if full_i[r, j] in (3, 7, 190)]
            expected = sort(allowed; by = x -> (-x[1], x[2]))
            @test got[r, :] == [ids[e[2]] for e in expected]
        end
    end

    @testset "allowlist nothing equals plain search" begin
        ids = UInt64.(1:n)
        m = IdMapIndex(dim, 4)
        add_with_ids!(m, X, ids)
        s1, g1 = search(m, Q, k)
        s2, g2 = search(m, Q, k; allowlist = nothing)
        @test s1 == s2 && g1 == g2
    end

    @testset "empty and unknown allowlists error" begin
        ids = UInt64.(1:n)
        m = IdMapIndex(dim, 4)
        add_with_ids!(m, X, ids)
        @test_throws AllowlistEmpty search(m, Q, k; allowlist = UInt64[])
        @test_throws UnknownId search(m, Q, k; allowlist = UInt64[999_999])
        empty = IdMapIndex(dim, 4)
        @test_throws AllowlistEmpty search(empty, Q, k; allowlist = UInt64[])
        @test_throws UnknownId search(empty, Q, k; allowlist = UInt64[1])
    end

    @testset "allowlist survives swap_remove" begin
        ids = UInt64.(1:n)
        m = IdMapIndex(dim, 4)
        add_with_ids!(m, X, ids)
        allowed = [ids[5], ids[9], ids[11]]
        _, before = search(m, Q, 3; allowlist = allowed)
        # Removing an id outside the allowlist keeps it valid.
        @test remove!(m, ids[25])
        _, after = search(m, Q, 3; allowlist = allowed)
        @test size(after, 2) == 3
        @test all(v -> v in allowed, after)
        @test Set(before[1, :]) == Set(after[1, :])
        # Removing an allowlisted id makes the old allowlist an error.
        @test remove!(m, ids[9])
        @test_throws UnknownId search(m, Q, 3; allowlist = allowed)
    end
end
