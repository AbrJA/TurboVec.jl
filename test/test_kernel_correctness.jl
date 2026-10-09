# Port of turbovec's tests/kernel_correctness.rs: behaviour every search
# path must share.

@testset "kernel correctness" begin
    rng = MersenneTwister(0xC0FFEE)
    dim, n, nq = 64, 400, 16
    X = unit_rows(rng, n, dim)
    Q = unit_rows(rng, nq, dim)

    for bits in (2, 4)
        idx = TurboQuantIndex(dim, bits)
        add!(idx, X)

        @testset "bits=$bits" begin
            s, i = search(idx, Q, 10)
            @test size(s) == (nq, 10)
            @test size(i) == (nq, 10)
            for r in 1:nq
                @test issorted(s[r, :]; rev = true)
                @test all(1 .<= i[r, :] .<= n)
            end

            # self-query: 4-bit must return self as top-1, 2-bit within top-3
            sself, iself = search(idx, X, 3)
            if bits == 4
                @test count(r -> iself[r, 1] == r, 1:n) >= n - 2
            else
                @test count(r -> r in iself[r, :], 1:n) >= n - 2
            end

            # single query equals the corresponding batch row
            for r in (1, 5, nq)
                sr, ir = search(idx, view(Q, r:r, :), 10)
                @test s[r, :] == sr[1, :]
                @test i[r, :] == ir[1, :]
            end

            # k above len clamps, and repeated searches are identical
            sbig, ibig = search(idx, Q[1:1, :], n + 50)
            @test size(sbig, 2) == n
            @test size(ibig, 2) == n
            s2, i2 = search(idx, Q, 10)
            @test s == s2 && i == i2
        end
    end

    @testset "empty query batch is not a panic" begin
        idx = TurboQuantIndex(32, 4)
        add!(idx, rand_rows(MersenneTwister(1), 10, 32))
        s, i = search(idx, zeros(Float32, 0, 32), 5)
        @test size(s) == (0, 5) && size(i) == (0, 5)
    end
end
