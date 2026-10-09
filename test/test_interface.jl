# Julia-idiomatic interface: Base integration and single-vector
# conveniences layered over the ported API.

@testset "interface" begin
    rng = MersenneTwister(0x1F)
    dim, bits, n = 32, 4, 40
    X = rand_rows(rng, n, dim)
    idx = TurboQuantIndex(dim, bits)
    add!(idx, X)

    @testset "size" begin
        @test size(idx) == (n, dim)
        @test size(idx, 1) == n
        @test size(idx, 2) == dim
        @test size(idx, 3) == 1
        lazy = TurboQuantIndex(bits)
        @test size(lazy) == (0, 0)
        m = IdMapIndex(dim, bits)
        add_with_ids!(m, X, UInt64.(1:n))
        @test size(m) == (n, dim)
    end

    @testset "show" begin
        s = sprint(show, idx)
        @test occursin("$dim features", s)
        @test occursin("$n vectors", s)
        @test occursin("uncalibrated", s)
        @test occursin("lazy", sprint(show, TurboQuantIndex(bits)))

        p = sprint(show, MIME"text/plain"(), idx)
        @test occursin("TurboQuantIndex:", p)
        @test occursin("$dim features", p)
        @test occursin("codes: ", p)

        cidx = TurboQuantIndex(dim, bits)
        calibrate!(cidx, X)
        @test occursin("calibrated", sprint(show, cidx))

        m = IdMapIndex(dim, bits)
        add_with_ids!(m, X, UInt64.(1:n))
        @test occursin("$n ids", sprint(show, m))
        pm = sprint(show, MIME"text/plain"(), m)
        @test occursin("IdMapIndex:", pm)
        @test occursin("$n ids", pm)
    end

    @testset "equality" begin
        a = TurboQuantIndex(dim, bits)
        b = TurboQuantIndex(dim, bits)
        @test a == b
        @test a != TurboQuantIndex(dim + 8, bits)
        @test a != TurboQuantIndex(dim, 2)
        add!(a, X)
        add!(b, X)
        @test a == b
        add!(b, X[1:1, :])
        @test a != b
        c = copy(a)
        empty!(c)
        @test a != c
        cidx = TurboQuantIndex(dim, bits)
        calibrate!(cidx, X)
        add!(cidx, X)
        @test cidx != a
        @test cidx == copy(cidx)

        m1 = IdMapIndex(dim, bits)
        add_with_ids!(m1, X, UInt64.(1:n))
        m2 = IdMapIndex(dim, bits)
        add_with_ids!(m2, X, UInt64.(1:n))
        @test m1 == m2
        m3 = copy(m1)
        remove!(m3, 1)
        @test m1 != m3
        m4 = IdMapIndex(dim, bits)
        add_with_ids!(m4, X, UInt64.(2:(n + 1)))
        @test m1 != m4
    end

    @testset "copy is independent" begin
        idx2 = copy(idx)
        @test length(idx2) == length(idx)
        s1, i1 = search(idx2, X, 3)
        add!(idx, X[1:1, :])
        @test length(idx) == n + 1
        @test length(idx2) == n
        s2, i2 = search(idx2, X, 3)
        @test s1 == s2 && i1 == i2

        m = IdMapIndex(dim, bits)
        add_with_ids!(m, X, UInt64.(1:n))
        m2 = copy(m)
        remove!(m, 3)
        @test length(m) == n - 1
        @test length(m2) == n
        @test 3 in m2
    end

    @testset "empty!" begin
        idx2 = copy(idx)
        empty!(idx2)
        @test isempty(idx2)
        @test size(idx2, 2) == dim      # geometry is kept
        add!(idx2, X)
        @test length(idx2) == n

        m = IdMapIndex(dim, bits)
        add_with_ids!(m, X, UInt64.(1:n))
        empty!(m)
        @test isempty(m)
        @test isempty(external_ids(m))
        @test size(m, 2) == dim
    end

    @testset "membership and iteration" begin
        m = IdMapIndex(dim, bits)
        ids = UInt64.(100:(100 + n - 1))
        add_with_ids!(m, X, ids)
        @test 100 in m
        @test !(999 in m)
        @test collect(m) == ids
        @test [id for id in m] == ids
        @test keys(m) == ids
    end

    @testset "is_calibrated and calibration" begin
        @test !is_calibrated(idx)
        @test calibration(idx) === nothing
        cidx = TurboQuantIndex(dim, bits)
        calibrate!(cidx, X)
        @test is_calibrated(cidx)
        cal = calibration(cidx)
        @test cal.shift === tqplus_shift(cidx)
        @test cal.scale === tqplus_scale(cidx)
        m = IdMapIndex(dim, bits)
        @test !is_calibrated(m)
    end

    @testset "single-vector conveniences" begin
        idx2 = TurboQuantIndex(dim, bits)
        add!(idx2, X[1, :])
        @test length(idx2) == 1
        s, i = search(idx2, X[1, :], 1)
        @test size(s) == (1, 1) && i[1, 1] == 1

        idx3 = TurboQuantIndex(dim, bits)
        add!(idx3, Float64.(X[1, :]))
        @test length(idx3) == 1

        m = IdMapIndex(dim, bits)
        add_with_ids!(m, X[1, :], 42)
        @test length(m) == 1
        @test 42 in m
        s2, g2 = search(m, X[1, :], 1)
        @test size(s2) == (1, 1) && g2[1, 1] == 42
    end
end
