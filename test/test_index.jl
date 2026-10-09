@testset "TurboQuantIndex" begin
    @testset "construction" begin
        idx = TurboQuantIndex(64, 4)
        @test length(idx) == 0
        @test isempty(idx)
        @test TurboVec.dim_opt(idx) == 64
        @test calibration_state(idx) == :uncalibrated

        lazy = TurboQuantIndex(4)
        @test TurboVec.dim_opt(lazy) === nothing

        @test_throws BitWidthOutOfRange TurboQuantIndex(64, 1)
        @test_throws BitWidthOutOfRange TurboQuantIndex(64, 5)
        @test_throws DimNotPositiveMultipleOf8 TurboQuantIndex(12, 4)
        @test_throws DimTooLarge TurboQuantIndex(MAX_DIM + 8, 4)
    end

    @testset "add and search" begin
        rng = MersenneTwister(1)
        dim, n = 64, 200
        X = rand_rows(rng, n, dim)
        idx = TurboQuantIndex(dim, 4)
        add!(idx, X)
        @test length(idx) == n

        # querying with database rows finds them
        k = 5
        scores, indices = search(idx, X, k)
        @test size(scores) == (n, k)
        @test size(indices) == (n, k)
        hits = count(i -> indices[i, 1] == i, 1:n)
        @test hits >= n - 2

        # scores are sorted descending within each row
        for i in 1:n
            @test issorted(scores[i, :]; rev = true)
        end

        # query batch and single query agree for every row (this also
        # catches shared-state races in the threaded batch path)
        s1, i1 = search(idx, X, k)
        for r in 1:n
            sr, ir = search(idx, view(X, r:r, :), k)
            @test s1[r, :] == sr[1, :]
            @test i1[r, :] == ir[1, :]
        end
        # repeated batch searches are bit-identical
        s1b, i1b = search(idx, X, k)
        @test s1 == s1b && i1 == i1b

        # k clamps to the index size
        s3, i3 = search(idx, X[1:1, :], n + 100)
        @test size(s3, 2) == n
        @test size(i3, 2) == n

        # k = 0 gives empty results
        s4, i4 = search(idx, X, 0)
        @test size(s4) == (n, 0)
    end

    @testset "lazy dim inference" begin
        rng = MersenneTwister(2)
        X = rand_rows(rng, 10, 32)
        idx = TurboQuantIndex(3)
        add!(idx, X)
        @test TurboVec.dim_opt(idx) == 32
        @test_throws DimMismatch add!(idx, rand_rows(rng, 2, 64))
    end

    @testset "zero and degenerate rows" begin
        dim = 32
        idx = TurboQuantIndex(dim, 4)
        X = zeros(Float32, 1, dim)
        X[1, 1] = 1.0f0
        add!(idx, X)
        add!(idx, zeros(Float32, 1, dim))
        q = zeros(Float32, 1, dim)
        q[1, 1] = 1.0f0
        scores, indices = search(idx, q, 2)
        @test indices[1, 1] == 1
        @test scores[1, 2] == 0.0f0
    end

    @testset "input validation" begin
        rng = MersenneTwister(3)
        dim = 32
        idx = TurboQuantIndex(dim, 4)
        @test_throws InvalidInputValue add!(idx, fill(Float32(NaN), 1, dim))
        @test_throws InvalidInputValue add!(idx, fill(Inf32, 1, dim))
        @test_throws InvalidInputValue add!(idx, fill(1.0f17, 1, dim))
        add!(idx, rand_rows(rng, 3, dim))
        @test_throws DimMismatch add!(idx, rand_rows(rng, 1, dim + 8))
        @test_throws QueryBufferNotMultipleOfDim search(idx, zeros(Float32, 1, dim + 8), 1)
        @test_throws InvalidQueryValue search(idx, fill(NaN32, 1, dim), 1)
    end

    @testset "validation reports the first row-major coordinate" begin
        idx = TurboQuantIndex(16, 4)
        # (5,1) and (2,3) are invalid: (2,3) comes first in row-major order.
        Y = ones(Float32, 6, 16)
        Y[5, 1] = NaN32
        Y[2, 3] = Inf32
        err = try
            add!(idx, Y)
        catch e
            e
        end
        @test err isa InvalidInputValue
        @test err.vector_index == 2
        @test err.coord_index == 3
        @test err.value == Inf32

        # Same rule for queries, including a last-column case.
        Q = ones(Float32, 4, 16)
        Q[3, 2] = NaN32
        Q[1, 16] = 1.0f17
        errq = try
            search(idx, Q, 1)
        catch e
            e
        end
        @test errq isa InvalidQueryValue
        @test errq.query_index == 1
        @test errq.coord_index == 16
    end

    @testset "calibration" begin
        rng = MersenneTwister(4)
        dim, n = 48, 300
        X = rand_rows(rng, n, dim)
        idx = TurboQuantIndex(dim, 2)
        calibrate!(idx, X)
        @test calibration_state(idx) == :calibrated
        add!(idx, X)
        scores, indices = search(idx, X, 5)
        hits = count(i -> indices[i, 1] == i, 1:n)
        @test hits >= n - 3

        # refit on a populated index re-encodes and stays searchable
        calibrate!(idx, X[1:64, :])
        @test calibration_state(idx) == :calibrated
        scores2, indices2 = search(idx, X, 5)
        hits2 = count(i -> indices2[i, 1] == i, 1:n)
        @test hits2 >= n - 3

        # degenerate sample is rejected and leaves state untouched
        idx2 = TurboQuantIndex(dim, 2)
        @test_throws DegenerateSample calibrate!(idx2, zeros(Float32, 10, dim))
        @test calibration_state(idx2) == :uncalibrated
        @test_throws EmptyCalibrationSample calibrate!(idx2, zeros(Float32, 1, dim))
    end

    @testset "swap_remove!" begin
        rng = MersenneTwister(5)
        dim, n = 32, 50
        X = rand_rows(rng, n, dim)
        idx = TurboQuantIndex(dim, 4)
        add!(idx, X)
        lastrow = copy(X[n, :])
        moved = swap_remove!(idx, 3)
        @test moved == n
        @test length(idx) == n - 1
        scores, indices = search(idx, reshape(lastrow, 1, dim), 1)
        @test indices[1, 1] == 3

        # removing the last slot
        swap_remove!(idx, length(idx))
        @test length(idx) == n - 2
        @test_throws BoundsError swap_remove!(idx, 0)
        @test_throws BoundsError swap_remove!(idx, length(idx) + 1)
    end
end
