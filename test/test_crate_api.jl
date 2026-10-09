# Port of turbovec's tests/crate_api.rs / api surface: errors compose,
# accessors report committed state, and result shapes are exact.

@testset "crate api" begin
    dim, bits, n = 48, 3, 40
    rng = MersenneTwister(0x0A91)
    X = rand_rows(rng, n, dim)
    idx = TurboQuantIndex(dim, bits)
    add!(idx, X)

    @testset "accessors report committed state" begin
        @test TurboVec.dim(idx) == dim
        @test dim_opt(idx) == dim
        @test bit_width(idx) == bits
        @test length(idx) == n
        @test length(scales(idx)) == n
        @test isempty(tqplus_shift(idx))
        @test isempty(tqplus_scale(idx))
        @test length(packed_codes(idx)) == n * bits * (dim ÷ 8)
        @test prepare(idx) === idx

        m = IdMapIndex(dim, bits)
        add_with_ids!(m, X, UInt64.(1:n))
        @test TurboVec.dim(m) == dim && bit_width(m) == bits
        @test contains_id(m, 1) && !contains_id(m, 999)
        @test length(external_ids(m)) == n
    end

    @testset "search errors are typed" begin
        @test_throws QueryBufferNotMultipleOfDim search(idx, zeros(Float32, 1, dim + 1), 1)
        @test_throws InvalidQueryValue search(idx, fill(NaN32, 1, dim), 1)
        @test_throws ArgumentError search(idx, X, -1)
        @test_throws MaskLengthMismatch search(idx, X, 1; mask = trues(n + 1))
    end

    @testset "add errors are typed" begin
        @test_throws InvalidInputValue add!(idx, fill(Inf32, 1, dim))
        @test_throws InvalidInputValue add!(idx, fill(1.0f16, 1, dim))
        @test_throws DimMismatch add!(idx, rand_rows(rng, 1, dim + 8))
        @test_throws ZeroDim add!(TurboQuantIndex(4), zeros(Float32, 1, 0))
    end

    @testset "calibration errors are typed" begin
        c = TurboQuantIndex(dim, bits)
        @test_throws EmptyCalibrationSample calibrate!(c, zeros(Float32, 1, dim))
        @test_throws DegenerateSample calibrate!(c, zeros(Float32, 8, dim))
        @test calibration_state(c) == :uncalibrated
    end

    @testset "id errors are typed" begin
        m = IdMapIndex(dim, bits)
        add_with_ids!(m, X[1:2, :], UInt64[7, 8])
        @test_throws IdAlreadyPresent add_with_ids!(m, X[1:1, :], UInt64[7])
        @test_throws DuplicateIdInBatch add_with_ids!(m, X[1:2, :], UInt64[9, 9])
        @test_throws IdsCountMismatch add_with_ids!(m, X[1:2, :], UInt64[9])
        @test remove!(m, 7)
        @test !remove!(m, 7)
        @test remove!(m, 8)
        @test isempty(m)
    end

    @testset "errors are TurboVecError subtypes" begin
        @test BitWidthOutOfRange(5) isa TurboVecError
        @test DimMismatch(1, 2) isa TurboVecError
        @test MaskLengthMismatch(1, 2) isa TurboVecError
        @test AllowlistEmpty() isa TurboVecError
        @test UnknownId(UInt64(3)) isa TurboVecError
        @test InvalidParts("x") isa TurboVecError
        @test InvalidFileFormat("x") isa TurboVecError
    end

    @testset "result shapes are row-major" begin
        Q = rand_rows(rng, 5, dim)
        s, i = search(idx, Q, 7)
        @test size(s) == (5, 7) && size(i) == (5, 7)
        s0, i0 = search(idx, Q, 0)
        @test size(s0) == (5, 0) && size(i0) == (5, 0)
    end
end
