# Port of turbovec's tests/from_parts.rs (the applicable subset).

@testset "from_parts" begin
    dim, bits, n = 64, 4, 80
    rng = MersenneTwister(0xF00D)
    X = rand_rows(rng, n, dim)
    idx = TurboQuantIndex(dim, bits)
    add!(idx, X)
    pc = packed_codes(idx)
    sc = copy(scales(idx))

    @testset "round trip matches normal build" begin
        built = from_parts(dim, bits, n, pc, sc)
        @test length(built) == n
        @test bit_width(built) == bits
        @test TurboVec.dim(built) == dim
        @test calibration_state(built) == :uncalibrated
        @test packed_codes(built) == pc
        @test scales(built) == sc
        s1, i1 = search(idx, X, 5)
        s2, i2 = search(built, X, 5)
        @test s1 == s2 && i1 == i2
        @test to_bytes(built) == to_bytes(idx)
    end

    @testset "accepts zero per-vector scale" begin
        sc2 = copy(sc)
        sc2[3] = 0.0f0
        @test from_parts(dim, bits, n, pc, sc2) isa TurboQuantIndex
    end

    @testset "lazy uncommitted" begin
        lazy = from_parts(0, bits, 0, UInt8[], Float32[])
        @test dim_opt(lazy) === nothing
        @test from_parts(0, bits, 0, UInt8[], Float32[], Float32[], Float32[]) isa
              TurboQuantIndex
    end

    @testset "calibration pair is carried and identity normalized" begin
        shift = fill(0.01f0, dim)
        scale = fill(1.02f0, dim)
        cal = from_parts(dim, bits, n, pc, sc, shift, scale)
        @test calibration_state(cal) == :calibrated
        @test tqplus_shift(cal) == shift
        @test tqplus_scale(cal) == scale
        ident = from_parts(dim, bits, n, pc, sc, zeros(Float32, dim), ones(Float32, dim))
        @test calibration_state(ident) == :uncalibrated
    end

    @testset "rejects bad shapes" begin
        @test_throws BitWidthOutOfRange from_parts(dim, 5, n, pc, sc)
        @test_throws BitWidthOutOfRange from_parts(dim, 0, n, pc, sc)
        @test_throws DimNotPositiveMultipleOf8 from_parts(12, bits, 0, UInt8[], Float32[])
        @test_throws DimTooLarge from_parts(MAX_DIM + 8, bits, 0, UInt8[], Float32[])
        @test_throws InvalidParts from_parts(dim, bits, n, pc[1:(end - 1)], sc)
        @test_throws InvalidParts from_parts(dim, bits, n, vcat(pc, UInt8[0]), sc)
        @test_throws InvalidParts from_parts(dim, bits, n, pc, sc[1:(end - 1)])
        @test_throws InvalidParts from_parts(dim, bits, n, pc, vcat(sc, Float32[1.0]))
        @test_throws InvalidParts from_parts(dim, bits, n, pc, sc, Float32[0.0], Float32[])
        @test_throws InvalidParts from_parts(dim, bits, n, pc, sc, Float32[], Float32[1.0])
        @test_throws InvalidParts from_parts(dim, bits, n, pc, sc,
                                             zeros(Float32, dim - 1),
                                             ones(Float32, dim - 1))
        @test_throws InvalidParts from_parts(0, bits, 1, pc, sc)
        @test_throws InvalidParts from_parts(0, bits, 0, UInt8[0], Float32[])
        @test_throws InvalidParts from_parts(dim, bits, -1, UInt8[], Float32[])
    end

    @testset "rejects bad scale values" begin
        for bad in (NaN32, Inf32, -1.0f0, 2.0f22)
            sc2 = copy(sc)
            sc2[1] = bad
            @test_throws InvalidParts from_parts(dim, bits, n, pc, sc2)
        end
    end

    @testset "rejects bad calibration values" begin
        good_shift = zeros(Float32, dim)
        good_scale = ones(Float32, dim)
        for bad in (NaN32, Inf32)
            @test_throws InvalidParts from_parts(dim, bits, n, pc, sc, fill(bad, dim),
                                                 good_scale)
            @test_throws InvalidParts from_parts(dim, bits, n, pc, sc, good_shift,
                                                 fill(bad, dim))
        end
        @test_throws InvalidParts from_parts(dim, bits, n, pc, sc, good_shift,
                                             zeros(Float32, dim))
        @test_throws InvalidParts from_parts(dim, bits, n, pc, sc, good_shift,
                                             -ones(Float32, dim))
    end
end
