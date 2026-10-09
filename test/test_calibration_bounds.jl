# Port of turbovec's tests/calibration_bounds.rs: inclusive calibration
# bounds are enforced on every construction path, and they scale with dim.

@testset "calibration bounds" begin
    dim, bits, n = 64, 4, 32
    rng = MersenneTwister(0xB0D)
    X = rand_rows(rng, n, dim)
    idx = TurboQuantIndex(dim, bits)
    add!(idx, X)
    pc = packed_codes(idx)
    sc = copy(scales(idx))

    @testset "bounds scale with dim" begin
        @test TurboVec.min_tqplus_scale(1024) > TurboVec.min_tqplus_scale(8)
        @test TurboVec.max_tqplus_shift(1024) < TurboVec.max_tqplus_shift(8)
    end

    @testset "TQ+ shift bound is inclusive" begin
        cap = TurboVec.max_tqplus_shift(dim)
        shift = fill(cap, dim)
        ok = from_parts(dim, bits, n, pc, sc, shift, ones(Float32, dim))
        @test tqplus_shift(ok) == shift
        @test_throws InvalidParts from_parts(
            dim, bits, n, pc, sc, fill(20.0f0 * cap, dim), ones(Float32, dim))
    end

    @testset "TQ+ scale floor is inclusive" begin
        floor = TurboVec.min_tqplus_scale(dim)
        scale = fill(floor, dim)
        shift = fill(1.0f-3, dim)
        ok = from_parts(dim, bits, n, pc, sc, shift, scale)
        @test tqplus_scale(ok) == scale
        @test_throws InvalidParts from_parts(
            dim, bits, n, pc, sc, shift, fill(floor / 20.0f0, dim))
    end

    @testset "per-vector scale bound is inclusive" begin
        sc2 = copy(sc)
        sc2[2] = TurboVec.MAX_VECTOR_SCALE
        @test from_parts(dim, bits, n, pc, sc2) isa TurboQuantIndex
        sc3 = copy(sc)
        sc3[2] = 2.0f0 * TurboVec.MAX_VECTOR_SCALE
        @test_throws InvalidParts from_parts(dim, bits, n, pc, sc3)
    end

    @testset "calibration at the bound round trips" begin
        cap = TurboVec.max_tqplus_shift(dim)
        shift = fill(cap, dim)
        scale = fill(1.0f0, dim)
        ok = from_parts(dim, bits, n, pc, sc, shift, scale)
        loaded = from_bytes(TurboQuantIndex, to_bytes(ok))
        @test tqplus_shift(loaded) == shift
        @test tqplus_scale(loaded) == scale
    end

    @testset "poisoned calibration is refused on load" begin
        good = from_parts(dim, bits, n, pc, sc,
                          fill(0.01f0, dim), fill(1.1f0, dim))
        bytes = to_bytes(good)
        # Header 27 bytes, then centroids (2^bits) and boundaries
        # (2^bits - 1) f32, then the TQ+ shift.
        shift_start = 27 + (1 << bits) * 4 + ((1 << bits) - 1) * 4 + 1
        bytes[shift_start:shift_start + 3] = UInt8[0x00, 0x00, 0xc0, 0x7f]  # NaN
        @test_throws InvalidFileFormat from_bytes(TurboQuantIndex, bytes)

        good_scale = to_bytes(from_parts(dim, bits, n, pc, sc,
                                         fill(0.01f0, dim), fill(1.1f0, dim)))
        scale_start = shift_start + dim * 4
        good_scale[scale_start:scale_start + 3] = UInt8[0x00, 0x00, 0x00, 0x00]  # 0.0
        @test_throws InvalidFileFormat from_bytes(TurboQuantIndex, good_scale)
    end
end
