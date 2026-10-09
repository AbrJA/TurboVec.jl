# Port of turbovec's tests/bytes_io.rs, adapted to the Julia-native
# byte layout: in-memory round trips must equal file round trips and
# preserve search results exactly.

function built_parts(; dim = 64, bits = 4, n = 120, calibrated = false, seed = 1)
    rng = MersenneTwister(seed)
    X = rand_rows(rng, n, dim)
    idx = TurboQuantIndex(dim, bits)
    calibrated && calibrate!(idx, X)
    add!(idx, X)
    (idx, X, packed_codes(idx), copy(scales(idx)),
     copy(tqplus_shift(idx)), copy(tqplus_scale(idx)))
end

@testset "bytes io" begin
    @testset "index round trip" begin
        idx, X, pc, sc, _, _ = built_parts()
        bytes = to_bytes(idx)
        loaded = from_bytes(TurboQuantIndex, bytes)
        @test length(loaded) == length(idx)
        s1, i1 = search(idx, X, 5)
        s2, i2 = search(loaded, X, 5)
        @test s1 == s2 && i1 == i2
        @test to_bytes(loaded) == bytes
    end

    @testset "id-map round trip" begin
        dim, n = 64, 120
        rng = MersenneTwister(2)
        X = rand_rows(rng, n, dim)
        ids = UInt64.(7000:(7000 + n - 1))
        m = IdMapIndex(dim, 4)
        add_with_ids!(m, X, ids)
        bytes = to_bytes(m)
        loaded = from_bytes(IdMapIndex, bytes)
        @test external_ids(loaded) == ids
        s1, g1 = search(m, X, 5)
        s2, g2 = search(loaded, X, 5)
        @test s1 == s2 && g1 == g2
        @test to_bytes(loaded) == bytes
    end

    @testset "empty and lazy indexes round trip" begin
        for idx in (TurboQuantIndex(64, 2), TurboQuantIndex(2))
            loaded = from_bytes(TurboQuantIndex, to_bytes(idx))
            @test dim_opt(loaded) == dim_opt(idx)
            @test isempty(loaded)
        end
        m = IdMapIndex(64, 2)
        loaded = from_bytes(IdMapIndex, to_bytes(m))
        @test isempty(loaded)
    end

    @testset "wrong kind is rejected" begin
        idx, _, _, _, _, _ = built_parts()
        m = IdMapIndex(64, 4)
        add_with_ids!(m, zeros(Float32, 1, 64), UInt64[1])
        @test_throws InvalidFileFormat from_bytes(IdMapIndex, to_bytes(idx))
        @test_throws InvalidFileFormat from_bytes(TurboQuantIndex, to_bytes(m))
    end

    @testset "from_bytes rejects duplicate ids" begin
        rng = MersenneTwister(3)
        X = rand_rows(rng, 4, 32)
        m = IdMapIndex(32, 4)
        add_with_ids!(m, X, UInt64[1, 2, 3, 4])
        bytes = to_bytes(m)
        # The id table is the final 4 * 8 bytes; make two entries equal.
        for b in 1:8
            bytes[end - 2 * 8 + b] = bytes[end - 3 * 8 + b]
        end
        @test_throws InvalidFileFormat from_bytes(IdMapIndex, bytes)
    end

    @testset "truncated buffers are rejected" begin
        idx, _, _, _, _, _ = built_parts()
        bytes = to_bytes(idx)
        @test_throws Exception from_bytes(TurboQuantIndex, bytes[1:div(length(bytes), 2)])
        @test_throws InvalidFileFormat from_bytes(TurboQuantIndex, UInt8[])
    end
end
