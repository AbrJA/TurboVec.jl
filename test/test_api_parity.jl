# Public-surface parity with turbovec: the accessors, constants and
# IO-generic entry points added on top of the core API.

@testset "api parity surface" begin
    rng = MersenneTwister(0xABC)
    dim, bits, n = 48, 4, 60
    X = rand_rows(rng, n, dim)
    idx = TurboQuantIndex(dim, bits)
    add!(idx, X)

    @testset "blocked_codes" begin
        cbs = blocked_codes(idx)
        @test cbs == idx.codes
        @test length(cbs) == TurboVec.blocked_len(n, bits, dim)
        @test isempty(blocked_codes(TurboQuantIndex(dim, bits)))
        @test isempty(blocked_codes(TurboQuantIndex(bits)))
    end

    @testset "codebook_for_write" begin
        b, c = codebook_for_write(idx)
        @test b == codebook(bits, dim)[1]
        @test c == codebook(bits, dim)[2]
        eb, ec = codebook_for_write(TurboQuantIndex(dim, bits))
        @test length(eb) == (1 << bits) - 1 && all(iszero, eb)
        @test length(ec) == 1 << bits && all(iszero, ec)
    end

    @testset "serialized_len is exact" begin
        @test serialized_len(idx) == length(to_bytes(idx))
        lazy = TurboQuantIndex(bits)
        @test serialized_len(lazy) == length(to_bytes(lazy))
        cidx = TurboQuantIndex(dim, bits)
        calibrate!(cidx, X)
        add!(cidx, X)
        @test serialized_len(cidx) == length(to_bytes(cidx))
        m = IdMapIndex(dim, bits)
        add_with_ids!(m, X, UInt64.(1:n))
        @test serialized_len(m) == length(to_bytes(m))
    end

    @testset "IO-generic write and load" begin
        io = IOBuffer()
        write_index(io, idx)
        @test take!(io) == to_bytes(idx)

        loaded = load_index(IOBuffer(to_bytes(idx)))
        s1, i1 = search(idx, X, 5)
        s2, i2 = search(loaded, X, 5)
        @test s1 == s2 && i1 == i2

        m = IdMapIndex(dim, bits)
        add_with_ids!(m, X, UInt64.(7:(7 + n - 1)))
        io2 = IOBuffer()
        write_idmap(io2, m)
        m2 = load_idmap(IOBuffer(take!(io2)))
        @test search(m, X, 3) == search(m2, X, 3)

        @test_throws InvalidFileFormat load_index(IOBuffer(to_bytes(m)))
        @test_throws InvalidFileFormat load_idmap(IOBuffer(to_bytes(idx)))
    end

    @testset "batch_addable" begin
        m = IdMapIndex(dim, bits)
        add_with_ids!(m, X[1:2, :], UInt64[10, 11])
        @test batch_addable(m, UInt64[12, 13])
        @test batch_addable(m, UInt64[])
        @test !batch_addable(m, UInt64[10])
        @test !batch_addable(m, UInt64[12, 12])
    end

    @testset "first_invalid_coord" begin
        clean = ones(Float32, dim * 3)
        @test first_invalid_coord(clean, dim) === nothing
        bad = copy(clean)
        bad[2 * dim + 6] = NaN32
        r = first_invalid_coord(bad, dim)
        @test r.vector_index == 3
        @test r.coord_index == 6
        @test isnan(r.value)
        huge = copy(clean)
        huge[1] = 1.0f16
        @test first_invalid_coord(huge, dim).value == 1.0f16
        @test_throws ArgumentError first_invalid_coord(clean, dim + 1)
    end

    @testset "state probes and constants" begin
        @test packed_ready(idx)
        @test slots_ready(IdMapIndex(dim, bits))
        @test MIN_INPUT_NORM == 1.0f-10
        @test MIN_CALIBRATION_ROWS == 2
        @test RECOMMENDED_CALIBRATION_ROWS == 1000
    end
end
