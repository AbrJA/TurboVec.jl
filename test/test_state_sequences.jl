# Port of turbovec's tests/state_sequences.rs: interleaved mutations and
# persistence sequences stay consistent.

@testset "state sequences" begin
    dim = 64
    rng = MersenneTwister(0x5E9)

    @testset "add after load extends the index" begin
        X = rand_rows(rng, 40, dim)
        X2 = rand_rows(rng, 10, dim)
        mktempdir() do dir
            path = joinpath(dir, "idx.tv")
            idx = TurboQuantIndex(dim, 4)
            add!(idx, X)
            write_index(path, idx)
            loaded = load_index(path)
            add!(loaded, X2)
            @test length(loaded) == 50
            # every vector still self-queries
            for r in (1, 40, 45, 50)
                row = r <= 40 ? X[r:r, :] : X2[(r - 40):(r - 40), :]
                _, i = search(loaded, row, 1)
                @test i[1, 1] == r
            end
        end
    end

    @testset "add, swap-remove, add: all phases findable" begin
        X = rand_rows(rng, 30, dim)
        Y = rand_rows(rng, 5, dim)
        Z = rand_rows(rng, 7, dim)
        idx = TurboQuantIndex(dim, 4)
        add!(idx, X)
        swap_remove!(idx, 4)          # last of X moves into slot 4
        add!(idx, Y)                  # slots 30..34
        add!(idx, Z)                  # slots 35..41
        @test length(idx) == 41
        # A phase-Y vector is at its expected slot.
        _, i = search(idx, Y[2:2, :], 1)
        @test i[1, 1] == 31
        # A phase-Z vector is at its expected slot.
        _, i2 = search(idx, Z[6:6, :], 1)
        @test i2[1, 1] == 40
        # The vector that moved into slot 4 self-queries there.
        _, i3 = search(idx, X[30:30, :], 1)
        @test i3[1, 1] == 4
    end

    @testset "id-map re-added id returns the new vector" begin
        m = IdMapIndex(dim, 4)
        A = rand_rows(MersenneTwister(1), 1, dim)
        B = rand_rows(MersenneTwister(2), 1, dim)
        add_with_ids!(m, A, UInt64[42])
        @test remove!(m, 42)
        add_with_ids!(m, B, UInt64[42])
        @test contains_id(m, 42)
        _, got = search(m, B, 1)
        @test got[1, 1] == 42
        sA, _ = search(m, A, 1)
        sB, _ = search(m, B, 1)
        @test sB[1, 1] > sA[1, 1]
    end

    @testset "id-map remove-last then add keeps tables consistent" begin
        n = 12
        X = rand_rows(rng, n, dim)
        ids = UInt64.(100:(100 + n - 1))
        m = IdMapIndex(dim, 4)
        add_with_ids!(m, X, ids)
        @test remove!(m, ids[12])
        @test length(m) == 11
        Y = rand_rows(rng, 3, dim)
        add_with_ids!(m, Y, UInt64[500, 501, 502])
        @test length(m) == 14
        for (slot, id) in enumerate(external_ids(m))
            row = if id in UInt64[500, 501, 502]
                Y[findfirst(isequal(id), UInt64[500, 501, 502]):findfirst(isequal(id),
                                                                          UInt64[500, 501,
                                                                                 502]), :]
            else
                X[findfirst(isequal(id), ids):findfirst(isequal(id), ids), :]
            end
            _, got = search(m, row, 1)
            @test got[1, 1] == id
        end
    end

    @testset "refit keeps every row searchable" begin
        X = rand_rows(rng, 100, dim)
        idx = TurboQuantIndex(dim, 2)
        calibrate!(idx, X)
        add!(idx, X)
        calibrate!(idx, X[1:32, :])
        hits = 0
        for r in 1:100
            _, i = search(idx, X[r:r, :], 5)
            hits += (r in i[1, :])
        end
        @test hits >= 95
    end
end
