@testset "persistence" begin
    rng = MersenneTwister(11)
    dim, n = 64, 150
    X = rand_rows(rng, n, dim)
    Q = rand_rows(rng, 4, dim)

    mktempdir() do dir
        for bits in (2, 3, 4)
            path = joinpath(dir, "idx_$bits.tv")
            idx = TurboQuantIndex(dim, bits)
            add!(idx, X)
            write_index(path, idx)
            loaded = load_index(path)
            @test length(loaded) == n
            @test loaded.bit_width == bits
            @test loaded.centroids == idx.centroids
            s1, i1 = search(idx, Q, 5)
            s2, i2 = search(loaded, Q, 5)
            @test reinterpret.(UInt32, s1) == reinterpret.(UInt32, s2)
            @test i1 == i2

            cpath = joinpath(dir, "idxc_$bits.tv")
            cidx = TurboQuantIndex(dim, bits)
            calibrate!(cidx, X)
            add!(cidx, X)
            write_index(cpath, cidx)
            cloaded = load_index(cpath)
            @test calibration_state(cloaded) == :calibrated
            cs1, ci1 = search(cidx, Q, 5)
            cs2, ci2 = search(cloaded, Q, 5)
            @test reinterpret.(UInt32, cs1) == reinterpret.(UInt32, cs2)
            @test ci1 == ci2
        end

        # idmap round-trip
        ids = UInt64.(5000:(5000 + n - 1))
        imap = IdMapIndex(dim, 4)
        add_with_ids!(imap, X, ids)
        ipath = joinpath(dir, "map.tvim")
        write_idmap(ipath, imap)
        iloaded = load_idmap(ipath)
        @test length(iloaded) == n
        s1, g1 = search(imap, Q, 4)
        s2, g2 = search(iloaded, Q, 4)
        @test s1 == s2 && g1 == g2

        # wrong-kind loads are rejected
        @test_throws InvalidFileFormat load_idmap(joinpath(dir, "idx_4.tv"))
        @test_throws InvalidFileFormat load_index(ipath)

        # truncated file is rejected rather than silently accepted
        bad = joinpath(dir, "bad.tv")
        write(bad, read(joinpath(dir, "idx_4.tv"))[1:20])
        @test_throws Exception load_index(bad)

        # empty index with committed dim round-trips
        epath = joinpath(dir, "empty.tv")
        eidx = TurboQuantIndex(dim, 2)
        write_index(epath, eidx)
        eloaded = load_index(epath)
        @test length(eloaded) == 0
        @test TurboVec.dim_opt(eloaded) == dim
    end
end
