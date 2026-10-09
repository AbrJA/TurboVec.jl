# Port of turbovec's tests/lazy_init.rs: dim inference, no-op empty
# adds, and lazy round-trips.

@testset "lazy init" begin
    rng = MersenneTwister(7)

    @testset "new_lazy starts with no dim" begin
        idx = TurboQuantIndex(4)
        @test dim_opt(idx) === nothing
        @test dim(idx) == 0
        @test isempty(idx)
        @test_throws BitWidthOutOfRange TurboQuantIndex(0)
        @test_throws BitWidthOutOfRange TurboQuantIndex(5)
    end

    @testset "add locks the dim" begin
        idx = TurboQuantIndex(4)
        X = rand_rows(rng, 5, 32)
        add!(idx, X)
        @test dim_opt(idx) == 32
        @test_throws DimMismatch add!(idx, rand_rows(rng, 2, 64))
        add!(idx, rand_rows(rng, 2, 32))
        @test length(idx) == 7
    end

    @testset "zero-row add does not commit and still validates dim" begin
        idx = TurboQuantIndex(4)
        add!(idx, zeros(Float32, 0, 32))
        @test dim_opt(idx) === nothing
        @test isempty(idx)

        idx2 = TurboQuantIndex(32, 4)
        add!(idx2, zeros(Float32, 0, 32))
        @test length(idx2) == 0
        @test_throws DimMismatch add!(idx2, zeros(Float32, 0, 64))

        m = IdMapIndex(4)
        add_with_ids!(m, zeros(Float32, 0, 32), UInt64[])
        @test dim_opt(m) === nothing
    end

    @testset "search on lazy uncommitted returns empty" begin
        idx = TurboQuantIndex(4)
        s, i = search(idx, zeros(Float32, 3, 32), 5)
        @test size(s) == (3, 0) && size(i) == (3, 0)
        m = IdMapIndex(4)
        s, i = search(m, zeros(Float32, 3, 32), 5)
        @test size(s) == (3, 0) && size(i) == (3, 0)
    end

    @testset "lazy round trips" begin
        mktempdir() do dir
            path = joinpath(dir, "lazy.tv")
            idx = TurboQuantIndex(3)
            write_index(path, idx)
            loaded = load_index(path)
            @test dim_opt(loaded) === nothing
            @test isempty(loaded)

            path2 = joinpath(dir, "lazy.tvim")
            m = IdMapIndex(2)
            write_idmap(path2, m)
            mloaded = load_idmap(path2)
            @test dim_opt(mloaded) === nothing
            @test isempty(mloaded)
        end
    end

    @testset "lazy after committed add" begin
        mktempdir() do dir
            path = joinpath(dir, "committed.tv")
            idx = TurboQuantIndex(4)
            X = rand_rows(rng, 4, 16)
            add!(idx, X)
            write_index(path, idx)
            loaded = load_index(path)
            @test dim_opt(loaded) == 16
            @test length(loaded) == 4
        end
    end

    @testset "prepare is idempotent and does not change results" begin
        idx = TurboQuantIndex(4)
        X = rand_rows(rng, 20, 32)
        add!(idx, X)
        prepare(idx)
        s1, i1 = search(idx, X[1:1, :], 3)
        prepare(idx)
        s2, i2 = search(idx, X[1:1, :], 3)
        @test s1 == s2 && i1 == i2
    end
end
