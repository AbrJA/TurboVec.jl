# Port of turbovec's tests/swap_remove.rs.

@testset "swap_remove" begin
    dim = 128

    @testset "shrinks length and returns last index" begin
        X = unit_rows(MersenneTwister(0xDE1E7E00), 10, dim)
        idx = TurboQuantIndex(dim, 4)
        add!(idx, X)
        @test length(idx) == 10
        moved_from = swap_remove!(idx, 4)
        @test moved_from == 10
        @test length(idx) == 9
    end

    @testset "last is no swap" begin
        X = unit_rows(MersenneTwister(0xDE1E7E01), 5, dim)
        idx = TurboQuantIndex(dim, 4)
        add!(idx, X)
        @test swap_remove!(idx, 5) == 5
        @test length(idx) == 4
    end

    @testset "search after swap_remove reflects the new layout" begin
        n = 100
        X = unit_rows(MersenneTwister(0xDE1E7E02), n, dim)
        idx = TurboQuantIndex(dim, 4)
        add!(idx, X)
        _, i = search(idx, X[6:6, :], 1)
        @test i[1, 1] == 6
        moved_from = swap_remove!(idx, 6)
        @test moved_from == n
        @test length(idx) == n - 1
        _, i2 = search(idx, X[n:n, :], 1)
        @test i2[1, 1] == 6
    end

    @testset "deleted vector no longer returned as top-1" begin
        n = 64
        X = unit_rows(MersenneTwister(0xDE1E7E03), n, dim)
        idx = TurboQuantIndex(dim, 4)
        add!(idx, X)
        swap_remove!(idx, 8)
        _, i = search(idx, X[8:8, :], length(idx))
        @test size(i, 2) == length(idx)
        @test !(i[1, 1] == 8)
    end

    @testset "remaining vectors still self-query correctly" begin
        n = 80
        X = unit_rows(MersenneTwister(0xDE1E7E04), n, dim)
        idx = TurboQuantIndex(dim, 4)
        add!(idx, X)
        live_at_slot = collect(1:n)
        for to_delete in (11, 6, 41, 1)
            last = length(live_at_slot)
            swap_remove!(idx, to_delete)
            live_at_slot[to_delete], live_at_slot[last] =
                live_at_slot[last], live_at_slot[to_delete]
            pop!(live_at_slot)
        end
        for (slot, orig) in enumerate(live_at_slot)
            _, i = search(idx, X[orig:orig, :], 1)
            @test i[1, 1] == slot
        end
    end

    @testset "out of bounds errors" begin
        X = unit_rows(MersenneTwister(0xDE1E7E05), 3, dim)
        idx = TurboQuantIndex(dim, 4)
        add!(idx, X)
        @test_throws BoundsError swap_remove!(idx, 4)
        @test_throws BoundsError swap_remove!(idx, 0)
    end
end
