@testset "IdMapIndex" begin
    rng = MersenneTwister(10)
    dim, n = 48, 120
    X = rand_rows(rng, n, dim)
    ids = UInt64.(1000:(1000 + n - 1))

    idx = IdMapIndex(dim, 4)
    add_with_ids!(idx, X, ids)
    @test length(idx) == n

    scores, got = search(idx, X, 3)
    @test size(scores) == (n, 3)
    @test size(got) == (n, 3)
    hits = count(i -> got[i, 1] == ids[i], 1:n)
    @test hits >= n - 2

    @test_throws IdAlreadyPresent add_with_ids!(idx, X[1:1, :], [ids[1]])
    @test_throws DuplicateIdInBatch add_with_ids!(idx, X[1:2, :], UInt64[7, 7])
    @test_throws IdsCountMismatch add_with_ids!(idx, X[1:2, :], UInt64[7])
    @test_throws DimMismatch add_with_ids!(idx, rand_rows(rng, 1, dim + 8), UInt64[7])

    # Ids outside the UInt64 domain raise a typed error, never InexactError.
    @test_throws InvalidIdValue add_with_ids!(idx, X[1:1, :], [-1])
    @test_throws InvalidIdValue add_with_ids!(idx, X[1:1, :], [big(2)^64])
    @test_throws InvalidIdValue add_with_ids!(idx, X[1, :], -5)
    @test_throws InvalidIdValue search(idx, X[1:1, :], 1; allowlist = [-3])
    # ...while predicates and removal treat them as simply absent.
    @test !is_addable(idx, [-1])
    @test !contains_id(idx, -1)
    @test !(-1 in idx)
    @test !remove!(idx, -1)
    @test !remove!(idx, big(2)^64)

    # O(1) remove by id
    @test remove!(idx, ids[5])
    @test !remove!(idx, ids[5])
    @test length(idx) == n - 1
    s2, g2 = search(idx, reshape(X[5, :], 1, dim), 1)
    @test g2[1, 1] != ids[5]

    # the moved last vector keeps its id and is searchable
    s3, g3 = search(idx, reshape(X[n, :], 1, dim), 1)
    @test g3[1, 1] == ids[n]

    # lazy construction
    lazy = IdMapIndex(2)
    add_with_ids!(lazy, X, ids)
    @test TurboVec.dim_opt(lazy) == dim

    empty = IdMapIndex(dim, 4)
    se, ie = search(empty, X[1:1, :], 3)
    @test size(se) == (1, 0) && size(ie) == (1, 0)
end
