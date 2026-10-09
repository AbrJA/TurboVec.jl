@testset "codebook" begin
    for bits in 2:4, dim in (8, 128, 200, 768, 1536)
        b, c = codebook(bits, dim)
        @test length(b) == (1 << bits) - 1
        @test length(c) == 1 << bits
        @test issorted(b)
        @test issorted(c)
        for i in eachindex(b)
            @test reinterpret(UInt32, b[i]) ==
                  reinterpret(UInt32, (c[i] + c[i + 1]) * 0.5f0)
        end
    end

    # memoized: identical arrays, no re-solve
    b1, c1 = codebook(4, 768)
    b2, c2 = codebook(4, 768)
    @test b1 == b2 && c1 == c2

    @test_throws ArgumentError codebook(1, 768)
    @test_throws ArgumentError codebook(5, 768)
    @test_throws ArgumentError codebook(2, 0)
    @test_throws ArgumentError codebook(2, 12)
    @test_throws ArgumentError codebook(2, MAX_DIM + 8)

    # centroids are symmetric around zero for the symmetric Beta
    _, c = codebook(4, 200)
    @test maximum(abs.(c .+ reverse(c))) < 1e-5
end
