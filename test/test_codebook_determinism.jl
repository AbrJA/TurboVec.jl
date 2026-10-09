# Port of turbovec's tests/codebook_determinism.rs: the published
# codebook bytes are frozen, and boundaries are exact f32 midpoints of
# the published centroids.

@testset "codebook determinism" begin
    @testset "boundaries are f32 midpoints of the published centroids" begin
        for bits in 2:4, dim in (8, 200, 768)
            b, c = codebook(bits, dim)
            for i in eachindex(b)
                @test reinterpret(UInt32, b[i]) ==
                      reinterpret(UInt32, (c[i] + c[i + 1]) * 0.5f0)
            end
        end
    end

    @testset "codebook bytes are frozen" begin
        # Rust turbovec 1.0 `expected_codebook`, bit patterns.
        b2, c2 = codebook(2, 200)
        @test reinterpret.(UInt32, c2) ==
              UInt32[0xbdda3f5c, 0xbd03142b, 0x3d03142b, 0x3dda3f5c]
        @test reinterpret.(UInt32, b2) == UInt32[0xbd8de4b9, 0x00000000, 0x3d8de4b9]

        b4, c4 = codebook(4, 768)
        @test reinterpret.(UInt32, c4) ==
              UInt32[0xbdc964ee, 0xbd989d1e, 0xbd6ecea6, 0xbd3975fa,
                     0xbd0b25c0, 0xbcc1fb3c, 0xbc653f5c, 0xbb97b5b7,
                     0x3b97b5b7, 0x3c653f5c, 0x3cc1fb3c, 0x3d0b25c0,
                     0x3d3975fa, 0x3d6ecea6, 0x3d989d1e, 0x3dc964ee]
        @test reinterpret.(UInt32, b4) ==
              UInt32[0xbdb10106, 0xbd880238, 0xbd542250, 0xbd224ddd,
                     0xbcec235e, 0xbc9a4d75, 0xbc188d1c, 0x00000000,
                     0x3c188d1c, 0x3c9a4d75, 0x3cec235e, 0x3d224ddd,
                     0x3d542250, 0x3d880238, 0x3db10106]
    end
end
