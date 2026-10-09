@testset "rotation" begin
    for dim in (8, 64, 200, 768, 1536)
        rot = TurboVec.Rotation(dim)

        # deterministic construction
        rot2 = TurboVec.Rotation(dim)
        @test rot.signs == rot2.signs
        @test rot.perms == rot2.perms

        # preserves norms (orthogonal transform, f32 rounding)
        rng = MersenneTwister(dim)
        row = rand(rng, Float32, dim) .- 0.5f0
        orig_norm = sqrt(sum(abs2, row))
        r = copy(row)
        TurboVec.apply_rotation!(rot, r)
        @test isapprox(sqrt(sum(abs2, r)), orig_norm; rtol = 1e-4)

        # preserves inner products between two rows
        row2 = rand(rng, Float32, dim) .- 0.5f0
        ip_orig = sum(row .* row2)
        r2 = copy(row2)
        TurboVec.apply_rotation!(rot, r2)
        @test isapprox(sum(r .* r2), ip_orig; rtol = 1e-3, atol = 1e-4)

        # in-place apply equals scaled-into with inv = 1
        dst = similar(row)
        scratch = similar(row)
        TurboVec.apply_scaled_into!(rot, row, 1.0f0, dst, scratch)
        r3 = copy(row)
        TurboVec.apply_rotation!(rot, r3)
        @test reinterpret.(UInt32, dst) == reinterpret.(UInt32, r3)
    end

    @test_throws ArgumentError TurboVec.Rotation(0)
    @test_throws ArgumentError TurboVec.Rotation(12)
    @test_throws ArgumentError TurboVec.Rotation(MAX_DIM + 8)
end
