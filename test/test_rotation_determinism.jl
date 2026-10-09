# Port of turbovec's tests/rotation_determinism.rs.
#
# The ChaCha8 stream and the block-Hadamard transform are pinned by
# golden bit patterns (they match the Rust implementation exactly), and
# the encode/search paths must be deterministic run to run and
# independent of how work is distributed across threads.

const GOLDEN_DIM8 = UInt32[
    1071644673, 3224371200, 1048575995, 1061158910, 1074790400,
    1083703296, 3214934014, 1067450369,
]
const GOLDEN_DIM24 = UInt32[
    1081081856, 1088421887, 3204448256, 1076887552, 3221225472, 3224371200,
    1061158914, 1084227583, 1052770317, 1063256064, 1072693248, 1052770303,
    3237216256, 3225944064, 1063256063, 3206545409, 3229089792, 3221749760,
    1085014015, 1040187401, 3225944064, 3222798336, 1040187401, 1072693248,
]
const GOLDEN_DIM64 = UInt32[
    1077149696, 3246260224, 1063256064, 1096482816, 1076625408, 3232497664,
    1081081856, 3244621824, 1089994752, 1093337088, 1096220672, 3229876224,
    3204448256, 1075052544, 1092812800, 3246391296, 1086062592, 1086193664,
    1095172096, 1095761920, 3225419776, 3238592512, 3207593984, 3230138368,
    1071120384, 1088159744, 3200253952, 3240820736, 3209691136, 3223322624,
    1093533696, 1091764224, 3232759808, 1078722560, 3233677312, 3225419776,
    1068498944, 3179282432, 1074528256, 3235905536, 3243048960, 3252027392,
    3233021952, 1091239936, 3221487616, 3236691968, 1087111168, 3224109056,
    1088552960, 1094189056, 3210739712, 1088028672, 1085669376, 3222274048,
    3240361984, 3225157632, 3242328064, 1096482816, 3226206208, 1063256064,
    3234070528, 3243704320, 3234201600, 1083703296,
]

function rotate_bits(dim::Int, input::AbstractVector{Float32})
    v = copy(input)
    TurboVec.apply_rotation!(TurboVec.Rotation(dim), v)
    reinterpret.(UInt32, v)
end

@testset "rotation determinism" begin
    @testset "golden rotation bits pin the ChaCha stream" begin
        input8 = Float32[i - 3.5 for i in 0:7]
        @test rotate_bits(8, input8) == GOLDEN_DIM8

        input24 = Float32[Float32(i * 7 % 11) - 5.0f0 for i in 0:23]
        @test rotate_bits(24, input24) == GOLDEN_DIM24

        input64 = Float32[Float32(i * 13 % 29) - 14.0f0 for i in 0:63]
        @test rotate_bits(64, input64) == GOLDEN_DIM64
    end

    @testset "encoded bytes and search are deterministic across runs" begin
        rng = MersenneTwister(0xBADC0DE)
        X = rand_rows(rng, 2000, 256)
        Q = rand_rows(rng, 8, 256)
        reference = nothing
        for _ in 1:4
            idx = TurboQuantIndex(256, 4)
            add!(idx, X)
            codes = packed_codes(idx)
            sc = copy(scales(idx))
            s, ix = search(idx, Q, 10)
            if reference === nothing
                reference = (codes, sc, s, ix)
            else
                @test codes == reference[1]
                @test reinterpret.(UInt32, sc) == reinterpret.(UInt32, reference[2])
                @test s == reference[3]
                @test ix == reference[4]
            end
        end
    end
end
