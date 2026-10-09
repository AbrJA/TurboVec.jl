# Opt-in static analysis of the public entry points:
#
#     julia --project=. -e 'using Pkg; Pkg.test(; test_args = ["jet"])'
#
# JET runs with `ignore_throws` (Julia exceptions are normal control flow)
# and `target_modules` limited to TurboVec, so Base-internal reports do
# not leak into the gate. Every call must be report-free.

using JET

@testset "JET public entry points" begin
    calls = [(TurboVec.TurboQuantIndex, (Int, Int)),
             (TurboVec.TurboQuantIndex, (Int,)),
             (TurboVec.add!, (TurboQuantIndex, Matrix{Float32})),
             (TurboVec.calibrate!, (TurboQuantIndex, Matrix{Float32})),
             (TurboVec.search, (TurboQuantIndex, Matrix{Float32}, Int)),
             (TurboVec.search, (IdMapIndex, Matrix{Float32}, Int)),
             (TurboVec.swap_remove!, (TurboQuantIndex, Int)),
             (TurboVec.from_parts, (Int, Int, Int, Vector{UInt8}, Vector{Float32})),
             (TurboVec.packed_codes, (TurboQuantIndex,)),
             (TurboVec.blocked_codes, (TurboQuantIndex,)),
             (TurboVec.add_with_ids!, (IdMapIndex, Matrix{Float32}, Vector{UInt64})),
             (TurboVec.remove!, (IdMapIndex, UInt64)),
             (TurboVec.is_addable, (IdMapIndex, Vector{UInt64})),
             (TurboVec.contains_id, (IdMapIndex, UInt64)),
             (TurboVec.to_bytes, (TurboQuantIndex,)),
             (TurboVec.from_bytes, (Type{TurboQuantIndex}, Vector{UInt8})),
             (TurboVec.write_index, (IOBuffer, TurboQuantIndex)),
             (TurboVec.load_index, (IOBuffer,)),
             (TurboVec.codebook, (Int, Int)),
             (TurboVec.first_invalid_coord, (Vector{Float32}, Int))]
    for (f, tt) in calls
        reports = JET.get_reports(JET.report_call(f, tt; ignore_throws = true,
                                                  target_modules = (TurboVec,)))
        @test isempty(reports)
        for r in reports
            @error "JET report" call = f report = r
        end
    end
end
