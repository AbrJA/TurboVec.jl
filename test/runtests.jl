using Test
using Random
using TurboVec

Random.seed!(20241002)

include("helpers.jl")

include("test_codebook.jl")
include("test_codebook_determinism.jl")
include("test_rotation.jl")
include("test_rotation_determinism.jl")
include("test_rust_golden.jl")
include("test_index.jl")
include("test_kernel_correctness.jl")
include("test_simd.jl")
include("test_query_scale_invariance.jl")
include("test_concurrent_search.jl")
include("test_swap_remove.jl")
include("test_lazy_init.jl")
include("test_filtering.jl")
include("test_idmap.jl")
include("test_state_sequences.jl")
include("test_io.jl")
include("test_bytes_io.jl")
include("test_from_parts.jl")
include("test_calibration_bounds.jl")
include("test_crate_api.jl")
include("test_api_parity.jl")
include("test_interface.jl")
include("test_recall.jl")
