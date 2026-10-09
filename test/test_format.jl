# Opt-in formatting check:
#
#     julia --project=. -e 'using Pkg; Pkg.test(; test_args = ["format"])'
#
# The formatter options live in .JuliaFormatter.toml at the package root.

using JuliaFormatter

@testset "formatting" begin
    root = normpath(joinpath(@__DIR__, ".."))
    @test format([joinpath(root, "src"), joinpath(root, "test")]; overwrite = false)
end
