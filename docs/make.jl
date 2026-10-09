using Documenter, TurboVec

makedocs(modules = [TurboVec],
         sitename = "TurboVec.jl",
         format = Documenter.HTML(),
         pages = ["Home" => "index.md"],
         )

deploydocs(
    repo = "github.com/AbrJA/TurboVec.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing,
    push_preview = true,
)
