# Package-quality checks (Aqua): ambiguities, undefined exports,
# unbound type parameters, type piracy, stale dependencies, compat
# bounds, and project/test-target hygiene.

using Aqua

@testset "code quality (Aqua)" begin
    Aqua.test_all(TurboVec; ambiguities = (; recursive = true))
end
