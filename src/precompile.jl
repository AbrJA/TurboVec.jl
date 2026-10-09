# Precompile workload: exercise the common entry points during package
# precompilation so the first `add!`/`search` in a session does not pay
# for inference and native codegen. The `llvmcall` scan kernels are part
# of these method bodies, so their native code is cached in the package
# image too.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    X = Matrix{Float32}(undef, 64, 32)
    for i in eachindex(X)
        X[i] = (i % 97) / 97.0f0 - 0.5f0
    end
    Q = X[1:4, :]
    ids = UInt64.(1:64)

    @compile_workload begin
        idx = TurboQuantIndex(32, 4)
        add!(idx, X)
        search(idx, Q, 5)
        search(idx, Q, 5; mask = trues(64))
        swap_remove!(idx, 3)

        cidx = TurboQuantIndex(32, 2)
        calibrate!(cidx, X)
        add!(cidx, X)
        search(cidx, Q, 5)

        m = IdMapIndex(32, 4)
        add_with_ids!(m, X, ids)
        search(m, Q, 5)
        search(m, Q, 5; allowlist = ids[1:3])
        remove!(m, ids[1])

        buf = to_bytes(m)
        from_bytes(IdMapIndex, buf)
        from_bytes(TurboQuantIndex, to_bytes(idx))
    end
end
