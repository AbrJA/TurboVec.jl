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

    # A second rotation geometry (768 = 3 × 256) compiles the other WHT
    # radix shapes.
    XW = Matrix{Float32}(undef, 8, 768)
    for i in eachindex(XW)
        XW[i] = (i % 89) / 89.0f0 - 0.5f0
    end

    @compile_workload begin
        idx = TurboQuantIndex(32, 4)
        add!(idx, X)
        search(idx, Q, 5)
        search(idx, Q, 5; mask = trues(64))
        search(idx, X[1, :], 5)
        blocked_codes(idx)
        from_parts(32, 4, 64, packed_codes(idx), copy(scales(idx)))
        swap_remove!(idx, 3)

        # 3-bit code path
        idx3 = TurboQuantIndex(32, 3)
        add!(idx3, X)
        search(idx3, Q, 5)

        # lazy index commits its dim on the first add
        lazyidx = TurboQuantIndex(2)
        add!(lazyidx, X)
        search(lazyidx, Q, 5)

        cidx = TurboQuantIndex(32, 2)
        calibrate!(cidx, X)
        add!(cidx, X)
        search(cidx, Q, 5)

        # wide-dim encode/search (other WHT radix shapes)
        widx = TurboQuantIndex(768, 4)
        add!(widx, XW)
        search(widx, XW[1:2, :], 5)

        m = IdMapIndex(32, 4)
        add_with_ids!(m, X, ids)
        search(m, Q, 5)
        search(m, Q, 5; allowlist = ids[1:3])
        remove!(m, ids[1])

        buf = to_bytes(m)
        from_bytes(IdMapIndex, buf)
        from_bytes(TurboQuantIndex, to_bytes(idx))

        # error path: the throw branch is compiled, not executed
        try
            add!(idx, fill(NaN32, 1, 32))
        catch
        end
    end
end
