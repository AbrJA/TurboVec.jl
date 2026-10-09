# SIMD kernel parity: every vectorized kernel must produce exactly the
# scores the scalar kernel does, for every bit width and layout geometry
# (including partial tail blocks and the 256-group flush boundaries).

# Build the scalar kernel's full input from a prepared LUT plus the
# combined table (SIMD hosts do not build `comb` in `_prepare_lut`).
function scalar_topk(idx, prep, comb, bfirst, blast, mask, k)
    prep_s = TurboVec.PreparedLut(prep.table, comb, prep.scale, prep.bias, prep.ng)
    h = TurboVec.TopK(k)
    TurboVec._scan_blocks_scalar!(h, prep_s, TurboVec.ScanCtx(idx), bfirst, blast,
                                  Vector{Int32}(undef, 32), mask)
    h
end

# One single-block scan kernel vs the scalar reference.
function block_parity(scan_kernel!, idx, prep, comb, ctx, k = 10)
    h1 = scalar_topk(idx, prep, comb, 0, idx.n_blocks, nothing, k)
    h2 = TurboVec.TopK(k)
    scan_kernel!(h2, prep, ctx, 0, idx.n_blocks, nothing,
                 Vector{Float32}(undef, 64))
    @test TurboVec.sorted_results(h1) == TurboVec.sorted_results(h2)
end

# Two-query pair kernel vs two scalar scans.
function pair_parity(pair_kernel!, pidx, prepA, prepB, combA, combB, masks)
    pctx = TurboVec.ScanCtx(pidx)
    for mask in masks
        hA = TurboVec.TopK(20)
        hB = TurboVec.TopK(20)
        pair_kernel!(hA, hB, prepA, prepB, pctx, pidx.n_blocks, mask,
                     Vector{Float32}(undef, 64), Vector{Float32}(undef, 64))
        rA = scalar_topk(pidx, prepA, combA, 0, pidx.n_blocks, mask, 20)
        rB = scalar_topk(pidx, prepB, combB, 0, pidx.n_blocks, mask, 20)
        @test TurboVec.sorted_results(hA) == TurboVec.sorted_results(rA)
        @test TurboVec.sorted_results(hB) == TurboVec.sorted_results(rB)
    end
end

# Pair-kernel geometries: no flush (ng < 256), flush + remainder
# (ng = 384), and flush exactly on the last group (ng = 512).
function pair_geometry_parity(pair_kernel!, rng)
    for (dim, bits, n) in ((128, 4, 130), (768, 4, 65), (2048, 2, 65))
        pidx = TurboQuantIndex(dim, bits)
        P = rand_rows(rng, n, dim)
        add!(pidx, P)
        qA = P[3, :]
        qB = P[5, :]
        prepA = TurboVec._prepare_lut(pidx, qA, zeros(Float32, dim),
                                      zeros(Float32, dim))
        prepB = TurboVec._prepare_lut(pidx, qB, zeros(Float32, dim),
                                      zeros(Float32, dim))
        combA = TurboVec._build_comb(prepA.table, prepA.ng)
        combB = TurboVec._build_comb(prepB.table, prepB.ng)
        m = falses(n)
        m[1:3:end] .= true
        pair_parity(pair_kernel!, pidx, prepA, prepB, combA, combB,
                    (nothing, TurboVec.pack_mask(m)))
    end
end

# Batch search pairs two queries per code pass on every SIMD host; the
# paired path must equal per-query searches, masked and not.
function batch_pairing_parity(rng)
    idx = TurboQuantIndex(128, 4)
    X = rand_rows(rng, 130, 128)
    add!(idx, X)
    for nq in (7, 8)
        Q = rand_rows(rng, nq, 128)
        s, i = search(idx, Q, 10)
        for r in 1:nq
            sr, ir = search(idx, view(Q, r:r, :), 10)
            @test s[r, :] == sr[1, :]
            @test i[r, :] == ir[1, :]
        end
        m = falses(130)
        m[1:5:end] .= true
        sm, im = search(idx, Q, 6; mask = m)
        for r in 1:nq
            sr, ir = search(idx, view(Q, r:r, :), 6; mask = m)
            @test sm[r, :] == sr[1, :]
            @test im[r, :] == ir[1, :]
        end
    end
end

# Single-kernel parity for one query per (bits, dim) geometry, plus a
# masked scan.
function single_geometry_parity(scan_kernel!, rng)
    for bits in (2, 3, 4), dim in (32, 64, 256)
        n = 100
        X = rand_rows(rng, n, dim)
        idx = TurboQuantIndex(dim, bits)
        add!(idx, X)
        ctx = TurboVec.ScanCtx(idx)
        for qi in 1:4
            qrow = X[qi, :]
            q = Vector{Float32}(undef, dim)
            scratch = Vector{Float32}(undef, dim)
            prep = TurboVec._prepare_lut(idx, qrow, q, scratch)
            comb = TurboVec._build_comb(prep.table, prep.ng)
            block_parity(scan_kernel!, idx, prep, comb, ctx)
        end
    end

    idx = TurboQuantIndex(128, 4)
    X = rand_rows(rng, 130, 128)
    add!(idx, X)
    ctx = TurboVec.ScanCtx(idx)
    mask = falses(130)
    mask[1:7:end] .= true
    pmask = TurboVec.pack_mask(mask)
    prep = TurboVec._prepare_lut(idx, X[1, :], zeros(Float32, 128),
                                 zeros(Float32, 128))
    comb = TurboVec._build_comb(prep.table, prep.ng)
    h1 = scalar_topk(idx, prep, comb, 0, idx.n_blocks, pmask, 20)
    h2 = TurboVec.TopK(20)
    scan_kernel!(h2, prep, ctx, 0, idx.n_blocks, pmask,
                 Vector{Float32}(undef, 64))
    @test TurboVec.sorted_results(h1) == TurboVec.sorted_results(h2)
end

@testset "simd kernel parity" begin
    rng = MersenneTwister(0x51D)
    if TurboVec.HAS_AVX2
        single_geometry_parity(TurboVec._scan_blocks_avx2!, rng)
        if TurboVec.HAS_AVX512BW
            single_geometry_parity(TurboVec._scan_blocks_avx512!, rng)
            # a range starting on an odd block exercises the tail path
            idx = TurboQuantIndex(64, 4)
            X = rand_rows(rng, 100, 64)
            add!(idx, X)
            prep = TurboVec._prepare_lut(idx, X[1, :], zeros(Float32, 64),
                                         zeros(Float32, 64))
            comb = TurboVec._build_comb(prep.table, prep.ng)
            ctx = TurboVec.ScanCtx(idx)
            h4 = TurboVec.TopK(10)
            TurboVec._scan_blocks_avx512!(h4, prep, ctx, 1, idx.n_blocks, nothing,
                                          Vector{Float32}(undef, 64))
            h5 = scalar_topk(idx, prep, comb, 1, idx.n_blocks, nothing, 10)
            @test TurboVec.sorted_results(h4) == TurboVec.sorted_results(h5)
        end
        pair_geometry_parity(TurboVec._scan_two_avx2!, rng)
        batch_pairing_parity(rng)
    elseif TurboVec.HAS_NEON
        single_geometry_parity(TurboVec._scan_blocks_neon!, rng)
        pair_geometry_parity(TurboVec._scan_two_neon!, rng)
        batch_pairing_parity(rng)
    else
        @info "no SIMD kernels available; scalar kernel only"
    end
end
