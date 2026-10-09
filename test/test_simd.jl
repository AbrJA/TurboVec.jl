# AVX2 scan kernel parity: the vectorized kernels must produce exactly
# the scores the scalar kernel does, for every bit width and layout
# geometry (including partial tail blocks).

# Build the scalar kernel's full input from a prepared LUT plus the
# combined table (AVX2 hosts do not build `comb` in `_prepare_lut`).
function scalar_topk(idx, prep, comb, bfirst, blast, mask, k)
    prep_s = TurboVec.PreparedLut(prep.table, comb, prep.scale, prep.bias, prep.ng)
    h = TurboVec.TopK(k)
    TurboVec._scan_blocks_scalar!(h, prep_s, TurboVec.ScanCtx(idx), bfirst, blast,
                                  Vector{Int32}(undef, 32), mask)
    h
end

@testset "simd kernel parity" begin
    if !TurboVec.HAS_AVX2
        @info "AVX2 not available; scalar kernel only"
    else
        rng = MersenneTwister(0x51D)
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

                h1 = scalar_topk(idx, prep, comb, 0, idx.n_blocks, nothing, 10)
                h2 = TurboVec.TopK(10)
                TurboVec._scan_blocks_avx2!(h2, prep, ctx, 0, idx.n_blocks, nothing,
                                            Vector{Float32}(undef, 64))

                s1, i1 = TurboVec.sorted_results(h1)
                s2, i2 = TurboVec.sorted_results(h2)
                @test s1 == s2
                @test i1 == i2

                if TurboVec.HAS_AVX512BW
                    h3 = TurboVec.TopK(10)
                    TurboVec._scan_blocks_avx512!(h3, prep, ctx, 0, idx.n_blocks,
                                                  nothing, Vector{Float32}(undef, 64))
                    @test TurboVec.sorted_results(h3) == (s1, i1)
                    # range starting on an odd block exercises the tail path
                    if idx.n_blocks > 1
                        h4 = TurboVec.TopK(10)
                        TurboVec._scan_blocks_avx512!(h4, prep, ctx, 1, idx.n_blocks,
                                                      nothing,
                                                      Vector{Float32}(undef, 64))
                        h5 = scalar_topk(idx, prep, comb, 1, idx.n_blocks, nothing, 10)
                        @test TurboVec.sorted_results(h4) ==
                              TurboVec.sorted_results(h5)
                    end
                end
            end
        end

        # Paired two-query batches (odd and even counts, masked and not)
        # must equal per-query searches.
        if TurboVec.HAS_AVX512BW
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

        # Masked scans agree too.
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
        TurboVec._scan_blocks_avx2!(h2, prep, ctx, 0, idx.n_blocks, pmask,
                                    Vector{Float32}(undef, 64))
        @test TurboVec.sorted_results(h1) == TurboVec.sorted_results(h2)

        # The AVX2 two-query pair kernel must equal two scalar scans for
        # every geometry: no flush (ng < 256), flush + remainder (ng = 384),
        # and flush exactly on the last group (ng = 512), masked or not.
        for (dim, bits, n) in ((128, 4, 130), (768, 4, 65), (2048, 2, 65))
            pidx = TurboQuantIndex(dim, bits)
            P = rand_rows(rng, n, dim)
            add!(pidx, P)
            pctx = TurboVec.ScanCtx(pidx)
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
            for mask in (nothing, TurboVec.pack_mask(m))
                hA = TurboVec.TopK(20)
                hB = TurboVec.TopK(20)
                TurboVec._scan_two_avx2!(hA, hB, prepA, prepB, pctx, pidx.n_blocks,
                                         mask, Vector{Float32}(undef, 64),
                                         Vector{Float32}(undef, 64))
                rA = scalar_topk(pidx, prepA, combA, 0, pidx.n_blocks, mask, 20)
                rB = scalar_topk(pidx, prepB, combB, 0, pidx.n_blocks, mask, 20)
                @test TurboVec.sorted_results(hA) == TurboVec.sorted_results(rA)
                @test TurboVec.sorted_results(hB) == TurboVec.sorted_results(rB)
            end
        end
    end
end
