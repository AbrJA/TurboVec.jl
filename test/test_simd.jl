# AVX2 scan kernel parity: the vectorized kernel must produce exactly
# the scores the scalar kernel does, for every bit width and layout
# geometry (including partial tail blocks).

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

            for qi in 1:4
                qrow = X[qi, :]
                q = Vector{Float32}(undef, dim)
                scratch = Vector{Float32}(undef, dim)
                prep = TurboVec._prepare_lut(idx, qrow, q, scratch)

                h1 = TurboVec.TopK(10)
                TurboVec._scan_blocks_scalar!(
                    h1, prep.comb, idx.codes, idx.scales, prep.ng, idx.n,
                    prep.bias, prep.scale, 0, idx.n_blocks,
                    Vector{Int32}(undef, 32), nothing)
                h2 = TurboVec.TopK(10)
                TurboVec._scan_blocks_avx2!(
                    h2, prep.table, idx.codes, idx.scales, prep.ng, idx.n,
                    prep.bias, prep.scale, 0, idx.n_blocks, nothing)

                s1, i1 = TurboVec.sorted_results(h1)
                s2, i2 = TurboVec.sorted_results(h2)
                @test s1 == s2
                @test i1 == i2

                if TurboVec.HAS_AVX512BW
                    h3 = TurboVec.TopK(10)
                    TurboVec._scan_blocks_avx512!(
                        h3, prep.table, idx.codes, idx.scales, prep.ng, idx.n,
                        prep.bias, prep.scale, 0, idx.n_blocks, nothing)
                    @test TurboVec.sorted_results(h3) == (s1, i1)
                    # range starting on an odd block exercises the tail path
                    if idx.n_blocks > 1
                        h4 = TurboVec.TopK(10)
                        TurboVec._scan_blocks_avx512!(
                            h4, prep.table, idx.codes, idx.scales, prep.ng, idx.n,
                            prep.bias, prep.scale, 1, idx.n_blocks, nothing)
                        h5 = TurboVec.TopK(10)
                        TurboVec._scan_blocks_scalar!(
                            h5, prep.comb, idx.codes, idx.scales, prep.ng, idx.n,
                            prep.bias, prep.scale, 1, idx.n_blocks,
                            Vector{Int32}(undef, 32), nothing)
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
        mask = falses(130)
        mask[1:7:end] .= true
        pmask = TurboVec.pack_mask(mask)
        prep = TurboVec._prepare_lut(idx, X[1, :], zeros(Float32, 128),
                                     zeros(Float32, 128))
        h1 = TurboVec.TopK(20)
        TurboVec._scan_blocks_scalar!(h1, prep.comb, idx.codes, idx.scales,
                                      prep.ng, idx.n, prep.bias, prep.scale,
                                      0, idx.n_blocks,
                                      Vector{Int32}(undef, 32), pmask)
        h2 = TurboVec.TopK(20)
        TurboVec._scan_blocks_avx2!(h2, prep.table, idx.codes, idx.scales,
                                    prep.ng, idx.n, prep.bias, prep.scale,
                                    0, idx.n_blocks, pmask)
        @test TurboVec.sorted_results(h1) == TurboVec.sorted_results(h2)
    end
end
