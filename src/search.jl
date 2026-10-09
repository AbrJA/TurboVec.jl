# Search: rotate the query, build the u8 nibble LUTs, scan the blocked
# codes with exact integer accumulation, and collect the top-k.
#
# Two scan kernels share the same semantics and produce bit-identical
# scores: a portable scalar kernel (one 256-entry combined table lookup
# per code byte) and an AVX2 kernel that resolves 32 code bytes per
# `vpshufb` against the 16-entry nibble tables. The AVX2 path is chosen
# once per process from the host CPU.

"""Per-query tables and score constants shared by both scan kernels."""
struct PreparedLut
    table::Vector{UInt8}   # 32 nibble-table bytes per group (the AVX2 input)
    comb::Vector{UInt8}    # 256 combined-entry bytes per group (the scalar input)
    scale::Float32
    bias::Float32
    ng::Int
end

"""Index-level scan state shared by every kernel: the blocked codes, the
per-vector scales, the vector count and the byte-group count. Built once
per search; per-query tables and constants live in `PreparedLut`."""
struct ScanCtx
    codes::Vector{UInt8}
    scales::Vector{Float32}
    n::Int
    ng::Int
end

function ScanCtx(index::TurboQuantIndex)
    ScanCtx(index.codes, index.scales, index.n,
            n_byte_groups(index.dim, index.bit_width))
end

@inline function _mask_allows(mask::Vector{UInt64}, slot0::Int)
    @inbounds (mask[(slot0 >> 6) + 1] >> (slot0 & 63)) & 0x1 != 0x0
end

@inline function _mask_block_allows(mask::Vector{UInt64}, base_vec::Int)
    @inbounds ((mask[(base_vec >> 6) + 1] >> (base_vec & 63)) & 0xFFFF_FFFF) != 0x0
end

"""Pack a one-bool-per-slot mask into the little-endian bitset the scan reads."""
function pack_mask(mask::AbstractVector{Bool})
    n = length(mask)
    words = zeros(UInt64, cld(n, 64))
    @inbounds for i in 1:n
        mask[i] || continue
        s = i - 1
        words[(s >> 6) + 1] |= UInt64(1) << (s & 63)
    end
    words
end

# ── scalar kernel ──────────────────────────────────────────────────────────

@inline function _scan_blocks_scalar!(topk::TopK, prep::PreparedLut, ctx::ScanCtx,
                                      bfirst::Int, blast::Int, acc::Vector{Int32},
                                      mask::M) where {M}
    comb = prep.comb
    codes = ctx.codes
    scales = ctx.scales
    ng = ctx.ng
    n = ctx.n
    bias = prep.bias
    scale = prep.scale
    @inbounds for b in bfirst:(blast - 1)
        base_vec = b * BLOCK
        if M !== Nothing
            _mask_block_allows(mask, base_vec) || continue
        end
        fill!(acc, 0)
        base = b * ng * BLOCK
        comb_base = 0
        for g in 0:(ng - 1)
            baseg = base + g * BLOCK
            if M === Nothing
                @simd for l in 1:BLOCK
                    acc[l] += Int32(comb[comb_base + Int(codes[baseg + l]) + 1])
                end
            else
                for l in 1:BLOCK
                    acc[l] += Int32(comb[comb_base + Int(codes[baseg + l]) + 1])
                end
            end
            comb_base += 256
        end
        vi0 = base_vec
        if M === Nothing
            for l in 1:BLOCK
                vi = vi0 + l
                vi > n && break
                insert_result!(topk, (scale * Float32(acc[l]) + bias) * scales[vi],
                               vi - 1)
            end
        else
            for l in 1:BLOCK
                vi = vi0 + l
                vi > n && break
                _mask_allows(mask, vi - 1) || continue
                insert_result!(topk, (scale * Float32(acc[l]) + bias) * scales[vi],
                               vi - 1)
            end
        end
    end
    nothing
end

# ── AVX2 kernel ────────────────────────────────────────────────────────────

function _scan_blocks_avx2!(topk::TopK, prep::PreparedLut, ctx::ScanCtx,
                            bfirst::Int, blast::Int, mask::M,
                            out::Vector{Float32}) where {M}
    table = prep.table
    codes = ctx.codes
    scales = ctx.scales
    ng = ctx.ng
    n = ctx.n
    bias = prep.bias
    scale = prep.scale
    GC.@preserve codes table out begin
        pc = pointer(codes)
        pl = pointer(table)
        po = pointer(out)
        for b in bfirst:(blast - 1)
            base_vec = b * BLOCK
            if M !== Nothing
                _mask_block_allows(mask, base_vec) || continue
            end
            scan_block_avx2!(pc + b * ng * BLOCK, pl, ng, scale, bias, po)
            vi0 = base_vec
            if M === Nothing
                @inbounds for l in 1:BLOCK
                    vi = vi0 + l
                    vi > n && break
                    insert_result!(topk, out[l] * scales[vi], vi - 1)
                end
            else
                @inbounds for l in 1:BLOCK
                    vi = vi0 + l
                    vi > n && break
                    _mask_allows(mask, vi - 1) || continue
                    insert_result!(topk, out[l] * scales[vi], vi - 1)
                end
            end
        end
    end
    nothing
end

@inline function _insert_lanes!(topk::TopK, out::Vector{Float32}, off::Int,
                                base_vec::Int, ctx::ScanCtx, mask::M) where {M}
    n = ctx.n
    scales = ctx.scales
    @inbounds for l in 1:BLOCK
        vi = base_vec + l
        vi > n && break
        if M !== Nothing
            _mask_allows(mask, vi - 1) || continue
        end
        insert_result!(topk, out[off + l - 1] * scales[vi], vi - 1)
    end
    nothing
end

# Two 32-vector blocks per pass through the AVX-512 pair kernel, falling
# back to the single-block AVX2 kernel for a leftover tail block.
function _scan_blocks_avx512!(topk::TopK, prep::PreparedLut, ctx::ScanCtx,
                              bfirst::Int, blast::Int, mask::M,
                              out::Vector{Float32}) where {M}
    table = prep.table
    codes = ctx.codes
    scales = ctx.scales
    ng = ctx.ng
    n = ctx.n
    bias = prep.bias
    scale = prep.scale
    stride = ng * BLOCK
    GC.@preserve codes table out begin
        pc = pointer(codes)
        pl = pointer(table)
        po = pointer(out)
        b = bfirst
        while b < blast
            base_vec = b * BLOCK
            if b + 1 < blast
                if M !== Nothing &&
                   !_mask_block_allows(mask, base_vec) &&
                   !_mask_block_allows(mask, base_vec + BLOCK)
                    b += 2
                    continue
                end
                scan_pair_avx512!(pc + b * stride, pc + (b + 1) * stride, pl,
                                  ng, scale, bias, po)
                _insert_lanes!(topk, out, 1, base_vec, ctx, mask)
                _insert_lanes!(topk, out, BLOCK + 1, base_vec + BLOCK, ctx, mask)
                b += 2
            else
                scan_block_avx2!(pc + b * stride, pl, ng, scale, bias, po)
                _insert_lanes!(topk, out, 1, base_vec, ctx, mask)
                b += 1
            end
        end
    end
    nothing
end

@inline function _scan_blocks!(topk::TopK, prep::PreparedLut, ctx::ScanCtx,
                               bfirst::Int, blast::Int, mask::M,
                               out::Vector{Float32}) where {M}
    if HAS_AVX512BW
        _scan_blocks_avx512!(topk, prep, ctx, bfirst, blast, mask, out)
    elseif HAS_AVX2
        _scan_blocks_avx2!(topk, prep, ctx, bfirst, blast, mask, out)
    else
        acc = Vector{Int32}(undef, BLOCK)
        _scan_blocks_scalar!(topk, prep, ctx, bfirst, blast, acc, mask)
    end
    nothing
end

# ── query preparation ──────────────────────────────────────────────────────

"""Rotate the query, apply inverse calibration, and build the scan tables."""
function _prepare_lut(index::TurboQuantIndex, qrow::AbstractVector{Float32},
                      q::Vector{Float32}, scratch::Vector{Float32})
    dim = index.dim
    bits = index.bit_width
    ng = n_byte_groups(dim, bits)
    @inbounds for d in 1:dim
        q[d] = qrow[d]
    end
    apply_scaled_into!(index.rotation::Rotation, q, 1.0f0, q, scratch)
    bias_corr = 0.0f0
    calibrated = !isempty(index.tqplus_shift)
    if calibrated
        shift = index.tqplus_shift
        scale_tq = index.tqplus_scale
        bc = 0.0
        @inbounds for d in 1:dim
            bc -= Float64(q[d]) * Float64(shift[d])
            q[d] = q[d] / scale_tq[d]
        end
        bias_corr = Float32(bc)
    end
    lut = build_query_lut(q, index.centroids, bits, dim)
    # The 256-entry combined table per byte group is the scalar kernel's
    # input only; AVX2/AVX-512 hosts never read it, so they skip building
    # it entirely (at dim 768 / 4-bit that is ~100 KB per query).
    comb = (HAS_AVX512BW || HAS_AVX2) ? UInt8[] : _build_comb(lut.table, ng)
    PreparedLut(lut.table, comb, lut.scale, lut.bias + bias_corr, ng)
end

"""Build the 256-entry combined table per byte group (scalar-kernel input)."""
function _build_comb(table::Vector{UInt8}, ng::Int)
    comb = Vector{UInt8}(undef, 256 * ng)
    @inbounds for g in 0:(ng - 1)
        lb = g * 32
        cb = g * 256
        for byte in 0:255
            comb[cb + byte + 1] = table[lb + (byte >> 4) + 1] +
                                  table[lb + 16 + (byte & 0x0f) + 1]
        end
    end
    comb
end

# ── per-query search ───────────────────────────────────────────────────────

"""Minimum number of 32-vector blocks at which a single query splits its scan."""
const SINGLE_QUERY_PARALLEL_MIN_BLOCKS = 1024

"""
    single_query_parallelizes(n_vectors) -> Bool

Whether a single-query search over `n_vectors` vectors splits its block
range across worker tasks: true once the index spans at least
`SINGLE_QUERY_PARALLEL_MIN_BLOCKS` 32-vector blocks (the first such
count is 32 737 vectors, since partial blocks count). Batch searches
parallelize per query pair instead and ignore this rule.
"""
function single_query_parallelizes(n_vectors::Integer)
    n_blocks(Int(n_vectors)) >= SINGLE_QUERY_PARALLEL_MIN_BLOCKS
end

function _search_one!(index::TurboQuantIndex, qrow::AbstractVector{Float32}, k::Int,
                      block_parallel::Bool = true,
                      mask::Union{Nothing,Vector{UInt64}} = nothing)
    dim = index.dim
    q = Vector{Float32}(undef, dim)
    scratch = Vector{Float32}(undef, dim)
    prep = _prepare_lut(index, qrow, q, scratch)
    ctx = ScanCtx(index)
    topk = TopK(k)
    if block_parallel && Threads.nthreads() > 1 && single_query_parallelizes(index.n)
        nt = min(Threads.nthreads(), index.n_blocks)
        stride = cld(index.n_blocks, nt)
        results = Vector{TopK}(undef, nt)
        tasks = Task[]
        for t in 1:nt
            results[t] = TopK(k)
            b0 = (t - 1) * stride
            b1 = min(b0 + stride, index.n_blocks)
            b0 >= b1 && continue
            r = results[t]
            push!(tasks,
                  Threads.@spawn begin
                      out = Vector{Float32}(undef, 64)
                      _scan_blocks!(r, prep, ctx, b0, b1, mask, out)
                  end)
        end
        foreach(wait, tasks)
        for t in 1:nt
            results[t].size > 0 && merge_topk!(topk, results[t])
        end
    else
        out = Vector{Float32}(undef, 64)
        _scan_blocks!(topk, prep, ctx, 0, index.n_blocks, mask, out)
    end
    sorted_results(topk)
end

@inline function _write_row!(scores::Matrix{Float32}, indices::Matrix{Int},
                             qi::Int, s::Vector{Float32}, ix::Vector{Int},
                             k_eff::Int)
    @inbounds for j in 1:k_eff
        scores[qi, j] = s[j]
        indices[qi, j] = ix[j] + 1
    end
    nothing
end

# Two queries per code pass: the pair kernel shuffles each 64-code chunk
# against both queries' tables, so a pair costs ~1.4 single-query scans
# and reads the codes once. Same integer sums, so results are identical
# to two independent searches.
function _scan_two_avx512!(topkA::TopK, topkB::TopK, prepA::PreparedLut,
                           prepB::PreparedLut, ctx::ScanCtx, n_blocks::Int,
                           mask::M, outA::Vector{Float32},
                           outB::Vector{Float32}) where {M}
    codes = ctx.codes
    scales = ctx.scales
    n = ctx.n
    ng = prepA.ng
    stride = ng * BLOCK
    GC.@preserve codes outA outB begin
        pc = pointer(codes)
        pla = pointer(prepA.table)
        plb = pointer(prepB.table)
        poa = pointer(outA)
        pob = pointer(outB)
        b = 0
        while b < n_blocks
            base_vec = b * BLOCK
            if b + 1 < n_blocks
                if M !== Nothing &&
                   !_mask_block_allows(mask, base_vec) &&
                   !_mask_block_allows(mask, base_vec + BLOCK)
                    b += 2
                    continue
                end
                scan_pair2_avx512!(pc + b * stride, pc + (b + 1) * stride,
                                   pla, plb, ng, prepA.scale, prepA.bias,
                                   prepB.scale, prepB.bias, poa, pob)
                _insert_lanes!(topkA, outA, 1, base_vec, ctx, mask)
                _insert_lanes!(topkA, outA, BLOCK + 1, base_vec + BLOCK, ctx, mask)
                _insert_lanes!(topkB, outB, 1, base_vec, ctx, mask)
                _insert_lanes!(topkB, outB, BLOCK + 1, base_vec + BLOCK, ctx, mask)
                b += 2
            else
                scan_block_avx2!(pc + b * stride, pla, ng, prepA.scale,
                                 prepA.bias, poa)
                _insert_lanes!(topkA, outA, 1, base_vec, ctx, mask)
                scan_block_avx2!(pc + b * stride, plb, ng, prepB.scale,
                                 prepB.bias, pob)
                _insert_lanes!(topkB, outB, 1, base_vec, ctx, mask)
                b += 1
            end
        end
    end
    nothing
end

function _search_two!(index::TurboQuantIndex, qrowA::AbstractVector{Float32},
                      qrowB::AbstractVector{Float32}, k::Int, mask::M) where {M}
    dim = index.dim
    prepA = _prepare_lut(index, qrowA, Vector{Float32}(undef, dim),
                         Vector{Float32}(undef, dim))
    prepB = _prepare_lut(index, qrowB, Vector{Float32}(undef, dim),
                         Vector{Float32}(undef, dim))
    ctx = ScanCtx(index)
    topkA = TopK(k)
    topkB = TopK(k)
    outA = Vector{Float32}(undef, 64)
    outB = Vector{Float32}(undef, 64)
    if HAS_AVX512BW
        _scan_two_avx512!(topkA, topkB, prepA, prepB, ctx, index.n_blocks,
                          mask, outA, outB)
    else
        _scan_two_avx2!(topkA, topkB, prepA, prepB, ctx, index.n_blocks,
                        mask, outA, outB)
    end
    (sorted_results(topkA), sorted_results(topkB))
end

# AVX2 twin of the two-query pass: one 32-code block per iteration is
# scored against both queries, so the codes are read once per pair.
function _scan_two_avx2!(topkA::TopK, topkB::TopK, prepA::PreparedLut,
                         prepB::PreparedLut, ctx::ScanCtx, n_blocks::Int,
                         mask::M, outA::Vector{Float32},
                         outB::Vector{Float32}) where {M}
    codes = ctx.codes
    scales = ctx.scales
    ng = prepA.ng
    stride = ng * BLOCK
    GC.@preserve codes outA outB begin
        pc = pointer(codes)
        pla = pointer(prepA.table)
        plb = pointer(prepB.table)
        poa = pointer(outA)
        pob = pointer(outB)
        for b in 0:(n_blocks - 1)
            base_vec = b * BLOCK
            if M !== Nothing && !_mask_block_allows(mask, base_vec)
                continue
            end
            scan_pair2_avx2!(pc + b * stride, pla, plb, ng, prepA.scale,
                             prepA.bias, prepB.scale, prepB.bias, poa, pob)
            _insert_lanes!(topkA, outA, 1, base_vec, ctx, mask)
            _insert_lanes!(topkB, outB, 1, base_vec, ctx, mask)
        end
    end
    nothing
end

function _search_batch!(index::TurboQuantIndex, queries::AbstractMatrix{Float32},
                        k_eff::Int, mask::Union{Nothing,Vector{UInt64}},
                        scores::Matrix{Float32}, indices::Matrix{Int}, nq::Int)
    if nq == 1
        s1, ix1 = _search_one!(index, view(queries, 1, :), k_eff, true, mask)
        _write_row!(scores, indices, 1, s1, ix1, k_eff)
    elseif HAS_AVX512BW || HAS_AVX2
        Threads.@threads for p in 1:cld(nq, 2)
            qa = 2p - 1
            qb = 2p
            if qb > nq
                s, ix = _search_one!(index, view(queries, qa, :), k_eff, false, mask)
                _write_row!(scores, indices, qa, s, ix, k_eff)
            else
                (sA, iA), (sB, iB) = _search_two!(index, view(queries, qa, :),
                                                  view(queries, qb, :), k_eff, mask)
                _write_row!(scores, indices, qa, sA, iA, k_eff)
                _write_row!(scores, indices, qb, sB, iB, k_eff)
            end
        end
    else
        Threads.@threads for qi in 1:nq
            s, ix = _search_one!(index, view(queries, qi, :), k_eff, false, mask)
            _write_row!(scores, indices, qi, s, ix, k_eff)
        end
    end
    nothing
end

"""
    search(index, queries, k; mask = nothing) -> (scores, indices)

Top-`k` nearest slots for each of the `nq × dim` `Float32` query rows.
Returns `nq × k_eff` matrices, `k_eff = min(k, length(index))`, sorted
by descending score within each row.

`mask`, when given, is a `Bool` vector with one entry per slot: only
`true` slots contribute, and `k_eff = min(k, length(index), count(mask))`.
The index is asked for exactly `k_eff` results — there is no masked
over-fetch.
"""
function search(index::TurboQuantIndex, queries::AbstractMatrix{Float32}, k::Integer;
                mask::Union{Nothing,AbstractVector{Bool}} = nothing)
    k >= 0 || throw(ArgumentError("k must be nonnegative, got $k"))
    dim = index.dim
    nq = size(queries, 1)
    if dim == 0
        return (Matrix{Float32}(undef, nq, 0), Matrix{Int}(undef, nq, 0))
    end
    size(queries, 2) == dim || throw(QueryBufferNotMultipleOfDim(size(queries, 2), dim))
    bad = _first_invalid_matrix(queries, dim)
    bad === nothing ||
        throw(InvalidQueryValue(bad.vector_index, bad.coord_index, bad.value))
    if index.n == 0
        mask !== nothing && length(mask) != 0 &&
            throw(MaskLengthMismatch(0, length(mask)))
        return (Matrix{Float32}(undef, nq, 0), Matrix{Int}(undef, nq, 0))
    end
    pmask = nothing
    n_allowed = index.n
    if mask !== nothing
        length(mask) == index.n ||
            throw(MaskLengthMismatch(index.n, length(mask)))
        pmask = pack_mask(mask)
        n_allowed = count(mask)
    end
    k_eff = min(Int(k), index.n, n_allowed)
    scores = Matrix{Float32}(undef, nq, k_eff)
    indices = Matrix{Int}(undef, nq, k_eff)
    k_eff == 0 && return (scores, indices)
    _search_batch!(index, queries, k_eff, pmask, scores, indices, nq)
    (scores, indices)
end

function search(index::TurboQuantIndex, queries::AbstractMatrix{Float32}, k::Integer,
                mask::AbstractVector{Bool})
    search(index, queries, k; mask = mask)
end

function search(index::TurboQuantIndex, queries::AbstractMatrix{<:Real}, k::Integer;
                mask::Union{Nothing,AbstractVector{Bool}} = nothing)
    search(index, Float32.(queries), k; mask = mask)
end

"""Single-query convenience: returns `1 × k_eff` matrices."""
function search(index::TurboQuantIndex, q::AbstractVector{Float32}, k::Integer;
                mask::Union{Nothing,AbstractVector{Bool}} = nothing)
    search(index, reshape(q, 1, :), k; mask = mask)
end

function search(index::TurboQuantIndex, q::AbstractVector{<:Real}, k::Integer;
                mask::Union{Nothing,AbstractVector{Bool}} = nothing)
    search(index, reshape(Float32.(q), 1, :), k; mask = mask)
end
