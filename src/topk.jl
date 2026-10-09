# Top-k collector: a size-k buffer with a linear rescan for the current
# minimum (the Rust `rescan_min` strategy; k is small, so the rescan wins
# over a heap).

mutable struct TopK
    k::Int
    scores::Vector{Float32}
    indices::Vector{Int}
    size::Int
    min_score::Float32
    min_pos::Int
end

function TopK(k::Int)
    TopK(k, Vector{Float32}(undef, k), Vector{Int}(undef, k), 0,
         -Inf32, 0)
end

@inline function rescan_min!(h::TopK)
    m = h.scores[1]
    mi = 1
    @inbounds for i in 2:h.size
        s = h.scores[i]
        if s < m
            m = s
            mi = i
        end
    end
    h.min_score = m
    h.min_pos = mi
    nothing
end

@inline function insert_result!(h::TopK, s::Float32, idx::Int)
    if h.size < h.k
        h.size += 1
        @inbounds h.scores[h.size] = s
        @inbounds h.indices[h.size] = idx
        if h.size == h.k
            rescan_min!(h)
        end
    elseif s > h.min_score
        @inbounds h.scores[h.min_pos] = s
        @inbounds h.indices[h.min_pos] = idx
        rescan_min!(h)
    end
    nothing
end

"""Merge `src` into `dst`, keeping the global top-k."""
function merge_topk!(dst::TopK, src::TopK)
    @inbounds for i in 1:src.size
        insert_result!(dst, src.scores[i], src.indices[i])
    end
    nothing
end

"""Results sorted by descending score (ties by ascending index)."""
function sorted_results(h::TopK)
    n = h.size
    scores = Vector{Float32}(undef, n)
    indices = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        scores[i] = h.scores[i]
        indices[i] = h.indices[i]
    end
    p = sortperm(1:n, by = i -> (-scores[i], indices[i]))
    (scores[p], indices[p])
end
