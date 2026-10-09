# IdMapIndex: stable external `UInt64` ids over a positional inner index.
# Port of `turbovec::IdMapIndex`.

mutable struct IdMapIndex
    inner::TurboQuantIndex
    slot_to_id::Vector{UInt64}
    id_to_slot::Dict{UInt64,Int}
end

"""
    IdMapIndex(dim, bit_width)

An id-map index with a known dimensionality.
"""
IdMapIndex(dim::Integer, bit_width::Integer) =
    IdMapIndex(TurboQuantIndex(dim, bit_width), UInt64[], Dict{UInt64,Int}())

"""
    IdMapIndex(bit_width)

An id-map index without a committed dim; inferred on the first add.
"""
IdMapIndex(bit_width::Integer) =
    IdMapIndex(TurboQuantIndex(bit_width), UInt64[], Dict{UInt64,Int}())

Base.length(index::IdMapIndex) = index.inner.n
Base.isempty(index::IdMapIndex) = index.inner.n == 0
dim_opt(index::IdMapIndex) = dim_opt(index.inner)
dim(index::IdMapIndex) = index.inner.dim
bit_width(index::IdMapIndex) = index.inner.bit_width
scales(index::IdMapIndex) = index.inner.scales
tqplus_shift(index::IdMapIndex) = index.inner.tqplus_shift
tqplus_scale(index::IdMapIndex) = index.inner.tqplus_scale
prepare(index::IdMapIndex) = index
calibration_state(index::IdMapIndex) = calibration_state(index.inner)

"""True if the index currently contains a vector with this external id."""
contains_id(index::IdMapIndex, id::Integer) = haskey(index.id_to_slot, UInt64(id))

"""
    batch_addable(index, ids) -> Bool

True when `ids` has no duplicates and none of them is already present —
exactly the pair of conditions [`add_with_ids!`](@ref) validates.
"""
function batch_addable(index::IdMapIndex, ids::AbstractVector{<:Integer})
    seen = Set{UInt64}()
    for id in ids
        u = UInt64(id)
        (haskey(index.id_to_slot, u) || u in seen) && return false
        push!(seen, u)
    end
    true
end

"""
    slots_ready(index) -> Bool

Always `true`: this port maintains the id tables eagerly, so there is no
lazy slot-map build to wait for. Kept for API parity with turbovec.
"""
slots_ready(index::IdMapIndex) = true

"""The external ids in slot order (slot order itself is an implementation detail)."""
external_ids(index::IdMapIndex) = copy(index.slot_to_id)

"""
    add_with_ids!(index, vectors, ids)

Add `n` vectors with `n` external ids. Rejects ids already present, and
duplicates within the batch, before mutating anything.
"""
function add_with_ids!(index::IdMapIndex, X::AbstractMatrix{Float32},
                       ids::AbstractVector{<:Integer})
    n = size(X, 1)
    length(ids) == n || throw(IdsCountMismatch(n, length(ids)))
    uids = Vector{UInt64}(undef, n)
    seen = Set{UInt64}()
    @inbounds for i in 1:n
        u = UInt64(ids[i])
        uids[i] = u
        haskey(index.id_to_slot, u) && throw(IdAlreadyPresent(u))
        u in seen && throw(DuplicateIdInBatch(u))
        push!(seen, u)
    end
    base = index.inner.n
    add!(index.inner, X)
    @inbounds for i in 1:n
        index.id_to_slot[uids[i]] = base + i
        push!(index.slot_to_id, uids[i])
    end
    index
end

add_with_ids!(index::IdMapIndex, X::AbstractMatrix{<:Real},
              ids::AbstractVector{<:Integer}) = add_with_ids!(index, Float32.(X), ids)

"""Remove the vector with external `id`. Returns `true` if present."""
function remove!(index::IdMapIndex, id::Integer)
    u = UInt64(id)
    slot = get(index.id_to_slot, u, 0)
    slot == 0 && return false
    last = index.inner.n
    swap_remove!(index.inner, slot)
    delete!(index.id_to_slot, u)
    if slot != last
        moved_id = index.slot_to_id[last]
        index.slot_to_id[slot] = moved_id
        index.id_to_slot[moved_id] = slot
    end
    pop!(index.slot_to_id)
    true
end

"""
    search(index, queries, k; allowlist = nothing) -> (scores, ids)

Top-`k` ids for each of the `nq × dim` query rows. `allowlist`, when
given, restricts results to those external ids; it is deduplicated, and
`k_eff = min(k, number of unique allowlisted ids)`. An empty allowlist
or an unknown id is an error, not a panic.
"""
function search(index::IdMapIndex, queries::AbstractMatrix{Float32}, k::Integer;
                allowlist::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    nq = size(queries, 1)
    if index.inner.dim == 0 || index.inner.n == 0
        # Ids are resolved before the inner search, so an empty index
        # still reports an empty or unknown allowlist.
        if allowlist !== nothing
            isempty(allowlist) && throw(AllowlistEmpty())
            throw(UnknownId(UInt64(first(allowlist))))
        end
        return (Matrix{Float32}(undef, nq, 0), Matrix{UInt64}(undef, nq, 0))
    end
    local scores::Matrix{Float32}, slots::Matrix{Int}
    if allowlist === nothing
        scores, slots = search(index.inner, queries, k)
    else
        isempty(allowlist) && throw(AllowlistEmpty())
        mask = fill(false, index.inner.n)
        for id in allowlist
            u = UInt64(id)
            slot = get(index.id_to_slot, u, 0)
            slot == 0 && throw(UnknownId(u))
            mask[slot] = true
        end
        scores, slots = search(index.inner, queries, k; mask = mask)
    end
    ids = Matrix{UInt64}(undef, size(slots, 1), size(slots, 2))
    @inbounds for j in eachindex(slots)
        ids[j] = index.slot_to_id[slots[j]]
    end
    (scores, ids)
end

search(index::IdMapIndex, queries::AbstractMatrix{Float32}, k::Integer,
       allowlist::AbstractVector{<:Integer}) =
    search(index, queries, k; allowlist = allowlist)

search(index::IdMapIndex, queries::AbstractMatrix{<:Real}, k::Integer;
       allowlist::Union{Nothing,AbstractVector{<:Integer}} = nothing) =
    search(index, Float32.(queries), k; allowlist = allowlist)

"""Fit a TQ+ calibration and re-encode stored rows."""
function calibrate!(index::IdMapIndex, sample::AbstractMatrix{Float32})
    calibrate!(index.inner, sample)
    index
end

calibrate!(index::IdMapIndex, sample::AbstractMatrix{<:Real}) =
    calibrate!(index, Float32.(sample))
