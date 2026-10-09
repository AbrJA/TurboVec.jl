# IdMapIndex: stable external `UInt64` ids over a positional inner index.
# Port of `turbovec::IdMapIndex`.

mutable struct IdMapIndex
    inner::TurboQuantIndex
    slot_to_id::Vector{UInt64}
    id_to_slot::Dict{UInt64,Int}
end

# Convert an external id, throwing a typed error when it cannot be a `UInt64`.
@inline function _to_uid(id::Integer)
    0 <= id <= typemax(UInt64) || throw(InvalidIdValue(id))
    UInt64(id)
end

# Predicate form: `false` when the id cannot name any stored vector.
@inline _uid_or_nothing(id::Integer) = 0 <= id <= typemax(UInt64) ? UInt64(id) : nothing

"""
    IdMapIndex(dim, bit_width)

An id-map index with a known dimensionality.
"""
function IdMapIndex(dim::Integer, bit_width::Integer)
    IdMapIndex(TurboQuantIndex(dim, bit_width), UInt64[], Dict{UInt64,Int}())
end

"""
    IdMapIndex(bit_width)

An id-map index without a committed dim; inferred on the first add.
"""
function IdMapIndex(bit_width::Integer)
    IdMapIndex(TurboQuantIndex(bit_width), UInt64[], Dict{UInt64,Int}())
end

Base.length(index::IdMapIndex) = index.inner.n
Base.isempty(index::IdMapIndex) = index.inner.n == 0
dim_opt(index::IdMapIndex) = dim_opt(index.inner)
is_lazy(index::IdMapIndex) = is_lazy(index.inner)
dim(index::IdMapIndex) = index.inner.dim
bit_width(index::IdMapIndex) = index.inner.bit_width
scales(index::IdMapIndex) = index.inner.scales
tqplus_shift(index::IdMapIndex) = index.inner.tqplus_shift
tqplus_scale(index::IdMapIndex) = index.inner.tqplus_scale
prepare(index::IdMapIndex) = index
calibration_state(index::IdMapIndex) = calibration_state(index.inner)
is_calibrated(index::IdMapIndex) = is_calibrated(index.inner)
calibration(index::IdMapIndex) = calibration(index.inner)

"""True if the index currently contains a vector with this external id.

Ids that cannot be represented as a `UInt64` (e.g. negative) are simply
not present, so this returns `false` rather than throwing.
"""
function contains_id(index::IdMapIndex, id::Integer)
    (u = _uid_or_nothing(id)) === nothing ? false : haskey(index.id_to_slot, u)
end

"""Idiomatic membership: `id in index`."""
Base.in(id::Integer, index::IdMapIndex) = contains_id(index, id)

"""Iterate the external ids in slot order."""
function Base.iterate(index::IdMapIndex, state::Int = 1)
    state > length(index) ? nothing : (index.slot_to_id[state], state + 1)
end

"""The external ids as a vector, in slot order (a copy)."""
Base.keys(index::IdMapIndex) = copy(index.slot_to_id)

Base.eltype(::Type{IdMapIndex}) = UInt64

"""`size(index) == (length(index), dim(index))`."""
Base.size(index::IdMapIndex) = (length(index), dim(index))
Base.size(index::IdMapIndex, d::Integer) = d == 1 ? length(index) : d == 2 ? dim(index) : 1

function Base.show(io::IO, index::IdMapIndex)
    print(io, "IdMapIndex(")
    if dim(index) == 0
        print(io, "lazy, $(bit_width(index))-bit, $(length(index)) ids)")
    else
        print(io, "$(dim(index)) features, $(bit_width(index))-bit, ",
              "$(length(index)) ids, ",
              is_calibrated(index) ? "calibrated" : "uncalibrated", ")")
    end
end

function Base.show(io::IO, ::MIME"text/plain", index::IdMapIndex)
    print(io, "IdMapIndex: ")
    if dim(index) == 0
        print(io, "lazy, $(bit_width(index))-bit, $(length(index)) ids")
    else
        print(io, "$(dim(index)) features, $(bit_width(index))-bit, ",
              "$(length(index)) ids, ",
              is_calibrated(index) ? "calibrated" : "uncalibrated")
        print(io, "\n  codes: ", length(index.inner.codes), " bytes")
    end
end

"""
    index == other

Structural equality: same inner index (geometry, codes, scales and
calibration) and the same id table in slot order.
"""
function Base.:(==)(a::IdMapIndex, b::IdMapIndex)
    a === b && return true
    a.inner == b.inner && a.slot_to_id == b.slot_to_id
end

"""Deep copy (inner index, id tables)."""
function Base.copy(index::IdMapIndex)
    IdMapIndex(copy(index.inner), copy(index.slot_to_id), copy(index.id_to_slot))
end

"""Drop every stored vector/id and the calibration; keep committed geometry."""
function Base.empty!(index::IdMapIndex)
    empty!(index.inner)
    empty!(index.slot_to_id)
    empty!(index.id_to_slot)
    index
end

# Convert and validate a batch of external ids, throwing on the first
# out-of-domain, already-present or duplicated id. Shared by `is_addable`
# and `add_with_ids!`.
function _checked_uids(index::IdMapIndex, ids::AbstractVector{<:Integer})
    uids = Vector{UInt64}(undef, length(ids))
    seen = Set{UInt64}()
    @inbounds for i in eachindex(ids)
        u = _to_uid(ids[i])
        uids[i] = u
        haskey(index.id_to_slot, u) && throw(IdAlreadyPresent(u))
        u in seen && throw(DuplicateIdInBatch(u))
        push!(seen, u)
    end
    uids
end

"""
    is_addable(index, ids) -> Bool

True when `ids` has no duplicates and none of them is already present —
exactly the pair of conditions [`add_with_ids!`](@ref) validates.
"""
function is_addable(index::IdMapIndex, ids::AbstractVector{<:Integer})
    try
        _checked_uids(index, ids)
        true
    catch e
        e isa TurboVecError ? false : rethrow()
    end
end

"""
    is_slots_ready(index) -> Bool

Always `true`: this port maintains the id tables eagerly, so there is no
lazy slot-map build to wait for. Kept for API parity with turbovec.
"""
is_slots_ready(index::IdMapIndex) = true

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
    uids = _checked_uids(index, ids)
    base = index.inner.n
    add!(index.inner, X)
    @inbounds for i in 1:n
        index.id_to_slot[uids[i]] = base + i
        push!(index.slot_to_id, uids[i])
    end
    index
end

function add_with_ids!(index::IdMapIndex, X::AbstractMatrix{<:Real},
                       ids::AbstractVector{<:Integer})
    add_with_ids!(index, Float32.(X), ids)
end

"""Add one vector with one external id."""
function add_with_ids!(index::IdMapIndex, x::AbstractVector{Float32}, id::Integer)
    add_with_ids!(index, reshape(x, 1, :), [id])
end

function add_with_ids!(index::IdMapIndex, x::AbstractVector{<:Real}, id::Integer)
    add_with_ids!(index, reshape(Float32.(x), 1, :), [id])
end

"""Remove the vector with external `id`. Returns `true` if present.

Ids that cannot be represented as a `UInt64` (e.g. negative) are never
present, so this returns `false`.
"""
function remove!(index::IdMapIndex, id::Integer)
    u = _uid_or_nothing(id)
    u === nothing && return false
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
            u = _to_uid(id)
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

function search(index::IdMapIndex, queries::AbstractMatrix{Float32}, k::Integer,
                allowlist::AbstractVector{<:Integer})
    search(index, queries, k; allowlist = allowlist)
end

function search(index::IdMapIndex, queries::AbstractMatrix{<:Real}, k::Integer;
                allowlist::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    search(index, Float32.(queries), k; allowlist = allowlist)
end

"""Single-query convenience: returns `1 × k_eff` matrices."""
function search(index::IdMapIndex, q::AbstractVector{Float32}, k::Integer;
                allowlist::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    search(index, reshape(q, 1, :), k; allowlist = allowlist)
end

function search(index::IdMapIndex, q::AbstractVector{<:Real}, k::Integer;
                allowlist::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    search(index, reshape(Float32.(q), 1, :), k; allowlist = allowlist)
end

"""Fit a TQ+ calibration and re-encode stored rows."""
function calibrate!(index::IdMapIndex, sample::AbstractMatrix{Float32})
    calibrate!(index.inner, sample)
    index
end

function calibrate!(index::IdMapIndex, sample::AbstractMatrix{<:Real})
    calibrate!(index, Float32.(sample))
end
