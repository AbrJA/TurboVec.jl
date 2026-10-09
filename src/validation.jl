# Value-level validation shared by every construction path.
#
# The bounds are physics, not policy: they are the largest values that
# cannot drive an overflow through the search path's per-coordinate
# division, bias dot product and score multiplication. The validators
# return `nothing` or a message so each caller raises its own error type
# (`InvalidParts` from `from_parts`, `InvalidFileFormat` from loaders)
# instead of one layer owning the other's errors.

"""Smallest TQ+ scale at `dim` that cannot drive a divided query to overflow."""
min_tqplus_scale(dim::Int) =
    Float32(max(dim, 1)) * MAX_INPUT_MAGNITUDE / floatmax(Float32) * 10.0f0

"""Largest TQ+ shift magnitude at `dim` whose bias dot product cannot overflow."""
max_tqplus_shift(dim::Int) =
    floatmax(Float32) / (Float32(max(dim, 1)) * MAX_INPUT_MAGNITUDE) / 10.0f0

"""Largest per-vector renormalization scale that cannot by itself overflow."""
const MAX_VECTOR_SCALE = 1.0f22

"""Message for the first invalid per-vector scale, or `nothing`."""
function _scale_error(scales::AbstractVector{Float32})
    @inbounds for (i, s) in enumerate(scales)
        if !isfinite(s) || s < 0.0f0 || s > MAX_VECTOR_SCALE
            return "invalid per-vector scale at slot $i: $s " *
                   "(must be finite and in [0, $MAX_VECTOR_SCALE])"
        end
    end
    nothing
end

"""Message for the first invalid TQ+ `(shift, scale)` entry, or `nothing`."""
function _calibration_error(shift::AbstractVector{Float32},
                            scale::AbstractVector{Float32})
    cap = max_tqplus_shift(length(shift))
    @inbounds for (i, v) in enumerate(shift)
        if !isfinite(v) || abs(v) > cap
            return "invalid TQ+ shift at coord $i: $v " *
                   "(must be finite and |shift| <= $cap)"
        end
    end
    floor = min_tqplus_scale(length(scale))
    @inbounds for (i, v) in enumerate(scale)
        if !isfinite(v) || v < floor
            return "invalid TQ+ scale at coord $i: $v " *
                   "(must be finite and >= $floor)"
        end
    end
    nothing
end

# First invalid coordinate of an `n × dim` matrix in **row-major**
# precedence (smallest row, then smallest column), scanning column-first
# so the inner loop walks contiguous memory. Equivalent to a row-major
# scan, without the cache-hostile `n`-element stride.
function _first_invalid_matrix(X::AbstractMatrix{Float32}, dim::Int)
    n = size(X, 1)
    best_i = typemax(Int)
    best_d = 0
    best_x = 0.0f0
    @inbounds for d in 1:dim
        for i in 1:min(n, best_i - 1)
            x = X[i, d]
            if !(abs(x) < MAX_INPUT_MAGNITUDE)
                best_i = i
                best_d = d
                best_x = x
                break
            end
        end
    end
    best_d == 0 ? nothing : (vector_index = best_i, coord_index = best_d, value = best_x)
end

"""
    first_invalid_coord(values, dim; max_magnitude = 1e16)
        -> Union{Nothing, NamedTuple}

Scan a flat `n * dim` buffer for the first coordinate that is not finite
or has magnitude >= `max_magnitude`, and return
`(vector_index, coord_index, value)` with **1-based** indices, or
`nothing` when the input is clean. This is the predicate `add!` and
`search` enforce.
"""
function first_invalid_coord(values::AbstractVector{Float32}, dim::Integer;
                             max_magnitude::Float32 = MAX_INPUT_MAGNITUDE)
    dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
    length(values) % dim == 0 ||
        throw(ArgumentError("values length $(length(values)) is not a multiple of dim $dim"))
    @inbounds for (i, x) in enumerate(values)
        if !(abs(x) < max_magnitude)
            vi = (i - 1) ÷ dim + 1
            ci = (i - 1) % dim + 1
            return (vector_index = vi, coord_index = ci, value = x)
        end
    end
    nothing
end
