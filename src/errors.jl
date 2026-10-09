# Typed errors mirroring turbovec's error surface. Every one is a
# `TurboVecError`, so `catch e; e isa TurboVecError` covers the package.

"""Base type for every error raised by TurboVec."""
abstract type TurboVecError <: Exception end

"""Bit width outside `{2, 3, 4}`."""
struct BitWidthOutOfRange <: TurboVecError
    bit_width::Int
end
Base.showerror(io::IO, e::BitWidthOutOfRange) =
    print(io, "bit_width must be 2, 3 or 4, got ", e.bit_width)

"""Index dimensionality not a positive multiple of 8."""
struct DimNotPositiveMultipleOf8 <: TurboVecError
    dim::Int
end
Base.showerror(io::IO, e::DimNotPositiveMultipleOf8) =
    print(io, "dim must be a positive multiple of 8, got ", e.dim)

"""Index dimensionality above `MAX_DIM`."""
struct DimTooLarge <: TurboVecError
    dim::Int
    max::Int
end
Base.showerror(io::IO, e::DimTooLarge) =
    print(io, "dim ", e.dim, " exceeds MAX_DIM (", e.max, ")")

"""Dimensionality argument was zero."""
struct ZeroDim <: TurboVecError end
Base.showerror(io::IO, ::ZeroDim) = print(io, "dim must be nonzero")

"""Operation's dimensionality does not match the index's committed dim."""
struct DimMismatch <: TurboVecError
    existing::Int
    got::Int
end
Base.showerror(io::IO, e::DimMismatch) =
    print(io, "dim mismatch: index has ", e.existing, ", got ", e.got)

"""Added vector has a non-finite or over-magnitude coordinate."""
struct InvalidInputValue <: TurboVecError
    vector_index::Int
    coord_index::Int
    value::Float32
end
Base.showerror(io::IO, e::InvalidInputValue) =
    print(io, "invalid input value at vector ", e.vector_index, ", coord ",
          e.coord_index, ": ", e.value,
          " (must be finite and |value| < 1e16 to avoid f32 norm overflow)")

"""Query matrix width does not equal the index dim."""
struct QueryBufferNotMultipleOfDim <: TurboVecError
    queries_len::Int
    dim::Int
end
Base.showerror(io::IO, e::QueryBufferNotMultipleOfDim) =
    print(io, "queries length ", e.queries_len, " is not a multiple of dim ", e.dim)

"""Query has a non-finite or over-magnitude coordinate."""
struct InvalidQueryValue <: TurboVecError
    query_index::Int
    coord_index::Int
    value::Float32
end
Base.showerror(io::IO, e::InvalidQueryValue) =
    print(io, "invalid query value at row ", e.query_index, ", coord ",
          e.coord_index, ": ", e.value)

"""`add_with_ids!` was given an id that is already in the index."""
struct IdAlreadyPresent <: TurboVecError
    id::UInt64
end
Base.showerror(io::IO, e::IdAlreadyPresent) = print(io, "id ", e.id, " is already present")

"""`add_with_ids!` was given the same id twice in one batch."""
struct DuplicateIdInBatch <: TurboVecError
    id::UInt64
end
Base.showerror(io::IO, e::DuplicateIdInBatch) = print(io, "id ", e.id, " is duplicated in batch")

"""Number of ids does not match the number of vectors."""
struct IdsCountMismatch <: TurboVecError
    expected::Int
    got::Int
end
Base.showerror(io::IO, e::IdsCountMismatch) =
    print(io, "expected ", e.expected, " ids, got ", e.got)

"""Calibration sample has fewer rows than `MIN_CALIBRATION_ROWS`."""
struct EmptyCalibrationSample <: TurboVecError
    rows::Int
    min::Int
end
Base.showerror(io::IO, e::EmptyCalibrationSample) =
    print(io, "calibration sample needs at least ", e.min, " rows, got ", e.rows)

"""Calibration sample has no per-coordinate spread anywhere (fits identity)."""
struct DegenerateSample <: TurboVecError end
Base.showerror(io::IO, ::DegenerateSample) =
    print(io, "calibration sample is degenerate (no per-coordinate spread anywhere)")

"""A persisted image is malformed, truncated, or fails validation."""
struct InvalidFileFormat <: TurboVecError
    msg::String
end
Base.showerror(io::IO, e::InvalidFileFormat) = print(io, "invalid TurboVec file: ", e.msg)

"""Slot mask length does not equal the number of slots."""
struct MaskLengthMismatch <: TurboVecError
    expected::Int
    got::Int
end
Base.showerror(io::IO, e::MaskLengthMismatch) =
    print(io, "mask length ", e.got, " does not match index size ", e.expected)

"""An allowlist was passed empty."""
struct AllowlistEmpty <: TurboVecError end
Base.showerror(io::IO, ::AllowlistEmpty) =
    print(io, "allowlist is empty; pass nothing to search the whole index")

"""An allowlist id is not present in the index."""
struct UnknownId <: TurboVecError
    id::UInt64
end
Base.showerror(io::IO, e::UnknownId) = print(io, "id ", e.id, " is not present in the index")

"""`from_parts` was given an inconsistent or out-of-bounds part."""
struct InvalidParts <: TurboVecError
    msg::String
end
Base.showerror(io::IO, e::InvalidParts) = print(io, "invalid index parts: ", e.msg)
