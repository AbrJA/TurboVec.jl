# Typed errors mirroring turbovec's error surface.

abstract type TurboVecError <: Exception end

struct BitWidthOutOfRange <: TurboVecError
    bit_width::Int
end
Base.showerror(io::IO, e::BitWidthOutOfRange) =
    print(io, "bit_width must be 2, 3 or 4, got ", e.bit_width)

struct DimNotPositiveMultipleOf8 <: TurboVecError
    dim::Int
end
Base.showerror(io::IO, e::DimNotPositiveMultipleOf8) =
    print(io, "dim must be a positive multiple of 8, got ", e.dim)

struct DimTooLarge <: TurboVecError
    dim::Int
    max::Int
end
Base.showerror(io::IO, e::DimTooLarge) =
    print(io, "dim ", e.dim, " exceeds MAX_DIM (", e.max, ")")

struct ZeroDim <: TurboVecError end
Base.showerror(io::IO, ::ZeroDim) = print(io, "dim must be nonzero")

struct DimMismatch <: TurboVecError
    existing::Int
    got::Int
end
Base.showerror(io::IO, e::DimMismatch) =
    print(io, "dim mismatch: index has ", e.existing, ", got ", e.got)

struct InvalidInputValue <: TurboVecError
    vector_index::Int
    coord_index::Int
    value::Float32
end
Base.showerror(io::IO, e::InvalidInputValue) =
    print(io, "invalid input value at vector ", e.vector_index, ", coord ",
          e.coord_index, ": ", e.value,
          " (must be finite and |value| < 1e16 to avoid f32 norm overflow)")

struct QueryBufferNotMultipleOfDim <: TurboVecError
    queries_len::Int
    dim::Int
end
Base.showerror(io::IO, e::QueryBufferNotMultipleOfDim) =
    print(io, "queries length ", e.queries_len, " is not a multiple of dim ", e.dim)

struct InvalidQueryValue <: TurboVecError
    query_index::Int
    coord_index::Int
    value::Float32
end
Base.showerror(io::IO, e::InvalidQueryValue) =
    print(io, "invalid query value at row ", e.query_index, ", coord ",
          e.coord_index, ": ", e.value)

struct IdAlreadyPresent <: TurboVecError
    id::UInt64
end
Base.showerror(io::IO, e::IdAlreadyPresent) = print(io, "id ", e.id, " is already present")

struct DuplicateIdInBatch <: TurboVecError
    id::UInt64
end
Base.showerror(io::IO, e::DuplicateIdInBatch) = print(io, "id ", e.id, " is duplicated in batch")

struct IdsCountMismatch <: TurboVecError
    expected::Int
    got::Int
end
Base.showerror(io::IO, e::IdsCountMismatch) =
    print(io, "expected ", e.expected, " ids, got ", e.got)

struct EmptyCalibrationSample <: TurboVecError
    rows::Int
    min::Int
end
Base.showerror(io::IO, e::EmptyCalibrationSample) =
    print(io, "calibration sample needs at least ", e.min, " rows, got ", e.rows)

struct DegenerateSample <: TurboVecError end
Base.showerror(io::IO, ::DegenerateSample) =
    print(io, "calibration sample is degenerate (no per-coordinate spread anywhere)")

struct InvalidFileFormat <: TurboVecError
    msg::String
end
Base.showerror(io::IO, e::InvalidFileFormat) = print(io, "invalid TurboVec file: ", e.msg)

struct MaskLengthMismatch <: TurboVecError
    expected::Int
    got::Int
end
Base.showerror(io::IO, e::MaskLengthMismatch) =
    print(io, "mask length ", e.got, " does not match index size ", e.expected)

struct AllowlistEmpty <: TurboVecError end
Base.showerror(io::IO, ::AllowlistEmpty) =
    print(io, "allowlist is empty; pass nothing to search the whole index")

struct UnknownId <: TurboVecError
    id::UInt64
end
Base.showerror(io::IO, e::UnknownId) = print(io, "id ", e.id, " is not present in the index")

struct InvalidParts <: TurboVecError
    msg::String
end
Base.showerror(io::IO, e::InvalidParts) = print(io, "invalid index parts: ", e.msg)
