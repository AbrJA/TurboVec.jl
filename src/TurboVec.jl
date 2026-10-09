"""
    TurboVec

Julia port of `turbovec`, a vector index built on Google Research's
TurboQuant algorithm: data-oblivious scalar quantization of rotated
unit vectors to 2-4 bits per coordinate, with SIMD-free but cache-aware
blocked scoring and a multi-threaded scan.

Public surface:

    TurboQuantIndex(dim, bit_width)
    add!(index, vectors)
    calibrate!(index, sample)
    search(index, queries, k)         -> (scores, indices)
    IdMapIndex(dim, bit_width)
    add_with_ids!(index, vectors, ids)
    remove!(index, id)
    write_index(path, index) / load_index(path)
    write_idmap(path, index) / load_idmap(path)
"""
module TurboVec

export TurboQuantIndex, IdMapIndex
export add!, calibrate!, search, swap_remove!, remove!
export add_with_ids!, calibration_state, codebook
export write_index, load_index, write_idmap, load_idmap, to_bytes, from_bytes
export from_parts, packed_codes, prepare, dim_opt, bit_width, scales,
       tqplus_shift, tqplus_scale, contains_id, external_ids, is_lazy
export blocked_codes, codebook_for_write, serialized_len, is_packed_ready,
       is_slots_ready, is_addable, first_invalid_coord, is_calibrated, calibration
export MIN_INPUT_NORM, MIN_CALIBRATION_ROWS, RECOMMENDED_CALIBRATION_ROWS
export TurboVecError, BitWidthOutOfRange, DimNotPositiveMultipleOf8, DimTooLarge,
       ZeroDim, DimMismatch, InvalidInputValue,
       QueryBufferNotMultipleOfDim, InvalidQueryValue, IdAlreadyPresent,
       DuplicateIdInBatch, IdsCountMismatch, EmptyCalibrationSample,
       DegenerateSample, InvalidFileFormat, InvalidParts, MaskLengthMismatch,
       AllowlistEmpty, UnknownId, MAX_DIM

"""Maximum supported dimensionality."""
const MAX_DIM = 16384

include("errors.jl")
include("chacha.jl")
include("rotation.jl")
include("codebook.jl")
include("pack.jl")
include("encode.jl")
include("validation.jl")
include("lut.jl")
include("simd.jl")
include("topk.jl")
include("index.jl")
include("search.jl")
include("id_map.jl")
include("io.jl")
include("precompile.jl")

end # module TurboVec
