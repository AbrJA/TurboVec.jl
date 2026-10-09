# Blocked code layout: 32 vectors per block, one code byte per byte
# group per lane. This is the layout the search kernel scans directly.
#
# For `bits` the extracted code byte packs `8/bits` codes, most
# significant field first, exactly as turbovec's `extract_codes_flat`
# (it uses a 4-bit field for 3-bit codes, so 2/3/4 all share the
# two-codes-per-byte nibble structure).
#
#   bits = 2 -> 4 codes/byte, groups cover 4 coordinates each
#   bits = 3 -> 2 codes/byte (3 significant bits), groups cover 2 coords
#   bits = 4 -> 2 codes/byte, groups cover 2 coordinates each

const BLOCK = 32

codes_per_byte(bits::Int) = 8 ÷ bits
n_byte_groups(dim::Int, bits::Int) = dim ÷ codes_per_byte(bits)
n_blocks(n::Int) = (n + BLOCK - 1) ÷ BLOCK

"""Total size in bytes of the blocked layout for `n` vectors."""
blocked_len(n::Int, bits::Int, dim::Int) = n_blocks(n) * n_byte_groups(dim, bits) * BLOCK

@inline function byte_offset(b::Int, g::Int, lane::Int, ng::Int)
    (b * ng + g) * BLOCK + lane + 1
end

"""
    write_codes_lane!(blocked, lane, codes, dim, bits)

Write one vector's `dim` code values (`0 <= code < 2^bits`) into the
blocked layout at `lane` (0-based).
"""
function write_codes_lane!(blocked::AbstractVector{UInt8}, lane::Int,
                           codes::AbstractVector{UInt8}, dim::Int, bits::Int)
    ng = n_byte_groups(dim, bits)
    b = lane ÷ BLOCK
    l = lane % BLOCK
    chunks = dim ÷ 8
    @inbounds if bits == 2
        for c in 0:(chunks - 1)
            o = 8c
            blocked[byte_offset(b, 2c, l, ng)] = (codes[o + 1] << 6) | (codes[o + 2] << 4) |
                                                 (codes[o + 3] << 2) | codes[o + 4]
            blocked[byte_offset(b, 2c + 1, l, ng)] = (codes[o + 5] << 6) |
                                                     (codes[o + 6] << 4) |
                                                     (codes[o + 7] << 2) | codes[o + 8]
        end
    else
        for c in 0:(chunks - 1)
            o = 8c
            for k in 0:3
                blocked[byte_offset(b, 4c + k, l, ng)] = (codes[o + 2k + 1] << 4) |
                                                         codes[o + 2k + 2]
            end
        end
    end
    nothing
end

"""Copy lane `src` to lane `dst` across every byte group."""
function move_lane!(blocked::AbstractVector{UInt8}, ng::Int, src::Int, dst::Int)
    src == dst && return nothing
    sb = src ÷ BLOCK
    sl = src % BLOCK
    db = dst ÷ BLOCK
    dl = dst % BLOCK
    @inbounds for g in 0:(ng - 1)
        blocked[byte_offset(db, g, dl, ng)] = blocked[byte_offset(sb, g, sl, ng)]
    end
    nothing
end

"""
    extract_codes_lane!(codes, blocked, lane, dim, bits)

Inverse of [`write_codes_lane!`]: recover the `dim` code values for a
lane from the blocked layout.
"""
function extract_codes_lane!(codes::AbstractVector{UInt8}, blocked::AbstractVector{UInt8},
                             lane::Int, dim::Int, bits::Int)
    ng = n_byte_groups(dim, bits)
    b = lane ÷ BLOCK
    l = lane % BLOCK
    chunks = dim ÷ 8
    @inbounds if bits == 2
        for c in 0:(chunks - 1)
            o = 8c
            b0 = blocked[byte_offset(b, 2c, l, ng)]
            b1 = blocked[byte_offset(b, 2c + 1, l, ng)]
            codes[o + 1] = (b0 >> 6) & 0x3
            codes[o + 2] = (b0 >> 4) & 0x3
            codes[o + 3] = (b0 >> 2) & 0x3
            codes[o + 4] = b0 & 0x3
            codes[o + 5] = (b1 >> 6) & 0x3
            codes[o + 6] = (b1 >> 4) & 0x3
            codes[o + 7] = (b1 >> 2) & 0x3
            codes[o + 8] = b1 & 0x3
        end
    else
        for c in 0:(chunks - 1)
            o = 8c
            for k in 0:3
                byte = blocked[byte_offset(b, 4c + k, l, ng)]
                codes[o + 2k + 1] = (byte >> 4) & 0x0f
                codes[o + 2k + 2] = byte & 0x0f
            end
        end
    end
    nothing
end
