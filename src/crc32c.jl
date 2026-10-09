# CRC-32C (Castagnoli) for the native file format's integrity footer.
#
# Pure Julia, slice-by-8: eight 256-entry tables are generated once at
# load time and eight bytes are consumed per iteration. No hardware
# intrinsics and no external dependency; the format's reader/writer and
# the corruption tests all use the same routine.

const _CRC32C_POLY = 0x82F63B78          # reflected 0x1EDC6F41
const _CRC32C_INIT = 0xffffffff

function _crc32c_tables()
    t1 = Vector{UInt32}(undef, 256)
    @inbounds for i in 0:255
        crc = UInt32(i)
        for _ in 1:8
            crc = (crc >> 1) ⊻ (UInt32(_CRC32C_POLY) & (zero(UInt32) - (crc & 0x00000001)))
        end
        t1[i + 1] = crc
    end
    tab = Vector{Vector{UInt32}}(undef, 8)
    tab[1] = t1
    @inbounds for k in 2:8
        prev = tab[k - 1]
        cur = Vector{UInt32}(undef, 256)
        for i in 0:255
            p = prev[i + 1]
            cur[i + 1] = t1[(p & 0xff) + 1] ⊻ (p >> 8)
        end
        tab[k] = cur
    end
    tab
end

const _CRC32C_TABLES = _crc32c_tables()

@inline function _crc32c_update(crc::UInt32, data::AbstractVector{UInt8})
    t1 = _CRC32C_TABLES[1]
    t2 = _CRC32C_TABLES[2]
    t3 = _CRC32C_TABLES[3]
    t4 = _CRC32C_TABLES[4]
    t5 = _CRC32C_TABLES[5]
    t6 = _CRC32C_TABLES[6]
    t7 = _CRC32C_TABLES[7]
    t8 = _CRC32C_TABLES[8]
    n = length(data)
    i = 1
    @inbounds while i + 7 <= n
        crc ⊻= UInt32(data[i]) | (UInt32(data[i + 1]) << 8) |
                (UInt32(data[i + 2]) << 16) | (UInt32(data[i + 3]) << 24)
        crc = t8[Int(crc & 0xff) + 1] ⊻
              t7[Int((crc >> 8) & 0xff) + 1] ⊻
              t6[Int((crc >> 16) & 0xff) + 1] ⊻
              t5[Int((crc >> 24) & 0xff) + 1] ⊻
              t4[Int(data[i + 4]) + 1] ⊻
              t3[Int(data[i + 5]) + 1] ⊻
              t2[Int(data[i + 6]) + 1] ⊻
              t1[Int(data[i + 7]) + 1]
        i += 8
    end
    @inbounds while i <= n
        crc = t1[Int((crc ⊻ data[i]) & 0xff) + 1] ⊻ (crc >> 8)
        i += 1
    end
    crc
end

"""One-shot CRC-32C of a byte buffer (internal; used by IO and tests)."""
_crc32c(data::AbstractVector{UInt8}) = _crc32c_update(_CRC32C_INIT, data) ⊻ _CRC32C_INIT
