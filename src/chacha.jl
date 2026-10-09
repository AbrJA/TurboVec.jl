# ChaCha8 keystream RNG.
#
# Bit-for-bit reproduction of `rand_chacha 0.3.1`'s `ChaCha8Rng` word
# stream, which turbovec freezes as the source of the rotation's sign
# flips and Fisher-Yates permutations. The buffer/read schedule mirrors
# `rand_core::block::BlockRng` (64-word buffer, `next_u64` reads two
# consecutive words least-significant first, with the buffer-boundary
# refill behaviour).

const CHACHA_CONSTANTS = (0x61707865, 0x3320646e, 0x79622d32, 0x6b206574)

mutable struct ChaCha8
    key::NTuple{8,UInt32}
    nonce::NTuple{2,UInt32}
    counter::UInt64
    buffer::Vector{UInt32}
    index::Int  # 1-based next word; length(buffer)+1 means empty
end

function ChaCha8(seed::AbstractVector{UInt8})
    length(seed) == 32 || throw(ArgumentError("ChaCha8 seed must be 32 bytes"))
    key = ntuple(i -> UInt32(seed[4i - 3]) | UInt32(seed[4i - 2]) << 8 |
                       UInt32(seed[4i - 1]) << 16 | UInt32(seed[4i]) << 24, 8)
    ChaCha8(key, (0x00000000, 0x00000000), 0x0000000000000000,
            Vector{UInt32}(undef, 64), 65)
end

@inline rotl32(x::UInt32, n::Int) = (x << n) | (x >> (32 - n))

@inline function chacha_qr(a::UInt32, b::UInt32, c::UInt32, d::UInt32)
    a += b; d ⊻= a; d = rotl32(d, 16)
    c += d; b ⊻= c; b = rotl32(b, 12)
    a += b; d ⊻= a; d = rotl32(d, 8)
    c += d; b ⊻= c; b = rotl32(b, 7)
    a, b, c, d
end

# One 16-word ChaCha8 block. `x` is the working state in the standard
# row-major layout (constants, key, counter, nonce).
function chacha8_block!(out::Vector{UInt32}, off::Int, key::NTuple{8,UInt32},
                        nonce::NTuple{2,UInt32}, counter::UInt64)
    @inbounds begin
        x0 = CHACHA_CONSTANTS[1]; x1 = CHACHA_CONSTANTS[2]
        x2 = CHACHA_CONSTANTS[3]; x3 = CHACHA_CONSTANTS[4]
        x4 = key[1]; x5 = key[2]; x6 = key[3]; x7 = key[4]
        x8 = key[5]; x9 = key[6]; x10 = key[7]; x11 = key[8]
        x12 = UInt32(counter & 0xffffffff); x13 = UInt32(counter >> 32)
        x14 = nonce[1]; x15 = nonce[2]
        w0, w1, w2, w3 = x0, x1, x2, x3
        w4, w5, w6, w7 = x4, x5, x6, x7
        w8, w9, w10, w11 = x8, x9, x10, x11
        w12, w13, w14, w15 = x12, x13, x14, x15
        for _ in 1:4  # ChaCha8 = 4 double rounds
            w0, w4, w8, w12 = chacha_qr(w0, w4, w8, w12)
            w1, w5, w9, w13 = chacha_qr(w1, w5, w9, w13)
            w2, w6, w10, w14 = chacha_qr(w2, w6, w10, w14)
            w3, w7, w11, w15 = chacha_qr(w3, w7, w11, w15)
            w0, w5, w10, w15 = chacha_qr(w0, w5, w10, w15)
            w1, w6, w11, w12 = chacha_qr(w1, w6, w11, w12)
            w2, w7, w8, w13 = chacha_qr(w2, w7, w8, w13)
            w3, w4, w9, w14 = chacha_qr(w3, w4, w9, w14)
        end
        out[off + 1] = w0 + x0;   out[off + 2] = w1 + x1
        out[off + 3] = w2 + x2;   out[off + 4] = w3 + x3
        out[off + 5] = w4 + x4;   out[off + 6] = w5 + x5
        out[off + 7] = w6 + x6;   out[off + 8] = w7 + x7
        out[off + 9] = w8 + x8;   out[off + 10] = w9 + x9
        out[off + 11] = w10 + x10; out[off + 12] = w11 + x11
        out[off + 13] = w12 + x12; out[off + 14] = w13 + x13
        out[off + 15] = w14 + x14; out[off + 16] = w15 + x15
    end
    nothing
end

# Refill the 64-word buffer with four consecutive blocks (counters
# `counter .. counter+3`), mirroring `refill4`.
function refill!(rng::ChaCha8)
    c = rng.counter
    @inbounds for b in 0:3
        chacha8_block!(rng.buffer, 16b, rng.key, rng.nonce, c + UInt64(b))
    end
    rng.counter = c + 4
    rng.index = 1
    nothing
end

@inline function next_u32!(rng::ChaCha8)
    rng.index > 64 && refill!(rng)
    @inbounds v = rng.buffer[rng.index]
    rng.index += 1
    v
end

# Mirrors `BlockRng::next_u64`: two consecutive words, least significant
# first, with the one-word-left boundary case spanning a refill.
@inline function next_u64!(rng::ChaCha8)
    i = rng.index
    if i < 64
        @inbounds v = UInt64(rng.buffer[i + 1]) << 32 | UInt64(rng.buffer[i])
        rng.index = i + 2
        return v
    elseif i > 64
        refill!(rng)
        @inbounds v = UInt64(rng.buffer[2]) << 32 | UInt64(rng.buffer[1])
        rng.index = 3
        return v
    else
        @inbounds x = UInt64(rng.buffer[64])
        refill!(rng)
        @inbounds y = UInt64(rng.buffer[1])
        rng.index = 2
        return (y << 32) | x
    end
end
