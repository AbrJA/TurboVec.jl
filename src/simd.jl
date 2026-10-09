# AVX2 scan kernel via `Base.llvmcall`.
#
# Scores one 32-vector block across all byte groups with `vpshufb`
# nibble-table lookups: each `vpshufb` resolves 32 code bytes (one per
# lane) at once, against a 16-entry u8 sub-table replicated into both
# 128-bit lanes. The integer sums (and the final f32 `scale*x + bias`)
# are exactly what the scalar kernel computes, so results are
# bit-identical; only the throughput differs.
#
# Runtime-gated on AVX2: on any other target the scalar path runs.
# The aarch64 NEON kernels at the bottom of this file follow the same
# integer-accumulation rules, so all three paths agree bit-for-bit.

const _PSHUFB_DECL = "declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)"

const _BROADCAST_MASK = join(["i32 $i" for i in vcat(0:15, 0:15)], ", ")
const _LOWHALF_MASK = join(["i32 $i" for i in 0:15], ", ")
const _HIGHHALF_MASK = join(["i32 $i" for i in 16:31], ", ")
const _LO8_MASK = join(["i32 $i" for i in 0:7], ", ")
const _HI8_MASK = join(["i32 $i" for i in 8:15], ", ")

const SCAN_IR_AVX2 = """
$_PSHUFB_DECL
define void @scan_block(ptr %codes, ptr %lut, i64 %ng,
                        float %scale, float %bias, ptr %out) #0 {
entry:
  br label %loop

loop:
  %g = phi i64 [ 0, %entry ], [ %g.next, %loop ]
  %a0 = phi <16 x i32> [ zeroinitializer, %entry ], [ %a0.next, %loop ]
  %a1 = phi <16 x i32> [ zeroinitializer, %entry ], [ %a1.next, %loop ]
  %off = mul i64 %g, 32
  %cp = getelementptr i8, ptr %codes, i64 %off
  %tp = getelementptr i8, ptr %lut, i64 %off
  %tp2 = getelementptr i8, ptr %tp, i64 16
  %v = load <32 x i8>, ptr %cp, align 1
  %hi.idx = lshr <32 x i8> %v, splat (i8 4)
  %lo.idx = and <32 x i8> %v, splat (i8 15)
  %hraw = load <16 x i8>, ptr %tp, align 1
  %htab = shufflevector <16 x i8> %hraw, <16 x i8> poison, <32 x i32> <$_BROADCAST_MASK>
  %lraw = load <16 x i8>, ptr %tp2, align 1
  %ltab = shufflevector <16 x i8> %lraw, <16 x i8> poison, <32 x i32> <$_BROADCAST_MASK>
  %hil = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %htab, <32 x i8> %hi.idx)
  %lol = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %ltab, <32 x i8> %lo.idx)
  %sum = add <32 x i8> %hil, %lol
  %s0 = shufflevector <32 x i8> %sum, <32 x i8> poison, <16 x i32> <$_LOWHALF_MASK>
  %s1 = shufflevector <32 x i8> %sum, <32 x i8> poison, <16 x i32> <$_HIGHHALF_MASK>
  %w0 = zext <16 x i8> %s0 to <16 x i32>
  %w1 = zext <16 x i8> %s1 to <16 x i32>
  %a0.next = add <16 x i32> %a0, %w0
  %a1.next = add <16 x i32> %a1, %w1
  %g.next = add i64 %g, 1
  %done = icmp eq i64 %g.next, %ng
  br i1 %done, label %fin, label %loop

fin:
  %f0 = uitofp <16 x i32> %a0.next to <16 x float>
  %f1 = uitofp <16 x i32> %a1.next to <16 x float>
  %scv = insertelement <16 x float> poison, float %scale, i32 0
  %scv2 = shufflevector <16 x float> %scv, <16 x float> poison, <16 x i32> zeroinitializer
  %biv = insertelement <16 x float> poison, float %bias, i32 0
  %biv2 = shufflevector <16 x float> %biv, <16 x float> poison, <16 x i32> zeroinitializer
  %m0 = fmul <16 x float> %f0, %scv2
  %m1 = fmul <16 x float> %f1, %scv2
  %b0 = fadd <16 x float> %m0, %biv2
  %b1 = fadd <16 x float> %m1, %biv2
  store <16 x float> %b0, ptr %out, align 4
  %out1 = getelementptr float, ptr %out, i64 16
  store <16 x float> %b1, ptr %out1, align 4
  ret void
}

attributes #0 = { "target-features"="+avx2" }
"""

function scan_block_avx2!(codes::Ptr{UInt8}, lut::Ptr{UInt8}, ng::Int,
                          scale::Float32, bias::Float32, out::Ptr{Float32})
    Base.llvmcall((SCAN_IR_AVX2, "scan_block"), Cvoid,
                  Tuple{Ptr{UInt8},Ptr{UInt8},Int,Float32,Float32,Ptr{Float32}},
                  codes, lut, ng, scale, bias, out)
    nothing
end

# Two queries per code pass on AVX2: one 32-code block per iteration is
# shuffled against both queries' nibble tables, so the panel of per-group
# indices is computed once and the codes are loaded once. Per-lane sums
# accumulate in u16 and are flushed into u32 every 256 groups
# (2 * 127 * 256 = 65024 < 2^16), keeping the final integer totals and
# the single `scale * Float32(total) + bias` bit-identical to the scalar
# kernel.
const SCAN_PAIR2_IR_AVX2 = """
$_PSHUFB_DECL
define void @scan_pair2(ptr %codes, ptr %la, ptr %lb, i64 %ng,
                        float %sa, float %ba, float %sb, float %bb,
                        ptr %oa, ptr %ob) #0 {
entry:
  br label %loop

loop:
  %g = phi i64 [ 0, %entry ], [ %gn, %cont ]
  %a0 = phi <16 x i16> [ zeroinitializer, %entry ], [ %a0c, %cont ]
  %a1 = phi <16 x i16> [ zeroinitializer, %entry ], [ %a1c, %cont ]
  %b0 = phi <16 x i16> [ zeroinitializer, %entry ], [ %b0c, %cont ]
  %b1 = phi <16 x i16> [ zeroinitializer, %entry ], [ %b1c, %cont ]
  %A0 = phi <16 x i32> [ zeroinitializer, %entry ], [ %A0c, %cont ]
  %A1 = phi <16 x i32> [ zeroinitializer, %entry ], [ %A1c, %cont ]
  %B0 = phi <16 x i32> [ zeroinitializer, %entry ], [ %B0c, %cont ]
  %B1 = phi <16 x i32> [ zeroinitializer, %entry ], [ %B1c, %cont ]
  %off = mul i64 %g, 32
  %cp = getelementptr i8, ptr %codes, i64 %off
  %ta = getelementptr i8, ptr %la, i64 %off
  %ta2 = getelementptr i8, ptr %ta, i64 16
  %tb = getelementptr i8, ptr %lb, i64 %off
  %tb2 = getelementptr i8, ptr %tb, i64 16
  %v = load <32 x i8>, ptr %cp, align 1
  %hi.idx = lshr <32 x i8> %v, splat (i8 4)
  %lo.idx = and <32 x i8> %v, splat (i8 15)
  %har = load <16 x i8>, ptr %ta, align 1
  %hat = shufflevector <16 x i8> %har, <16 x i8> poison, <32 x i32> <$_BROADCAST_MASK>
  %lar = load <16 x i8>, ptr %ta2, align 1
  %lat = shufflevector <16 x i8> %lar, <16 x i8> poison, <32 x i32> <$_BROADCAST_MASK>
  %hbr = load <16 x i8>, ptr %tb, align 1
  %hbt = shufflevector <16 x i8> %hbr, <16 x i8> poison, <32 x i32> <$_BROADCAST_MASK>
  %lbr = load <16 x i8>, ptr %tb2, align 1
  %lbt = shufflevector <16 x i8> %lbr, <16 x i8> poison, <32 x i32> <$_BROADCAST_MASK>
  %hhA = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %hat, <32 x i8> %hi.idx)
  %llA = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %lat, <32 x i8> %lo.idx)
  %hhB = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %hbt, <32 x i8> %hi.idx)
  %llB = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %lbt, <32 x i8> %lo.idx)
  %sA = add <32 x i8> %hhA, %llA
  %sB = add <32 x i8> %hhB, %llB
  %sA0 = shufflevector <32 x i8> %sA, <32 x i8> poison, <16 x i32> <$_LOWHALF_MASK>
  %sA1 = shufflevector <32 x i8> %sA, <32 x i8> poison, <16 x i32> <$_HIGHHALF_MASK>
  %sB0 = shufflevector <32 x i8> %sB, <32 x i8> poison, <16 x i32> <$_LOWHALF_MASK>
  %sB1 = shufflevector <32 x i8> %sB, <32 x i8> poison, <16 x i32> <$_HIGHHALF_MASK>
  %wA0 = zext <16 x i8> %sA0 to <16 x i16>
  %wA1 = zext <16 x i8> %sA1 to <16 x i16>
  %wB0 = zext <16 x i8> %sB0 to <16 x i16>
  %wB1 = zext <16 x i8> %sB1 to <16 x i16>
  %a0n = add <16 x i16> %a0, %wA0
  %a1n = add <16 x i16> %a1, %wA1
  %b0n = add <16 x i16> %b0, %wB0
  %b1n = add <16 x i16> %b1, %wB1
  %gn = add i64 %g, 1
  %dof = and i64 %gn, 255
  %doflush = icmp eq i64 %dof, 0
  br i1 %doflush, label %flush, label %cont

flush:
  %eA0 = zext <16 x i16> %a0n to <16 x i32>
  %eA1 = zext <16 x i16> %a1n to <16 x i32>
  %eB0 = zext <16 x i16> %b0n to <16 x i32>
  %eB1 = zext <16 x i16> %b1n to <16 x i32>
  %A0b = add <16 x i32> %A0, %eA0
  %A1b = add <16 x i32> %A1, %eA1
  %B0b = add <16 x i32> %B0, %eB0
  %B1b = add <16 x i32> %B1, %eB1
  br label %cont

cont:
  %A0c = phi <16 x i32> [ %A0b, %flush ], [ %A0, %loop ]
  %A1c = phi <16 x i32> [ %A1b, %flush ], [ %A1, %loop ]
  %B0c = phi <16 x i32> [ %B0b, %flush ], [ %B0, %loop ]
  %B1c = phi <16 x i32> [ %B1b, %flush ], [ %B1, %loop ]
  %a0c = phi <16 x i16> [ zeroinitializer, %flush ], [ %a0n, %loop ]
  %a1c = phi <16 x i16> [ zeroinitializer, %flush ], [ %a1n, %loop ]
  %b0c = phi <16 x i16> [ zeroinitializer, %flush ], [ %b0n, %loop ]
  %b1c = phi <16 x i16> [ zeroinitializer, %flush ], [ %b1n, %loop ]
  %done = icmp eq i64 %gn, %ng
  br i1 %done, label %fin, label %loop

fin:
  %feA0 = zext <16 x i16> %a0c to <16 x i32>
  %feA1 = zext <16 x i16> %a1c to <16 x i32>
  %feB0 = zext <16 x i16> %b0c to <16 x i32>
  %feB1 = zext <16 x i16> %b1c to <16 x i32>
  %fA0i = add <16 x i32> %A0c, %feA0
  %fA1i = add <16 x i32> %A1c, %feA1
  %fB0i = add <16 x i32> %B0c, %feB0
  %fB1i = add <16 x i32> %B1c, %feB1
  %fA0 = uitofp <16 x i32> %fA0i to <16 x float>
  %fA1 = uitofp <16 x i32> %fA1i to <16 x float>
  %fB0 = uitofp <16 x i32> %fB0i to <16 x float>
  %fB1 = uitofp <16 x i32> %fB1i to <16 x float>
  %sav = insertelement <16 x float> poison, float %sa, i32 0
  %savv = shufflevector <16 x float> %sav, <16 x float> poison, <16 x i32> zeroinitializer
  %bav = insertelement <16 x float> poison, float %ba, i32 0
  %bavv = shufflevector <16 x float> %bav, <16 x float> poison, <16 x i32> zeroinitializer
  %sbv = insertelement <16 x float> poison, float %sb, i32 0
  %sbvv = shufflevector <16 x float> %sbv, <16 x float> poison, <16 x i32> zeroinitializer
  %bbv = insertelement <16 x float> poison, float %bb, i32 0
  %bbvv = shufflevector <16 x float> %bbv, <16 x float> poison, <16 x i32> zeroinitializer
  %mA0 = fmul <16 x float> %fA0, %savv
  %mA1 = fmul <16 x float> %fA1, %savv
  %mB0 = fmul <16 x float> %fB0, %sbvv
  %mB1 = fmul <16 x float> %fB1, %sbvv
  %rA0 = fadd <16 x float> %mA0, %bavv
  %rA1 = fadd <16 x float> %mA1, %bavv
  %rB0 = fadd <16 x float> %mB0, %bbvv
  %rB1 = fadd <16 x float> %mB1, %bbvv
  store <16 x float> %rA0, ptr %oa, align 4
  %oa1 = getelementptr float, ptr %oa, i64 16
  store <16 x float> %rA1, ptr %oa1, align 4
  store <16 x float> %rB0, ptr %ob, align 4
  %ob1 = getelementptr float, ptr %ob, i64 16
  store <16 x float> %rB1, ptr %ob1, align 4
  ret void
}

attributes #0 = { "target-features"="+avx2" }
"""

function scan_pair2_avx2!(codes::Ptr{UInt8}, la::Ptr{UInt8}, lb::Ptr{UInt8},
                          ng::Int, sa::Float32, ba::Float32, sb::Float32,
                          bb::Float32, oa::Ptr{Float32}, ob::Ptr{Float32})
    Base.llvmcall((SCAN_PAIR2_IR_AVX2, "scan_pair2"), Cvoid,
                  Tuple{Ptr{UInt8},Ptr{UInt8},Ptr{UInt8},Int,Float32,
                        Float32,Float32,Float32,Ptr{Float32},Ptr{Float32}},
                  codes, la, lb, ng, sa, ba, sb, bb, oa, ob)
    nothing
end

# CPU feature probe. `Base.BinaryPlatforms.CPUID` is the only portable
# feature source in Base (there is no public API), so it is wrapped
# defensively: any missing binding, unexpected constant type or probe
# failure falls back to `false`, i.e. the scalar path, rather than
# breaking package load on an unknown toolchain.
function _cpu_feature(name::Symbol)
    Sys.ARCH === :x86_64 || return false
    try
        cp = Base.BinaryPlatforms.CPUID
        isdefined(cp, name) || return false
        cp.test_cpu_feature(getfield(cp, name))
    catch
        false
    end
end

const HAS_AVX2 = _cpu_feature(:JL_X86_avx2)
const HAS_AVX512BW = _cpu_feature(:JL_X86_avx512f) &&
                     _cpu_feature(:JL_X86_avx512bw)

# NEON (Advanced SIMD) is mandatory on aarch64, so no probe is needed;
# every other architecture takes the scalar path.
const HAS_NEON = Sys.ARCH === :aarch64

const _CAT64_MASK = join(["i32 $i" for i in 0:63], ", ")
const _BR16X4_MASK = join(["i32 $i" for i in vcat(0:15, 0:15, 0:15, 0:15)], ", ")
const _LO32_MASK = join(["i32 $i" for i in 0:31], ", ")
const _HI32_MASK = join(["i32 $i" for i in 32:63], ", ")

# 64-lane pair kernel: two 32-vector blocks scored per group against one
# pair of table broadcasts. Bit-identical integer sums to the single-block
# kernel; u16 lanes are flushed into u32 every 256 groups (2 * 127 * 256
# < 2^16).
const SCAN_PAIR_IR_AVX512 = """
declare <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8>, <64 x i8>)
define void @scan_pair(ptr %c0, ptr %c1, ptr %lut, i64 %ng,
                       float %scale, float %bias, ptr %out) #0 {
entry:
  br label %loop
loop:
  %g = phi i64 [ 0, %entry ], [ %g.next, %cont ]
  %a0 = phi <32 x i16> [ zeroinitializer, %entry ], [ %a0c, %cont ]
  %a1 = phi <32 x i16> [ zeroinitializer, %entry ], [ %a1c, %cont ]
  %A0 = phi <32 x i32> [ zeroinitializer, %entry ], [ %A0c, %cont ]
  %A1 = phi <32 x i32> [ zeroinitializer, %entry ], [ %A1c, %cont ]
  %off = mul i64 %g, 32
  %p0 = getelementptr i8, ptr %c0, i64 %off
  %p1 = getelementptr i8, ptr %c1, i64 %off
  %tp = getelementptr i8, ptr %lut, i64 %off
  %tp2 = getelementptr i8, ptr %tp, i64 16
  %v0 = load <32 x i8>, ptr %p0, align 1
  %v1 = load <32 x i8>, ptr %p1, align 1
  %v = shufflevector <32 x i8> %v0, <32 x i8> %v1, <64 x i32> <$_CAT64_MASK>
  %hi.idx = lshr <64 x i8> %v, splat (i8 4)
  %lo.idx = and <64 x i8> %v, splat (i8 15)
  %hr = load <16 x i8>, ptr %tp, align 1
  %ht = shufflevector <16 x i8> %hr, <16 x i8> poison, <64 x i32> <$_BR16X4_MASK>
  %lr = load <16 x i8>, ptr %tp2, align 1
  %lt = shufflevector <16 x i8> %lr, <16 x i8> poison, <64 x i32> <$_BR16X4_MASK>
  %hil = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %ht, <64 x i8> %hi.idx)
  %lol = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %lt, <64 x i8> %lo.idx)
  %sum = add <64 x i8> %hil, %lol
  %s0 = shufflevector <64 x i8> %sum, <64 x i8> poison, <32 x i32> <$_LO32_MASK>
  %s1 = shufflevector <64 x i8> %sum, <64 x i8> poison, <32 x i32> <$_HI32_MASK>
  %w0 = zext <32 x i8> %s0 to <32 x i16>
  %w1 = zext <32 x i8> %s1 to <32 x i16>
  %a0n = add <32 x i16> %a0, %w0
  %a1n = add <32 x i16> %a1, %w1
  %g.next = add i64 %g, 1
  %dof = and i64 %g.next, 255
  %doflush = icmp eq i64 %dof, 0
  br i1 %doflush, label %flush, label %cont

flush:
  %ea0 = zext <32 x i16> %a0n to <32 x i32>
  %ea1 = zext <32 x i16> %a1n to <32 x i32>
  %A0b = add <32 x i32> %A0, %ea0
  %A1b = add <32 x i32> %A1, %ea1
  br label %cont

cont:
  %A0c = phi <32 x i32> [ %A0b, %flush ], [ %A0, %loop ]
  %A1c = phi <32 x i32> [ %A1b, %flush ], [ %A1, %loop ]
  %a0c = phi <32 x i16> [ zeroinitializer, %flush ], [ %a0n, %loop ]
  %a1c = phi <32 x i16> [ zeroinitializer, %flush ], [ %a1n, %loop ]
  %done = icmp eq i64 %g.next, %ng
  br i1 %done, label %fin, label %loop

fin:
  %ea0f = zext <32 x i16> %a0c to <32 x i32>
  %ea1f = zext <32 x i16> %a1c to <32 x i32>
  %F0 = add <32 x i32> %A0c, %ea0f
  %F1 = add <32 x i32> %A1c, %ea1f
  %f0 = uitofp <32 x i32> %F0 to <32 x float>
  %f1 = uitofp <32 x i32> %F1 to <32 x float>
  %scv = insertelement <32 x float> poison, float %scale, i32 0
  %scv2 = shufflevector <32 x float> %scv, <32 x float> poison, <32 x i32> zeroinitializer
  %biv = insertelement <32 x float> poison, float %bias, i32 0
  %biv2 = shufflevector <32 x float> %biv, <32 x float> poison, <32 x i32> zeroinitializer
  %m0 = fmul <32 x float> %f0, %scv2
  %m1 = fmul <32 x float> %f1, %scv2
  %b0 = fadd <32 x float> %m0, %biv2
  %b1 = fadd <32 x float> %m1, %biv2
  store <32 x float> %b0, ptr %out, align 4
  %out1 = getelementptr float, ptr %out, i64 32
  store <32 x float> %b1, ptr %out1, align 4
  ret void
}

attributes #0 = { "target-features"="+avx512f,+avx512bw" }
"""

function scan_pair_avx512!(c0::Ptr{UInt8}, c1::Ptr{UInt8}, lut::Ptr{UInt8}, ng::Int,
                           scale::Float32, bias::Float32, out::Ptr{Float32})
    Base.llvmcall((SCAN_PAIR_IR_AVX512, "scan_pair"), Cvoid,
                  Tuple{Ptr{UInt8},Ptr{UInt8},Ptr{UInt8},Int,Float32,Float32,
                        Ptr{Float32}},
                  c0, c1, lut, ng, scale, bias, out)
    nothing
end

# Two queries per code pass: the same 64 code bytes are shuffled against
# both queries' tables, so a pair of queries costs ~1.4 single-query
# passes instead of ~2.0 and reads the codes once. Scores remain
# bit-identical to the single-query kernel.
const SCAN_PAIR2_IR_AVX512 = """
declare <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8>, <64 x i8>)
define void @scan_pair2(ptr %c0, ptr %c1, ptr %la, ptr %lb, i64 %ng,
                        float %sa, float %ba, float %sb, float %bb,
                        ptr %oa, ptr %ob) #0 {
entry:
  br label %loop
loop:
  %g = phi i64 [ 0, %entry ], [ %gn, %cont ]
  %a0 = phi <32 x i16> [ zeroinitializer, %entry ], [ %a0c, %cont ]
  %a1 = phi <32 x i16> [ zeroinitializer, %entry ], [ %a1c, %cont ]
  %b0 = phi <32 x i16> [ zeroinitializer, %entry ], [ %b0c, %cont ]
  %b1 = phi <32 x i16> [ zeroinitializer, %entry ], [ %b1c, %cont ]
  %A0 = phi <32 x i32> [ zeroinitializer, %entry ], [ %A0c, %cont ]
  %A1 = phi <32 x i32> [ zeroinitializer, %entry ], [ %A1c, %cont ]
  %B0 = phi <32 x i32> [ zeroinitializer, %entry ], [ %B0c, %cont ]
  %B1 = phi <32 x i32> [ zeroinitializer, %entry ], [ %B1c, %cont ]
  %off = mul i64 %g, 32
  %p0 = getelementptr i8, ptr %c0, i64 %off
  %p1 = getelementptr i8, ptr %c1, i64 %off
  %ta = getelementptr i8, ptr %la, i64 %off
  %ta2 = getelementptr i8, ptr %ta, i64 16
  %tb = getelementptr i8, ptr %lb, i64 %off
  %tb2 = getelementptr i8, ptr %tb, i64 16
  %v0 = load <32 x i8>, ptr %p0, align 1
  %v1 = load <32 x i8>, ptr %p1, align 1
  %v = shufflevector <32 x i8> %v0, <32 x i8> %v1, <64 x i32> <$_CAT64_MASK>
  %hidx = lshr <64 x i8> %v, splat (i8 4)
  %loidx = and <64 x i8> %v, splat (i8 15)
  %har = load <16 x i8>, ptr %ta, align 1
  %hat = shufflevector <16 x i8> %har, <16 x i8> poison, <64 x i32> <$_BR16X4_MASK>
  %lar = load <16 x i8>, ptr %ta2, align 1
  %lat = shufflevector <16 x i8> %lar, <16 x i8> poison, <64 x i32> <$_BR16X4_MASK>
  %hbr = load <16 x i8>, ptr %tb, align 1
  %hbt = shufflevector <16 x i8> %hbr, <16 x i8> poison, <64 x i32> <$_BR16X4_MASK>
  %lbr = load <16 x i8>, ptr %tb2, align 1
  %lbt = shufflevector <16 x i8> %lbr, <16 x i8> poison, <64 x i32> <$_BR16X4_MASK>
  %hhA = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %hat, <64 x i8> %hidx)
  %llA = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %lat, <64 x i8> %loidx)
  %hhB = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %hbt, <64 x i8> %hidx)
  %llB = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %lbt, <64 x i8> %loidx)
  %sA = add <64 x i8> %hhA, %llA
  %sB = add <64 x i8> %hhB, %llB
  %sA0 = shufflevector <64 x i8> %sA, <64 x i8> poison, <32 x i32> <$_LO32_MASK>
  %sA1 = shufflevector <64 x i8> %sA, <64 x i8> poison, <32 x i32> <$_HI32_MASK>
  %sB0 = shufflevector <64 x i8> %sB, <64 x i8> poison, <32 x i32> <$_LO32_MASK>
  %sB1 = shufflevector <64 x i8> %sB, <64 x i8> poison, <32 x i32> <$_HI32_MASK>
  %wA0 = zext <32 x i8> %sA0 to <32 x i16>
  %wA1 = zext <32 x i8> %sA1 to <32 x i16>
  %wB0 = zext <32 x i8> %sB0 to <32 x i16>
  %wB1 = zext <32 x i8> %sB1 to <32 x i16>
  %a0n = add <32 x i16> %a0, %wA0
  %a1n = add <32 x i16> %a1, %wA1
  %b0n = add <32 x i16> %b0, %wB0
  %b1n = add <32 x i16> %b1, %wB1
  %gn = add i64 %g, 1
  %dof = and i64 %gn, 255
  %doflush = icmp eq i64 %dof, 0
  br i1 %doflush, label %flush, label %cont

flush:
  %eA0 = zext <32 x i16> %a0n to <32 x i32>
  %eA1 = zext <32 x i16> %a1n to <32 x i32>
  %eB0 = zext <32 x i16> %b0n to <32 x i32>
  %eB1 = zext <32 x i16> %b1n to <32 x i32>
  %A0b = add <32 x i32> %A0, %eA0
  %A1b = add <32 x i32> %A1, %eA1
  %B0b = add <32 x i32> %B0, %eB0
  %B1b = add <32 x i32> %B1, %eB1
  br label %cont

cont:
  %A0c = phi <32 x i32> [ %A0b, %flush ], [ %A0, %loop ]
  %A1c = phi <32 x i32> [ %A1b, %flush ], [ %A1, %loop ]
  %B0c = phi <32 x i32> [ %B0b, %flush ], [ %B0, %loop ]
  %B1c = phi <32 x i32> [ %B1b, %flush ], [ %B1, %loop ]
  %a0c = phi <32 x i16> [ zeroinitializer, %flush ], [ %a0n, %loop ]
  %a1c = phi <32 x i16> [ zeroinitializer, %flush ], [ %a1n, %loop ]
  %b0c = phi <32 x i16> [ zeroinitializer, %flush ], [ %b0n, %loop ]
  %b1c = phi <32 x i16> [ zeroinitializer, %flush ], [ %b1n, %loop ]
  %done = icmp eq i64 %gn, %ng
  br i1 %done, label %fin, label %loop

fin:
  %ea0 = zext <32 x i16> %a0c to <32 x i32>
  %ea1 = zext <32 x i16> %a1c to <32 x i32>
  %eb0 = zext <32 x i16> %b0c to <32 x i32>
  %eb1 = zext <32 x i16> %b1c to <32 x i32>
  %fA0i = add <32 x i32> %A0c, %ea0
  %fA1i = add <32 x i32> %A1c, %ea1
  %fB0i = add <32 x i32> %B0c, %eb0
  %fB1i = add <32 x i32> %B1c, %eb1
  %fA0 = uitofp <32 x i32> %fA0i to <32 x float>
  %fA1 = uitofp <32 x i32> %fA1i to <32 x float>
  %fB0 = uitofp <32 x i32> %fB0i to <32 x float>
  %fB1 = uitofp <32 x i32> %fB1i to <32 x float>
  %sav = insertelement <32 x float> poison, float %sa, i32 0
  %savv = shufflevector <32 x float> %sav, <32 x float> poison, <32 x i32> zeroinitializer
  %bav = insertelement <32 x float> poison, float %ba, i32 0
  %bavv = shufflevector <32 x float> %bav, <32 x float> poison, <32 x i32> zeroinitializer
  %sbv = insertelement <32 x float> poison, float %sb, i32 0
  %sbvv = shufflevector <32 x float> %sbv, <32 x float> poison, <32 x i32> zeroinitializer
  %bbv = insertelement <32 x float> poison, float %bb, i32 0
  %bbvv = shufflevector <32 x float> %bbv, <32 x float> poison, <32 x i32> zeroinitializer
  %mA0 = fmul <32 x float> %fA0, %savv
  %mA1 = fmul <32 x float> %fA1, %savv
  %mB0 = fmul <32 x float> %fB0, %sbvv
  %mB1 = fmul <32 x float> %fB1, %sbvv
  %rA0 = fadd <32 x float> %mA0, %bavv
  %rA1 = fadd <32 x float> %mA1, %bavv
  %rB0 = fadd <32 x float> %mB0, %bbvv
  %rB1 = fadd <32 x float> %mB1, %bbvv
  store <32 x float> %rA0, ptr %oa, align 4
  %oa1 = getelementptr float, ptr %oa, i64 32
  store <32 x float> %rA1, ptr %oa1, align 4
  store <32 x float> %rB0, ptr %ob, align 4
  %ob1 = getelementptr float, ptr %ob, i64 32
  store <32 x float> %rB1, ptr %ob1, align 4
  ret void
}

attributes #0 = { "target-features"="+avx512f,+avx512bw" }
"""

function scan_pair2_avx512!(c0::Ptr{UInt8}, c1::Ptr{UInt8}, la::Ptr{UInt8},
                            lb::Ptr{UInt8}, ng::Int, sa::Float32, ba::Float32,
                            sb::Float32, bb::Float32, oa::Ptr{Float32},
                            ob::Ptr{Float32})
    Base.llvmcall((SCAN_PAIR2_IR_AVX512, "scan_pair2"), Cvoid,
                  Tuple{Ptr{UInt8},Ptr{UInt8},Ptr{UInt8},Ptr{UInt8},Int,
                        Float32,Float32,Float32,Float32,Ptr{Float32},
                        Ptr{Float32}},
                  c0, c1, la, lb, ng, sa, ba, sb, bb, oa, ob)
    nothing
end

# ── aarch64 NEON kernels ───────────────────────────────────────────────────
#
# Same contract as the AVX2 kernels, one 128-bit lane at a time: each
# `tbl1` resolves 16 code bytes against a 16-entry nibble sub-table, the
# two nibble results are byte-added, widened to u16 and flushed into u32
# accumulators every 256 groups (254 * 256 < 2^16), so the final integer
# totals and the single `scale * Float32(total) + bias` stay bit-identical
# to the scalar kernel. NEON is mandatory on aarch64, so `HAS_NEON` needs
# no probe; every other architecture runs the scalar path.

const _TBL1_DECL = "declare <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8>, <16 x i8>)"

const SCAN_IR_NEON = """
$_TBL1_DECL
define void @scan_block(ptr %codes, ptr %lut, i64 %ng,
                        float %scale, float %bias, ptr %out) #0 {
entry:
  br label %loop

loop:
  %g = phi i64 [ 0, %entry ], [ %gn, %cont ]
  %a0 = phi <8 x i16> [ zeroinitializer, %entry ], [ %a0c, %cont ]
  %a1 = phi <8 x i16> [ zeroinitializer, %entry ], [ %a1c, %cont ]
  %a2 = phi <8 x i16> [ zeroinitializer, %entry ], [ %a2c, %cont ]
  %a3 = phi <8 x i16> [ zeroinitializer, %entry ], [ %a3c, %cont ]
  %A0 = phi <8 x i32> [ zeroinitializer, %entry ], [ %A0c, %cont ]
  %A1 = phi <8 x i32> [ zeroinitializer, %entry ], [ %A1c, %cont ]
  %A2 = phi <8 x i32> [ zeroinitializer, %entry ], [ %A2c, %cont ]
  %A3 = phi <8 x i32> [ zeroinitializer, %entry ], [ %A3c, %cont ]
  %off = mul i64 %g, 32
  %cp = getelementptr i8, ptr %codes, i64 %off
  %tp = getelementptr i8, ptr %lut, i64 %off
  %tp2 = getelementptr i8, ptr %tp, i64 16
  %htab = load <16 x i8>, ptr %tp, align 1
  %ltab = load <16 x i8>, ptr %tp2, align 1
  %v0 = load <16 x i8>, ptr %cp, align 1
  %hi0 = lshr <16 x i8> %v0, splat (i8 4)
  %lo0 = and <16 x i8> %v0, splat (i8 15)
  %h0 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %htab, <16 x i8> %hi0)
  %l0 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %ltab, <16 x i8> %lo0)
  %s0 = add <16 x i8> %h0, %l0
  %cp1 = getelementptr i8, ptr %cp, i64 16
  %v1 = load <16 x i8>, ptr %cp1, align 1
  %hi1 = lshr <16 x i8> %v1, splat (i8 4)
  %lo1 = and <16 x i8> %v1, splat (i8 15)
  %h1 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %htab, <16 x i8> %hi1)
  %l1 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %ltab, <16 x i8> %lo1)
  %s1 = add <16 x i8> %h1, %l1
  %s0lo = shufflevector <16 x i8> %s0, <16 x i8> poison, <8 x i32> <$_LO8_MASK>
  %s0hi = shufflevector <16 x i8> %s0, <16 x i8> poison, <8 x i32> <$_HI8_MASK>
  %s1lo = shufflevector <16 x i8> %s1, <16 x i8> poison, <8 x i32> <$_LO8_MASK>
  %s1hi = shufflevector <16 x i8> %s1, <16 x i8> poison, <8 x i32> <$_HI8_MASK>
  %w0 = zext <8 x i8> %s0lo to <8 x i16>
  %w1 = zext <8 x i8> %s0hi to <8 x i16>
  %w2 = zext <8 x i8> %s1lo to <8 x i16>
  %w3 = zext <8 x i8> %s1hi to <8 x i16>
  %a0n = add <8 x i16> %a0, %w0
  %a1n = add <8 x i16> %a1, %w1
  %a2n = add <8 x i16> %a2, %w2
  %a3n = add <8 x i16> %a3, %w3
  %gn = add i64 %g, 1
  %dof = and i64 %gn, 255
  %doflush = icmp eq i64 %dof, 0
  br i1 %doflush, label %flush, label %cont

flush:
  %e0 = zext <8 x i16> %a0n to <8 x i32>
  %e1 = zext <8 x i16> %a1n to <8 x i32>
  %e2 = zext <8 x i16> %a2n to <8 x i32>
  %e3 = zext <8 x i16> %a3n to <8 x i32>
  %A0b = add <8 x i32> %A0, %e0
  %A1b = add <8 x i32> %A1, %e1
  %A2b = add <8 x i32> %A2, %e2
  %A3b = add <8 x i32> %A3, %e3
  br label %cont

cont:
  %A0c = phi <8 x i32> [ %A0b, %flush ], [ %A0, %loop ]
  %A1c = phi <8 x i32> [ %A1b, %flush ], [ %A1, %loop ]
  %A2c = phi <8 x i32> [ %A2b, %flush ], [ %A2, %loop ]
  %A3c = phi <8 x i32> [ %A3b, %flush ], [ %A3, %loop ]
  %a0c = phi <8 x i16> [ zeroinitializer, %flush ], [ %a0n, %loop ]
  %a1c = phi <8 x i16> [ zeroinitializer, %flush ], [ %a1n, %loop ]
  %a2c = phi <8 x i16> [ zeroinitializer, %flush ], [ %a2n, %loop ]
  %a3c = phi <8 x i16> [ zeroinitializer, %flush ], [ %a3n, %loop ]
  %done = icmp eq i64 %gn, %ng
  br i1 %done, label %fin, label %loop

fin:
  %fe0 = zext <8 x i16> %a0c to <8 x i32>
  %fe1 = zext <8 x i16> %a1c to <8 x i32>
  %fe2 = zext <8 x i16> %a2c to <8 x i32>
  %fe3 = zext <8 x i16> %a3c to <8 x i32>
  %F0i = add <8 x i32> %A0c, %fe0
  %F1i = add <8 x i32> %A1c, %fe1
  %F2i = add <8 x i32> %A2c, %fe2
  %F3i = add <8 x i32> %A3c, %fe3
  %f0 = uitofp <8 x i32> %F0i to <8 x float>
  %f1 = uitofp <8 x i32> %F1i to <8 x float>
  %f2 = uitofp <8 x i32> %F2i to <8 x float>
  %f3 = uitofp <8 x i32> %F3i to <8 x float>
  %scv = insertelement <8 x float> poison, float %scale, i32 0
  %scv2 = shufflevector <8 x float> %scv, <8 x float> poison, <8 x i32> zeroinitializer
  %biv = insertelement <8 x float> poison, float %bias, i32 0
  %biv2 = shufflevector <8 x float> %biv, <8 x float> poison, <8 x i32> zeroinitializer
  %m0 = fmul <8 x float> %f0, %scv2
  %m1 = fmul <8 x float> %f1, %scv2
  %m2 = fmul <8 x float> %f2, %scv2
  %m3 = fmul <8 x float> %f3, %scv2
  %r0 = fadd <8 x float> %m0, %biv2
  %r1 = fadd <8 x float> %m1, %biv2
  %r2 = fadd <8 x float> %m2, %biv2
  %r3 = fadd <8 x float> %m3, %biv2
  store <8 x float> %r0, ptr %out, align 4
  %out1 = getelementptr float, ptr %out, i64 8
  store <8 x float> %r1, ptr %out1, align 4
  %out2 = getelementptr float, ptr %out, i64 16
  store <8 x float> %r2, ptr %out2, align 4
  %out3 = getelementptr float, ptr %out, i64 24
  store <8 x float> %r3, ptr %out3, align 4
  ret void
}

attributes #0 = { "target-features"="+neon" }
"""

function scan_block_neon!(codes::Ptr{UInt8}, lut::Ptr{UInt8}, ng::Int,
                          scale::Float32, bias::Float32, out::Ptr{Float32})
    Base.llvmcall((SCAN_IR_NEON, "scan_block"), Cvoid,
                  Tuple{Ptr{UInt8},Ptr{UInt8},Int,Float32,Float32,Ptr{Float32}},
                  codes, lut, ng, scale, bias, out)
    nothing
end

# Two queries per code pass: the same 32 code bytes and nibble indices are
# resolved against both queries' tables, so a pair shares the code reads.
const SCAN_PAIR2_IR_NEON = """
$_TBL1_DECL
define void @scan_pair2(ptr %codes, ptr %ta, ptr %tb, i64 %ng,
                        float %sa, float %ba, float %sb, float %bb,
                        ptr %oa, ptr %ob) #0 {
entry:
  br label %loop

loop:
  %g = phi i64 [ 0, %entry ], [ %gn, %cont ]
  %a0 = phi <8 x i16> [ zeroinitializer, %entry ], [ %a0c, %cont ]
  %a1 = phi <8 x i16> [ zeroinitializer, %entry ], [ %a1c, %cont ]
  %a2 = phi <8 x i16> [ zeroinitializer, %entry ], [ %a2c, %cont ]
  %a3 = phi <8 x i16> [ zeroinitializer, %entry ], [ %a3c, %cont ]
  %b0 = phi <8 x i16> [ zeroinitializer, %entry ], [ %b0c, %cont ]
  %b1 = phi <8 x i16> [ zeroinitializer, %entry ], [ %b1c, %cont ]
  %b2 = phi <8 x i16> [ zeroinitializer, %entry ], [ %b2c, %cont ]
  %b3 = phi <8 x i16> [ zeroinitializer, %entry ], [ %b3c, %cont ]
  %A0 = phi <8 x i32> [ zeroinitializer, %entry ], [ %A0c, %cont ]
  %A1 = phi <8 x i32> [ zeroinitializer, %entry ], [ %A1c, %cont ]
  %A2 = phi <8 x i32> [ zeroinitializer, %entry ], [ %A2c, %cont ]
  %A3 = phi <8 x i32> [ zeroinitializer, %entry ], [ %A3c, %cont ]
  %B0 = phi <8 x i32> [ zeroinitializer, %entry ], [ %B0c, %cont ]
  %B1 = phi <8 x i32> [ zeroinitializer, %entry ], [ %B1c, %cont ]
  %B2 = phi <8 x i32> [ zeroinitializer, %entry ], [ %B2c, %cont ]
  %B3 = phi <8 x i32> [ zeroinitializer, %entry ], [ %B3c, %cont ]
  %off = mul i64 %g, 32
  %cp = getelementptr i8, ptr %codes, i64 %off
  %tap = getelementptr i8, ptr %ta, i64 %off
  %tap2 = getelementptr i8, ptr %tap, i64 16
  %tbp = getelementptr i8, ptr %tb, i64 %off
  %tbp2 = getelementptr i8, ptr %tbp, i64 16
  %ha = load <16 x i8>, ptr %tap, align 1
  %la = load <16 x i8>, ptr %tap2, align 1
  %hb = load <16 x i8>, ptr %tbp, align 1
  %lb = load <16 x i8>, ptr %tbp2, align 1
  %v0 = load <16 x i8>, ptr %cp, align 1
  %hi0 = lshr <16 x i8> %v0, splat (i8 4)
  %lo0 = and <16 x i8> %v0, splat (i8 15)
  %hA0 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %ha, <16 x i8> %hi0)
  %lA0 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %la, <16 x i8> %lo0)
  %sA0 = add <16 x i8> %hA0, %lA0
  %hB0 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %hb, <16 x i8> %hi0)
  %lB0 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %lb, <16 x i8> %lo0)
  %sB0 = add <16 x i8> %hB0, %lB0
  %cp1 = getelementptr i8, ptr %cp, i64 16
  %v1 = load <16 x i8>, ptr %cp1, align 1
  %hi1 = lshr <16 x i8> %v1, splat (i8 4)
  %lo1 = and <16 x i8> %v1, splat (i8 15)
  %hA1 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %ha, <16 x i8> %hi1)
  %lA1 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %la, <16 x i8> %lo1)
  %sA1 = add <16 x i8> %hA1, %lA1
  %hB1 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %hb, <16 x i8> %hi1)
  %lB1 = call <16 x i8> @llvm.aarch64.neon.tbl1(<16 x i8> %lb, <16 x i8> %lo1)
  %sB1 = add <16 x i8> %hB1, %lB1
  %sA0lo = shufflevector <16 x i8> %sA0, <16 x i8> poison, <8 x i32> <$_LO8_MASK>
  %sA0hi = shufflevector <16 x i8> %sA0, <16 x i8> poison, <8 x i32> <$_HI8_MASK>
  %sA1lo = shufflevector <16 x i8> %sA1, <16 x i8> poison, <8 x i32> <$_LO8_MASK>
  %sA1hi = shufflevector <16 x i8> %sA1, <16 x i8> poison, <8 x i32> <$_HI8_MASK>
  %sB0lo = shufflevector <16 x i8> %sB0, <16 x i8> poison, <8 x i32> <$_LO8_MASK>
  %sB0hi = shufflevector <16 x i8> %sB0, <16 x i8> poison, <8 x i32> <$_HI8_MASK>
  %sB1lo = shufflevector <16 x i8> %sB1, <16 x i8> poison, <8 x i32> <$_LO8_MASK>
  %sB1hi = shufflevector <16 x i8> %sB1, <16 x i8> poison, <8 x i32> <$_HI8_MASK>
  %wA0 = zext <8 x i8> %sA0lo to <8 x i16>
  %wA1 = zext <8 x i8> %sA0hi to <8 x i16>
  %wA2 = zext <8 x i8> %sA1lo to <8 x i16>
  %wA3 = zext <8 x i8> %sA1hi to <8 x i16>
  %wB0 = zext <8 x i8> %sB0lo to <8 x i16>
  %wB1 = zext <8 x i8> %sB0hi to <8 x i16>
  %wB2 = zext <8 x i8> %sB1lo to <8 x i16>
  %wB3 = zext <8 x i8> %sB1hi to <8 x i16>
  %a0n = add <8 x i16> %a0, %wA0
  %a1n = add <8 x i16> %a1, %wA1
  %a2n = add <8 x i16> %a2, %wA2
  %a3n = add <8 x i16> %a3, %wA3
  %b0n = add <8 x i16> %b0, %wB0
  %b1n = add <8 x i16> %b1, %wB1
  %b2n = add <8 x i16> %b2, %wB2
  %b3n = add <8 x i16> %b3, %wB3
  %gn = add i64 %g, 1
  %dof = and i64 %gn, 255
  %doflush = icmp eq i64 %dof, 0
  br i1 %doflush, label %flush, label %cont

flush:
  %eA0 = zext <8 x i16> %a0n to <8 x i32>
  %eA1 = zext <8 x i16> %a1n to <8 x i32>
  %eA2 = zext <8 x i16> %a2n to <8 x i32>
  %eA3 = zext <8 x i16> %a3n to <8 x i32>
  %eB0 = zext <8 x i16> %b0n to <8 x i32>
  %eB1 = zext <8 x i16> %b1n to <8 x i32>
  %eB2 = zext <8 x i16> %b2n to <8 x i32>
  %eB3 = zext <8 x i16> %b3n to <8 x i32>
  %A0b = add <8 x i32> %A0, %eA0
  %A1b = add <8 x i32> %A1, %eA1
  %A2b = add <8 x i32> %A2, %eA2
  %A3b = add <8 x i32> %A3, %eA3
  %B0b = add <8 x i32> %B0, %eB0
  %B1b = add <8 x i32> %B1, %eB1
  %B2b = add <8 x i32> %B2, %eB2
  %B3b = add <8 x i32> %B3, %eB3
  br label %cont

cont:
  %A0c = phi <8 x i32> [ %A0b, %flush ], [ %A0, %loop ]
  %A1c = phi <8 x i32> [ %A1b, %flush ], [ %A1, %loop ]
  %A2c = phi <8 x i32> [ %A2b, %flush ], [ %A2, %loop ]
  %A3c = phi <8 x i32> [ %A3b, %flush ], [ %A3, %loop ]
  %B0c = phi <8 x i32> [ %B0b, %flush ], [ %B0, %loop ]
  %B1c = phi <8 x i32> [ %B1b, %flush ], [ %B1, %loop ]
  %B2c = phi <8 x i32> [ %B2b, %flush ], [ %B2, %loop ]
  %B3c = phi <8 x i32> [ %B3b, %flush ], [ %B3, %loop ]
  %a0c = phi <8 x i16> [ zeroinitializer, %flush ], [ %a0n, %loop ]
  %a1c = phi <8 x i16> [ zeroinitializer, %flush ], [ %a1n, %loop ]
  %a2c = phi <8 x i16> [ zeroinitializer, %flush ], [ %a2n, %loop ]
  %a3c = phi <8 x i16> [ zeroinitializer, %flush ], [ %a3n, %loop ]
  %b0c = phi <8 x i16> [ zeroinitializer, %flush ], [ %b0n, %loop ]
  %b1c = phi <8 x i16> [ zeroinitializer, %flush ], [ %b1n, %loop ]
  %b2c = phi <8 x i16> [ zeroinitializer, %flush ], [ %b2n, %loop ]
  %b3c = phi <8 x i16> [ zeroinitializer, %flush ], [ %b3n, %loop ]
  %done = icmp eq i64 %gn, %ng
  br i1 %done, label %fin, label %loop

fin:
  %feA0 = zext <8 x i16> %a0c to <8 x i32>
  %feA1 = zext <8 x i16> %a1c to <8 x i32>
  %feA2 = zext <8 x i16> %a2c to <8 x i32>
  %feA3 = zext <8 x i16> %a3c to <8 x i32>
  %feB0 = zext <8 x i16> %b0c to <8 x i32>
  %feB1 = zext <8 x i16> %b1c to <8 x i32>
  %feB2 = zext <8 x i16> %b2c to <8 x i32>
  %feB3 = zext <8 x i16> %b3c to <8 x i32>
  %FA0i = add <8 x i32> %A0c, %feA0
  %FA1i = add <8 x i32> %A1c, %feA1
  %FA2i = add <8 x i32> %A2c, %feA2
  %FA3i = add <8 x i32> %A3c, %feA3
  %FB0i = add <8 x i32> %B0c, %feB0
  %FB1i = add <8 x i32> %B1c, %feB1
  %FB2i = add <8 x i32> %B2c, %feB2
  %FB3i = add <8 x i32> %B3c, %feB3
  %fA0 = uitofp <8 x i32> %FA0i to <8 x float>
  %fA1 = uitofp <8 x i32> %FA1i to <8 x float>
  %fA2 = uitofp <8 x i32> %FA2i to <8 x float>
  %fA3 = uitofp <8 x i32> %FA3i to <8 x float>
  %fB0 = uitofp <8 x i32> %FB0i to <8 x float>
  %fB1 = uitofp <8 x i32> %FB1i to <8 x float>
  %fB2 = uitofp <8 x i32> %FB2i to <8 x float>
  %fB3 = uitofp <8 x i32> %FB3i to <8 x float>
  %sav = insertelement <8 x float> poison, float %sa, i32 0
  %savv = shufflevector <8 x float> %sav, <8 x float> poison, <8 x i32> zeroinitializer
  %bav = insertelement <8 x float> poison, float %ba, i32 0
  %bavv = shufflevector <8 x float> %bav, <8 x float> poison, <8 x i32> zeroinitializer
  %sbv = insertelement <8 x float> poison, float %sb, i32 0
  %sbvv = shufflevector <8 x float> %sbv, <8 x float> poison, <8 x i32> zeroinitializer
  %bbv = insertelement <8 x float> poison, float %bb, i32 0
  %bbvv = shufflevector <8 x float> %bbv, <8 x float> poison, <8 x i32> zeroinitializer
  %mA0 = fmul <8 x float> %fA0, %savv
  %mA1 = fmul <8 x float> %fA1, %savv
  %mA2 = fmul <8 x float> %fA2, %savv
  %mA3 = fmul <8 x float> %fA3, %savv
  %mB0 = fmul <8 x float> %fB0, %sbvv
  %mB1 = fmul <8 x float> %fB1, %sbvv
  %mB2 = fmul <8 x float> %fB2, %sbvv
  %mB3 = fmul <8 x float> %fB3, %sbvv
  %rA0 = fadd <8 x float> %mA0, %bavv
  %rA1 = fadd <8 x float> %mA1, %bavv
  %rA2 = fadd <8 x float> %mA2, %bavv
  %rA3 = fadd <8 x float> %mA3, %bavv
  %rB0 = fadd <8 x float> %mB0, %bbvv
  %rB1 = fadd <8 x float> %mB1, %bbvv
  %rB2 = fadd <8 x float> %mB2, %bbvv
  %rB3 = fadd <8 x float> %mB3, %bbvv
  store <8 x float> %rA0, ptr %oa, align 4
  %oa1 = getelementptr float, ptr %oa, i64 8
  store <8 x float> %rA1, ptr %oa1, align 4
  %oa2 = getelementptr float, ptr %oa, i64 16
  store <8 x float> %rA2, ptr %oa2, align 4
  %oa3 = getelementptr float, ptr %oa, i64 24
  store <8 x float> %rA3, ptr %oa3, align 4
  store <8 x float> %rB0, ptr %ob, align 4
  %ob1 = getelementptr float, ptr %ob, i64 8
  store <8 x float> %rB1, ptr %ob1, align 4
  %ob2 = getelementptr float, ptr %ob, i64 16
  store <8 x float> %rB2, ptr %ob2, align 4
  %ob3 = getelementptr float, ptr %ob, i64 24
  store <8 x float> %rB3, ptr %ob3, align 4
  ret void
}

attributes #0 = { "target-features"="+neon" }
"""

function scan_pair2_neon!(codes::Ptr{UInt8}, la::Ptr{UInt8}, lb::Ptr{UInt8},
                          ng::Int, sa::Float32, ba::Float32, sb::Float32,
                          bb::Float32, oa::Ptr{Float32}, ob::Ptr{Float32})
    Base.llvmcall((SCAN_PAIR2_IR_NEON, "scan_pair2"), Cvoid,
                  Tuple{Ptr{UInt8},Ptr{UInt8},Ptr{UInt8},Int,Float32,
                        Float32,Float32,Float32,Ptr{Float32},Ptr{Float32}},
                  codes, la, lb, ng, sa, ba, sb, bb, oa, ob)
    nothing
end
