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

const _PSHUFB_DECL = "declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)"

const _BROADCAST_MASK = join(["i32 $i" for i in vcat(0:15, 0:15)], ", ")
const _LOWHALF_MASK = join(["i32 $i" for i in 0:15], ", ")
const _HIGHHALF_MASK = join(["i32 $i" for i in 16:31], ", ")

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
                  Tuple{Ptr{UInt8}, Ptr{UInt8}, Int, Float32, Float32, Ptr{Float32}},
                  codes, lut, ng, scale, bias, out)
    nothing
end

function _cpu_feature(name::Symbol)
    Sys.ARCH === :x86_64 || return false
    cp = Base.BinaryPlatforms.CPUID
    isdefined(cp, name) && return cp.test_cpu_feature(getfield(cp, name))
    false
end

const HAS_AVX2 = _cpu_feature(:JL_X86_avx2)
const HAS_AVX512BW = _cpu_feature(:JL_X86_avx512f) &&
                     _cpu_feature(:JL_X86_avx512bw)

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
                  Tuple{Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Int, Float32, Float32,
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
                  Tuple{Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Int,
                        Float32, Float32, Float32, Float32, Ptr{Float32},
                        Ptr{Float32}},
                  c0, c1, la, lb, ng, sa, ba, sb, bb, oa, ob)
    nothing
end
