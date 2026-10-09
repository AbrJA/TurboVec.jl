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

function _detect_avx2()
    Sys.ARCH === :x86_64 || return false
    if isdefined(Base, :BinaryPlatforms) && isdefined(Base.BinaryPlatforms, :CPUID)
        cp = Base.BinaryPlatforms.CPUID
        return cp.test_cpu_feature(cp.JL_X86_avx2)
    end
    try
        Sys.islinux() && return occursin("avx2", lowercase(read("/proc/cpuinfo", String)))
    catch
    end
    false
end

const HAS_AVX2 = _detect_avx2()
