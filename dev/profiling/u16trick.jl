using Random
using Printf
using TurboVec

const BR = join(["i32 $i" for i in vcat(0:15, 0:15, 0:15, 0:15)], ", ")
const CAT = join(["i32 $i" for i in 0:63], ", ")
const B0MASK = join(["i32 $v" for v in vcat([[i, 32 + i] for i in 0:15]...)], ", ")
const B1MASK = join(["i32 $v" for v in vcat([[16 + i, 48 + i] for i in 0:15]...)], ", ")

const FLUSH = """
  %s1 = shl <32 x i16> %a1n, splat (i16 8)
  %L0 = sub <32 x i16> %a0n, %s1
  %s2 = shl <32 x i16> %a3n, splat (i16 8)
  %L1 = sub <32 x i16> %a2n, %s2
  %t0b0 = shufflevector <32 x i16> %L0, <32 x i16> %a1n, <32 x i32> <$B0MASK>
  %t0b1 = shufflevector <32 x i16> %L0, <32 x i16> %a1n, <32 x i32> <$B1MASK>
  %t1b0 = shufflevector <32 x i16> %L1, <32 x i16> %a3n, <32 x i32> <$B0MASK>
  %t1b1 = shufflevector <32 x i16> %L1, <32 x i16> %a3n, <32 x i32> <$B1MASK>
  %e0b0 = uitofp <32 x i16> %t0b0 to <32 x float>
  %e0b1 = uitofp <32 x i16> %t0b1 to <32 x float>
  %e1b0 = uitofp <32 x i16> %t1b0 to <32 x float>
  %e1b1 = uitofp <32 x i16> %t1b1 to <32 x float>
  %F0x = fadd <32 x float> %F0v, %e0b0
  %F1x = fadd <32 x float> %F1v, %e0b1
  %F2x = fadd <32 x float> %F2v, %e1b0
  %F3x = fadd <32 x float> %F3v, %e1b1
"""

const IR = """
declare <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8>, <64 x i8>)
define void @u16trick(ptr %c0, ptr %c1, ptr %lut, i64 %ng,
                      float %scale, float %bias, ptr %out) #0 {
entry:
  br label %loop
loop:
  %g = phi i64 [ 0, %entry ], [ %gn, %cont ]
  %a0v = phi <32 x i16> [ zeroinitializer, %entry ], [ %a0c, %cont ]
  %a1v = phi <32 x i16> [ zeroinitializer, %entry ], [ %a1c, %cont ]
  %a2v = phi <32 x i16> [ zeroinitializer, %entry ], [ %a2c, %cont ]
  %a3v = phi <32 x i16> [ zeroinitializer, %entry ], [ %a3c, %cont ]
  %F0v = phi <32 x float> [ zeroinitializer, %entry ], [ %F0c, %cont ]
  %F1v = phi <32 x float> [ zeroinitializer, %entry ], [ %F1c, %cont ]
  %F2v = phi <32 x float> [ zeroinitializer, %entry ], [ %F2c, %cont ]
  %F3v = phi <32 x float> [ zeroinitializer, %entry ], [ %F3c, %cont ]
  %off = mul i64 %g, 32
  %p0 = getelementptr i8, ptr %c0, i64 %off
  %p1 = getelementptr i8, ptr %c1, i64 %off
  %tl = getelementptr i8, ptr %lut, i64 %off
  %th = getelementptr i8, ptr %tl, i64 16
  %v0 = load <32 x i8>, ptr %p0, align 1
  %v1 = load <32 x i8>, ptr %p1, align 1
  %v = shufflevector <32 x i8> %v0, <32 x i8> %v1, <64 x i32> <$CAT>
  %vshift = lshr <64 x i8> %v, splat (i8 4)
  %clo = and <64 x i8> %v, splat (i8 15)
  %chi = and <64 x i8> %vshift, splat (i8 15)
  %lr = load <16 x i8>, ptr %tl, align 1
  %lt = shufflevector <16 x i8> %lr, <16 x i8> poison, <64 x i32> <$BR>
  %hr = load <16 x i8>, ptr %th, align 1
  %ht = shufflevector <16 x i8> %hr, <16 x i8> poison, <64 x i32> <$BR>
  %r0 = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %lt, <64 x i8> %chi)
  %r1 = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %ht, <64 x i8> %clo)
  %r0w = bitcast <64 x i8> %r0 to <32 x i16>
  %r1w = bitcast <64 x i8> %r1 to <32 x i16>
  %r0h = lshr <32 x i16> %r0w, splat (i16 8)
  %r1h = lshr <32 x i16> %r1w, splat (i16 8)
  %a0n = add <32 x i16> %a0v, %r0w
  %a1n = add <32 x i16> %a1v, %r0h
  %a2n = add <32 x i16> %a2v, %r1w
  %a3n = add <32 x i16> %a3v, %r1h
  %gn = add i64 %g, 1
  %dof = and i64 %gn, 255
  %doflush = icmp eq i64 %dof, 0
  br i1 %doflush, label %flush, label %cont

flush:
$FLUSH
  br label %cont

cont:
  %a0c = phi <32 x i16> [ zeroinitializer, %flush ], [ %a0n, %loop ]
  %a1c = phi <32 x i16> [ zeroinitializer, %flush ], [ %a1n, %loop ]
  %a2c = phi <32 x i16> [ zeroinitializer, %flush ], [ %a2n, %loop ]
  %a3c = phi <32 x i16> [ zeroinitializer, %flush ], [ %a3n, %loop ]
  %F0c = phi <32 x float> [ %F0x, %flush ], [ %F0v, %loop ]
  %F1c = phi <32 x float> [ %F1x, %flush ], [ %F1v, %loop ]
  %F2c = phi <32 x float> [ %F2x, %flush ], [ %F2v, %loop ]
  %F3c = phi <32 x float> [ %F3x, %flush ], [ %F3v, %loop ]
  %done = icmp eq i64 %gn, %ng
  br i1 %done, label %fin, label %loop

fin:
  %fs1 = shl <32 x i16> %a1c, splat (i16 8)
  %fL0 = sub <32 x i16> %a0c, %fs1
  %fs2 = shl <32 x i16> %a3c, splat (i16 8)
  %fL1 = sub <32 x i16> %a2c, %fs2
  %ft0b0 = shufflevector <32 x i16> %fL0, <32 x i16> %a1c, <32 x i32> <$B0MASK>
  %ft0b1 = shufflevector <32 x i16> %fL0, <32 x i16> %a1c, <32 x i32> <$B1MASK>
  %ft1b0 = shufflevector <32 x i16> %fL1, <32 x i16> %a3c, <32 x i32> <$B0MASK>
  %ft1b1 = shufflevector <32 x i16> %fL1, <32 x i16> %a3c, <32 x i32> <$B1MASK>
  %fe0b0 = uitofp <32 x i16> %ft0b0 to <32 x float>
  %fe0b1 = uitofp <32 x i16> %ft0b1 to <32 x float>
  %fe1b0 = uitofp <32 x i16> %ft1b0 to <32 x float>
  %fe1b1 = uitofp <32 x i16> %ft1b1 to <32 x float>
  %T0 = fadd <32 x float> %F0c, %fe0b0
  %T1 = fadd <32 x float> %F1c, %fe0b1
  %T2 = fadd <32 x float> %F2c, %fe1b0
  %T3 = fadd <32 x float> %F3c, %fe1b1
  %totb0 = fadd <32 x float> %T0, %T2
  %totb1 = fadd <32 x float> %T1, %T3
  %scv = insertelement <32 x float> poison, float %scale, i32 0
  %scv2 = shufflevector <32 x float> %scv, <32 x float> poison, <32 x i32> zeroinitializer
  %biv = insertelement <32 x float> poison, float %bias, i32 0
  %biv2 = shufflevector <32 x float> %biv, <32 x float> poison, <32 x i32> zeroinitializer
  %m0 = fmul <32 x float> %totb0, %scv2
  %m1 = fmul <32 x float> %totb1, %scv2
  %b0 = fadd <32 x float> %m0, %biv2
  %b1 = fadd <32 x float> %m1, %biv2
  store <32 x float> %b0, ptr %out, align 4
  %out1 = getelementptr float, ptr %out, i64 32
  store <32 x float> %b1, ptr %out1, align 4
  ret void
}
attributes #0 = { "target-features"="+avx512f,+avx512bw" }
"""

function u16trick(c0::Ptr{UInt8}, c1::Ptr{UInt8}, lut::Ptr{UInt8}, ng::Int,
                  scale::Float32, bias::Float32, out::Ptr{Float32})
    Base.llvmcall((IR, "u16trick"), Cvoid,
                  Tuple{Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Int, Float32, Float32,
                        Ptr{Float32}},
                  c0, c1, lut, ng, scale, bias, out)
    nothing
end

function scalar_pair(codes, lut, ng, base, stride, scale, bias)
    out = zeros(Float32, 64)
    for l in 1:32
        for (bi, off) in ((0, base), (1, base + stride))
            a = Int32(0)
            for g in 0:(ng - 1)
                byte = codes[off + g * 32 + l]
                a += Int32(lut[g * 32 + Int(byte >> 4) + 1]) +
                     Int32(lut[g * 32 + 16 + Int(byte & 15) + 1])
            end
            out[bi * 32 + l] = scale * Float32(a) + bias
        end
    end
    out
end

function main()
    # Deterministic single-group check: codes all 0x12 (hi=1, lo=2),
    # table entry hi=1 -> 5, lo=2 -> 7, so every lane should total 12.
    let ng = 1
        codes = fill(0x12, 700 + 32)
        lut = zeros(UInt8, 32)
        lut[2] = 5        # 0-based entry 1 (high-nibble table)
        lut[16 + 3] = 7   # 0-based entry 2 (low-nibble table)
        out = zeros(Float32, 64)
        GC.@preserve codes lut out begin
            u16trick(pointer(codes), pointer(codes) + 700, pointer(lut), ng,
                     0.5f0, 0.0f0, pointer(out))
        end
        println("simple: uniq=", unique(out), " expect 6.0")
    end
    Random.seed!(1)
    for ng in (1, 2, 3, 255, 256, 257, 384, 385, 600)
        stride = 700
        codes = rand(UInt8, stride + ng * 32)
        lut = UInt8.(rand(0:127, ng * 32))
        out = zeros(Float32, 64)
        GC.@preserve codes lut out begin
            u16trick(pointer(codes), pointer(codes) + stride, pointer(lut), ng,
                     0.31f0, -7.5f0, pointer(out))
        end
        want = scalar_pair(codes, lut, ng, 0, stride, 0.31f0, -7.5f0)
        ok = reinterpret.(UInt32, out) == reinterpret.(UInt32, want)
        @printf("ng=%d exact=%s maxdiff=%g\n", ng, ok, maximum(abs.(out .- want)))
        flush(stdout)
    end

    # Timing on a full index vs the shipped pair kernel.
    n, ng = 100_000, 384
    nb = n ÷ 32
    stride = ng * 32
    codes = rand(UInt8, nb * stride)
    lut = UInt8.(rand(0:127, ng * 32))
    out = Vector{Float32}(undef, 64)
    GC.@preserve codes lut out begin
        pc = pointer(codes)
        pl = pointer(lut)
        po = pointer(out)
        t_new = @elapsed for _ in 1:3
            for b in 0:2:(nb - 2)
                u16trick(pc + b * stride, pc + (b + 1) * stride, pl, ng,
                         1.0f0, 0.0f0, po)
            end
        end
        t_old = @elapsed for _ in 1:3
            for b in 0:2:(nb - 2)
                TurboVec.scan_pair_avx512!(pc + b * stride, pc + (b + 1) * stride,
                                           pl, ng, 1.0f0, 0.0f0, po)
            end
        end
        @printf("u16 trick:   %.2f ms/scan\n", t_new / 3 * 1000)
        @printf("shipped:     %.2f ms/scan\n", t_old / 3 * 1000)
        @printf("speedup:     %.2fx\n", t_old / t_new)
    end
end

main()
