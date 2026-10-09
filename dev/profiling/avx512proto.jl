using Random
using Printf

const BR_MASK = join(["i32 $i" for i in vcat(0:15, 0:15, 0:15, 0:15)], ", ")
const LO32 = join(["i32 $i" for i in 0:31], ", ")
const HI32 = join(["i32 $i" for i in 32:63], ", ")
const CAT = join(["i32 $i" for i in 0:63], ", ")

const IR512 = """
declare <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8>, <64 x i8>)
define void @scan_pair(ptr %codes, ptr %lut, i64 %ng, i64 %stride,
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
  %cp = getelementptr i8, ptr %codes, i64 %off
  %cp2 = getelementptr i8, ptr %cp, i64 %stride
  %tp = getelementptr i8, ptr %lut, i64 %off
  %tp2 = getelementptr i8, ptr %tp, i64 16
  %v0 = load <32 x i8>, ptr %cp, align 1
  %v1 = load <32 x i8>, ptr %cp2, align 1
  %v = shufflevector <32 x i8> %v0, <32 x i8> %v1, <64 x i32> <$CAT>
  %hi.idx = lshr <64 x i8> %v, splat (i8 4)
  %lo.idx = and <64 x i8> %v, splat (i8 15)
  %hr = load <16 x i8>, ptr %tp, align 1
  %ht = shufflevector <16 x i8> %hr, <16 x i8> poison, <64 x i32> <$BR_MASK>
  %lr = load <16 x i8>, ptr %tp2, align 1
  %lt = shufflevector <16 x i8> %lr, <16 x i8> poison, <64 x i32> <$BR_MASK>
  %hil = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %ht, <64 x i8> %hi.idx)
  %lol = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %lt, <64 x i8> %lo.idx)
  %sum = add <64 x i8> %hil, %lol
  %s0 = shufflevector <64 x i8> %sum, <64 x i8> poison, <32 x i32> <$LO32>
  %s1 = shufflevector <64 x i8> %sum, <64 x i8> poison, <32 x i32> <$HI32>
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

function scan_pair(codes::Ptr{UInt8}, lut::Ptr{UInt8}, ng::Int, stride::Int,
                   scale::Float32, bias::Float32, out::Ptr{Float32})
    Base.llvmcall((IR512, "scan_pair"), Cvoid,
                  Tuple{Ptr{UInt8}, Ptr{UInt8}, Int, Int, Float32, Float32, Ptr{Float32}},
                  codes, lut, ng, stride, scale, bias, out)
    nothing
end

function scalar_pair(codes, lut, ng, b, nb, stride, scale, bias)
    out = zeros(Float32, 64)
    for l in 1:32
        a = Int32(0)
        for g in 0:(ng - 1)
            byte = codes[g * 32 + l]
            a += Int32(lut[g * 32 + Int(byte >> 4) + 1]) +
                 Int32(lut[g * 32 + 16 + Int(byte & 15) + 1])
        end
        out[l] = scale * Float32(a) + bias
        a2 = Int32(0)
        for g in 0:(ng - 1)
            off = stride + g * 32 + l
            byte = codes[off]
            a2 += Int32(lut[g * 32 + Int(byte >> 4) + 1]) +
                  Int32(lut[g * 32 + 16 + Int(byte & 15) + 1])
        end
        out[32 + l] = scale * Float32(a2) + bias
    end
    out
end

function main()
    Random.seed!(1)
    for ng in (1, 2, 3, 255, 256, 257, 384, 385, 512)
        stride = 700
        codes = rand(UInt8, stride + ng * 32)
        lut = UInt8.(rand(0:127, ng * 32))
        out = zeros(Float32, 64)
        GC.@preserve codes lut out begin
            scan_pair(pointer(codes), pointer(lut), ng, stride, 0.31f0, -7.5f0,
                      pointer(out))
        end
        want = scalar_pair(codes, lut, ng, 0, 0, stride, 0.31f0, -7.5f0)
        @printf("ng=%d equal=%s maxdiff=%g\n", ng, out == want,
                maximum(abs.(out .- want)))
        flush(stdout)
    end
    # timing on a full index
    n, ng = 100_000, 384
    nb = n ÷ 32
    stride = ng * 32
    codes = rand(UInt8, (nb + 1) * stride)
    lut = UInt8.(rand(0:127, ng * 32))
    out = Vector{Float32}(undef, 64)
    GC.@preserve codes lut out begin
        pc = pointer(codes); pl = pointer(lut); po = pointer(out)
        t = @elapsed for _ in 1:3
            for b in 0:2:(nb - 2)
                scan_pair(pc + b * stride, pl, ng, stride, 1.0f0, 0.0f0, po)
            end
        end
        @printf("pair kernel: %.2f ms/scan\n", t / 3 * 1000)
    end
end

main()
