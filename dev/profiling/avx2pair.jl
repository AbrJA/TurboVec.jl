using Random, Printf
const BR = join(["i32 $i" for i in vcat(0:15, 0:15)], ", ")
const LO = join(["i32 $i" for i in 0:15], ", ")
const HI = join(["i32 $i" for i in 16:31], ", ")
const IR = """
declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)
define void @pair(ptr %c0, ptr %c1, ptr %lut, i64 %ng, float %scale, float %bias, ptr %out) #0 {
entry:
  br label %loop
loop:
  %g = phi i64 [0, %entry], [%gn, %loop]
  %a0 = phi <16 x i32> [zeroinitializer, %entry], [%a0a, %loop]
  %a1 = phi <16 x i32> [zeroinitializer, %entry], [%a1a, %loop]
  %b0 = phi <16 x i32> [zeroinitializer, %entry], [%b0a, %loop]
  %b1 = phi <16 x i32> [zeroinitializer, %entry], [%b1a, %loop]
  %off = mul i64 %g, 32
  %p0 = getelementptr i8, ptr %c0, i64 %off
  %p1 = getelementptr i8, ptr %c1, i64 %off
  %tp = getelementptr i8, ptr %lut, i64 %off
  %tp2 = getelementptr i8, ptr %tp, i64 16
  %hr = load <16 x i8>, ptr %tp, align 1
  %ht = shufflevector <16 x i8> %hr, <16 x i8> poison, <32 x i32> <$BR>
  %lr = load <16 x i8>, ptr %tp2, align 1
  %lt = shufflevector <16 x i8> %lr, <16 x i8> poison, <32 x i32> <$BR>
  %v0 = load <32 x i8>, ptr %p0, align 1
  %v1 = load <32 x i8>, ptr %p1, align 1
  %hi0 = lshr <32 x i8> %v0, splat (i8 4)
  %lo0 = and <32 x i8> %v0, splat (i8 15)
  %hi1 = lshr <32 x i8> %v1, splat (i8 4)
  %lo1 = and <32 x i8> %v1, splat (i8 15)
  %hh0 = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %ht, <32 x i8> %hi0)
  %ll0 = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %lt, <32 x i8> %lo0)
  %hh1 = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %ht, <32 x i8> %hi1)
  %ll1 = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %lt, <32 x i8> %lo1)
  %s0 = add <32 x i8> %hh0, %ll0
  %s1 = add <32 x i8> %hh1, %ll1
  %s0l = shufflevector <32 x i8> %s0, <32 x i8> poison, <16 x i32> <$LO>
  %s0h = shufflevector <32 x i8> %s0, <32 x i8> poison, <16 x i32> <$HI>
  %s1l = shufflevector <32 x i8> %s1, <32 x i8> poison, <16 x i32> <$LO>
  %s1h = shufflevector <32 x i8> %s1, <32 x i8> poison, <16 x i32> <$HI>
  %w00 = zext <16 x i8> %s0l to <16 x i32>
  %w01 = zext <16 x i8> %s0h to <16 x i32>
  %w10 = zext <16 x i8> %s1l to <16 x i32>
  %w11 = zext <16 x i8> %s1h to <16 x i32>
  %a0a = add <16 x i32> %a0, %w00
  %a1a = add <16 x i32> %a1, %w01
  %b0a = add <16 x i32> %b0, %w10
  %b1a = add <16 x i32> %b1, %w11
  %gn = add i64 %g, 1
  %d = icmp eq i64 %gn, %ng
  br i1 %d, label %fin, label %loop
fin:
  %scv = insertelement <16 x float> poison, float %scale, i32 0
  %scv2 = shufflevector <16 x float> %scv, <16 x float> poison, <16 x i32> zeroinitializer
  %biv = insertelement <16 x float> poison, float %bias, i32 0
  %biv2 = shufflevector <16 x float> %biv, <16 x float> poison, <16 x i32> zeroinitializer
  %fa0 = uitofp <16 x i32> %a0a to <16 x float>
  %fa1 = uitofp <16 x i32> %a1a to <16 x float>
  %fb0 = uitofp <16 x i32> %b0a to <16 x float>
  %fb1 = uitofp <16 x i32> %b1a to <16 x float>
  store <16 x float> %fa0, ptr %out, align 4
  %o1 = getelementptr float, ptr %out, i64 16
  store <16 x float> %fa1, ptr %o1, align 4
  %o2 = getelementptr float, ptr %out, i64 32
  store <16 x float> %fb0, ptr %o2, align 4
  %o3 = getelementptr float, ptr %out, i64 48
  store <16 x float> %fb1, ptr %o3, align 4
  ret void
}
attributes #0 = { "target-features"="+avx2" }
"""
function pair(c0::Ptr{UInt8}, c1::Ptr{UInt8}, lut::Ptr{UInt8}, ng::Int,
              scale::Float32, bias::Float32, out::Ptr{Float32})
    Base.llvmcall((IR, "pair"), Cvoid,
                  Tuple{Ptr{UInt8},Ptr{UInt8},Ptr{UInt8},Int,Float32,Float32,Ptr{Float32}},
                  c0, c1, lut, ng, scale, bias, out)
end
function main()
    Random.seed!(1)
    n, ng = 100_000, 384
    nb = n ÷ 32
    stride = ng * 32
    codes = rand(UInt8, nb * stride)
    lut = UInt8.(rand(0:127, ng * 32))
    out = Vector{Float32}(undef, 64)
    GC.@preserve codes lut out begin
        pc = pointer(codes); pl = pointer(lut); po = pointer(out)
        t = @elapsed for _ in 1:3
            for b in 0:2:(nb - 2)
                pair(pc + b*stride, pc + (b+1)*stride, pl, ng, 1.0f0, 0.0f0, po)
            end
        end
        @printf("avx2 pair: %.2f ms/scan\n", t/3*1000)
        # correctness spot check vs scalar
        ok = true
        for b in (0, 2, 100)
            pair(pc + b*stride, pc + (b+1)*stride, pl, ng, 1.0f0, 0.0f0, po)
            for l in 1:32
                a = Int32(0)
                for g in 0:ng-1
                    byte = codes[g*32 + b*32 + l]
                    a += Int32(lut[g*32+Int(byte>>4)+1]) + Int32(lut[g*32+16+Int(byte&15)+1])
                end
                ok &= (out[l] == Float32(a))
                a2 = Int32(0)
                for g in 0:ng-1
                    byte = codes[(b+1)*stride + g*32 + l]
                    a2 += Int32(lut[g*32+Int(byte>>4)+1]) + Int32(lut[g*32+16+Int(byte&15)+1])
                end
                ok &= (out[32+l] == Float32(a2))
            end
        end
        println("correct=", ok)
    end
end
main()
