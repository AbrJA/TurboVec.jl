using Random
using Printf
using TurboVec

const BR = join(["i32 $i" for i in vcat(0:15, 0:15, 0:15, 0:15)], ", ")
const CAT = join(["i32 $i" for i in 0:63], ", ")
const LO = join(["i32 $i" for i in 0:31], ", ")
const HI = join(["i32 $i" for i in 32:63], ", ")

const IR2Q = """
declare <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8>, <64 x i8>)
define void @pair2(ptr %c0, ptr %c1, ptr %la, ptr %lb, i64 %ng,
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
  %v = shufflevector <32 x i8> %v0, <32 x i8> %v1, <64 x i32> <$CAT>
  %hidx = lshr <64 x i8> %v, splat (i8 4)
  %loidx = and <64 x i8> %v, splat (i8 15)
  %har = load <16 x i8>, ptr %ta, align 1
  %hat = shufflevector <16 x i8> %har, <16 x i8> poison, <64 x i32> <$BR>
  %lar = load <16 x i8>, ptr %ta2, align 1
  %lat = shufflevector <16 x i8> %lar, <16 x i8> poison, <64 x i32> <$BR>
  %hbr = load <16 x i8>, ptr %tb, align 1
  %hbt = shufflevector <16 x i8> %hbr, <16 x i8> poison, <64 x i32> <$BR>
  %lbr = load <16 x i8>, ptr %tb2, align 1
  %lbt = shufflevector <16 x i8> %lbr, <16 x i8> poison, <64 x i32> <$BR>
  %hhA = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %hat, <64 x i8> %hidx)
  %llA = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %lat, <64 x i8> %loidx)
  %hhB = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %hbt, <64 x i8> %hidx)
  %llB = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %lbt, <64 x i8> %loidx)
  %sA = add <64 x i8> %hhA, %llA
  %sB = add <64 x i8> %hhB, %llB
  %sA0 = shufflevector <64 x i8> %sA, <64 x i8> poison, <32 x i32> <$LO>
  %sA1 = shufflevector <64 x i8> %sA, <64 x i8> poison, <32 x i32> <$HI>
  %sB0 = shufflevector <64 x i8> %sB, <64 x i8> poison, <32 x i32> <$LO>
  %sB1 = shufflevector <64 x i8> %sB, <64 x i8> poison, <32 x i32> <$HI>
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

function pair2(c0::Ptr{UInt8}, c1::Ptr{UInt8}, la::Ptr{UInt8}, lb::Ptr{UInt8}, ng::Int,
               sa::Float32, ba::Float32, sb::Float32, bb::Float32,
               oa::Ptr{Float32}, ob::Ptr{Float32})
    Base.llvmcall((IR2Q, "pair2"), Cvoid,
                  Tuple{Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Int,
                        Float32, Float32, Float32, Float32, Ptr{Float32}, Ptr{Float32}},
                  c0, c1, la, lb, ng, sa, ba, sb, bb, oa, ob)
    nothing
end

function scalar_pair1(codes, lut, ng, base, scale, bias)
    out = zeros(Float32, 32)
    for l in 1:32
        a = Int32(0)
        for g in 0:(ng - 1)
            byte = codes[base + g * 32 + l]
            a += Int32(lut[g * 32 + Int(byte >> 4) + 1]) +
                 Int32(lut[g * 32 + 16 + Int(byte & 15) + 1])
        end
        out[l] = scale * Float32(a) + bias
    end
    out
end

function single_kernel_scan(pc::Ptr{UInt8}, pl::Ptr{UInt8}, ng::Int, nb::Int,
                           stride::Int, po::Ptr{Float32})
    for b in 0:2:(nb - 2)
        TurboVec.scan_pair_avx512!(pc + b * stride, pc + (b + 1) * stride, pl, ng,
                                   1.0f0, 0.0f0, po)
    end
    nothing
end

function main()
    Random.seed!(1)
    n, ng = 100_000, 384
    nb = n ÷ 32
    stride = ng * 32
    codes = rand(UInt8, (nb + 1) * stride)
    la = UInt8.(rand(0:127, ng * 32))
    lb = UInt8.(rand(0:127, ng * 32))
    oa = zeros(Float32, 64)
    ob = zeros(Float32, 64)

    # correctness for one pair of blocks
    GC.@preserve codes la lb oa ob begin
        pair2(pointer(codes), pointer(codes) + stride, pointer(la), pointer(lb), ng,
              0.5f0, 1.0f0, -0.25f0, 2.0f0, pointer(oa), pointer(ob))
    end
    wa = scalar_pair1(codes, la, ng, 0, 0.5f0, 1.0f0)
    wb = scalar_pair1(codes, lb, ng, 0, -0.25f0, 2.0f0)
    println("correct A: ", oa[1:32] == wa, "  B: ", ob[1:32] == wb)

    GC.@preserve codes la lb oa ob begin
        pc = pointer(codes)
        pla = pointer(la)
        plb = pointer(lb)
        poa = pointer(oa)
        pob = pointer(ob)
        # 1-query baseline: the shipped single-query pair kernel.
        single = zeros(Float32, 64)
        psingle = pointer(single)
        GC.@preserve single single_kernel_scan(pc, pla, ng, nb, stride, psingle)
        t1 = @elapsed for _ in 1:3
            single_kernel_scan(pc, pla, ng, nb, stride, psingle)
        end
        t2 = @elapsed for _ in 1:3
            for b in 0:2:(nb - 2)
                pair2(pc + b * stride, pc + (b + 1) * stride, pla, plb, ng,
                      1.0f0, 0.0f0, 1.0f0, 0.0f0, poa, pob)
            end
        end
        @printf("1-query scan (shipped pair kernel): %.2f ms\n", t1 / 3 * 1000)
        @printf("2-query scan (pair2 kernel):        %.2f ms\n", t2 / 3 * 1000)
        @printf("ratio 2q/1q = %.2f  (two separate 1q scans would be ~2.0)\n", t2 / t1)
    end
end

main()
