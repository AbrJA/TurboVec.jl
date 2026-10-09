using Random, Printf, TurboVec

const BR = join(["i32 $i" for i in vcat(0:15, 0:15, 0:15, 0:15)], ", ")
const CAT = join(["i32 $i" for i in 0:63], ", ")
const LOW = join(["i32 $i" for i in 0:31], ", ")
const HIGH = join(["i32 $i" for i in 32:63], ", ")

function quad_ir()
    io = IOBuffer()
    print(io, "declare <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8>, <64 x i8>)\n")
    print(io, "define void @scan_quad(ptr %c0, ptr %c1, ")
    for q in 0:3
        print(io, "ptr %l$q, ")
    end
    print(io, "i64 %ng, ")
    for q in 0:3
        print(io, "float %s$q, float %b$q, ")
    end
    for q in 0:3
        print(io, "ptr %o$q", q == 3 ? ")" : ", ")
    end
    print(io, " #0 {\nentry:\n  br label %loop\n\nloop:\n")
    print(io, "  %g = phi i64 [ 0, %entry ], [ %gn, %cont ]\n")
    for q in 0:3
        print(io, "  %a$q.0 = phi <32 x i16> [ zeroinitializer, %entry ], [ %a$q.0c, %cont ]\n")
        print(io, "  %a$q.1 = phi <32 x i16> [ zeroinitializer, %entry ], [ %a$q.1c, %cont ]\n")
        print(io, "  %F$q.0 = phi <32 x float> [ zeroinitializer, %entry ], [ %F$q.0c, %cont ]\n")
        print(io, "  %F$q.1 = phi <32 x float> [ zeroinitializer, %entry ], [ %F$q.1c, %cont ]\n")
    end
    print(io, """
  %off = mul i64 %g, 32
  %p0 = getelementptr i8, ptr %c0, i64 %off
  %p1 = getelementptr i8, ptr %c1, i64 %off
  %v0 = load <32 x i8>, ptr %p0, align 1
  %v1 = load <32 x i8>, ptr %p1, align 1
  %v = shufflevector <32 x i8> %v0, <32 x i8> %v1, <64 x i32> <$CAT>
  %vs = lshr <64 x i8> %v, splat (i8 4)
  %clo = and <64 x i8> %v, splat (i8 15)
  %chi = and <64 x i8> %vs, splat (i8 15)
""")
    for q in 0:3
        print(io, "  %tl$q = getelementptr i8, ptr %l$q, i64 %off\n")
        print(io, "  %th$q = getelementptr i8, ptr %tl$q, i64 16\n")
        print(io, "  %lr$q = load <16 x i8>, ptr %tl$q, align 1\n")
        print(io, "  %lt$q = shufflevector <16 x i8> %lr$q, <16 x i8> poison, <64 x i32> <$BR>\n")
        print(io, "  %hr$q = load <16 x i8>, ptr %th$q, align 1\n")
        print(io, "  %ht$q = shufflevector <16 x i8> %hr$q, <16 x i8> poison, <64 x i32> <$BR>\n")
        print(io, "  %r$q.0 = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %lt$q, <64 x i8> %chi)\n")
        print(io, "  %r$q.1 = call <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8> %ht$q, <64 x i8> %clo)\n")
        print(io, "  %s$q.0 = shufflevector <64 x i8> %r$q.0, <64 x i8> poison, <32 x i32> <$LOW>\n")
        print(io, "  %s$q.1 = shufflevector <64 x i8> %r$q.0, <64 x i8> poison, <32 x i32> <$HIGH>\n")
        print(io, "  %t$q.0 = shufflevector <64 x i8> %r$q.1, <64 x i8> poison, <32 x i32> <$LOW>\n")
        print(io, "  %t$q.1 = shufflevector <64 x i8> %r$q.1, <64 x i8> poison, <32 x i32> <$HIGH>\n")
        print(io, "  %w$q.0 = zext <32 x i8> %s$q.0 to <32 x i16>\n")
        print(io, "  %w$q.1 = zext <32 x i8> %s$q.1 to <32 x i16>\n")
        print(io, "  %x$q.0 = zext <32 x i8> %t$q.0 to <32 x i16>\n")
        print(io, "  %x$q.1 = zext <32 x i8> %t$q.1 to <32 x i16>\n")
        print(io, "  %a$q.0n = add <32 x i16> %a$q.0, %w$q.0\n")
        print(io, "  %a$q.1n = add <32 x i16> %a$q.1, %w$q.1\n")
        print(io, "  %u$q.0n = add <32 x i16> %a$q.0n, %x$q.0\n")
        print(io, "  %u$q.1n = add <32 x i16> %a$q.1n, %x$q.1\n")
        # rename: keep u as the running accumulator, a as alias for phi naming
        print(io, "  %a$q.0u = add <32 x i16> %u$q.0n, zeroinitializer\n")
        print(io, "  %a$q.1u = add <32 x i16> %u$q.1n, zeroinitializer\n")
    end
    print(io, """
  %gn = add i64 %g, 1
  %dof = and i64 %gn, 255
  %doflush = icmp eq i64 %dof, 0
  br i1 %doflush, label %flush, label %cont

flush:
""")
    for q in 0:3
        print(io, "  %e$q.0 = uitofp <32 x i16> %a$q.0u to <32 x float>\n")
        print(io, "  %e$q.1 = uitofp <32 x i16> %a$q.1u to <32 x float>\n")
        print(io, "  %F$q.0f = fadd <32 x float> %F$q.0, %e$q.0\n")
        print(io, "  %F$q.1f = fadd <32 x float> %F$q.1, %e$q.1\n")
    end
    print(io, "  br label %cont\n\ncont:\n")
    for q in 0:3
        print(io, "  %a$q.0c = phi <32 x i16> [ zeroinitializer, %flush ], [ %a$q.0u, %loop ]\n")
        print(io, "  %a$q.1c = phi <32 x i16> [ zeroinitializer, %flush ], [ %a$q.1u, %loop ]\n")
        print(io, "  %F$q.0c = phi <32 x float> [ %F$q.0f, %flush ], [ %F$q.0, %loop ]\n")
        print(io, "  %F$q.1c = phi <32 x float> [ %F$q.1f, %flush ], [ %F$q.1, %loop ]\n")
    end
    print(io, """
  %done = icmp eq i64 %gn, %ng
  br i1 %done, label %fin, label %loop

fin:
""")
    for q in 0:3
        print(io, "  %fv$q.0 = uitofp <32 x i16> %a$q.0c to <32 x float>\n")
        print(io, "  %fv$q.1 = uitofp <32 x i16> %a$q.1c to <32 x float>\n")
        print(io, "  %tot$q.0 = fadd <32 x float> %F$q.0c, %fv$q.0\n")
        print(io, "  %tot$q.1 = fadd <32 x float> %F$q.1c, %fv$q.1\n")
        print(io, "  %sq$q = insertelement <32 x float> poison, float %s$q, i32 0\n")
        print(io, "  %sqv$q = shufflevector <32 x float> %sq$q, <32 x float> poison, <32 x i32> zeroinitializer\n")
        print(io, "  %bq$q = insertelement <32 x float> poison, float %b$q, i32 0\n")
        print(io, "  %bqv$q = shufflevector <32 x float> %bq$q, <32 x float> poison, <32 x i32> zeroinitializer\n")
        print(io, "  %m$q.0 = fmul <32 x float> %tot$q.0, %sqv$q\n")
        print(io, "  %m$q.1 = fmul <32 x float> %tot$q.1, %sqv$q\n")
        print(io, "  %r$q.0f = fadd <32 x float> %m$q.0, %bqv$q\n")
        print(io, "  %r$q.1f = fadd <32 x float> %m$q.1, %bqv$q\n")
        print(io, "  store <32 x float> %r$q.0f, ptr %o$q, align 4\n")
        print(io, "  %o$q.p = getelementptr float, ptr %o$q, i64 32\n")
        print(io, "  store <32 x float> %r$q.1f, ptr %o$q.p, align 4\n")
    end
    print(io, "  ret void\n}\n\nattributes #0 = { \"target-features\"=\"+avx512f,+avx512bw\" }\n")
    String(take!(io))
end

const QUAD_IR = quad_ir()

function scan_quad(c0, c1, l0, l1, l2, l3, ng, s0, b0, s1, b1, s2, b2, s3, b3, o0, o1, o2, o3)
    Base.llvmcall((QUAD_IR, "scan_quad"), Cvoid,
        Tuple{Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8},
              Int, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32,
              Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}},
        c0, c1, l0, l1, l2, l3, ng, s0, b0, s1, b1, s2, b2, s3, b3, o0, o1, o2, o3)
    nothing
end

function scalar_pair(codes, lut, ng, base, stride, scale, bias)
    out = zeros(Float32, 64)
    for l in 1:32, (bi, off) in ((0, base), (1, base + stride))
        a = Int32(0)
        for g in 0:(ng - 1)
            byte = codes[off + g * 32 + l]
            a += Int32(lut[g * 32 + Int(byte >> 4) + 1]) +
                 Int32(lut[g * 32 + 16 + Int(byte & 15) + 1])
        end
        out[bi * 32 + l] = scale * Float32(a) + bias
    end
    out
end

function main()
    Random.seed!(1)
    for ng in (2, 3, 255, 256, 257, 384, 385, 600)
        stride = 700
        codes = rand(UInt8, stride + ng * 32)
        luts = [UInt8.(rand(0:127, ng * 32)) for _ in 1:4]
        outs = [zeros(Float32, 64) for _ in 1:4]
        GC.@preserve codes luts outs begin
            scan_quad(pointer(codes), pointer(codes) + stride,
                      pointer(luts[1]), pointer(luts[2]), pointer(luts[3]), pointer(luts[4]),
                      ng, 0.31f0, -7.5f0, 0.5f0, 1.0f0, -0.25f0, 2.0f0, 1.5f0, 0.5f0,
                      pointer(outs[1]), pointer(outs[2]), pointer(outs[3]), pointer(outs[4]))
        end
        ok = true
        for q in 1:4
            want = scalar_pair(codes, luts[q], ng, 0, stride,
                               (0.31f0, 0.5f0, -0.25f0, 1.5f0)[q], (-7.5f0, 1.0f0, 2.0f0, 0.5f0)[q])
            ok &= reinterpret.(UInt32, outs[q]) == reinterpret.(UInt32, want)
        end
        @printf("ng=%d exact=%s\n", ng, ok)
        flush(stdout)
    end

    n, ng = 100_000, 384
    nb = n ÷ 32
    stride = ng * 32
    codes = rand(UInt8, nb * stride)
    luts = [UInt8.(rand(0:127, ng * 32)) for _ in 1:4]
    outs = [zeros(Float32, 64) for _ in 1:4]
    GC.@preserve codes luts outs begin
        pc = pointer(codes); pls = [pointer(l) for l in luts]; pos = [pointer(o) for o in outs]
        t4 = @elapsed for _ in 1:3
            for b in 0:2:(nb - 2)
                scan_quad(pc + b * stride, pc + (b + 1) * stride,
                          pls[1], pls[2], pls[3], pls[4], ng,
                          1.0f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0, 0.0f0,
                          pos[1], pos[2], pos[3], pos[4])
            end
        end
        t2 = @elapsed for _ in 1:3
            for b in 0:2:(nb - 2)
                TurboVec.scan_pair2_avx512!(pc + b * stride, pc + (b + 1) * stride,
                                            pls[1], pls[2], ng, 1.0f0, 0.0f0,
                                            1.0f0, 0.0f0, pos[1], pos[2])
            end
        end
        @printf("quad4:  %.3f ms per 4-query sweep (%.3f ms/query)\n", t4/3*1000, t4/3*1000/4)
        @printf("pair2:  %.3f ms per 2-query sweep (%.3f ms/query)\n", t2/3*1000, t2/3*1000/2)
    end
end

main()
