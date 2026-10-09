using TurboVec, Random, Printf

const BASE = TurboVec.SCAN_PAIR_IR_AVX512

function add_prefetch(ir, dist)
    ir = replace(ir,
        "declare <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8>, <64 x i8>)" =>
        "declare <64 x i8> @llvm.x86.avx512.pshuf.b.512(<64 x i8>, <64 x i8>)\n" *
        "declare void @llvm.prefetch(ptr, i32, i32, i32)")
    ir = replace(ir,
        "  %v0 = load <32 x i8>, ptr %p0, align 1" =>
        "  %pfa = getelementptr i8, ptr %c0, i64 $dist\n" *
        "  %pfb = getelementptr i8, ptr %c1, i64 $dist\n" *
        "  call void @llvm.prefetch(ptr %pfa, i32 0, i32 3, i32 1)\n" *
        "  call void @llvm.prefetch(ptr %pfb, i32 0, i32 3, i32 1)\n" *
        "  %v0 = load <32 x i8>, ptr %p0, align 1")
    ir
end

function mk(ir, name)
    @eval function $(Symbol(name))(c0::Ptr{UInt8}, c1::Ptr{UInt8}, lut::Ptr{UInt8},
                                   ng::Int, scale::Float32, bias::Float32, out::Ptr{Float32})
        Base.llvmcall(($ir, "scan_pair"), Cvoid,
                      Tuple{Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}, Int, Float32, Float32, Ptr{Float32}},
                      c0, c1, lut, ng, scale, bias, out)
    end
end

mk(BASE, :k_base)
mk(add_prefetch(BASE, 256), :k_pf256)
mk(add_prefetch(BASE, 512), :k_pf512)
mk(add_prefetch(BASE, 1024), :k_pf1024)

function scan(k, pc, pl, ng, nb, stride, po)
    for b in 0:2:(nb - 2)
        k(pc + b * stride, pc + (b + 1) * stride, pl, ng, 1.0f0, 0.0f0, po)
    end
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
        for (name, k) in (("base", k_base), ("pf256", k_pf256), ("pf512", k_pf512),
                          ("pf1024", k_pf1024))
            scan(k, pc, pl, ng, nb, stride, po)
            t = @elapsed for _ in 1:3
                scan(k, pc, pl, ng, nb, stride, po)
            end
            @printf("%-7s %.2f ms/scan\n", name, t / 3 * 1000)
            flush(stdout)
        end
    end
end

main()
