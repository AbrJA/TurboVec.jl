using Random
using Printf

const BR_MASK = join(["i32 $i" for i in vcat(0:15, 0:15)], ", ")
const LO_MASK = join(["i32 $i" for i in 0:15], ", ")
const HI_MASK = join(["i32 $i" for i in 16:31], ", ")

function make_ir(bcast::Symbol, acc::Symbol)
    bcast_decl = bcast === :intr ?
        "declare <4 x i64> @llvm.x86.avx2.vbroadcasti128(ptr)" : ""
    bcast_fn = bcast === :intr ? "intr" : "shuf"
    accw = acc === :u16 ? "i16" : "i32"
    flush = acc === :u16 ? """
    %dof = and i64 %g, 255
    %doflush = icmp eq i64 %dof, 0
    br i1 %doflush, label %flush, label %cont

flush:
    %ea0 = zext <16 x i16> %a0 to <16 x i32>
    %ea1 = zext <16 x i16> %a1 to <16 x i32>
    %A0b = add <16 x i32> %A0, %ea0
    %A1b = add <16 x i32> %A1, %ea1
    br label %cont

cont:
    %A0n = phi <16 x i32> [ %A0b, %flush ], [ %A0, %loop ]
    %A1n = phi <16 x i32> [ %A1b, %flush ], [ %A1, %loop ]
    """ : ""
    acc_phi = acc === :u16 ? """
  %a0 = phi <16 x $accw> [ zeroinitializer, %entry ], [ %a0n, %cont ]
  %a1 = phi <16 x $accw> [ zeroinitializer, %entry ], [ %a1n, %cont ]
  %A0 = phi <16 x i32> [ zeroinitializer, %entry ], [ %A0n, %cont ]
  %A1 = phi <16 x i32> [ zeroinitializer, %entry ], [ %A1n, %cont ]
""" : ""
    acc_init32 = acc === :u32 ? """
  %a0 = phi <16 x i32> [ zeroinitializer, %entry ], [ %a0n, %loop ]
  %a1 = phi <16 x i32> [ zeroinitializer, %entry ], [ %a1n, %loop ]
""" : ""
    widen = acc === :u16 ? """
  %wa0 = zext <16 x i8> %s0 to <16 x i16>
  %wa1 = zext <16 x i8> %s1 to <16 x i16>
  %a0n = add <16 x i16> %a0, %wa0
  %a1n = add <16 x i16> %a1, %wa1
""" : """
  %wa0 = zext <16 x i8> %s0 to <16 x i32>
  %wa1 = zext <16 x i8> %s1 to <16 x i32>
  %a0n = add <16 x i32> %a0, %wa0
  %a1n = add <16 x i32> %a1, %wa1
"""
    fin = acc === :u16 ? """
  %fa0 = zext <16 x i16> %a0 to <16 x i32>
  %fa1 = zext <16 x i16> %a1 to <16 x i32>
  %F0 = add <16 x i32> %A0n, %fa0
  %F1 = add <16 x i32> %A1n, %fa1
  %f0 = uitofp <16 x i32> %F0 to <16 x float>
  %f1 = uitofp <16 x i32> %F1 to <16 x float>
""" : """
  %f0 = uitofp <16 x i32> %a0n to <16 x float>
  %f1 = uitofp <16 x i32> %a1n to <16 x float>
"""
    ht_load = bcast === :intr ?
        "%ht = bitcast <4 x i64> %hraw to <32 x i8>" :
        "%ht = shufflevector <16 x i8> %hr, <16 x i8> poison, <32 x i32> <$BR_MASK>"
    lt_load = bcast === :intr ?
        "%lt = bitcast <4 x i64> %lraw to <32 x i8>" :
        "%lt = shufflevector <16 x i8> %lr, <16 x i8> poison, <32 x i32> <$BR_MASK>"
    raw_load_h = bcast === :intr ?
        "%hraw = call <4 x i64> @llvm.x86.avx2.vbroadcasti128(ptr %tp)" :
        "%hr = load <16 x i8>, ptr %tp, align 1"
    raw_load_l = bcast === :intr ?
        "%lraw = call <4 x i64> @llvm.x86.avx2.vbroadcasti128(ptr %tp2)" :
        "%lr = load <16 x i8>, ptr %tp2, align 1"
    br_to = acc === :u16 ? "%cont" : "%loop"
    inc = acc === :u16 ? "%a0n, %cont" : "%a0n, %loop"
    """
$bcast_decl
declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)
define void @scan_block(ptr %codes, ptr %lut, i64 %ng, float %scale, float %bias, ptr %out) #0 {
entry:
  br label %loop
loop:
  %g = phi i64 [ 0, %entry ], [ %g.next, $br_to ]
$acc_init32$acc_phi  %off = mul i64 %g, 32
  %cp = getelementptr i8, ptr %codes, i64 %off
  %tp = getelementptr i8, ptr %lut, i64 %off
  %tp2 = getelementptr i8, ptr %tp, i64 16
  %v = load <32 x i8>, ptr %cp, align 1
  %hi.idx = lshr <32 x i8> %v, splat (i8 4)
  %lo.idx = and <32 x i8> %v, splat (i8 15)
  $raw_load_h
  $ht_load
  $raw_load_l
  $lt_load
  %hil = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %ht, <32 x i8> %hi.idx)
  %lol = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %lt, <32 x i8> %lo.idx)
  %sum = add <32 x i8> %hil, %lol
  %s0 = shufflevector <32 x i8> %sum, <32 x i8> poison, <16 x i32> <$LO_MASK>
  %s1 = shufflevector <32 x i8> %sum, <32 x i8> poison, <16 x i32> <$HI_MASK>
$widen$flush  %g.next = add i64 %g, 1
  %done = icmp eq i64 %g.next, %ng
  br i1 %done, label %fin, label %loop
fin:
$fin  %scv = insertelement <16 x float> poison, float %scale, i32 0
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
end

function make_kernel(ir)
    @eval function $(gensym(:k))(codes::Ptr{UInt8}, lut::Ptr{UInt8}, ng::Int,
                                 scale::Float32, bias::Float32, out::Ptr{Float32})
        Base.llvmcall(($ir, "scan_block"), Cvoid,
                      Tuple{Ptr{UInt8}, Ptr{UInt8}, Int, Float32, Float32, Ptr{Float32}},
                      codes, lut, ng, scale, bias, out)
    end
end

function scan_all(k, codes, lut, ng, nb)
    out = Vector{Float32}(undef, 32)
    total = 0.0f0
    GC.@preserve codes lut out begin
        pc = pointer(codes); pl = pointer(lut); po = pointer(out)
        for b in 0:(nb - 1)
            k(pc + b * ng * 32, pl, ng, 1.0f0, 0.0f0, po)
            total += out[1]
        end
    end
    total
end

const VARIANTS = [
    ("A shuf-u32", make_kernel(make_ir(:shuf, :u32))),
    ("B intr-u32", make_kernel(make_ir(:intr, :u32))),
    ("C shuf-u16", make_kernel(make_ir(:shuf, :u16))),
    ("D intr-u16", make_kernel(make_ir(:intr, :u16))),
]

function main()
    Random.seed!(1)
    n, ng = 100_000, 384
    nb = n ÷ 32
    codes = rand(UInt8, nb * ng * 32)
    lut = UInt8.(rand(0:127, ng * 32))
    variants = VARIANTS
    ref = nothing
    for (name, k) in variants
        got = scan_all(k, codes, lut, ng, nb)
        if ref === nothing
            ref = got
        else
            @printf("%s checksum match=%s\n", name, got == ref)
        end
    end
    for (name, k) in variants
        t = @elapsed for _ in 1:3
            scan_all(k, codes, lut, ng, nb)
        end
        @printf("%s: %.2f ms/scan\n", name, t / 3 * 1000)
        flush(stdout)
    end
end

main()
