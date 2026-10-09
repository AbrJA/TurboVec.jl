using Random, Printf
using LoopVectorization

function lv_scan(codes::Vector{UInt8}, comb::Vector{UInt8}, ng::Int, nb::Int)
    acc = zeros(Int32, 32)
    idx = Vector{Int32}(undef, 32)
    @inbounds for b in 0:(nb - 1)
        fill!(acc, 0)
        base = b * ng * 32
        cb = 0
        for g in 0:(ng - 1)
            baseg = base + g * 32
            for l in 1:32
                idx[l] = Int32(cb + Int(codes[baseg + l]) + 1)
            end
            @turbo for l in 1:32
                acc[l] += comb[idx[l]]
            end
            cb += 256
        end
    end
    sum(acc)
end

function scalar_scan(codes::Vector{UInt8}, comb::Vector{UInt8}, ng::Int, nb::Int)
    acc = zeros(Int32, 32)
    total = Int64(0)
    @inbounds for b in 0:(nb - 1)
        fill!(acc, 0)
        base = b * ng * 32
        cb = 0
        for g in 0:(ng - 1)
            baseg = base + g * 32
            for l in 1:32
                acc[l] += Int32(comb[cb + Int(codes[baseg + l]) + 1])
            end
            cb += 256
        end
        for l in 1:32
            total += acc[l]
        end
    end
    total
end

function main()
    Random.seed!(1)
    n, ng = 100_000, 384
    nb = n ÷ 32
    codes = rand(UInt8, nb * ng * 32)
    comb = UInt8.(rand(0:254, 256 * ng))
    lv_scan(codes, comb, ng, nb)
    scalar_scan(codes, comb, ng, nb)
    t1 = @elapsed for _ in 1:3; lv_scan(codes, comb, ng, nb); end
    t2 = @elapsed for _ in 1:3; scalar_scan(codes, comb, ng, nb); end
    @printf("LV gather:  %.2f ms/scan\n", t1/3*1000)
    @printf("scalar:     %.2f ms/scan\n", t2/3*1000)
end
main()
