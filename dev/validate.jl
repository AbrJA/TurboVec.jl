using TurboVec
TV = TurboVec
REF = joinpath(@__DIR__, "out")

mutable struct Xs
    x::UInt64
end
function next_f32!(r::Xs)
    x = r.x
    x ⊻= x << 13
    x ⊻= x >> 7
    x ⊻= x << 17
    r.x = x
    Float32(Float64(x) / Float64(typemax(UInt64))) - 0.5f0
end

mutable struct Res
    fail::Int
end
function check(r::Res, name, cond)
    if cond
        println("ok   $name")
    else
        println("FAIL $name")
        r.fail += 1
    end
end

function canonical_row(index, lane)
    dim = index.dim
    bits = index.bit_width
    codes = Vector{UInt8}(undef, dim)
    TV.extract_codes_lane!(codes, index.codes, lane, dim, bits)
    bpp = dim ÷ 8
    row = zeros(UInt8, bits * bpp)
    for c in 0:(bpp - 1)
        for k in 0:7
            code = codes[8c + k + 1]
            for p in 0:(bits - 1)
                bit = (code >> p) & 0x1
                row[p * bpp + c + 1] |= bit << (7 - k)
            end
        end
    end
    row
end

function build_index(dim, bits, n, nq; calibrated)
    seed = (0x9E3779B97F4A7C15 ⊻ UInt64(dim) ⊻ (UInt64(bits) << 32)) % UInt64
    rng = Xs(seed)
    X = Matrix{Float32}(undef, n, dim)
    for i in 1:n, d in 1:dim
        X[i, d] = next_f32!(rng)
    end
    Q = Matrix{Float32}(undef, nq, dim)
    for i in 1:nq, d in 1:dim
        Q[i, d] = next_f32!(rng)
    end
    idx = TurboQuantIndex(dim, bits)
    calibrated && calibrate!(idx, X)
    add!(idx, X)
    (idx, Q)
end

function main()
    r = Res(0)
    lines = readlines(joinpath(REF, "rotation.txt"))
    li = 1
    while li <= length(lines)
        startswith(lines[li], "dim ") || (li += 1; continue)
        dim = parse(Int, split(lines[li])[2])
        rot = TV.Rotation(dim)
        for rd in 1:2
            signs_line = split(lines[li + 2rd - 1])
            perm_line = split(lines[li + 2rd])
            sref = signs_line[2]
            pref = parse.(Int, split(perm_line[2], ","))
            sgot = join([rot.signs[rd][i] < 0 ? '1' : '0' for i in 1:dim])
            pgot = [Int(p) for p in rot.perms[rd]]
            check(r, "rotation dim=$dim round=$rd signs", sref == sgot)
            check(r, "rotation dim=$dim round=$rd perm", pref == pgot)
        end
        li += 5
    end

    for f in sort(readdir(joinpath(REF, "codebook")))
        m = match(r"b(\d+)_d(\d+)\.txt", f)
        m === nothing && continue
        bits = parse(Int, m.captures[1])
        dim = parse(Int, m.captures[2])
        ls = readlines(joinpath(REF, "codebook", f))
        cen_start = findfirst(==("centroids"), ls)
        bnd_start = findfirst(==("boundaries"), ls)
        cref = [reinterpret(Float32, parse(UInt32, ls[i], base = 16)) for i in cen_start+1:bnd_start-1]
        bref = [reinterpret(Float32, parse(UInt32, ls[i], base = 16)) for i in bnd_start+1:length(ls)]
        b, c = TV.codebook(bits, dim)
        check(r, "codebook bits=$bits dim=$dim centroids",
              all(reinterpret(UInt32, cref) .== reinterpret(UInt32, c)))
        check(r, "codebook bits=$bits dim=$dim boundaries",
              all(reinterpret(UInt32, bref) .== reinterpret(UInt32, b)))
    end

    for (dim, bits, n, nq, k, tag) in [(768, 4, 256, 4, 10, "cal"), (768, 4, 256, 4, 10, "raw"),
                                       (768, 2, 256, 4, 10, "cal"), (768, 2, 256, 4, 10, "raw"),
                                       (200, 2, 200, 2, 5, "cal"), (200, 2, 200, 2, 5, "raw")]
        idx, Q = build_index(dim, bits, n, nq; calibrated = tag == "cal")
        ref_codes = read(joinpath(REF, "codes_b$(bits)_d$(dim)_n$(n)_$(tag).bin"))
        got_codes = UInt8[]
        for lane in 0:(n - 1)
            append!(got_codes, canonical_row(idx, lane))
        end
        check(r, "codes bits=$bits dim=$dim n=$n $tag", ref_codes == got_codes)

        ref = readlines(joinpath(REF, "search_b$(bits)_d$(dim)_n$(n)_$(tag).txt"))
        nqref = parse(Int, split(ref[1])[2])
        kref = parse(Int, split(ref[1])[4])
        nsc = nqref * kref
        rscores = [reinterpret(Float32, parse(UInt32, ref[1 + i], base = 16)) for i in 1:nsc]
        ridx = [parse(Int, ref[1 + nsc + i]) for i in 1:nsc]
        scores, indices = search(idx, Q, k)
        setmatch = true
        ov = 0
        tot = 0
        maxaligned = 0.0f0
        for q in 1:nqref
            gref = collect(ridx[(q-1)*kref+1:q*kref])
            gidx = indices[q, :] .- 1
            ov += length(intersect(gref, gidx))
            tot += length(gref)
            if sort(gidx) != sort(gref)
                setmatch = false
            end
            for (pos, ix) in enumerate(gidx)
                j = findfirst(==(ix), gref)
                if j !== nothing
                    maxaligned = max(maxaligned, abs(scores[q, pos] - rscores[(q-1)*kref+j]))
                end
            end
        end
        check(r, "search sets bits=$bits dim=$dim n=$n $tag", setmatch)
        println("      overlap $(ov)/$(tot)  max aligned |score diff| = $(maxaligned)")
    end

    println(r.fail == 0 ? "ALL OK" : "$(r.fail) FAILURES")
end

main()
