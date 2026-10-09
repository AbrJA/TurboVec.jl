using TurboVec, Random
const TV = TurboVec

# Current: flat scan over all boundaries.
function flat_code(calib::Float32, boundaries, limits)
    v = 0
    for bi in 1:limits
        calib > boundaries[bi] && (v += 1)
    end
    v
end

# Two-level scan with a branch on the median boundary.
function twolevel_branchy(calib::Float32, boundaries, limits)
    m = calib > boundaries[8]
    v = m ? 8 : 0
    lo = m ? 8 : 0
    for bi in 1:7
        calib > boundaries[lo + bi] && (v += 1)
    end
    v
end

# Two-level scan with a select (branchless).
function twolevel_select(calib::Float32, boundaries, limits)
    m = calib > boundaries[8]
    v = m ? 8 : 0
    for bi in 1:7
        b = m ? boundaries[8 + bi] : boundaries[bi]
        calib > b && (v += 1)
    end
    v
end

function bench(name, f, calibs, boundaries, limits)
    # warm
    s = 0
    for c in calibs
        s += f(c, boundaries, limits)
    end
    t = @elapsed for _ in 1:3
        acc = 0
        for c in calibs
            acc += f(c, boundaries, limits)
        end
        s += acc
    end
    println(name, ": ", round(t / 3 / length(calibs) * 1e9, digits = 3), " ns/coord",
          "  (checksum ", s, ")")
end

function main()
    dim = 768
    _, boundaries = TV.codebook(4, dim)
    calibs = Float32.(4 .* randn(MersenneTwister(1), dim * 2000))
    for (name, f) in (("flat           ", flat_code),
                      ("twolevel-branch", twolevel_branchy),
                      ("twolevel-select", twolevel_select))
        bench(name, f, calibs, boundaries, 15)
    end
    # verify all three agree on codes
    ok = true
    for c in calibs
        a = flat_code(c, boundaries, 15)
        ok &= a == twolevel_branchy(c, boundaries, 15) == twolevel_select(c, boundaries, 15)
    end
    println("codes agree: ", ok)
end

main()
