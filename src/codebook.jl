# Lloyd-Max scalar quantizer for the Beta((d-1)/2, (d-1)/2) distribution
# that each coordinate of a rotated unit vector follows. Port of
# `turbovec::codebook`.
#
# The f64 iteration matches the Rust solver's algorithm; centroids are
# then rounded to f32 and boundaries are taken as the f32 midpoints of
# the f32 centroids (the property that makes the codebook reproducible).

const LANCZOS_G = 7.0
const LANCZOS_COEF = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                      771.32342877765313, -176.61502916214059, 12.507343278686905,
                      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)

function loggamma(z::Float64)
    if z < 0.5
        return log(π / sin(π * z)) - loggamma(1.0 - z)
    end
    z -= 1.0
    x = LANCZOS_COEF[1]
    @inbounds for i in 1:8
        x += LANCZOS_COEF[i + 1] / (z + i)
    end
    t = z + LANCZOS_G + 0.5
    0.5 * log(2π) + (z + 0.5) * log(t) - t + log(x)
end

"""Beta pdf on [0,1]."""
function beta_pdf(x::Float64, a::Float64, b::Float64)
    (x <= 0.0 || x >= 1.0) && return 0.0
    exp((a - 1.0) * log(x) + (b - 1.0) * log1p(-x) -
        (loggamma(a) + loggamma(b) - loggamma(a + b)))
end

# Continued fraction for the incomplete beta function (Lentz's method).
function beta_cf(a::Float64, b::Float64, x::Float64)
    maxit = 300
    eps = 3.0e-16
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    abs(d) < fpmin && (d = fpmin)
    d = 1.0 / d
    h = d
    for m in 1:maxit
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        abs(d) < fpmin && (d = fpmin)
        c = 1.0 + aa / c
        abs(c) < fpmin && (c = fpmin)
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        abs(d) < fpmin && (d = fpmin)
        c = 1.0 + aa / c
        abs(c) < fpmin && (c = fpmin)
        d = 1.0 / d
        del = d * c
        h *= del
        abs(del - 1.0) <= eps && break
    end
    h
end

"""Regularized incomplete beta function I_x(a,b)."""
function beta_inc(a::Float64, b::Float64, x::Float64)
    x <= 0.0 && return 0.0
    x >= 1.0 && return 1.0
    bt = exp(loggamma(a + b) - loggamma(a) - loggamma(b) +
             a * log(x) + b * log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0)
        bt * beta_cf(a, b, x) / a
    else
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    end
end

"""Adaptive Simpson's rule, matching the Rust solver's recursion and stopping rule."""
function adaptive_simpson(f::F, a::Float64, b::Float64, tol::Float64,
                          max_depth::Int) where {F}
    mid = (a + b) / 2.0
    fa = f(a)
    fb = f(b)
    fm = f(mid)
    whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb)
    _adaptive_simpson_rec(f, a, b, fa, fb, fm, whole, tol, max_depth)
end

function _adaptive_simpson_rec(f::F, a::Float64, b::Float64, fa::Float64, fb::Float64,
                               fm::Float64, whole::Float64, tol::Float64,
                               depth::Int) where {F}
    mid = (a + b) / 2.0
    m1 = (a + mid) / 2.0
    m2 = (mid + b) / 2.0
    fm1 = f(m1)
    fm2 = f(m2)
    left = (mid - a) / 6.0 * (fa + 4.0 * fm1 + fm)
    right = (b - mid) / 6.0 * (fm + 4.0 * fm2 + fb)
    refined = left + right
    if depth == 0 || abs(refined - whole) < 15.0 * tol
        refined + (refined - whole) / 15.0
    else
        _adaptive_simpson_rec(f, a, mid, fa, fm, fm1, left, tol / 2.0, depth - 1) +
        _adaptive_simpson_rec(f, mid, b, fm, fb, fm2, right, tol / 2.0, depth - 1)
    end
end

function lloyd_max(bits::Int, dim::Int, max_iter::Int, tol::Float64)
    a = (Float64(dim) - 1.0) / 2.0
    n_levels = 1 << bits

    std_dev = sqrt(2.0 * a / ((2.0 * a + 1.0) * 4.0 * a))
    spread = 3.0 * std_dev
    centroids = Float64[-spread + 2.0 * spread * Float64(i) / Float64(n_levels - 1)
                        for i in 0:(n_levels - 1)]

    edges = Vector{Float64}(undef, n_levels + 1)
    new_centroids = Vector{Float64}(undef, n_levels)
    for _ in 1:max_iter
        edges[1] = -1.0
        edges[end] = 1.0
        @inbounds for i in 1:(n_levels - 1)
            edges[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0
        end

        @inbounds for i in 1:n_levels
            lo = edges[i]
            hi = edges[i + 1]
            cdf_lo = beta_inc(a, a, (lo + 1.0) / 2.0)
            cdf_hi = beta_inc(a, a, (hi + 1.0) / 2.0)
            prob = cdf_hi - cdf_lo
            if prob < 1.0e-15
                new_centroids[i] = centroids[i]
            else
                mean = adaptive_simpson(x -> x * beta_pdf((x + 1.0) / 2.0, a, a) / 2.0,
                                        lo, hi, 1.0e-14, 50)
                new_centroids[i] = mean / prob
            end
        end

        max_change = 0.0
        @inbounds for i in 1:n_levels
            d = abs(centroids[i] - new_centroids[i])
            d > max_change && (max_change = d)
        end
        centroids, new_centroids = new_centroids, centroids
        max_change < tol && break
    end

    centroids_f32 = Float32.(centroids)
    boundaries = Float32[(centroids_f32[i] + centroids_f32[i + 1]) * 0.5f0
                         for i in 1:(n_levels - 1)]
    (boundaries, centroids_f32)
end

"""Named codebook arrays: `(boundaries = …, centroids = …)`."""
const Codebook = NamedTuple{(:boundaries, :centroids),
                            Tuple{Vector{Float32},Vector{Float32}}}

const CODEBOOK_MEMO = Dict{Tuple{Int,Int},Codebook}()
const CODEBOOK_LOCK = ReentrantLock()

"""
    codebook(bits, dim) -> (; boundaries, centroids)

The canonical Lloyd-Max codebook for `bits ∈ {2,3,4}` and a positive
multiple-of-8 `dim`. Memoized per `(bits, dim)`.
"""
function codebook(bits::Int, dim::Int)
    (bits >= 2 && bits <= 4) ||
        throw(ArgumentError("bit_width must be 2, 3 or 4, got $bits"))
    (dim > 0 && dim % 8 == 0) ||
        throw(ArgumentError("dim must be a positive multiple of 8, got $dim"))
    dim <= MAX_DIM || throw(ArgumentError("dim must be <= $MAX_DIM (MAX_DIM), got $dim"))
    lock(CODEBOOK_LOCK)
    try
        get!(CODEBOOK_MEMO, (bits, dim)) do
            boundaries, centroids = lloyd_max(bits, dim, 200, 1.0e-12)
            (boundaries = boundaries, centroids = centroids)
        end
    finally
        unlock(CODEBOOK_LOCK)
    end
end
