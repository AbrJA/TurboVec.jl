function rand_rows(rng::AbstractRNG, n::Int, dim::Int)
    Matrix{Float32}(randn(rng, Float32, n, dim))
end

"""Gaussian rows normalized to unit L2 norm, as `Float32`."""
function unit_rows(rng::AbstractRNG, n::Int, dim::Int)
    X = Matrix{Float32}(undef, n, dim)
    for i in 1:n
        s = 0.0
        @inbounds for d in 1:dim
            v = randn(rng, Float64)
            X[i, d] = Float32(v)
            s += v * v
        end
        inv = Float32(1.0 / max(sqrt(s), 1e-12))
        @inbounds for d in 1:dim
            X[i, d] *= inv
        end
    end
    X
end

function brute_topk(Q::AbstractMatrix{Float32}, X::AbstractMatrix{Float32}, k::Int)
    S = Float64.(Q) * transpose(Float64.(X))
    nq = size(S, 1)
    out = Vector{Vector{Int}}(undef, nq)
    for i in 1:nq
        p = partialsortperm(view(S, i, :), 1:k; rev = true)
        out[i] = collect(p)
    end
    out
end
