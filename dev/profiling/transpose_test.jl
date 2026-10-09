using Random
using Printf

function copy_rows_strided!(src, X, n, dim)
    for i in 1:n
        @inbounds @simd for d in 1:dim
            src[d] = X[i, d]
        end
    end
    nothing
end

function copy_rows_contig!(src, Xt, n, dim)
    for i in 1:n
        @inbounds @simd for d in 1:dim
            src[d] = Xt[d, i]
        end
    end
    nothing
end

function main()
    Random.seed!(1)
    for (n, dim) in ((5_000, 768), (20_000, 768), (100_000, 768))
        X = randn(MersenneTwister(2), Float32, n, dim)      # n × dim (column-major)
        Xt = permutedims(X)                                  # dim × n
        src = Vector{Float32}(undef, dim)
        copy_rows_strided!(src, X, n, dim)
        copy_rows_contig!(src, Xt, n, dim)
        t1 = @elapsed copy_rows_strided!(src, X, n, dim)
        t2 = @elapsed copy_rows_contig!(src, Xt, n, dim)
        t3 = @elapsed permutedims(X)
        @printf("n=%d dim=%d  strided row-copy=%.2f us/row  contiguous=%.2f us/row  permutedims once=%.3f s (%.2f us/row)\n",
                n, dim, t1/n*1e6, t2/n*1e6, t3, t3/n*1e6)
        flush(stdout)
    end
end
main()
