using TurboVec, Random
const TV = TurboVec
function main()
    for n in (5_000, 20_000, 50_000, 100_000, 200_000)
        dim = 768
        X = randn(MersenneTwister(1), Float32, n, dim)
        rot = TV.Rotation(dim)
        r = Matrix{Float32}(undef, dim, n)
        norms = Vector{Float32}(undef, n)
        t1 = @elapsed TV.rotate_batch!(r, norms, X, rot)
        t2 = @elapsed TV.rotate_batch!(r, norms, X, rot)
        t3 = @elapsed TV.rotate_batch!(r, norms, X, rot)
        println("n=$n first=", round(t1/n*1e6, digits=2), " then=",
                round(t2/n*1e6, digits=2), ",", round(t3/n*1e6, digits=2), " us/row")
        flush(stdout)
    end
end
main()
