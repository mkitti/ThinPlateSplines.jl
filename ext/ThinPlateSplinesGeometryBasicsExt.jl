module ThinPlateSplinesGeometryBasicsExt
    using GeometryBasics: Point
    import ThinPlateSplines: tps_solve, tps_deform, ThinPlateSpline

    # Convert a vector of N-dimensional points into a K×N matrix.
    # Unlike `stack`, this also works for empty vectors.
    _pointmatrix(x::AbstractVector{<:Point{N}}) where {N} = [p[j] for p in x, j in 1:N]

    function tps_solve(
        x::AbstractVector{<:Point},
        y::AbstractVector{<:Point},
        λ;
        compute_affine = true
    )
        return tps_solve(_pointmatrix(x), _pointmatrix(y), λ; compute_affine)
    end

    function tps_deform(x2::AbstractVector{<:Point{N}}, tps::ThinPlateSpline) where {N}
        deformed = tps_deform(_pointmatrix(x2), tps)
        return Point{size(deformed, 2)}.(eachrow(deformed))
    end
end
