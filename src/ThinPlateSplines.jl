module ThinPlateSplines

using LinearAlgebra
using Tullio

export ThinPlateSpline
export tps_solve, tps_solve!, tps_energy, tps_deform
export TPSWorkspace

"""
	ThinPlateSpline{Λ,X,M}

the type (structure) holding the deformation information. This is needed to apply the deformation to other points using `tps_deform`.
(see http://en.wikipedia.org/wiki/Thin_plate_spline for more information)
	
# Members
`λ::Λ`  # Stiffness.
`x1::X` # control points
`Y::M`  # Homogeneous control point coordinates
`Φ::M`  # TPS kernel
`d::M`  # Affine component
`c::M`  # Non-affine component
"""
struct ThinPlateSpline{Λ,X <: AbstractMatrix,M}
	λ::Λ  # Stiffness.
	x1::X # control points
	Y::M  # Homogeneous control point coordinates
	Φ::M  # TPS kernel
	d::M  # Affine component
	c::M  # Non-affine component
end

# Thin-plate splines.
# based on http://en.wikipedia.org/wiki/Thin_plate_spline

# tps_basis is applied to the result of a norm which is either positive or zero
# using ifelse is faster than the (? x:y) notation
tps_basis(r::T) where {T}  = ifelse(r < eps(r), zero(T), r*r*log(r))

my_norm(a) = sqrt(sum(a.^2))

# x: matrix of size KxD
# Allocation-free kernel: the naive comprehension `x[i,:] - x[j,:]` allocates two
# row slices + a diff + `.^2` for each of K² entries (564 MB / ~120 ms at K=1312).
# Fuse the pairwise squared-distance into a single @tullio loop (as tps_deform
# already does), then map tps_basis over it.
function tps_kernel(x)
    @tullio sumsqr[i,j] := abs2(x[i,m] - x[j,m])
    @tullio K[i,j] := tps_basis(sqrt(sumsqr[i,j]))
    return K
end

# In-place variant for `tps_solve!`: writes the K×K kernel into `Kbuf` using a
# caller-supplied K×K `sumsqr` scratch buffer, so neither array is allocated per
# call (the dominant ~28 MB/call at K=1312). Numerically identical to `tps_kernel`.
function tps_kernel!(Kbuf::AbstractMatrix, sumsqr::AbstractMatrix, x)
    @tullio sumsqr[i,j] = abs2(x[i,m] - x[j,m])
    @tullio Kbuf[i,j]   = tps_basis(sqrt(sumsqr[i,j]))
    return Kbuf
end

"""
    TPSWorkspace{T}(K, D)

Preallocated scratch for [`tps_solve!`](@ref) at a fixed problem size (`K` control
points, `D` dimensions). Reuse one workspace across many solves of the same size to
avoid re-allocating the K×K kernel, the (K+D+1)² KKT matrix, and the RHS on every
call — useful when warping repeatedly against a fixed-size control lattice.

A workspace is **not** thread-safe: use one per task/thread.
"""
struct TPSWorkspace{T}
    sumsqr::Matrix{T}   # K×K kernel scratch
    A::Matrix{T}        # (K+D+1)² KKT matrix; bunchkaufman! overwrites it
    rhs::Matrix{T}      # (K+D+1)×(D+1) RHS; ldiv! overwrites it with the solution
    K::Int
    D::Int
end

function TPSWorkspace{T}(K::Integer, D::Integer) where {T}
    n = K + D + 1
    TPSWorkspace{T}(Matrix{T}(undef, K, K), Matrix{T}(undef, n, n),
                    Matrix{T}(undef, n, D + 1), Int(K), Int(D))
end

"""
	tps_solve(x,y,λ,compute_affine=true)

find solution of tps transformation 
(required for some operations but takes additional time.)

# Arguments
	`x`: control points
	`y`: deformed (warped) control points
	`λ`: stiffness coefficient
	`compute_affine`: computes affine component if `true`

returns a `ThinPlateSpline` structure which can be supplied as an argument to `tps_deform`.

# See almost
	`ThinPlateSpline`
"""
function tps_solve(x,y,λ; compute_affine=true)
	# D: number of dimensions
	# K: number of data points
	K,D = size(x)

	# homogeneous coordinates
	X=hcat(ones(K,1),x)
	Y=hcat(ones(K,1),y)

	# compute TPS kernel
	Φ = tps_kernel(x)

	# Solve the regularized TPS saddle-point (KKT) system directly:
	#     [ Φ+λI   X ] [c]   [Y]
	#     [ X'     0 ] [d] = [0]
	# This is mathematically identical to the QR null-space method (projecting
	# the top block onto null(X') with orthonormal q2 gives
	# q2'(Φ+λI)q2 = λI + q2'Φq2) but needs a single symmetric-indefinite
	# factorization instead of a full QR plus the O(K³) q2'Φq2 double matmul and
	# a separate Φ*c affine backsolve.
	A = [Φ + λ*I                       X
	     X'        zeros(eltype(X), D+1, D+1)]
	rhs = vcat(Y, zeros(eltype(Y), D+1, size(Y, 2)))
	sol = bunchkaufman!(Symmetric(A)) \ rhs
	c = sol[1:K, :]
	d = compute_affine ? sol[(K+1):end, :] : eltype(c)[;;]
	return ThinPlateSpline(λ,x,Y,Φ,d,c)
end

"""
    tps_solve!(ws::TPSWorkspace, x, y, λ; compute_affine=true)

Allocation-light [`tps_solve`](@ref): solves the same augmented (KKT) system but
reuses the buffers in `ws` (built with `TPSWorkspace{T}(K, D)` for the matching
problem size) for the K×K kernel, the (K+D+1)² KKT matrix, and the RHS. The
returned `ThinPlateSpline` carries `c` and `d` (freshly allocated, small) and is
fully usable with [`tps_deform`](@ref); its `Φ` field is left **empty** (the kernel
lives in `ws.A`, which the factorization overwrites). If you need `tps.Φ`, use the
allocating `tps_solve`.

Results are numerically identical to `tps_solve` up to floating-point rounding.
A workspace must not be shared across threads.
"""
function tps_solve!(ws::TPSWorkspace{T}, x, y, λ; compute_affine=true) where {T}
	K, D = size(x)
	(K == ws.K && D == ws.D) ||
		throw(DimensionMismatch("workspace is for (K=$(ws.K), D=$(ws.D)) but got (K=$K, D=$D)"))

	# homogeneous coordinates (small: K×(D+1)) — fresh allocation is cheap
	X = hcat(ones(T, K), x)
	Y = hcat(ones(T, K), y)

	# Build the KKT matrix directly in ws.A (no [block; block] literal, which would
	# allocate a fresh (K+D+1)² matrix every call):
	#     [ Φ+λI   X ] [c]   [Y]
	#     [ X'     0 ] [d] = [0]
	Φ = @view ws.A[1:K, 1:K]
	tps_kernel!(Φ, ws.sumsqr, x)              # write kernel into the top-left block
	@inbounds for i in 1:K
		Φ[i, i] += λ                          # Φ + λI
	end
	@views ws.A[1:K, (K+1):end]   .= X
	@views ws.A[(K+1):end, 1:K]   .= transpose(X)
	@views ws.A[(K+1):end, (K+1):end] .= zero(T)

	@views ws.rhs[1:K, :]      .= Y
	@views ws.rhs[(K+1):end, :] .= zero(T)

	ldiv!(bunchkaufman!(Symmetric(ws.A)), ws.rhs)   # solution overwrites ws.rhs

	c = ws.rhs[1:K, :]                         # copy out (escapes ws) — small
	d = compute_affine ? ws.rhs[(K+1):end, :] : eltype(c)[;;]
	# Φ field intentionally left empty: ws.A was overwritten by the factorization.
	return ThinPlateSpline(λ, x, Y, similar(c, 0, 0), d, c)
end

"""
	tps_energy(tps::ThinPlateSpline)

Thin-plate spline bending energy at the minimum.

# Arguments
	`tps`: `ThinPlateSpline` structure defining the deformation
"""
tps_energy(tps::ThinPlateSpline) = tps.λ*tr(tps.c*tps.Y')

"""
	tps_deform(x2::AbstractMatrix, tps::ThinPlateSpline) 

calculate deformed locations of an input vector of positions`x2` according to thin-plate spline.
Note that in combination with suitable interpolation packages can this be used to warp an image, but this function does not apply a warp by itself.
Yet it can be used to transform other mathematical structures rather than points.

# Arguments
	`x2`: coordinates of points to be deformed.
	`tps`: the thin-plate spline structure of type `ThinPlateSpline` defining the deformation.
"""
function tps_deform(x2::AbstractMatrix{T}, tps::ThinPlateSpline) where {T}
    x1,d,c = tps.x1,tps.d,tps.c
	d==[] && throw(ArgumentError("Affine component not available; run tps_solve with compute_affine=true."))
	D = size(x2, 2)
    all_homo_z = hcat(ones(T, size(x2,1)), x2)
    # calculate sum of squares. Note that the summation is done outside the abs2
	# it may be useful to join the terms below, but this seems a little harder than first thought
    @tullio sumsqr[k,j] := abs2(all_homo_z[k,m+1] .- x1[j,m])
    @tullio yt[i,j] := (tps_basis(sqrt(sumsqr[i,k])) * c[k,j+1]) (j in 1:D)
    @tullio yt[i,j] += d[l,j+1] * all_homo_z[i,l] 

	# it would save some memory and possibly make it a bit faster, if the order of dimensions is changed for x and yt due to cache utilization. 
	# but the advantage is relatively minor (10%?) since @tullio seems to take care of this as well as possible
    return yt
end

"""
	tps_deform(x1, x2, y, λ; compute_affine=true) 

computes a tps deformation based on `x1`, `y` and `λ` and applies it to `x2`.
First `tps_solve` is applied and then `tps_deform`.

# Arguments
	`x1`: control points for the tps generation
	`x2`: coordinates of points to be deformed by the tps.
	`y`: deformed control points for the tpd generation
	`λ`: stiffness coefficient
	`compute_affine`: optional argument defining whether the affine transform is computed.

returned are the deformed coordinates as a matrix
"""
function tps_deform(x1, x2, y, λ; compute_affine=true)
	tps = tps_solve(x1, y, λ; compute_affine=compute_affine)
	tps_deform(x2, tps)
end

end
