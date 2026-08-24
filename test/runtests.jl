using Test
using ThinPlateSplines
using LinearAlgebra

@testset "tps generation" begin
    x1 = [0.0 1.0 
    1.0 0.0
    1.0 1.0]
    x2 = [0.0 1.0
      1.1 0.0
      1.2 1.5]
    tps = tps_solve(x1, x2, 1.0)
    @test tps.Y == [ 1.0  0.0  1.0
    1.0  1.1  0.0
    1.0  1.2  1.5]
    @test isapprox(tps.c, zeros((3,3)); atol=1e-10)  # ~0 to machine precision (KKT solve gives ~1e-16, QR gave exact 0)
    @test tps.d ≈ [  1.0          -0.1  -0.5
    0   1.2   0.5
    0   0.1   1.5]
    @test isapprox(tps.Φ, [ 0.0       0.693147  0.0
    0.693147  0.0       0.0
    0.0       0.0       0.0], atol=1e-5)
end

x1 = [0.0 1.0 
1.0 0.0
1.0 1.0]
x2 = [0.0 1.0
  1.1 0.0
  1.2 1.5]
tps = tps_solve(x1, x2, 1.0)

@testset "tps_deform" begin
    
    x = [1.0 0.0
     2.0 2.0]
    y = tps_deform(x,tps)
    @test y ≈ [ 1.1  0
    2.5   3.5]
    y = tps_deform(x1, x, x2, 1.0)
    @test y ≈ [ 1.1  0
    2.5   3.5]
end

@testset "Different input and output dimensions" begin
  start_pts = [0.0 0.0; 1.0 0.0; 0.0 1.0]
  end_pts = [0.0 0.0 1.0; 1.0 0.0 2.0; 0.0 1.0 3.0]
  tps = tps_solve(start_pts, end_pts, 1.0)
  deformed = tps_deform([0.5 0.25], tps)
  @test size(deformed) == (1, 3)
  @test deformed ≈ [0.5 0.25 2.0]
end

@testset "tps_energy" begin
    @test isapprox(tps_energy(tps), 0; atol=1e-10)  # ~0 to machine precision (energy ∝ c, which is ~1e-16 here)
end

@testset "Three dimensions" begin
  start_pts = [0 0 0; 0 0 1; 0 1 0; 1 0 0]
  end_pts = [-0.7 -0.7 0; 0 0 1; 0 1 0; 1 0 0]
  tps = tps_solve(start_pts, end_pts, 1.0)
  deformed = tps_deform(start_pts, tps)
  @test size(deformed,1) == size(start_pts, 1)
  @test size(deformed,2) == size(start_pts, 2)
  @test deformed ≈ end_pts
  @test tps_deform([0.5 0.5 0.5], tps) ≈ [0.85 0.85 0.5]
end

@testset "Four dimensions" begin
  start_pts = [0 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0]
  end_pts = [-0.7 -0.7 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0]
  tps = tps_solve(start_pts, end_pts, 1.0)
  deformed = tps_deform(start_pts, tps)
  @test size(deformed,1) == size(start_pts, 1)
  @test size(deformed,2) == size(start_pts, 2)
  @test deformed ≈ end_pts
  @test tps_deform([0.5 0.5 0.5 0.5], tps) ≈ [1.2 1.2 0.5 0.5]
end

# Guards the allocation-free tps_kernel rewrite: it must match the original
# naive comprehension `[tps_basis(my_norm(x[i,:]-x[j,:]))…]` (the only difference
# is float summation order). Exercise enough points to run the fused loop.
@testset "tps_kernel matches naive comprehension" begin
  naive_basis(r) = ifelse(r < eps(r), zero(r), r*r*log(r))
  naive_kernel(x) =
      [naive_basis(sqrt(sum(abs2, x[i, :] - x[j, :]))) for i in axes(x, 1), j in axes(x, 1)]
  for (K, D) in ((40, 3), (60, 2), (33, 4))
      # Deterministic spread of points (no Random dep needed).
      x = Float64[sin(0.7i + 1.3d) + 0.11i - 0.05d for i in 1:K, d in 1:D]
      Φ = ThinPlateSplines.tps_kernel(x)
      @test Φ ≈ naive_kernel(x)
      @test Φ == Φ'                            # symmetric
      @test all(iszero, @view Φ[diagind(Φ)])   # zero diagonal (r == 0)
  end
end

# Guards the augmented (KKT) solve: it must reproduce the classic QR-nullspace
# TPS coefficients (and hence the same deformation) to fp tolerance.
@testset "tps_solve matches QR-nullspace reference" begin
  function ref_solve(x, y, λ)
      K, D = size(x)
      X = hcat(ones(K, 1), x); Y = hcat(ones(K, 1), y)
      Φ = ThinPlateSplines.tps_kernel(x)
      Q, r = qr(X)
      Qf = Q * Matrix{Float64}(I, K, K)
      q1 = Qf[:, 1:(D+1)]; q2 = Qf[:, (D+2):end]
      c = q2 * ((UniformScaling(λ) + q2'*Φ*q2) \ (q2'*Y))
      d = r \ (q1' * (Y - Φ*c))
      (c, d)
  end
  for (K, D) in ((40, 3), (60, 2))
      x = Float64[sin(0.7i + 1.3d) + 0.11i - 0.05d for i in 1:K, d in 1:D]
      y = Float64[cos(0.5i - 0.9d) + 0.07i        for i in 1:K, d in 1:D]
      tps = tps_solve(x, y, 1.0)
      rc, rd = ref_solve(x, y, 1.0)
      @test isapprox(tps.c, rc; atol=1e-8)
      @test isapprox(tps.d, rd; atol=1e-8)
      ref_tps = ThinPlateSpline(1.0, x, hcat(ones(K, 1), y),
                                ThinPlateSplines.tps_kernel(x), rd, rc)
      pts = x[1:5, :]
      @test isapprox(tps_deform(pts, tps), tps_deform(pts, ref_tps); atol=1e-8)
  end
end

# Guards the in-place tps_solve! / TPSWorkspace path: it must reproduce the
# allocating tps_solve's c, d, and deform output — and reusing the SAME workspace
# for a second, different solve must leave no stale state (every buffer is fully
# overwritten each call).
@testset "tps_solve! matches tps_solve and reuses workspace cleanly" begin
  for (K, D) in ((40, 3), (60, 2))
      x  = Float64[sin(0.7i + 1.3d) + 0.11i - 0.05d for i in 1:K, d in 1:D]
      y1 = Float64[cos(0.5i - 0.9d) + 0.07i        for i in 1:K, d in 1:D]
      y2 = Float64[sin(0.3i + 0.4d) - 0.02i        for i in 1:K, d in 1:D]
      pts = x[1:5, :]

      ws = TPSWorkspace{Float64}(K, D)

      ref1 = tps_solve(x, y1, 1.0)
      got1 = tps_solve!(ws, x, y1, 1.0)
      @test isapprox(got1.c, ref1.c; atol=1e-9)
      @test isapprox(got1.d, ref1.d; atol=1e-9)
      @test isempty(got1.Φ)                                   # Φ intentionally dropped
      @test isapprox(tps_deform(pts, got1), tps_deform(pts, ref1); atol=1e-9)

      # Reuse the SAME workspace with different control points.
      ref2 = tps_solve(x, y2, 1.0)
      got2 = tps_solve!(ws, x, y2, 1.0)
      @test isapprox(got2.c, ref2.c; atol=1e-9)
      @test isapprox(got2.d, ref2.d; atol=1e-9)
      @test isapprox(tps_deform(pts, got2), tps_deform(pts, ref2); atol=1e-9)

      # compute_affine=false still gives identical c, empty d.
      no_aff = tps_solve!(ws, x, y1, 1.0; compute_affine=false)
      @test isapprox(no_aff.c, ref1.c; atol=1e-9)
      @test isempty(no_aff.d)

      # Wrong-size workspace is rejected.
      @test_throws DimensionMismatch tps_solve!(TPSWorkspace{Float64}(K+1, D), x, y1, 1.0)
  end
end
