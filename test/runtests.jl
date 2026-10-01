using Test
using ThinPlateSplines

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
    @test tps.c == zeros((3,3))
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
    @test tps_energy(tps) ≈ 0
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

@testset "GeometryBasics extension" begin
  using GeometryBasics: Point, Point2, Point3

  # The extension is loaded because GeometryBasics is a test dependency
  ext = Base.get_extension(ThinPlateSplines, :ThinPlateSplinesGeometryBasicsExt)
  @test ext !== nothing

  p1 = [Point(0.0, 1.0), Point(1.0, 0.0), Point(1.0, 1.0)]
  p2 = [Point(0.0, 1.0), Point(1.1, 0.0), Point(1.2, 1.5)]

  @testset "tps_solve matches the matrix method" begin
    tps_pts = tps_solve(p1, p2, 1.0)
    @test tps_pts isa ThinPlateSpline
    @test tps_pts.x1 == x1
    @test tps_pts.Y == tps.Y
    @test tps_pts.d ≈ tps.d
    @test tps_pts.c ≈ tps.c
    @test tps_pts.Φ ≈ tps.Φ
    @test tps_energy(tps_pts) ≈ 0
  end

  @testset "compute_affine=false" begin
    tps_noaff = tps_solve(p1, p2, 1.0; compute_affine=false)
    @test isempty(tps_noaff.d)
  end

  @testset "tps_deform returns points" begin
    pts = [Point(1.0, 0.0), Point(2.0, 2.0)]
    y = tps_deform(pts, tps)
    @test y isa Vector{Point2{Float64}}
    @test length(y) == 2
    @test stack(y; dims=1) ≈ [1.1 0.0; 2.5 3.5]
    @test stack(y; dims=1) ≈ tps_deform([1.0 0.0; 2.0 2.0], tps)
  end

  @testset "four-argument tps_deform" begin
    pts = [Point(1.0, 0.0), Point(2.0, 2.0)]
    y = tps_deform(p1, pts, p2, 1.0)
    @test y isa Vector{Point2{Float64}}
    @test stack(y; dims=1) ≈ [1.1 0.0; 2.5 3.5]
    # control points deform onto their targets for an affine (exact) fit
    @test stack(tps_deform(p1, p1, p2, 1.0); dims=1) ≈ stack(p2; dims=1)
  end

  @testset "integer and Float32 points" begin
    y = tps_deform([Point(1, 0)], tps)
    @test y isa Vector{Point2{Float64}}
    @test stack(y; dims=1) ≈ [1.1 0.0]
    y32 = tps_deform([Point(1f0, 0f0)], tps)
    @test stack(y32; dims=1) ≈ [1.1 0.0]
  end

  @testset "empty input" begin
    y = tps_deform(Point2{Float64}[], tps)
    @test y isa Vector{Point2{Float64}}
    @test isempty(y)
  end

  @testset "different input and output dimensions" begin
    s = [Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)]
    e = [Point(0.0, 0.0, 1.0), Point(1.0, 0.0, 2.0), Point(0.0, 1.0, 3.0)]
    tps23 = tps_solve(s, e, 1.0)
    y = tps_deform([Point(0.5, 0.25)], tps23)
    @test y isa Vector{Point3{Float64}}
    @test only(y) ≈ Point(0.5, 0.25, 2.0)
  end

  @testset "three dimensions" begin
    s = [Point(0, 0, 0), Point(0, 0, 1), Point(0, 1, 0), Point(1, 0, 0)]
    e = [Point(-0.7, -0.7, 0.0), Point(0.0, 0.0, 1.0), Point(0.0, 1.0, 0.0), Point(1.0, 0.0, 0.0)]
    tps3 = tps_solve(s, e, 1.0)
    y = tps_deform(s, tps3)
    @test y isa Vector{Point3{Float64}}
    @test y ≈ e
    @test only(tps_deform([Point(0.5, 0.5, 0.5)], tps3)) ≈ Point(0.85, 0.85, 0.5)
  end
end
