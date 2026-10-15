# C13 Option 1 decision rule (written before the robustness matrix was run)

A family other than B-spline gets Option 1 (its W = U*C input-VJP GEMM in FP32 math under
TF32) when its dense TF32 input VJP exceeds the C1 TF32 tolerance (1e-2|e| + 1e-2 max|e|
against the FP64 CPU at the executor's parameters; ratio > 1) in more than one cell of the
matrix {fit, decay coefficient profiles} x {sizes 8, 16, 32, 64} x {inputs 64, 256} x
{input distributions sin-spread, uniform}, counting only cells whose forward outputs are within
the same tolerance (forward ratio <= 1; otherwise the representation itself is ill-conditioned
and an FP32 W GEMM cannot repair it). The rule is applied mechanically to Chebyshev, Legendre
and Jacobi (the polynomial families that fail or nearly fail in the first sweep); Hermite
functions, Fourier, Gaussian RBF and Mexican hat stay on TF32 by the first sweep (fit profile
<= 0.38 except the ill-conditioned Mexican-hat fits, whose forward fails).
