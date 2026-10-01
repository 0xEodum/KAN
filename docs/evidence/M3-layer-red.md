# M3 CPU integration RED checkpoint

2026-10-01: `scripts/build.ps1 -BuildDirectory build-m3-layer-red` compiled
successfully with MSVC Release. `build-m3-layer-red/m3_layer_test.exe` exited 1,
**0/5 passed**. Failures: nonlinear gradient vectors missing, update/refinement
APIs throwing `M3 pending`, new spline kind rejected, L2 API throwing. Tests cover
shared nonlinear finite differences, atomic network updates, exact spline
refinement, sample-driven refinement and coefficient L2 VJP. New methods are
throwing stubs at this checkpoint; no acceptance claim is made.
