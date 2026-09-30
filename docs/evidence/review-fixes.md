# Independent review corrections

Independent read-only reviewer exercised cases beyond the original suites.
Each production correction follows an executable regression RED checkpoint.

## Moved-from objects and custom installation include directory: RED

`./scripts/build.ps1 -Test` after adding regressions on 2026-09-30 compiled
successfully and ran all five CPU CTest targets. Results: 2/5 passed, 3/5 failed:

- `layer`: SegFault in `moved_from_layer_rejects_numerical_and_update_operations`.
- `network`: SegFault in `moved_from_network_and_layers_are_rejected`.
- `package_custom_include`: installed package consumer configuration fails because
  imported target references nonexistent `install/include` while headers installed
  into custom `install/kan-headers`.

The review also independently reproduced moved-from null accesses with MSVC
AddressSanitizer and custom-package failure with GCC. These failures are the
intended absent safety guard / incorrect installed path, not missing dependencies.

CUDA fused-arithmetic RED/GREEN evidence is in [cuda.md](cuda.md).
Jacobi asymmetric endpoint RED/GREEN evidence is in [basis.md](basis.md).
