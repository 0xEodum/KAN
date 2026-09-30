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

## Moved-from objects and custom include directory: GREEN

Added Layer invariant guards before numerical/parameter update operations, Network
empty/moved-from guards, and validation of supplied Layer values at network construction.
Reassignment restores normal operation; no moved-from null access is permitted.
Moved-from semantics are explicit in docs/CONTRACT.md.

Moved GNUInstallDirs initialization before target definitions and use its include
directory in the exported target interface. The regression configures an actual
CPU-only package with `CMAKE_INSTALL_INCLUDEDIR=kan-headers`, installs it, compiles
an external `find_package(KAN)` consumer and runs its numerical assertion.

`./scripts/build.ps1 -Test` after these changes: **5/5 CTest targets passed**,
including layer 12/12 and network 8/8, custom installed consumer and holdout fitting.
Post-fix AddressSanitizer/coverage and reviewer recheck are recorded in M1.md.
