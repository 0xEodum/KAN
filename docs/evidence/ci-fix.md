# Hosted CI configuration regression

First published run [36733073655](https://github.com/0xEodum/KAN/actions/runs/36733073655)
passed Windows and Ubuntu Release build/test/install/consumer jobs. Coverage
compiled and passed basis/layer/network/fitting, but its package integration
harness failed because the root single-configuration build type was unset.

The harness passed `--config` with an empty value to the nested build, causing
`CMake Error: Invalid value used with --config`. This is a test-infrastructure
defect, not a numerical RED claim. A dedicated `package_default_configuration`
CTest target now preserves this regression independently of the root build type.

Local harness RED: configured `build-ci-empty` using GCC/Ninja without
`CMAKE_BUILD_TYPE`, then ran
`ctest --test-dir build-ci-empty -R '^package_default_configuration$' --output-on-failure`.
The regression actually executed and failed **0/1** with that same invalid
empty `--config` argument.

Correction: when the root build type is empty, the isolated package/consumer pair
explicitly selects matching Release configurations. The root build type is unchanged.
Same targeted invocation then passed **1/1**, and the complete no-build-type GCC
suite passed **6/6**. An independent reviewer built a separate no-build-type tree,
also passed **6/6**, checked the root remained unconfigured and the isolated
package selected Release, and approved the harness correction without findings.

After correction, local MSVC CPU passed **6/6**, CUDA-enabled CTest **7/7**,
and the GCC coverage suite **6/6**, with the same numerical coverage summary.
Hosted rerun [36733786401](https://github.com/0xEodum/KAN/actions/runs/36733786401)
at source commit `73f207492a8bea5635ed0e10dd70516a7f70df9c` completed **success**:
Windows Release build/test/install/consumer, Ubuntu Release build/test/install/consumer,
and Ubuntu coverage all passed. The subsequent evidence commit changes documentation only.
