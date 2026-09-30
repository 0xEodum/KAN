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
