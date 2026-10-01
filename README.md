# KAN: modular C++/CUDA numerical library

An original implementation of learned univariate edge expansions, built from
[the project brief](introduction.md). C++20 owns mathematics, parameters, forward,
backward and training; optional Python bindings expose the same implementation.

## Current scope (M1 and M2)

| Category | Available | Planned |
|---|---|---|
| Orthogonal polynomials | Chebyshev T, Legendre P, Jacobi, physicists' Hermite | Normalization and additional families |
| Harmonic / wave | Fourier | Wavelets |
| Radial / rational | Fixed-center Gaussian RBF | Trainable centers/widths, Padé/rational edges |
| Local / adaptive | — | B-splines, adaptive knots |
| Quantum carriers | — | Experimental PQC/Fock contracts and adapters |

CPU supports all six available families and compatible networks of any depth.
The optional CUDA target provides both the original synchronous Chebyshev layer
API and a resident executor supporting all six families, mixed networks, reusable
workspaces, and GPU SGD. Optional NumPy bindings support CPU and resident CUDA use.
Performance conclusions require the frozen, matched full-call benchmark; see
[M2 benchmark evidence](docs/evidence/M2-benchmark.md).

## Build and test

The CPU C++ library has no third-party runtime dependencies. CMake >=3.24 and a C++20 compiler
are required. Windows helper defaults to the Visual Studio location in the brief:

```powershell
./scripts/build.ps1 -Test
./build/fit_polynomial.exe
./scripts/build.ps1 -Asan -Configuration Debug -Test
```

Pass `-VisualStudio 'path/to/installation'` when different. CPU builds do not load CUDA.
CUDA additionally requires an NVIDIA toolkit and GPU:

```powershell
./scripts/build.ps1 -Cuda -Test -CudaArchitectures 86
```

On this machine CUDA 13.1 rejects the installed VS 2026/MSVC 19.50 version guard.
A compile probe succeeds with the explicit override below; local CPU/GPU parity
and Compute Sanitizer results are recorded in [M1 evidence](docs/evidence/M1.md).
Prefer a supported compiler/toolkit combination for production use.

```powershell
./scripts/build.ps1 -Cuda -AllowUnsupportedCudaCompiler -Test
```

Portable CPU build (Linux or a compiler developer shell):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/fit_polynomial
```

Portable CUDA build adds `-DKAN_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`.
Choose the architecture for your GPU. GPU tests require actual hardware and fail
explicitly when unavailable. `-DBUILD_TESTING=OFF` disables tests, and
`-DKAN_BUILD_EXAMPLES=OFF` disables the example.

## C++ use

```cpp
#include <kan/network.hpp>

kan::BasisConfig basis;
basis.kind = kan::BasisKind::Chebyshev;
basis.size = 3; // three terms: T0, T1, T2 (degree two)
kan::Layer layer(1, 1, basis);
const std::vector<double> coefficients{0.0, 0.7, -0.2}, bias{0.0};
layer.set_parameters(coefficients, bias);
kan::Network model({layer});
const std::vector<double> input{-1.0, 0.0, 1.0};
auto prediction = model.forward(input, 3); // [-0.9, 0.2, 0.5]
// Supply the gradient of your own loss, including any batch averaging:
const std::vector<double> upstream{0.1, -0.2, 0.1};
model.sgd(model.backward(input, 3, upstream), 0.01);
```

Include `<kan/cuda.hpp>` and link `kan::cuda` to call
`kan::cuda::forward(layer, input, batch)` or `kan::cuda::backward(...)`.
Query `kan::cuda::available()` for runtime availability. CUDA accepts Chebyshev
layers only; other families are rejected explicitly.

For persistent execution, include `<kan/resident.hpp>` and link `kan::cuda`:

```cpp
kan::cuda::ResidentNetwork gpu(model, 3); // reserve maximum batch once
gpu.upload_input(input, 3);
gpu.upload_output_gradient(upstream);
gpu.forward();
auto output = gpu.download_output();
gpu.backward();
gpu.sgd(0.01); // validate and update all parameters on the GPU
// Inputs/upstream remain resident; forward/backward/SGD can be repeated.
auto trained = gpu.download_parameters(); // owned CPU snapshot when requested
```

Each executor owns its stream and storage. Uploads/downloads and numerical calls
complete before returning; numerical calls transfer a small error status to check
overflow. Forward retains activations for backward. Input uploads invalidate prior
outputs/gradients/upstream; SGD invalidates outputs/gradients. Batch capacity is
fixed; create a new executor to increase it. Different executors are independent;
serialize access to the same executor. No execution call allocates GPU storage.

Parameters initialize to zero. Initialize multilayer parameters to nonzero
values explicitly so gradients can propagate. Polynomial inputs are not
automatically normalized or clipped. Fourier uses angular frequency and the
order `[1, cos(wx), sin(wx), ...]`. RBF width is the denominator in
`exp(-((x-center)/width)^2)`, not a standard deviation.
See [the numerical contract](docs/CONTRACT.md) for layouts, derivatives,
exceptions, finite-data requirements and atomic optimizer updates.

## Install / consume

The build produces static libraries and an exported CMake package:

```sh
cmake --install build --prefix /path/to/kan-install
```

In a consumer:

```cmake
find_package(KAN 0.1 CONFIG REQUIRED)
target_link_libraries(your_program PRIVATE kan::kan) # or kan::cuda
```

Set `CMAKE_PREFIX_PATH` to the installation directory. On MSVC, build the consumer
with the same configuration/runtime as the installed static library (Release in
the examples above); mixing Debug and Release STL objects is unsupported.
A CPU package requires
no CUDA discovery. See [tests/consumer](tests/consumer) for a standalone consumer.
Set `-DKAN_CONSUMER_CUDA=ON` for that consumer to additionally validate an installed
`kan::cuda` package on actual GPU hardware.

## Python use

Bindings are optional. Install their build/test dependencies in your environment:

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements-python.txt
./scripts/build.ps1 -Python -BuildDirectory build-python -Test
# Add -Cuda -AllowUnsupportedCudaCompiler for this machine's GPU build.
$env:PYTHONPATH = "$PWD/build-python/python"
./.venv/Scripts/python.exe -c "import kan; print(kan.cuda_available())"
```

Portable builds add `-DKAN_BUILD_PYTHON=ON -DPython_EXECUTABLE=/path/to/python`
to CMake. CMake locates pybind11 installed in that interpreter. Add the resulting
`build/python` directory to `PYTHONPATH` (for multi-configuration generators use
the directory containing `_kan` as well). `cmake --install` places the extension and
package in `<prefix>/python`; `KAN_PYTHON_INSTALL_DIR` overrides that relative path.
Python wheels and distribution packaging remain M6 work.

The NumPy interface requires C-contiguous float64 arrays, preserving caller dtype
and layout decisions. Tensor results own their memory. See
[Python test examples](tests/python_test.py) for layer/network construction,
gradients, training, and resident uploads/downloads.

## Full-call benchmarks and profiling

```powershell
./scripts/build.ps1 -Cuda -AllowUnsupportedCudaCompiler -Benchmarks -BuildDirectory build-m2-bench
./build-m2-bench/m2_benchmark.exe
```

The benchmark freezes deterministic inputs and parameter initialization, verifies
outputs/gradients/SGD against CPU, and reports wall time for complete public calls.
It distinguishes setup, resident execution, and transfer-inclusive execution.
[The benchmark evidence](docs/evidence/M2-benchmark.md) records exact commands,
sampling, profile results, hardware limits, and measured conclusions.

## Development evidence

Tests use a small dependency-free C++ harness registered with CTest. They include
independent polynomial identities, finite-difference gradients, mixed-family
network composition, actual fitting and invalid-data/overflow cases. GCC coverage:

```sh
python -m pip install -r requirements-dev.txt
cmake -S . -B build-coverage -DCMAKE_CXX_COMPILER=g++ -DKAN_ENABLE_COVERAGE=ON
cmake --build build-coverage --parallel
ctest --test-dir build-coverage --output-on-failure
python -m gcovr --root . --filter 'src/' --exclude-unreachable-branches --exclude-throw-branches --txt --fail-under-line 80 build-coverage
```

Coverage excludes CUDA and compiler exception machinery; GPU kernels are checked
through CPU parity and Compute Sanitizer. See [the roadmap](docs/ROADMAP.md)
and [M1 evidence](docs/evidence/M1.md) for status and exact validation. M2's
[acceptance plan](docs/evidence/M2-plan.md) links execution scope to its tests;
its final evidence records closure only after all gates pass.

The original [KAN paper](https://arxiv.org/abs/2404.19756) motivates the edge-function
architecture; [NIST DLMF](https://dlmf.nist.gov/18.9) specifies polynomial conventions.
External KAN projects informed discovery only; no implementation or interface was ported.
