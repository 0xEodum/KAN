# KAN: modular C++/CUDA numerical library

An original implementation of learned univariate edge expansions, built from
[the project brief](introduction.md). C++20 owns mathematics, parameters, forward,
backward and training; optional Python bindings expose the same implementation.

## Current scope (M1 through M4)

| Category | Available | Planned |
|---|---|---|
| Orthogonal polynomials | Chebyshev T, Legendre P, Jacobi, physicists' Hermite | Normalization and additional families |
| Harmonic / wave | Fourier, normalized Mexican-hat wavelets | Additional wavelets |
| Radial / rational | Fixed or trainable Gaussian RBF centers/widths, nonlinear Padé-compatible rational edges | Additional rational parameterizations |
| Local / adaptive | B-splines, exact adaptive knot refinement, coefficient L2 | Additional grid policies |
| Input maps | Explicit affine (fixed, from data range or moments), tanh, LayerNorm (trainable gain/bias) layers | — |
| Quantum carriers | — | Experimental PQC/Fock contracts and adapters |

CPU supports all eight available families and compatible networks of any depth.
The optional CUDA target provides both the original synchronous Chebyshev layer
API and a resident executor supporting all eight families, mixed networks, reusable
workspaces, and GPU SGD. CPU and resident execution also support rational layers
and mixed rational/basis networks. Optional NumPy bindings expose both.
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

// Each family has its own configuration type holding only its parameters;
// kan::BasisConfig is a std::variant of them.
const kan::ChebyshevConfig basis{3}; // three terms: T0, T1, T2 (degree two)
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

Link `kan::cuda` and query `kan::cuda::available()` for runtime availability; it is
declared in `<kan/cuda_runtime.hpp>`, which `<kan/resident.hpp>` and `<kan/cuda.hpp>` both
include (backlog R8). The M1 single-layer calls `kan::cuda::forward(layer, input, batch)`
and `kan::cuda::backward(...)` in `<kan/cuda.hpp>` are deprecated (backlog R7): each
call builds a one-layer resident executor, runs it once and releases it, so it accepts
every carrier but pays the executor construction on every call. Use the persistent
executor below instead.

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
// Reuse the executor with weights trained or restored elsewhere (same structure):
gpu.upload_parameters(model); // then forward() again before backward()
```

`upload_parameters(network)` (backlog R9; Python `gpu.upload_parameters(network)`) replaces
every trainable parameter of the executor without reallocating anything: coefficients,
biases, trainable RBF centers/log widths, rational denominators and LayerNorm gain/bias.
The network must have the structure the executor was built for (layer kinds and sizes,
carriers, basis and rational configuration including spline knots, fixed input maps);
anything else, or a value the executor precision cannot represent, raises
`std::invalid_argument` (`ValueError`) and changes nothing. After `insert_knot`/`adapt_grid`
the structure differs: build a new executor. Uploading costs a host-to-device copy of the
parameters, a small fraction of construction for small networks (see the contract).

Each executor owns its stream and storage. Uploads/downloads and numerical calls
complete before returning; numerical calls transfer a small error status to check
overflow. Forward retains activations for backward. Input uploads invalidate prior
outputs/gradients/upstream; SGD invalidates outputs/gradients. Batch capacity is
fixed; create a new executor to increase it. Different executors are independent;
serialize access to the same executor. No execution call allocates GPU storage.
Dense contractions of basis layers run on cuBLAS (`kan::cuda` links `CUDA::cublas`,
CUDA 12+; the cuBLAS runtime library, e.g. `cublas64_13.dll` on Windows, must be on the
loader path at run time; the Python package registers the build's CUDA toolkit
directories and `CUDA_PATH`, so a module copied to another machine needs a CUDA runtime
there). GPU results match the CPU reference within floating-point tolerance, not bitwise.

The executor computes in double precision by default. For training throughput pass
`kan::cuda::Precision::Float32` (FP32 storage, kernels and cuBLAS SGEMM; on an RTX 3090 the
256-wide review step is about 26x faster than FP64) or `Precision::TensorFloat32` (FP32 with
TF32 tensor-core contractions, looser tolerance) as the third constructor argument
(`precision=kan.Precision.FLOAT32` in Python). Host data stays `double`; values FP32 cannot
represent are rejected, and results follow the FP32 tolerance of
[the contract](docs/CONTRACT.md). Configure with `-DKAN_CUDA_FMA=ON`
(`scriptsuild.ps1 -CudaFma`) for the performance build, which lets nvcc fuse multiply-adds
in the kernels; the default parity build keeps `--fmad=false`.

Parameters initialize to zero. Initialize multilayer parameters to nonzero
values explicitly so gradients can propagate. Polynomial inputs are not
automatically normalized or clipped; use an explicit input map (below). Fourier uses angular frequency and the
order `[1, cos(wx), sin(wx), ...]`. RBF width is the denominator in
`exp(-((x-center)/width)^2)`, not a standard deviation.
See [the numerical contract](docs/CONTRACT.md) for layouts, derivatives,
exceptions, finite-data requirements and atomic optimizer updates.

## Input maps

Inputs are never rescaled implicitly. Polynomials explode for |x| >> 1 and localized
bases are dead outside their support, so put an explicit input map in front:

```cpp
#include <kan/network.hpp>
// Map each feature's sample range onto the Chebyshev domain [-1, 1].
const auto affine = kan::affine_from_range(samples, batch, features, -1.0, 1.0);
kan::Network model({kan::InputMap(features, affine), kan::Layer(features, 1, kan::ChebyshevConfig{5})});
// Alternatives: kan::TanhMap{0.01} (squash), kan::LayerNormMap{1e-5, gain, bias}
// (per-sample normalization, trainable gain/bias), kan::affine_from_moments(...).
const auto& map = std::get<kan::InputMap>(model.layers()[0]); // layers() holds Layer or InputMap
```

Layer indices (`insert_knot`, `adapt_grid`, gradients) are positions in `layers()`,
maps included; `NetworkGradients::layers[i]` holds `kan::LayerGradients` or
`kan::InputMapGradients`. Maps run on the resident executor as well. See the
[contract](docs/CONTRACT.md#input-maps-backlog-m1).

## Localized and adaptive use

```cpp
#include <kan/families.hpp> // family operations: insert_knot, adapt_grid, setters
// Localized families derive their term count: knots.size() - degree - 1 = 4.
const kan::BSplineConfig spline{3, {0,0,0,0,1,1,1,1}};
kan::Layer localized(1,1,spline);
localized.set_parameters(std::vector<double>{0,0,1.0/3,1}, std::vector<double>{0});
kan::insert_knot(localized, 0.4); // preserves the existing x^2 edge
kan::adapt_grid(localized, std::vector<double>{0.1,0.2,0.3}); // sample-driven refinement
auto penalty = localized.regularization(0.01); // value and coefficient VJP

// Trainable RBF: shared centers and log widths are nonlinear parameters.
const kan::TrainableRbfConfig radial{{-0.5,0.5}, {std::log(0.4),std::log(0.6)}}; // include <cmath>
kan::Layer learnable(1,1,radial); // explicitly initialize coefficients for training
// A layer holds one carrier (kan::BasisEdges, TrainableRbfEdges or RationalEdges):
const auto& edges = std::get<kan::BasisEdges>(localized.carrier());
const auto& knots = std::get<kan::BSplineConfig>(edges.basis).knots;
// backward returns kan::TrainableRbfGradients{centers, log_widths} in
// LayerGradients::nonlinear; sgd updates them atomically.
```

Splines are zero outside their explicit domain; repeated interior knots permit
reduced continuity. Mexican-hat terms use explicit translations (`centers`) and
positive `scales`, with continuous L2 normalization. RBF centers/widths are shared
per layer and widths use log parameters. Refinement changes coefficient shapes,
so compute new gradients afterward. For a resident model, download its snapshot,
refine explicitly, then construct a new executor. `gpu.backward(0.01)` adds
coefficient L2 gradients on the GPU; CPU callers explicitly add the regularization
VJP to their loss gradients. Python exposes the same operations (`kan.insert_knot(layer, x)`,
`layer.carrier`); see
[localized Python examples](tests/m3_python_test.py).

## Rational use

```cpp
kan::RationalConfig rational;
rational.numerator_degree = 1; rational.denominator_degree = 1;
kan::Layer pade(1,1,rational);
kan::set_rational_parameters(pade, std::vector<double>{1,0.5},
                             std::vector<double>{-0.5}, std::vector<double>{0});
auto y = pade.forward(std::vector<double>{-0.5,0,0.5},3);
auto g = pade.backward(std::vector<double>{-0.5,0,0.5},3,
                       std::vector<double>{0.1,-0.2,0.1});
pade.sgd(g,0.001); // learns both numerator and denominator
```

An edge computes `P(z)/Q(z)`, with `z=(x-center)/scale` and fixed `Q(0)=1`.
Degrees range independently from zero to sixteen. Explicit center/scale and a
relative denominator guard control conditioning: unsafe denominators raise
`domain_error`, including removable poles. The guard checks evaluated samples;
choose an input domain and validate it for your application. Numerator parameters
use `coefficients`; denominators live in `kan::RationalEdges::denominators` and their
VJPs in `kan::RationalGradients`. Python exposes
the same typed constructor and strict float64 arrays; see
[rational Python examples](tests/m4_python_test.py). Resident CUDA accepts rational
layers in any compatible network without host numerical fallback. The original
synchronous CUDA layer API remains Chebyshev-only. See the
[rational numerical contract](docs/CONTRACT.md#rational-edges-m4).

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
gradients, training, and resident uploads/downloads, and
[input map examples](tests/input_map_python_test.py) for
`kan.Network([kan.InputMap(2, kan.affine_from_range(x)), kan.Layer(2, 1, ...)])`.

## Full-call benchmarks and profiling

```powershell
./scripts/build.ps1 -Cuda -AllowUnsupportedCudaCompiler -Benchmarks -BuildDirectory build-m2-bench
./build-m2-bench/m2_benchmark.exe
./build-m2-bench/m3_benchmark.exe
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
M2 is complete; [final M2 evidence](docs/evidence/M2.md) records its acceptance
results and measured limits. [M3 acceptance plan](docs/evidence/M3-plan.md)
defines the localized/adaptive stage and its validation gates.
M3 is complete; [M3 final evidence](docs/evidence/M3.md) records numerical,
installation, sanitizer, independent review and hardware-scoped performance
results. M4 is complete; [M4 final evidence](docs/evidence/M4.md) records rational
contracts, independent review, numerical/installation/sanitizer gates and measured
CPU/GPU improvements. M5 experimental quantum carriers is next.

The original [KAN paper](https://arxiv.org/abs/2404.19756) motivates the edge-function
architecture; [NIST DLMF](https://dlmf.nist.gov/18.9) specifies polynomial conventions.
External KAN projects informed discovery only; no implementation or interface was ported.
