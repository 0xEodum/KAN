# M2 Python binding evidence

Scope: optional NumPy bindings for the existing six basis families, CPU layers
and networks, and the optional persistent CUDA network. No wheel distribution
or model serialization is introduced; those remain M6 work. CPU C++ builds do
not acquire Python or pybind11 dependencies. Python builds use pybind11 3.0.2;
runtime arrays are NumPy float64, native endian and C-contiguous.

## Executable RED

2026-10-01, Windows x64, repository `.venv` Python 3.13.10, NumPy 2.5.2:

```
$env:PYTHONPATH='python'
.\.venv\Scripts\python.exe tests\python_test.py
```

Exit 1: `ModuleNotFoundError: No module named '_kan'`. The complete numerical,
shape, ownership and optional GPU tests are present before the extension.
The Hermite oracle uses NumPy's physicists' Hermite polynomials, matching M1.

## Array and ownership contract

Inputs are explicit `(batch, inputs)` arrays and output gradients are
`(batch, outputs)` arrays. Coefficients have `(outputs, inputs, basis.size)`
shape, biases have `(outputs,)` shape. Batch is inferred from the first axis;
an explicit `batch=` argument must agree. Lists, incompatible dtype/endian,
noncontiguous arrays and incorrect ranks or dimensions are rejected, with no
implicit casting. Nonfinite values follow the C++ rejection policy.

Outputs, gradient properties and parameter properties are owned NumPy snapshots.
`Network.layers` returns layer snapshots; modifying one does not modify the
network. CPU SGD accepts the gradient object returned by backward. GPU SGD uses
the most recent resident backward result. Compute releases the GIL; callers must
serialize operations on the same model instance and must not mutate borrowed
input arrays during a call. Different model instances can execute concurrently.

The binding checks `py::array` and dtype/flags directly with `.noconvert()`;
this avoids pybind11's default array force-casting behavior described in the
[official NumPy interface documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html).

## GREEN and installation

Bindings use `_kan` as the compiled module, with the small `kan` reexport package.
`cuda_enabled` reports compile-time support; `cuda_available()` probes runtime
hardware availability and returns false from CPU-only builds. Basis configuration
is copied before GIL release so later Python configuration mutation cannot race
with that evaluation. GPU construction also releases the GIL.

2026-10-01: `.\scripts\build.ps1 -Python -BuildDirectory build-m2-python -Test`
passed **7/7 CTest targets**, with Python **4 numerical/validation cases passed**
and the explicitly optional GPU case skipped. Follow-up adversarial tests also
passed, including network coefficient finite differences, unaligned buffers,
gradient topology mismatch and dtype validation on parameter arrays. One immediate
post-link import returned a Windows file-sharing violation; the dedicated Python
CTest rerun passed without source changes.

The actual CPU polynomial objective gives holdout MSE **1.787191650958667e-5**
after 180 SGD steps and **6.963329110696405e-11** after 600. The GPU training test
uses 600 steps and a frozen acceptance bound of 1e-6. The initial 180-step trial
failed that bound on GPU with the matching MSE **1.787191650958717e-5**, while all
six basis-family numerical and optimizer parity checks passed. This was insufficient
optimization time rather than a CPU/GPU numerical disagreement.

Install is opt-in along with `KAN_BUILD_PYTHON=ON`. The default
`KAN_PYTHON_INSTALL_DIR=python` is relative to the CMake install prefix, and can
be overridden by the consumer. Both `_kan` and `kan` are placed there; adding
this directory to `PYTHONPATH` enables imports without a wheel or `pip install`.

```
cmake --install build-m2-python --prefix build-m2-python/install
$env:PYTHONPATH='build-m2-python/install/python'
.\.venv\Scripts\python.exe tests\python_test.py
```

The installed CPU package passed **4/4** applicable Python cases (GPU skipped).
The CUDA build used `.\scripts\build.ps1 -Python -Cuda
-AllowUnsupportedCudaCompiler -BuildDirectory build-m2-python-cuda -Test`.
All nine C++ CTest targets passed and the initial 180-step Python holdout case
failed as described above. After selecting 600 steps based on independent CPU
convergence, `ctest --test-dir build-m2-python-cuda --output-on-failure -R '^python$'`
passed **1/1** with all **5/5 Python cases**, including real RTX 3090 execution.
GPU execution tested all six families through multilayer forward, input/parameter
gradients and three GPU optimizer updates, frozen allocation counts, shape/dtype
rejection, and an independently evaluated polynomial holdout after 600 updates.

```
cmake --install build-m2-python-cuda --prefix build-m2-python-cuda/install
$env:PYTHONPATH='build-m2-python-cuda/install/python'
.\.venv\Scripts\python.exe tests\python_test.py --cuda
```

The installed GPU package independently passed **5/5** Python cases. Validation
uses native Windows Python 3.13.10, NumPy 2.5.2, pybind11 3.0.2, MSVC 19.50,
CUDA 13.1.115 and the real RTX 3090. CUDA's host compiler override is explicit
and does not claim vendor support for this local compiler pairing.

The resident CUDA author independently reviewed the binding code and found no
blocking correctness or lifetime defect. Its constructor-GIL observation was
resolved before the passing GPU run. This lane independently reviewed the complete
resident CUDA source against the M1 scalar numerical rules and state/ownership
contracts, finding no blocking issue.
