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

Pending implementation and actual CPU/GPU/import validation.
