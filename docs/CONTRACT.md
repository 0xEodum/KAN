# KAN numerical contract (M1 and M2)

An edge is a learned univariate expansion. A layer computes
`y[b,o] = bias[o] + sum_i sum_k coefficients[o,i,k] * basis_k(x[b,i])`.
Arrays are contiguous, batch-major; coefficients use `(o * inputs + i) * basis.size + k`.
All arithmetic and storage in M1 use `double`. Bias and coefficients initialize to zero.
Topology and parameters are owned values, never global state. Initialization for training
is explicit: zero initialization is useful for a single linear-in-coefficients layer, but
multilayer training needs nonzero parameters to propagate gradients.

`BasisConfig::size` is the number of terms, never a polynomial degree. Chebyshev
uses T_n, Legendre uses P_n, Jacobi uses P_n^(alpha,beta), Hermite uses physicists'
H_n. Polynomials are evaluated by recurrence, including endpoints and outside [-1,1].
There is no implicit clipping, normalization, tanh, or extrapolation policy.
Fourier is `[1, cos(w*x), sin(w*x), cos(2*w*x), sin(2*w*x), ...]` and size is odd.
Gaussian RBF is `exp(-((x-center)/width)^2)` with explicit finite centers and width > 0.
Jacobi alpha and beta are finite and greater than -1; angular frequency is finite and positive.
Only parameters relevant to the selected family participate in validation/evaluation.
Each evaluator returns values and derivatives with respect to x.

Layer backward is a vector-Jacobian product, summing parameter gradients over the batch.
It does not average or mutate parameters. Caller supplies any loss scaling.
Network backward recomputes intermediate activations, returning gradients in layer order.
SGD applies `parameter -= learning_rate * gradient`; learning rate is finite and positive.
Updates validate all parameter gradients and the complete resulting parameter vectors
before mutating, including across a network. Input gradients are not consumed by SGD.
Inputs, upstream gradients, and parameters must be finite. Nonfinite computed results
raise `std::overflow_error`; invalid configuration, sizes and data raise `std::invalid_argument`.
Dimension multiplication is checked before allocation (`std::overflow_error`).
Layer dimensions are positive; batch zero accepts only empty inputs/upstream and returns
empty outputs/input gradients and zero parameter gradients. Networks are nonempty,
adjacent dimensions match, and can mix basis families.
Moved-from layers and networks remain assignable; their numerical/parameter-update
operations raise `std::invalid_argument`. A network rejects moved-from layer values.
Accessors are safe but their moved-from values are unspecified.

CUDA is an optional separate `kan::cuda` target; CPU has no CUDA dependency.
M1 CUDA supports Chebyshev only, with the same mathematical/validation contract.
Other families raise `std::invalid_argument` rather than selecting another backend.
The host API is synchronous and transfers inputs/parameters/results per call; it makes
no performance claim. Device memory is owned per invocation, exceptions release it,
and reductions have fixed summation order, without floating-point atomic accumulation.
No device returns `available() == false`; actual operations fail explicitly with
`std::runtime_error`. CPU/CUDA equivalence is tolerance-based, not bitwise promised.

## Persistent CUDA execution (M2)

`kan::cuda::ResidentNetwork` is a move-only executor built from an owned snapshot
of a CPU `Network`, with a fixed maximum batch capacity. It supports every M1
basis family and mixed compatible networks. All storage remains double precision.
The M1 synchronous Chebyshev layer functions keep their original contract.

Each resident executor owns its CUDA stream and preallocated device storage for
parameters, inputs, upstream gradients, activations, basis values/derivatives,
input/parameter gradients, and candidate SGD parameters. Construction uploads the
model; `upload_input` and `upload_output_gradient` explicitly transfer finite host
data. No numerical call allocates device storage. Batches above capacity fail.
`workspace_allocations()` reports the executor's construction-time device allocation
count, which remains unchanged by subsequent operations.

Uploads, computations, and downloads complete before returning. Computations check
a small device error status on the host without copying full tensors. There is no
asynchronous-submission guarantee. Different instances can run independently;
callers must serialize operations on the same instance. Host input buffers need
only remain alive for their upload call. Downloads return independently owned values.

After input upload, `forward()` saves the activations for `backward()`. Backward
requires a successful current forward and a correctly sized uploaded upstream.
`download_output()` and `download_gradients()` require their corresponding current
successful computation. Input upload invalidates output/gradients and the uploaded
upstream. Upstream upload invalidates gradients. SGD requires current gradients
and a finite positive learning rate. It validates every candidate parameter on
the GPU before committing any layer, preserving network-wide atomicity. Successful
SGD invalidates output/gradients while retaining input/upstream for another iteration.
Batch zero produces empty outputs/input gradients and zero parameter gradients.
Invalid lifecycle and moved-from operations raise `std::logic_error`; invalid
host shapes/data raise `std::invalid_argument`, dimension/numerical overflow raises
`std::overflow_error`, and absent hardware/runtime failures raise `std::runtime_error`.
All M1 mathematical domain and finite-result
requirements apply; CPU/GPU equivalence is tolerance-based.

Python bindings expose the same mathematics through an optional `kan` module.
Numerical tensor arguments must be NumPy C-contiguous float64 arrays of the declared
shape. No implicit float32 promotion or layout conversion is performed. Returned
arrays own their storage; expensive computation releases the GIL. Serialize access
to shared model instances and do not mutate borrowed input arrays during a call.
Building CPU
static libraries needs neither Python nor pybind11. Python package/wheel distribution
and model serialization remain outside M2.

## Extension boundaries

Basis mathematics lives in `include/kan/basis.hpp` and `src/basis.cpp`; CPU edge
contraction lives in `src/layer.cpp`; topology in `src/network.cpp`; kernels in
`src/cuda.cu`. No symbolic parser, Eigen, Torch, Python runtime or imported KAN
implementation is required. Rational edges will need nonlinear parameter gradients
and pole policy; quantum carriers need separate physical/measurement contracts.

## Mathematical sources

- [Original KAN paper](https://arxiv.org/abs/2404.19756): learned univariate edge functions.
- [NIST DLMF 18.9](https://dlmf.nist.gov/18.9): polynomial recurrence and derivative conventions.
- [Awesome KAN](https://github.com/mintisan/awesome-kan): variant discovery only.
- [Quantum-KAN](https://github.com/wtroy2/Quantum-KAN): referenced by the brief; no code reused.
