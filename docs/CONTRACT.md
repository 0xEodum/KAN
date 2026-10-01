# KAN numerical contract (M1 through M4)

An edge is a learned univariate function. A basis layer computes
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

## Localized and adaptive bases (M3)

`BSpline` uses explicit clamped nondecreasing finite knots, length
`size+degree+1`, degree 0..16 and size at least degree+1. Endpoints have exactly
degree+1 repetitions and a positive domain `[knots[degree],knots[size]]`.
Interior multiplicity is at most degree+1; full multiplicity permits a jump.
Values and input derivatives are zero outside the domain. Interior knots use
right-hand values/derivatives; the upper endpoint uses the inward left-hand
convention. Degree-zero derivatives are zero, including at jumps (a convention,
not a claim of differentiability there). Cox-de Boor terms with zero denominators
are zero. No implicit extrapolation or clipping occurs.

`MexicanHat` has explicit translations `centers` and positive `scales`, each
length size. For `q=(x-center)/scale`, each term is
`A*(1-q*q)*exp(-q*q/2)`, where `A=2/(sqrt(3)*pi^(1/4)*sqrt(scale))`.
Its derivative is `A/scale*q*(q*q-3)*exp(-q*q/2)`. These are L2-normalized
continuous wavelets; configuration is fixed, with learned edge coefficients.
Extreme representable tails use log-space evaluation to preserve derivatives
even when the basis value underflows. Nonfinite mathematical results fail.

Gaussian configuration remains fixed by default, using scalar width. With
`trainable_rbf=true`, `log_widths` has length size and each exponent must be
finite and positive; the scalar width is unused. Centers/log widths are shared
across a layer's edges, not per edge. At each term, the center derivative is the
negative input derivative, and log-width derivative is `2*q*q*exp(-q*q)`.
`BasisValues` returns these vectors only for trainable RBFs. `LayerGradients`
returns their VJPs, summing over batches and edges; fixed families return empty
vectors. `set_rbf_parameters` validates and atomically replaces both vectors.
SGD validates finite candidate vectors and finite positive exponentiated widths
before committing any parameter; candidate width overflow/underflow raises
`overflow_error`, while invalid user configuration/gradients raises
`invalid_argument`. Network SGD retains whole-network atomicity.

`Layer::insert_knot(x)` accepts a strictly interior spline knot whose new
multiplicity is allowed. Boehm insertion adds one term and transforms every edge's
coefficients, preserving the represented function and its derivatives wherever
the declared derivative convention applies, within floating-point tolerance.
`adapt_grid(samples)` validates all samples, counts in-domain samples per nonzero
span (upper endpoint belongs to the final span), chooses the most populated span
(ties choose the lowest), and inserts the median (mean of middle pair for even
count) if strictly inside that span, otherwise its midpoint. Empty/outside-only
samples or a span without a representable interior value fail explicitly.
Updates are atomic; stale gradient shapes are rejected after refinement. Network
methods accept an explicit layer index and samples in that layer's input domain.
Samples do not implicitly propagate through preceding layers.

`regularization(lambda)` returns the coefficient L2 penalty
`lambda/2*sum(coefficients^2)` and its parameter VJP `lambda*coefficients`.
Lambda is finite and nonnegative. Input gradients are empty; bias and trainable
RBF gradients are correctly shaped zero vectors. Network penalties sum layers.
Add this VJP to a loss VJP explicitly before CPU SGD. Resident `backward(lambda=0)`
adds coefficient L2 gradients on GPU. Resident execution supports all eight
families, nonlinear VJPs and candidate width validation with persistent storage.
Grid refinement changes storage shape: explicitly download a model, refine it,
and reconstruct the resident executor. Numerical execution never reallocates or
silently falls back to the host. Python exposes these operations and snapshots;
`regularization` returns `(value, gradients)`. Its compatible `evaluate_basis`
continues returning `(values,input_derivatives)`; nonlinear gradients are
accessible through layer/network backward.

## Rational edges (M4)

A typed `RationalConfig` selects a nonlinear rational Layer, separately from
`BasisKind`. Each edge is `r(x)=P(z)/Q(z)`, `z=(x-center)/scale`,
`P=sum(a[k]*z^k,k=0..m)`, `Q=1+sum(b[k-1]*z^k,k=1..n)`.
Degrees m,n are independently 0..16, default 3,2. Center is finite (default zero),
scale finite and positive (default one). Fixing Q's constant to one removes common
scale ambiguity. All a/b values are trainable per edge. Zero initialization gives
P=0,Q=1. This Padé-compatible parameterization does not automatically construct a
Taylor-series approximant. Supplied [1/1] exponential coefficients, for example,
are `a={1,0.5}, b={-0.5}`. Horner evaluation and explicit scaling condition the
polynomial evaluation; inputs are not clipped.

`RationalEvaluation` exposes value, input derivative, and numerator/denominator
partial derivatives. With primes denoting derivatives with respect to z:
`dr/dx=(P'/Q-r*Q'/Q)/scale`, `dr/da[k]=z^k/Q`,
`dr/db[k-1]=-r*z^k/Q`. Layer backward contracts these partials with upstream
gradients and sums over the batch. Center/scale/epsilon are fixed configuration.
Rare log-space paths preserve representable parameter/input derivatives when
powers or intermediate quotients become zero or subnormal; the ordinary Horner
path handles normal values. These paths do not hide overflowing intermediates.

For every executed sample/edge, require
`abs(Q)>epsilon*(1+sum(abs(b[k-1]*z^k)))`, epsilon finite and strictly between zero
and one (default 1e-8). A finite denominator failing this relative cancellation
guard raises `std::domain_error`, including exact or removable poles, zero
numerators and zero upstreams. Nonfinite intermediates/results raise
`std::overflow_error`. No clipping or pole removal changes the represented
function. The guard checks executed samples; it does not prove pole freedom
between them. Finite setters/SGD candidates can therefore fail later execution.

Numerator layout is `(outputs,inputs,m+1)` and denominator layout is
`(outputs,inputs,n)`; bias is per output. `coefficients()` exposes numerator a.
`denominators()` exposes b (empty for basis layers). `set_rational_parameters`
atomically replaces a,b,bias. `set_parameters`, RBF setters and spline operations
reject rational layers. `is_rational()` identifies layer type; `basis()` rejects
rational layers and `rational_config()` rejects basis layers. The constrained
rational constructor preserves existing `Layer(inputs,outputs,{})` basis usage.
LayerGradients appends `denominators`, empty for basis layers and correctly shaped
for rational layers, including zero batch. SGD validates shapes/data and all
finite candidate vectors before network-wide commit. Numerator coefficient L2
keeps its existing definition; denominators and bias are unpenalized.

Resident CUDA executes mixed rational/basis networks with persistent a/b storage,
analytic VJPs and atomic GPU SGD. Unsafe denominators are reported to the host as
domain_error; a failed execution invalidates its output/gradient state. Numerical
calls do not allocate GPU storage or fall back to CPU evaluation. CPU/GPU parity
remains tolerance-based. The original synchronous Chebyshev CUDA API rejects
rational layers.

Python exposes `RationalConfig`, the rational Layer constructor, owned config,
parameter and gradient snapshots, and `set_rational_parameters`. Rational
denominator arrays have shape `(outputs,inputs,n)`, even for n=0; basis-layer
denominator arrays have shape `(0,)`. `evaluate_rational(config,x,a,b)` accepts
strict one-dimensional float64 arrays and returns `(value,input_derivative,da,db)`.
`domain_error` maps to Python ValueError. All existing strict array/layout and
owned-snapshot rules apply.

## Extension boundaries

Basis and rational formulas have a single source shared by the CPU backend and
the resident CUDA kernels: `KAN_HOST_DEVICE` templates in `src/detail/basis_formulas.hpp`
and `src/detail/rational_formulas.hpp`, parameterized by a finiteness guard (CPU throws,
device records status). Public declarations and validation live in
`include/kan/basis.hpp`/`src/basis.cpp` and `include/kan/rational.hpp`/`src/rational.cpp`;
CPU edge contraction lives in `src/layer.cpp`; topology in `src/network.cpp`; persistent
kernels in `src/resident.cu`, with one basis kernel instantiation per family; the
legacy M1 Chebyshev kernels in `src/cuda.cu`. No symbolic parser, Eigen, Torch,
Python runtime or imported KAN implementation is required. Quantum carriers need
separate physical/measurement contracts at M5.

## Mathematical sources

- [Original KAN paper](https://arxiv.org/abs/2404.19756): learned univariate edge functions.
- [NIST DLMF 18.9](https://dlmf.nist.gov/18.9): polynomial recurrence and derivative conventions.
- [NIST DLMF 3.11](https://dlmf.nist.gov/3.11): rational and Padé approximation definitions.
- [SciPy BSpline mathematical notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html): spline recurrence and partition of unity.
- [Boehm insertion references](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.insert_knot.html): exact spline refinement.
- [Ricker definition](https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.signal.ricker.html): continuous Mexican-hat normalization.
- [Awesome KAN](https://github.com/mintisan/awesome-kan): variant discovery only.
- [Quantum-KAN](https://github.com/wtroy2/Quantum-KAN): referenced by the brief; no code reused.
