# M5 experimental quantum carriers: scope and acceptance plan

2026-10-22, branch `cpp-foundation`, starting head `6fb482b`, clean tree. Stage B is closed
([closure](backlog-closure.md)); every [backlog](../BACKLOG.md) item is closed, so M5 is the
`NEXT` stage (AGENTS rule 3). This plan is M5's first task, as for M2–M4: it resolves the
terse roadmap entry ("Typed PQC/Fock carrier interfaces and simulator adapters, physical
normalization, measurement/gradient semantics and independently verified
QUBO/quadratization. No string hashing or symbolic parsing in numerical kernels.") into
scope, recorded defaults, tests written before implementation and acceptance gates. It
writes no library code. An independent review of this plan is summarized at the end.

Machine at planning time: GB10 (sm_121, aarch64, 48 SMs, 128 GB unified memory), CUDA
13.0.88, driver 580.159.03, GCC 13.3, `.venv` with numpy 2.5.2, pybind11 3.0.2, clang-format
18.1.8 (no scipy, mpmath, cuQuantum, Qiskit, PennyLane or Strawberry Fields installed). No
RTX 3090 (sm_86) is attached; see owner question Q7.

## Amendment 2026-10-23 (supersedes the text below where they differ)

Recorded in the [2026-10-23 decision](roadmap-2026-10-23.md) (owner agreement, AGENTS rule 5),
with the plan-review dispositions:

- **Required core:** phases 1, 2a, 5, 7 and a new phase 8 (time-to-quality study, measurement
  only, PQC against FourierEdges of the matching degree and other classical carriers, FP32,
  on GB10).
- **Fock (phases 3-4)** only after a go/no-go gate: numerical preflight (R6) with the
  norm-deficit criterion (R1) and a CPU prototype compared with normalized Hermite functions;
  owner decision, default no-go.
- **Deferred outside M5 acceptance:** phase 2b (cuStateVec) and phase 6 (finite shots); their
  text below remains the specification if reinstated.
- **Before phase 2a:** `docs/evidence/device-adaptation-contract.md`; the resident quantum
  kernels follow it ("no GPU-specific constants" made checkable).
- **Q7 resolved:** an RTX 3090 (sm_86, MSVC) is available; GPU phases record runtime results
  on it in addition to GB10.
- **D2 / Q1:** third-party simulators allowed in evidence and optional, cleanly skipping
  tests; the in-repo references remain the required gate.
- **Phase 1 review fixes:** R3 (parameter-shift near-zero bound scaled by epsilon, gate count
  and expectation magnitude) and R5 (tape preparation measured separately).
- **Phase 3 (if taken):** R1 norm-deficit rejection and FP32 strategy; **phase 5:** R2
  deterministic per-monomial summation order and the `energy()` constant convention;
  **phase 6 (if taken):** R4 explicit resampling.
- **Profiling scope (Q11 / AGENTS rule 6):** matched complete-step measurements, removal of
  pathological bottlenecks and contract compliance; no exhaustive tuning requirement.

## Sources and what they fix

- **The brief** (`introduction.md`, commit `af1c96b`), which CONTRACT cites as "the brief".
  It lists quantum carriers ("PQC / parametrised schemes, Fock modes – experimental
  section") and names [Quantum-KAN](https://github.com/wtroy2/Quantum-KAN) as an example
  with defects: converting the AST to strings and hashing via `__str__`, parsing strings
  during numerical computation, memory UB in Eigen, hard-coded topology, copy-paste,
  namespace pollution, and "Rosenberg's quadratisation is naive; is a symbol engine
  (SymEngine) even needed here?". Nothing from it is ported. This plan answers each point:
  typed enumerations compiled to an integer opcode tape (no strings, no hashing, no
  parsing), no Eigen, topology from the existing `Network`, one carrier protocol (no
  per-carrier copy of Layer/Network logic), everything in `kan::`, and **no symbolic
  engine**: a pseudo-Boolean polynomial is a canonical list of integer-indexed monomials and
  quadratization is a typed algorithm on it, with a penalty-free reduction for negative
  monomials and greedy pair selection for positive ones instead of naive Rosenberg alone.
- **Quantum-KAN / Troy**, [Sparks of Quantum Advantage and Rapid Retraining in Machine
  Learning (arXiv 2407.16020)](https://arxiv.org/abs/2407.16020): KAN training recast as a
  QUBO over binary-encoded coefficients of Bézier (linear-in-coefficient) edges, solved on
  D-Wave annealers. This is the concrete role of QUBO/quadratization in this library.
- **QKAN / DARUAN**, Jiang, Huang, Chen, Goan, [Quantum Variational Activation Functions
  Empower Kolmogorov-Arnold Networks (arXiv 2509.14026)](https://arxiv.org/abs/2509.14026):
  each edge is a single-qubit data re-uploading circuit
  `U(x) = W(r+1) S(w_r x + b_r) W(r) ... S(w_1 x + b_1) W(1)`, `S(φ) = exp(-iφσ_j/2)`,
  trainable preprocessing weights `w_l` and biases `b_l`, trainable single-qubit blocks
  `W` (example `R_z R_y R_z`), initial state `|0>`, edge value `<ψ|σ_z|ψ>`; edges sum at
  nodes. The PQC carrier is this construction, generalized to a few qubits.
- **Fock / CV**: Killoran et al., [Continuous-variable quantum neural networks (arXiv
  1806.06871)](https://arxiv.org/abs/1806.06871) (layer = interferometer, squeezing,
  displacement, Kerr); Miatto and Quesada, [Fast optimization of parametrized quantum
  optical circuits (arXiv 2004.11002)](https://arxiv.org/abs/2004.11002) (recurrences for
  truncated Fock matrix elements of displacement/squeezing and their derivatives). No
  published "Fock KAN" edge was found; the Fock carrier is this library's own experimental
  construction (the brief's "experimental section") and is labelled so.
- **Gradients**: parameter-shift rule for gates `exp(-iθG)` with `G` of eigenvalues ±1/2
  ([Schuld et al. 2019, arXiv 1811.11184](https://arxiv.org/abs/1811.11184)); adjoint
  differentiation ([Jones and Gacon 2020, arXiv 2009.02823](https://arxiv.org/abs/2009.02823)).
- **Quadratization**: Rosenberg (1975) substitution `y = x_i x_j` with penalty
  `M(x_i x_j - 2x_i y - 2x_j y + 3y)`; Freedman–Drineas / Kolmogorov–Zabih reduction of a
  negative monomial `-a·x_1…x_k = min_y -a·y(Σx_i - (k-1))`, `a > 0`, which needs no
  penalty; Ishikawa (2011) for positive monomials (recorded, not in scope); survey: Dattani,
  [Quadratization in discrete optimization and quantum mechanics (arXiv 1901.04405)](https://arxiv.org/abs/1901.04405).
- **cuStateVec** ([release notes](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/release-notes.html)):
  `custatevecApplyMatrixBatched` (cuStateVec 1.4+) applies per-state matrices to batched
  state vectors, `custatevecComputeExpectationBatched` (1.6+) batched expectations. CUDA 13
  aarch64 wheels of cuQuantum 25.x are listed on the NVIDIA package index; installation,
  sm_121 support and the API set on this machine are validated in phase 2b, not assumed.

## Scope

In scope: two new edge carriers (`PqcEdges`, `FockEdges`) through the existing carrier
protocol (CPU, resident CUDA, initializers, L2, `set_parameters`, `set_carrier`, Python); an
internal CPU reference simulator and resident CUDA kernels for them; exact expectation
values with adjoint gradients (default) and parameter-shift gradients; opt-in finite-shot
sampling for the PQC carrier; a CPU-only pseudo-Boolean / QUBO module with quadratization
and a single-layer KAN-to-QUBO compiler, verified exhaustively and by separately written
references; a matched comparison of the custom batched kernel against cuStateVec.

Out of scope (recorded): annealer/D-Wave/dimod connectivity; noise models and density
matrices; multi-mode Fock circuits; Fock shot sampling (homodyne/photon counting); Ishikawa's
positive-monomial reduction; a library compiler for multi-layer KAN objectives (one
two-layer instance is a test-built case, phase 5); any symbolic algebra; porting code from
Quantum-KAN, PennyLane, Qiskit or Strawberry Fields; model serialization (roadmap M10).

## Recorded defaults (AGENTS rule 5)

Defaults let work proceed; those marked **(Qn)** are hard to reverse and are listed with
their defaults at the end; none blocks a phase.

D1. **Location.** Public headers `include/kan/pqc.hpp`, `include/kan/fock.hpp`,
`include/kan/qubo.hpp`; shared host/device formulas (gate matrices, tape interpretation,
recurrences) in `src/carriers/quantum/` (as CONTRACT "Extension boundaries" plans); CPU loops
in `src/backends/cpu/pqc_edges.cpp`, `fock_edges.cpp`; resident kernels in a **new
translation unit** `src/backends/cuda/quantum.cu`. Its host launchers take raw device
pointers, sizes, the stream and the status word pointer, and are declared in a private
header called from the plans in `resident.cu` (no `-rdc`). Device helpers the kernels need
from `resident.cu` (status reporting, finiteness, reduction trees) are moved verbatim into a
private header in a separate extraction commit that must pass the SASS-identity gate on its
own. QUBO is not a carrier and has no device code: `src/qubo/`, in the CPU library `kan`.
**(Q10)**

D2. **No external simulator dependency in the library.** The always-available simulator is
the internal CPU reference. The GPU adapter is the existing optional `kan::cuda` target with
custom kernels. cuStateVec is installed only into an evidence prefix for the phase-2b
comparison. If it wins that comparison, adopting it (a build option `KAN_ENABLE_CUSTATEVEC`,
OFF by default, selecting the library path inside `kan::cuda`, with both paths kept
parity-tested against each other) is an owner decision, because it adds a dependency.
Independent references are written in this repository (dense Kronecker-product qubit
simulator and dense matrix-exponential Fock reference in `tests/support/`, NumPy references
in Python tests); Python simulators are not used. The CPU-only build stays free of CUDA and
cuQuantum. **(Q1)**

D3. **Typed structure, per-edge parameters** (the `WaveletConfig` pattern). The circuit
structure is per-layer configuration (value type, `operator==`, checked by
`upload_parameters`); every edge has its own parameters. Gates and observables are
`enum class` values. The configuration is compiled into a flat tape of
`{opcode, operands, parameter slot}` integers (private, `src/carriers/quantum/tape.hpp`),
identical for every edge of the layer, so a device kernel's opcode stream is warp-uniform.
On the CPU the tape is compiled per call into a bounded stack array (at most 815 entries for
PQC at the limits, 16 for Fock under D9), never stored in the public carrier aggregate; the
resident plan caches it in device memory at construction. Numerical code never sees a
string; error messages are string literals only.
**Source check** (phase 1 RED, a CTest): in `src/carriers/quantum/*`,
`src/backends/cpu/pqc_edges.cpp`, `src/backends/cpu/fock_edges.cpp`,
`src/backends/cuda/quantum.cu` and `src/qubo/*`, the regular expression
`std::(string|string_view|to_string|sto[dfil]|hash|unordered_\w+|regex|\w*stringstream)|\b(strtod|strtof|sscanf|atof|atoi)\b|#include <(string|sstream|regex|unordered_map|unordered_set)>`
must not match; the test reference headers `tests/support/dense_qubit_reference.hpp` and
`dense_fock_reference.hpp` must not include anything from `src/` or `kan/` (independence).

D4. **Carrier coefficient tensor.** Each carrier keeps a per-edge linear readout coefficient
tensor `(outputs, inputs, terms)`, so L2, `coefficients()`, `set_parameters` and the
initializers act on it unchanged: edge value `phi(x) = Σ_t c[o,i,t]·m_t(x)`, `m_t` the
measured expectations. For DARUAN this is a reparametrization (its node sum has no free
scale), recorded as the wavelet normalization was.

D5. **Exact expectations are the default measurement**; shots are opt-in (PQC only).

D6. **Adjoint differentiation is the default gradient** in exact mode; the parameter-shift
rule is implemented on the CPU as an independent exact gradient (`evaluate_pqc(...,
GradientMethod::ParameterShift)`) and is the gradient of shot mode. Both are verified against
finite differences and against each other.

D7. **Precision.** The FP64 executor is the parity reference; FP32 uses `complex<float>`
amplitudes with accurate `sincosf`/`expf` (no fast intrinsics unless a measured accuracy bound
is recorded, as the wavelet FP32 envelope was); TF32 has no contraction in these carriers and
computes as FP32 (only a residual branch's GEMMs use TF32), as for `WaveletEdges`.

D8. **Initializers** (a stated deviation from "nonlinear parameters are kept"; CONTRACT
records it). Reference measure for both quantum carriers: uniform on `[-1, 1]` (variance 1/3;
the range of a `TanhMap`). PQC moments depend on readout and entangler: with
`Entangler::None` the qubits are independent DARUAN circuits, `m = 1/3` for PauliZ (the
many-block limit of `E[<Z>²]` for uniformly drawn ZYZ blocks) and `3^-n` for Parity; with an
entangler `m = 1/(2^n + 1)`, the Haar average of `<P>²` for one Pauli observable on n qubits
(an assumption about the drawn circuits); Fock at the constructor placement (block parameters zero), with
`W = Σ_l w_l`, `B = Σ_l b_l`: Quadrature `m = 2(W²/3 + B²)`, PhotonNumber
`m = W⁴/5 + 2W²B² + B⁴` (exact for displacement-only circuits, used as the convention
otherwise). VarianceScaling `σ_t² = gain²·(1/3)/(inputs·terms·m_t)`. NoiseInit `G = terms`.
Draw order: coefficients in layout order, then (PQC only) block angles in layout order,
uniform in `[-π, π)` (`π·(2u - 1)`), then residual weights; a pending second normal value of
the coefficient draws survives the uniform angle draws and is the first residual draw (the
denominator rule). Encoding weights/biases and every Fock nonlinear parameter are kept. Only
basic operations are used, so new digests are portable and existing digests unchanged.
**(Q3)**

D9. **Limits** (validated, `std::invalid_argument`): PQC qubits 1..5, repetitions 1..32,
block length 1..3; Fock cutoff 8..32, repetitions 1..16, block length 0..6 with
`r·(B + 1) <= 16` until phase 4 measures per-gate state storage, guard 2..cutoff/2.
**(Q2)**

D10. **Fock truncation policy**: true truncated matrix elements, no renormalization, a
guard-band check that rejects (never clips); default cutoff 32, guard 2, `max_leakage`
1e-10, supported input domain stated (below). **(Q4)**

D11. **Shot keys** do not include a training-step counter; the seed is uploadable state, so
resampling does not need a new executor. **(Q5)**

D12. **QUBO encoding**: offset binary `c = lo + step·Σ_k 2^k z_k` per coefficient; Freedman
for negative monomials, then Rosenberg with greedy pair selection for positive ones. **(Q6)**

D13. **Protocol completeness per phase.** A phase that adds a carrier alternative also adds,
in the same RED/GREEN series, everything that must compile or dispatch on the variant:
resident host stubs that reject the new carrier with `std::invalid_argument` ("resident
execution of PQC layers is not available") until its resident phase, initializer dispatch,
a Python stub (a minimal read-only snapshot class, or `TypeError` with a test) until
phase 7, and for PQC `shots > 0` rejected with `std::invalid_argument("finite shots are not
available")` until phase 6. Both CUDA builds and the Python suites stay green at every phase.

## PQC carrier (`PqcEdges`)

```cpp
enum class Rotation : std::uint8_t { X, Y, Z };          // R_a(θ) = exp(-iθσ_a/2)
enum class Entangler : std::uint8_t { None, CnotChain, CnotRing, CzChain };
enum class PqcReadout : std::uint8_t { PauliZ, Parity }; // terms = qubits, or 1
enum class GradientMethod : std::uint8_t { Adjoint, ParameterShift };
struct PqcConfig {
    std::size_t qubits = 1;                 // n, 1..5
    std::size_t repetitions = 2;            // r, 1..32 (DARUAN's r)
    Rotation encoding = Rotation::Y;        // S(φ) = R_enc(φ)
    std::vector<Rotation> block{Rotation::Z, Rotation::Y, Rotation::Z}; // W, 1..3 axes
    Entangler entangler = Entangler::None;  // must be None for n = 1
    PqcReadout readout = PqcReadout::PauliZ;
    std::size_t shots = 0;                  // 0: exact expectations (phase 6)
    bool operator==(const PqcConfig&) const = default;
};
struct PqcEdges {
    PqcConfig config;
    std::vector<double> coefficients;     // c, (outputs, inputs, terms)
    std::vector<double> angles;           // θ, (outputs, inputs, (r+1)·n·B), index (l·n + q)·B + k
    std::vector<double> encoding_weights; // w, (outputs, inputs, r·n), index (l-1)·n + q
    std::vector<double> encoding_biases;  // b, same layout
    std::uint64_t shot_seed = 0;          // state (uploadable), used only when shots > 0
    bool operator==(const PqcEdges&) const = default;
};
struct PqcGradients { std::vector<double> angles, encoding_weights, encoding_biases; };
struct PqcEvaluation { std::vector<double> values; // m_t
                       std::vector<double> input_derivatives, angle_derivatives,
                           encoding_weight_derivatives, encoding_bias_derivatives; }; // per t, row-major
void validate_pqc(const PqcConfig& config);
PqcEvaluation evaluate_pqc(const PqcConfig& config, double x, std::span<const double> angles,
                           std::span<const double> encoding_weights,
                           std::span<const double> encoding_biases,
                           GradientMethod method = GradientMethod::Adjoint);
void set_pqc_parameters(Layer&, std::span<const double> coefficients, std::span<const double> angles,
                        std::span<const double> encoding_weights,
                        std::span<const double> encoding_biases, std::span<const double> bias);
```

`Layer(inputs, outputs, PqcConfig)` uses the constrained template constructor pattern of
`WaveletConfig`; `Carrier` gains `PqcEdges` and `NonlinearGradients` gains `PqcGradients`,
appended at the end. Validation rejects out-of-range sizes, unknown enumerator values,
adjacent equal block axes (their angles simply add), an entangler other than `None` on one
qubit, `CnotRing` below 3 qubits; `set_carrier` rejects nonfinite parameters and a nonzero
`shot_seed` with `shots = 0`. Structurally inert angles remain expressible and are documented
rather than rejected (as in DARUAN): with the default ZYZ block the first Z rotations act on
`|0…0>` (a phase) and the last Z rotations commute with Z/parity readouts, so their gradients
are identically zero; tests assert this.

**Edge function.** Little-endian basis (bit q of the index is qubit q), initial state
`|0…0>`. `W_l = ⊗_q R_{a_{B-1}}(θ[l,q,B-1]) ⋯ R_{a_0}(θ[l,q,0])` (axis 0 applied first),
qubits in ascending order. `E` = `CNOT(q→q+1)` for q ascending (`CnotRing` then adds
`CNOT(n-1→0)`), or `CZ(q,q+1)` for q ascending; `E` is the identity for `None`.

```
|ψ(x)> = W_r · S_r(x) · E · W_{r-1} · … · S_1(x) · E · W_0 |0…0>
S_l(x) = ⊗_q R_enc(w[l,q]·x + b[l,q]),  l = 1..r
m_q(x) = <ψ|Z_q|ψ> (PauliZ, terms n)   or   m(x) = <ψ|Z⊗…⊗Z|ψ> (Parity, terms 1)
phi_{o,i}(x) = Σ_t c[o,i,t]·m_t(x)
```

With `n = 1` this is DARUAN with `W(1..r+1) = W_0..W_r`. The tape lists, in application
order, every rotation, encoding rotation and two-qubit gate; `G` is its length (two-qubit
gates included).

**Constructor:** `c = 0`, bias 0, `θ = 0`, `w = 1`, `b = 0`, `shot_seed = 0`. With `c = 1`,
`n = 1` and `Y` encoding the edge is `cos(r·x)`. At this symmetric point (Y encoding,
real-amplitude circuit) Z-angle derivatives are exactly zero on the adjoint path (and within
a few ulp on the parameter-shift path), while input, encoding and Y-angle derivatives are
nonzero at generic `x`. With Z encoding the constructor circuit is diagonal and the edge is
constant in x (documented; the initializer's angle draws leave this point).

**Derivatives.** For a rotation angle `θ_g`: `∂m/∂θ_g = (m(θ_g+π/2) - m(θ_g-π/2))/2` (exact,
`G = σ/2`). The input enters every encoding gate:
`∂m/∂x = Σ_{l,q} w[l,q]·∂m/∂φ_{l,q}`, `∂m/∂w[l,q] = x·∂m/∂φ_{l,q}`, `∂m/∂b[l,q] = ∂m/∂φ_{l,q}`.
Adjoint (per sample and edge, upstream `u = u[b,o]`), exact order:

```
ψ ← U_G ⋯ U_1 |0…0>                       (forward; m_t from ψ)
λ ← u · Σ_t c_t M_t ψ
for g = G … 1:
    if gate g has a parameter p_g (U_g = exp(-i p_g σ/2)):
        d_g ← Im(<λ|σ_g|ψ>)               (= ∂(u·Σ c_t m_t)/∂p_g; 2Re<λ|(-iσ/2)ψ>)
    ψ ← U_g† ψ;  λ ← U_g† λ
angle VJP += d_g; encoding gate (l,q): b VJP += d_g, w VJP += x·d_g, input VJP += w·d_g
```

Accumulation orders over samples, outputs, inputs and gates are fixed and written into
CONTRACT (as the wavelet CPU order). States are recovered by inversion, so nothing is
stored per gate; phase 2a measures the FP32 drift of this inversion against FP64 at the
limits and, if it exceeds the C1 tolerances, switches to checkpointed recomputation.

**Physical normalization.** States are unitary images of `|0…0>`; Born probabilities
`p_k = |ψ_k|²`. Tests require `|Σ_k p_k - 1| <= (4G + 2^n)·2^-52` (FP64) and each
`p_k ∈ [0, 1]` within rounding; FP32 bounds are measured and recorded. No renormalization
is applied (it would hide defects).

## Fock carrier (`FockEdges`)

```cpp
enum class FockGate : std::uint8_t { Rotation, Squeezing, Kerr }; // trainable block gates
enum class FockReadout : std::uint8_t { Quadrature, PhotonNumber };
struct FockConfig {
    std::size_t cutoff = 32;               // D, levels 0..D-1, 8..32
    std::size_t repetitions = 1;           // r, 1..16
    std::vector<FockGate> block{FockGate::Rotation, FockGate::Squeezing,
                                FockGate::Rotation, FockGate::Kerr};
    std::vector<FockReadout> readouts{FockReadout::Quadrature}; // terms; nonempty, distinct
    std::size_t guard = 2;                 // 2..D/2 guard-band levels
    double max_leakage = 1e-10;            // finite, in (0, 1e-2]
    bool operator==(const FockConfig&) const = default;
};
struct FockEdges {
    FockConfig config;
    std::vector<double> coefficients;     // c, (outputs, inputs, terms)
    std::vector<double> gate_parameters;  // (outputs, inputs, r·B), index l·B + k
    std::vector<double> encoding_weights; // w, (outputs, inputs, r), index l
    std::vector<double> encoding_biases;  // b, same layout
    bool operator==(const FockEdges&) const = default;
};
struct FockGradients { std::vector<double> gate_parameters, encoding_weights, encoding_biases; };
// evaluate_fock / set_fock_parameters / validate_fock: as the PQC declarations (no method).
```

Validation also rejects two adjacent equal diagonal gates (`Rotation, Rotation` or
`Kerr, Kerr`: parameters add).

**Edge function** (ħ = 1, `x̂ = (a + a†)/√2`, `n̂ = a†a`), gates applied left to right in
time:

```
|ψ> = [G_{r,B-1} ⋯ G_{r,0} D(α_r)] ⋯ [G_{1,B-1} ⋯ G_{1,0} D(α_1)] |0>,   α_l = w[l]·x + b[l]
D(α) = exp(α(a† - a)) (real α),  R(φ) = exp(iφn̂),  S(ζ) = exp(ζ(a² - a†²)/2) (real ζ),  K(κ) = exp(iκn̂²)
m_t = <ψ|M_t|ψ>, M_t the truncated x̂ (tridiagonal) or n̂ (diagonal), on the unnormalized truncated state
```

`PhotonNumber` is insensitive to rotations and Kerr gates (they commute with n̂); the
default readout is the quadrature for that reason. Constructor: `c = 0`, gate parameters 0,
`w = 1`, `b = 0`; with `c = 1` the edge is `√2·r·x` up to truncation.

**Truncation and leakage.** Gate matrices are the true matrix elements `<m|G|n>`, `n < D`,
from recurrences; rotation and Kerr are diagonal and exact. Derivatives need rows beyond the
cutoff: row `D` for displacement (`∂D/∂α = (a† - a)D` couples `m` to `m ± 1`) and rows `D`
and `D + 1` for squeezing (`(a² - a†²)/2` couples `m` to `m ± 2`). Truncated D and S are not
unitary; norm lost to levels `>= D` is not restored. **Check:** after every displacement and
squeezing gate, the guard-band population `Σ_{m=D-g}^{D-1} |ψ_m|²` must be
`<= max_leakage`; otherwise the CPU raises `std::domain_error("Fock state leaves the
truncation guard band")` and the resident executor sets status bit 256 (the next free bit:
1 nonfinite, 2 rational pole, 4 parameter range, 8–128 grid refit), decoded in both the eager
`result()` and the deferred step check after bit 1 and the parameter-range bit, raising
`std::domain_error("resident Fock state leaves the truncation guard band")`; failed training
steps roll back (C9). No clipping, no renormalization. Guard must be at least 2 because
squeezing preserves photon-number parity (a guard of 1 on an even cutoff never sees a
squeezed vacuum). This is an **empirical indicator**, not a bound: the evidence measures,
over the tested parameter ranges, the largest `|m_t(D) - m_t(D + 32)|` among passing states
and CONTRACT states that relation. The norm satisfies `||ψ||² <= 1 + (4G + D)·D·2^-52`
(FP64).

**Supported input domain.** A coherent displacement α puts `e^{-α²}α^{2m}/m!` in level m;
with D = 32 and the default guard and threshold, `|α| <= 2` passes (`|α| = 3` does not), and
squeezing narrows this. Inputs are not clipped: a Fock layer is meant to follow an explicit
`TanhMap` (or other bounded input map), which the documentation and examples use; a RED test
trains through a trajectory approaching the guard and records how often steps are rejected.

**Derivatives.** Adjoint through the truncated linear maps (the exact derivative of the
computed truncated function, which is what finite differences of the implementation
measure). Truncated gates cannot be inverted, so the CPU stores the per-gate states of one
edge evaluation (at most `(1 + 16)·32` amplitudes under D9, a fixed per-call workspace). The
device layout (registers, shared memory, or checkpointed recomputation) is a phase-4 decision
with profiling; D9's `r·(B + 1) <= 16` bound is revisited only with that evidence.
`∂/∂x = Σ_l w[l]·∂/∂α_l`.

## Measurement semantics

- **Exact (default, `shots = 0`).** Expectations from amplitudes; deterministic and
  bitwise-reproducible on one build (CPU) or one device and batch (resident), with the
  existing parity tolerances between them.
- **Shots (PQC only, phase 6).** `shots = S > 0`: each circuit evaluation draws S basis
  outcomes from its Born distribution by inverse CDF over basis indices in ascending order,
  with `u = (mix(key) >> 11)·2^-53` (the initializers' SplitMix64 `mix`, `γ =
  0x9E3779B97F4A7C15`). The key chains the indices in a fixed order:
  `h = shot_seed`, then for `v` in `(layer position, sample, output, input, circuit, shot)`:
  `h = mix(h + (v + 1)·γ)`; `key = h`. Layer position is the KAN layer index in a `Network`
  (0 for a lone `Layer`), sample is the position in the batch, circuit is 0 for the forward
  circuit and `1 + 2p`, `2 + 2p` for the `+π/2`, `-π/2` shifts of parameter slot p (angles,
  then encoding slots). If `u` is not below the computed CDF total, the outcome is the last
  basis state with `p_k > 0`. `m_q` is the mean of the `±1` eigenvalues of `Z_q` (or the
  parity). The backward pass in shot mode is the parameter-shift rule with sampled
  expectations (adjoint is not defined for samples). `shots` is structure (part of
  `operator==`); `shot_seed` is state, replaced by `set_carrier`/`upload_parameters` without
  rebuilding an executor. Guarantees: same seed and inputs give bitwise-identical estimates on
  one build/device; keys depend on logical indices only, so results are independent of
  launch geometry, and eager and captured-graph resident steps are bitwise identical; CPU and
  GPU draw the same `u`, but an outcome whose `u` lies within rounding of a CDF boundary may
  differ, so CPU/GPU parity is "identical outcomes except boundary flips", counted and
  bounded in tests.

## Simulator adapters

The adapter boundary is the compiled tape plus per-edge parameter and input arrays; a
simulator consumes it and returns expectations and VJPs. Adapters: (1) CPU reference
(always built, `kan`); (2) resident CUDA custom kernels (`kan::cuda`); (3) cuStateVec, an
evaluation harness in `docs/evidence/m5/` during phase 2b, adopted into `kan::cuda` only by
owner decision (D2). For Fock no CUDA-ecosystem library simulates truncated bosonic modes;
custom kernels are justified by the absence of a library facility (AGENTS rule 8), stated in
phase-4 evidence. Any adopted or measured library records its version, package source,
SHA256, GPU/toolchain requirements and an installation validation in evidence.

## QUBO and quadratization

**Role in this library.** Following Troy (2407.16020), compile the training objective of a
KAN layer with binary-encoded coefficients into a QUBO an annealer or exact solver can
minimize, and reduce higher-order pseudo-Boolean objectives to quadratic ones. For a layer
linear in its coefficients (`BasisEdges`) with fixed inputs the loss is quadratic in the
coefficients, so offset-binary encoding gives a QUBO directly (no quadratization).
Quadratization is needed for genuinely higher-order objectives; the KAN-derived instance is
a test-built two-layer case (phase 5, test 7). The module is CPU only and independent of the
carriers.

**API (`include/kan/qubo.hpp`).** `PseudoBooleanPolynomial` (variable count; canonical
monomials: strictly increasing `std::uint32_t` indices, `x² = x` applied, equal monomials
merged, zero coefficients removed, ordered by degree then lexicographically; a constant),
`Qubo{variables, constant, linear, quadratic (i < j, sorted)}`, `energy(...)`,
`quadratize(polynomial, QuadratizationOptions{penalty_margin = 1})` returning the QUBO and
the auxiliary definitions, `exhaustive_minimize(...)` (Gray-code incremental evaluation, at
most 30 variables, ties to the lowest assignment index), and
`compile_layer_qubo(layer, input, target, batch, FixedPointEncoding, l2)` with
`decode(...)`. The compiled objective is
`L = Σ_{b,o}(y[b,o] - t[b,o])²/(batch·outputs) + 0.5·l2·Σc²` (the resident MSE plus the CPU
coefficient penalty) with the layer's bias fixed; layers other than `BasisEdges`, and layers
with a residual branch, are rejected (`std::invalid_argument`). Ordered containers only;
output is bitwise-deterministic.

**Quadratization algorithm.** Pass 1: each negative monomial of degree >= 3 is replaced by
the Freedman form with one auxiliary (no penalty). Pass 2: while a positive monomial of degree
>= 3 remains, choose the pair occurring in most such monomials (ties: lexicographically
smallest), introduce `y` for it in every positive monomial of degree >= 3 that contains it, and
add the Rosenberg penalty with `M = penalty_margin + Σ|a|` over the monomials changed by this
substitution, computed on the current polynomial (the inductive proof in phase 5 evidence).
Substitution preserves signs, so pass 2 never creates negative higher-order monomials.
`penalty_margin` is finite and >= 0; with 0, ties in y can occur (the energy equality still
holds), and `decode` always sets `y = x_i x_j`.

**Independently verified** means, for every test instance:
1. *Exhaustive equality*: for every original assignment `x ∈ {0,1}^n`,
   `min_y Q(x, y) == f(x)` exactly, enumerating all `2^(n+m)` assignments; generated
   instances keep `n + m <= 24`, and all data are dyadic so every value is exact in double.
2. *Argmin preservation*: the minimizing `x` of `Q` (projected) equal those of `f`.
3. *Penalty necessity*: a constructed instance with the penalty below the bound violates (1).
4. *Second reference*: a separately written evaluator that does not use the library's
   polynomial types: in C++ tests, the objective from the raw test specification, and for
   KAN compilation the loss computed from `Layer::forward` / `Network::forward` outputs of the
   decoded coefficients (an independently verified path); in Python (phase 7) a NumPy
   enumerator written from the definitions. Self-consistency of the library with itself does
   not count.

## GPU design requirements (phases 2a, 2b, 4, 6)

- Per-edge circuits are small; work is batched over samples × edges. Candidate layouts: one
  thread per (sample, edge) with the state in registers for small `2^n`/`D`; one warp per
  (sample, edge) with lanes over amplitudes for larger ones. Per-edge rotation matrices,
  squeezing matrices and Kerr phases depend on parameters only and are computed once per
  call (as `wavelet_scales_kernel`); encoding gates and displacement matrices depend on the
  sample and are formed in the kernel. The tape is in device memory and warp-uniform.
- Parameter VJPs are reduced over samples deterministically (fixed partial slots and fixed
  trees, no floating-point atomics), as the wavelet backward.
- Launch geometry comes from device properties at construction (`cudaGetDeviceProperties`,
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` per instantiation), as the wavelet cost
  model; no GPU-specific constants. Both sm_86 and sm_121 are compiled with `-Xptxas -v`
  (registers, no spills recorded); runtime results on sm_86 only if a device is available
  (Q7).
- Profiling with nsys (ncu counters are not accessible here: `ERR_NVGPUCTRPERM`); identify and
  address a measured bottleneck; accept tuning only with matched complete-step timings.
- **Library comparison (phase 2b).** cuStateVec materializes one state vector per
  (sample, edge): a 256→256 layer at batch 8192 is 5.4e8 states, about 137 GB of FP64
  amplitudes at n = 4, beyond the 128 GB of GB10, before per-state encoding matrices. The phase
  therefore starts with a feasibility spike (install, sm_121 support, presence of the batched
  APIs, memory per state, determinism of its reductions), then a harness that processes the
  batch in chunks, on workloads capped to what fits (layers 16→16 and 64→64, n ∈ {1, 2, 4},
  r ∈ {2, 4}, batch 256–4096, FP64 and FP32), measuring complete `train_step`s (forward, loss,
  backward, SGD) with the same numerical checks; any missing library piece (e.g. adjoint inner
  products) is a counted custom kernel. If cuStateVec cannot meet the contract (fixed-order
  reductions, per-state matrices without prohibitive memory), the evidence records "not
  covered by the library facilities" under rule 8. Otherwise the library is recommended
  unless the custom path is faster in the matched complete step by more than the run-to-run
  spread (ABBA, at least 3 pairs).

## Isolation gates (every phase)

- The 8 golden dumps (`docs/evidence/backlog/golden/`) byte-identical to the
  [R6 hashes](backlog/R6-bench/golden-sha256.txt) in both CUDA builds (parity, FMA).
- Every pre-existing `resident.cu` kernel SASS-identical
  ([`sass_cmp.py`](backlog/C5-bench/sass_cmp.py), before vs after objects, both builds),
  including the D1 extraction commit. This gate is for device code; host functions may
  change because the variants grow (new alternatives are appended, existing indices kept).
- Existing pinned initializer digests unchanged; frozen benchmark suites (`basis`, `local`,
  `rational`) ABBA within noise with 0 checksum mismatches (phases touching CUDA or CPU hot
  paths).
- Existing CTest suites (CPU, CUDA parity, CUDA FMA) and Python suites pass unchanged (D13).
- `scripts/check_format.sh` exits 0; the D3 source check passes.
- CPU-only configure/build/install has no CUDA or cuQuantum include, link or symbol.

## Phases

Order: (1) PQC CPU → (2a) PQC resident CUDA → (2b) cuStateVec comparison → (3) Fock CPU →
(4) Fock resident CUDA → (5) QUBO/quadratization → (6) PQC shots → (7) Python and
acceptance. Justification: PQC is the published variant (DARUAN) and establishes the quantum
protocol (tape, simulator, gradient verification) that Fock reuses; its resident design and
the library comparison come right after, so Fock's kernels follow a measured design. Carrier
phases change the shared variants and the resident executor and run serially. QUBO is
independent and CPU-only: it is placed after the carriers, but it touches disjoint files and
the coordinator may run it in parallel with phases 2a–4. Shots come late because they are
opt-in and need the parameter-shift path on the device. Python bindings land once, against
the stable C++ surface (with the D13 stubs before), together with the integrated gates. Each
phase is one implementer + independent reviewer + patch series: RED commit (failing for the
right reason, retained log), GREEN commits, evidence commit, CONTRACT update, journal line.

### Phase 1: PQC carrier on the CPU (exact, adjoint and parameter-shift)

RED tests (`tests/carriers/quantum/pqc_test.cpp`, `pqc_layer_test.cpp`, reference
`tests/support/dense_qubit_reference.hpp`: Kronecker products of explicitly written 2×2/4×4
matrices, a different algorithm from the in-place tape simulator, no `src/` includes):
1. Validation: qubits 0/6, repetitions 0/33, empty or 4-axis block, adjacent equal axes,
   non-`None` entangler with 1 qubit (and `None` with 3 qubits accepted), `CnotRing` with 2,
   out-of-range enumerators (`static_cast<Rotation>(7)`), nonzero seed with `shots = 0`,
   `shots > 0` (D13, until phase 6) → `std::invalid_argument`.
2. Closed forms: 1 qubit, Y encoding, block `{Y}`: `<Z> = cos(Σ angles + Σ(w x + b))`; ZYZ
   single layer against its analytic Bloch-vector form; 2 qubits `RY(a)⊗RY(b)` + CNOT:
   `<Z0> = cos a`, `<Z1> = cos a cos b`, parity; CZ chain on computational states.
   Tolerance 1e-15 absolute.
3. Dense reference on 200 random configurations (all axes, entanglers, readouts, n <= 4,
   r <= 4): amplitudes and expectations within 1e-13.
4. Normalization and Born-rule bounds (above) on random circuits up to 32 repetitions.
5. Gradients: adjoint vs parameter-shift within 1e-12; each vs central finite differences
   (h = 1e-6, relative 1e-7 with absolute floor 1e-9) for every angle, encoding weight and
   bias, coefficient and the input; inert angles of the default block exactly zero.
6. Layer protocol: constructor values and the `n = 1` `cos(r x)` edge; symmetric-point
   property (Y encoding, `c = 1`: Z-angle adjoint derivatives exactly 0, parameter-shift
   within 4 ulp of 0, input/encoding/Y-angle derivatives nonzero at generic x); batch 0;
   nonfinite input and parameters; `set_carrier` validation; `set_parameters` keeps nonlinear
   parameters; `set_pqc_parameters` atomic; SGD atomic (nonfinite candidate leaves the layer
   unchanged); L2 penalizes coefficients (and residual weights) only, `PqcGradients` of the
   penalty zero; residual branch; mixed network (Chebyshev → PQC → wavelet) finite
   differences.
7. Initializers: all existing digests unchanged; new pinned PQC digest; draw order (D8);
   VarianceScaling `E[y²]/target` within 0.15 over a 16-seed mean (two-sided, as M7) for
   n = 1 and n = 2 with `Entangler::None`, r = 4; measured ratios recorded (not asserted)
   for r = 1, 2 and entangled circuits; angles in `[-π, π)`.
8. Deterministic training: toy regression with independent holdout; MSE threshold fixed in
   the RED commit.
9. No per-edge heap allocation in forward/backward hot loops (allocation probe as M4).
10. D3 source check; D13 stubs (resident construction with a PQC layer raises
    `std::invalid_argument`; Python behaviour tested).

Exit: all RED tests GREEN; full CPU suite, both CUDA builds, Python suites, ASan; CONTRACT
section "Quantum carriers (M5): PQC"; evidence `M5-pqc-cpu.md` with RED/GREEN logs,
measured tolerances, coverage of the new files, isolation gates.

### Phase 2a: PQC on the resident executor

RED tests (`tests/backends/cuda/pqc_resident_test.cpp`):
1. FP64 forward/backward/SGD parity with the CPU within `1e-11|e| + 1e-12 max|e|`; FP32/TF32
   within the C1 tolerances; mixed networks with every existing carrier.
2. `train_step` (OutputGradient and MSE) bitwise equal to the eager sequence; results under
   every forced geometry of a private test hook (as `testing::WaveletLaunch`) bitwise
   identical, or within the documented split-order exception.
3. `upload_parameters`: changed `PqcConfig` rejected; FP32 representability; executor
   unchanged on failure. Zero batch; capacity; unchanged `workspace_allocations()`.
4. Nonfinite status → `std::overflow_error`, rollback of failed steps, deferred status
   interval unchanged.
5. FP32 adjoint-inversion drift against FP64 at the D9 limits within the C1 tolerances (or
   the checkpointed fallback).
6. Existing resident suites unchanged.

Work: D1 extraction commit (SASS gate on its own); freeze quantum benchmark fixtures
(`benchmarks/quantum_benchmark.cpp`, frozen sources under `docs/evidence/m5/`) before tuning;
launch model; nsys profile; ptxas reports for sm_86 and sm_121. Exit: GREEN in parity and
FMA builds; Compute Sanitizer (memcheck, racecheck, initcheck, synccheck) clean on the new
suite; isolation gates; evidence `M5-pqc-cuda.md`.

### Phase 2b: cuStateVec feasibility and matched comparison

Feasibility spike, chunked harness and comparison as specified above; evidence
`M5-custatevec.md` with version, source, SHA256, installation validation, memory per state,
the comparison table (raw CSVs) and the recommendation. It proceeds on the Q1 default (an
evidence-only install is a CUDA-ecosystem library use AGENTS rule 8 permits); library code
changes only if the owner adopts the library (D2).

### Phase 3: Fock carrier on the CPU

RED tests (`tests/carriers/quantum/fock_test.cpp`, `fock_layer_test.cpp`, reference
`tests/support/dense_fock_reference.hpp`: generators truncated at `D + 40`, exponentiated by
Taylor scaling-and-squaring in double-double arithmetic written in the test, independent of
the recurrences and of the platform's `long double`, no `src/` includes):
1. Validation (limits, `r·(B + 1) <= 16`, enumerators, distinct readouts, adjacent equal
   diagonal gates, `max_leakage`, guard 1 rejected).
2. Matrix elements of D(α) and S(ζ) for `n < D` and the extra rows (D for displacement;
   D and D+1 for squeezing) against the reference within 1e-13; derivative matrices
   `∂S/∂ζ` at rows D−2 and D−1 and `∂D/∂α` at row D−1 against reference finite differences;
   rotation/Kerr exact phases.
3. Physics: coherent state `P(m) = e^{-α²}α^{2m}/m!`; squeezed vacuum
   `P(2m) = (2m)!/(2^m m!)²·tanh^{2m}ζ/cosh ζ`, `P(2m+1) = 0`; `<x̂> = √2α` after `D(α)`;
   Kerr leaves populations unchanged; the norm bound.
4. Leakage: constructed states just above/below `max_leakage` raise/pass (exact message);
   layer state unchanged after the error; a squeezed state that a guard of 1 would miss is
   caught by guard 2; passing states agree with the `D + 32` reference within the tolerance
   recorded in the RED commit; the supported-domain statement (`|α| = 2` passes, `3` fails
   at the defaults).
5. Gradients: adjoint vs central finite differences for every gate parameter, encoding
   weight and bias, coefficient and the input; and vs finite differences of the reference.
6. Layer protocol, initializer digest and moments (D8), training toy behind a `TanhMap`
   with a recorded rejection count on a trajectory approaching the guard, D13 stubs.

Exit and evidence `M5-fock-cpu.md` as phase 1, including the measured leakage-indicator
table (largest `|m(D) - m(D+32)|` of passing states per parameter range).

### Phase 4: Fock on the resident executor (Q11)

RED tests (`tests/backends/cuda/fock_resident_test.cpp`): as phase 2a (1–4, 6) plus status
bit 256 (`std::domain_error`, the resident message, rollback, deferred-status decoding) and
accept/reject decisions identical to the CPU on samples away from the threshold. Work:
per-gate state storage or checkpointed recomputation chosen by profiling; launch model;
frozen fixtures extended before tuning. Exit and evidence `M5-fock-cuda.md` (custom kernels
justified: no library facility).

### Phase 5: QUBO and quadratization (CPU only)

RED tests (`tests/qubo/pseudo_boolean_test.cpp`, `quadratization_test.cpp`,
`kan_qubo_test.cpp`):
1. Canonicalization: `x² = x`, merging, ordering, zero removal, index bounds, constants;
   bitwise-identical output for permuted input monomials.
2. Quadratization on 500 random polynomials (n <= 8, degree <= 4, at most 12 monomials,
   dyadic coefficients of both signs, `n + m <= 24` asserted): verification items 1–2;
   auxiliary counts reported; greedy pair rule uses fewer auxiliaries than first-pair on a
   fixed instance.
3. Freedman pass alone on negative monomials: same checks, no penalty used.
4. Penalty necessity (item 3).
5. Second reference (item 4) from the raw test specification.
6. KAN compiler: `BasisEdges` Chebyshev layer, 2 inputs, 1 output, batch a power of two,
   dyadic inputs/targets/`lo`/`step`, at most 16 bits: `energy(z) + constant == L(decode(z))`
   for all z, L computed from `Layer::forward`; L2 term; with a test-written exact
   least-squares solution `c*`: `L(c*) <= exhaustive min <= L(round(c*))`; non-`BasisEdges`
   and residual layers rejected.
7. Two-layer KAN instance (test-built): layer 1 binary-encoded `BasisEdges`, fixed layer 2
   Chebyshev size 3; the test expands its loss into a degree-4 polynomial through the public
   API, quadratizes it and verifies exhaustively against `Network::forward` losses.
8. `exhaustive_minimize`: 31 variables rejected, tie rule, Gray-code result equal to plain
   enumeration (test instances <= 20 variables).
Exit and evidence `M5-qubo.md` (penalty bound proof, instance statistics, CONTRACT section).

### Phase 6: finite shots (PQC, CPU and resident)

RED tests (`tests/carriers/quantum/pqc_shots_test.cpp`,
`tests/backends/cuda/pqc_shots_resident_test.cpp`):
1. `shots = 0` paths bitwise unchanged.
2. Estimator: mean over 4096 seeds within 5σ of the exact value; variance `(1 - m²)/S` within
   chi-square bounds; parity readout; outcome frequencies vs Born probabilities (chi-square,
   fixed seeds); the key formula reproduced by a test-written SplitMix64 chain.
3. Reproducibility: same seed identical, different seeds differ; `shot_seed` upload changes
   the samples without a new executor; results independent of forced launch geometries.
4. Parameter-shift gradient under shots: mean within 5σ of the exact gradient.
5. Resident: eager vs graph bitwise; CPU/GPU outcome equality except counted boundary flips
   (bound recorded).
Exit and evidence `M5-shots.md` (cost of the parameter-shift backward vs adjoint measured).

### Phase 7: Python and milestone acceptance

RED tests (`tests/python/pqc_python_test.py`, `fock_python_test.py`, `qubo_python_test.py`):
configuration enums/classes, carrier snapshots, setters, `evaluate_pqc`/`evaluate_fock`,
gradients, resident execution, and NumPy statevector/enumeration references written from the
definitions (second reference for PQC and QUBO). Then the acceptance gates below.

## Milestone acceptance gates

1. Every phase's RED-then-GREEN commits, evidence and review findings resolved; CONTRACT
   updated (quantum carriers, measurement/gradient semantics, QUBO).
2. Integrated CTest in CPU, CUDA parity and CUDA FMA builds; Python suites in the build tree
   and from an installed prefix; CPU-only Python build.
3. AddressSanitizer (CPU suites) and Compute Sanitizer (memcheck, racecheck, initcheck,
   synccheck) on every resident quantum suite: 0 errors.
4. GCC CPU line coverage >= 80% overall and reported for the new files.
5. Installed C++ consumer (CPU and CUDA) and installed Python use of the carriers in scope and QUBO.
6. Matched complete-call CPU/GPU profiling on the frozen quantum benchmark (outputs, VJPs and
   parameters verified), a measured and addressed bottleneck, runtime results on GB10 and
   the RTX 3090, stated limitations (no ncu counters); existing frozen benchmarks as
   regressions. (Amended 2026-10-23: the phase-2b library recommendation applies only if
   phase 2b is reinstated.)
7. Isolation gates on the final head.
8. Independent acceptance review of source and evidence; blocking findings resolved; then
   ROADMAP M5 DONE and roadmap M6 (external baseline) handed off.

## Owner questions

| | Question | Default if unanswered | Waits for the owner |
|---|---|---|---|
| Q1 | Download/install cuQuantum (cuStateVec) into an evidence prefix; adopt it as an optional `KAN_ENABLE_CUSTATEVEC` (OFF) backend if it wins; allow a PennyLane/Qiskit/Strawberry Fields cross-check in tests | install for evidence only; adoption not decided; no Python simulators | adoption into `kan::cuda` only |
| Q2 | Limits: PQC qubits <= 5; Fock cutoff <= 32, `r·(B+1) <= 16`; repetition caps | as D9; revisited with phase-4 evidence | none |
| Q3 | Initializers draw PQC angles in `[-π, π)`; moments of D8 (Haar assumption for PQC, displacement-only convention for Fock) | as D8 | none |
| Q4 | Fock truncation: empirical guard band with rejection vs a rigorous norm-deficit bound (FP32 cannot resolve deficits below ~1e-6) vs renormalization; defaults D = 32, guard 2, 1e-10 | as D10 | none |
| Q5 | Shots: fold a training-step counter into keys (fresh noise per resident step; CPU/GPU semantics diverge) vs seed-only keys; shots on the GPU at all | seed-only keys, CPU and GPU | none |
| Q6 | QUBO: annealer export formats (dimod-compatible arrays), solvers beyond exhaustive, Ishikawa positive reduction, two's-complement encoding | none of these; offset binary | none |
| Q7 | Access to an RTX 3090 (sm_86) for runtime measurements | compile-only sm_86 evidence | none |
| Q8 | Python names (`kan.PqcConfig`, `kan.Rotation`, `kan.FockConfig`, `kan.qubo.*`) | as listed | 7 (naming only) |
| Q9 | Make the tape/adapter boundary public API | private | none |
| Q10 | QUBO location `src/qubo/` + `include/kan/qubo.hpp` | as stated | none |
| Q11 | Fock on the resident executor within M5, or recorded as future work | included (phase 4) | 4 (descoping only) |

## Plan review

An independent read-only reviewer (planning subagent) checked this plan against AGENTS,
CONTRACT, the carrier protocol, the M3/M4 plans, the M7 template and the brief, and verified
repository facts (head, brief commit, toolchain, golden hashes, `sass_cmp.py`, status bits,
initializer and binding dispatch). 24 findings (9 HIGH, 12 MEDIUM, 3 LOW), all resolved in
the text above:

| # | Finding | Resolution |
|---|---|---|
| 1 H | Squeezing derivatives need rows D and D+1, not one extra row | Truncation paragraph; phase 3 test 2 checks `∂S/∂ζ` at rows D−2, D−1 |
| 2 H | Fock defaults (D = 16) reject inputs beyond about \|x\| 1.1 | Default cutoff 32; supported input domain stated (\|α\| <= 2), `TanhMap` documented, no clipping; RED test records rejections on a trajectory near the guard |
| 3 H | Adding variant alternatives breaks `resident.cu` overloads, initializer dispatch, Python views before their phases | D13: resident rejection stubs, initializer dispatch, Python stubs in the same phase; both CUDA builds and Python suites green every phase |
| 4 H | Shot seed in structural config, key formula, sample index, lone-layer position and CDF fallback unspecified | `shot_seed` moved to uploadable carrier state; exact SplitMix64 key chain; sample = batch position (offset claim dropped); lone layer position 0; fallback = last outcome with p > 0 |
| 5 H | cuStateVec comparison at the planned sizes exceeds GB10 memory | Feasibility spike first, chunked harness, capped workloads, "not covered" outcome allowed |
| 6 H | Phase 2 too large; a library "win" did not change any path | Split into 2a (custom resident) and 2b (comparison); adoption is an owner decision with both paths parity-tested |
| 7 H | Status bit unspecified, unknown bits decode as rational; "names the layer" precedent false | Bit 256, decoded in eager and deferred paths after bit 1 and the range bit; exact CPU and resident messages; layer naming dropped |
| 8 H | Quantum VarianceScaling variance/moments/G undefined, one-sided test, deviation unstated | D8: uniform [-1,1] measure, PQC moments per readout and entangler, Fock closed forms, `G = terms`, draw order with the pending-pair rule, deviation recorded; two-sided 0.15 band test |
| 9 H | Exhaustive QUBO checks infeasible as sized; exactness, loss, bias, residual, two-layer API undefined | `n + m <= 24`, smaller instances; dyadic data, power-of-two batch; loss defined; bias fixed; residual rejected; two-layer case test-built |
| 10 M | "Exhaustive min within a quantization step of LS" is false | `L(c*) <= min <= L(round(c*))` with a test-written exact LS solution |
| 11 M | Pass order, pair counting, M recomputation, margin bounds ambiguous | Quadratization algorithm paragraph |
| 12 M | Redundancy principle contradicts inert ZYZ angles; missing `None` test | Inert angles documented and tested; only adjacent equal axes/diagonal gates rejected; `None` allowed for n > 1 and tested |
| 13 M | Symmetric-point claims partly false (Z encoding, PS rounding, c = 0) | Restricted to Y encoding, c = 1; adjoint exact, PS within 4 ulp; Z encoding documented |
| 14 M | Adjoint order and upstream notation unspecified; FP32 inversion drift | Pseudocode with exact order; phase 2a test 5 with checkpointing fallback |
| 15 M | Fock per-gate state storage does not scale at the limits | `r·(B + 1) <= 16` until phase 4 measures; device layout decided by profiling |
| 16 M | Guard 1 blind to squeezing (parity) | Guard >= 2; RED test |
| 17 M | TU split hides a refactor of `resident.cu` helpers | D1: raw-pointer launchers, verbatim helper extraction commit under the SASS gate |
| 18 M | Dense references could share code with the library | No `src/`/`kan/` includes in reference headers, enforced by the source check |
| 19 M | Source check mis-scoped and incomplete | Exact file list and regular expression; literal-only error messages |
| 20 M | API declarations and Fock layouts unspecified; tape storage | Declarations and layouts written; tape compiled per call on the CPU, cached in the resident plan |
| 21 M | Owner questions lack blocking information; Fock-on-GPU is an owner choice | Table with the default and what waits for the owner; Q11 added (default: included). The suggestion that Q1/Q5 block phases 2b/6 was turned into recorded defaults instead: the owner asked work to proceed autonomously, and an evidence-only cuStateVec install is permitted by AGENTS rule 8; only library adoption waits |
| 22 L | `long double` reference loses precision on MSVC | Double-double arithmetic in the test reference |
| 23 L | G and probability-sum bound undefined; 30-variable exhaustive search time | G defined (tape length); bound `(4G + 2^n)·2^-52`; Gray-code search, test instances <= 20 variables |
| 24 L | ROADMAP claimed shots verified against finite differences | Reworded: exact gradients vs FD and each other; shots statistically |

One reviewer suggestion was not adopted as stated: making CPU-only Fock the default if Q11 is
unanswered. The roadmap scope ("carrier interfaces and simulator adapters") and the AGENTS
CUDA rules favour a resident Fock carrier, so phase 4 stays in scope by default and Q11 only
allows the owner to descope it.
