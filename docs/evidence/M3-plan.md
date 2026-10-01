# M3 localized/adaptive bases: execution and acceptance plan

2026-10-01, `cpp-foundation`, starting head `00a93c7`, clean tree. The full
M2 integrated replay passed 10/10 CTest targets on the RTX 3090. M3 alone is
in scope; no BACKLOG.md exists. This resolves the terse M3 roadmap entry.

## Numerical defaults and APIs

- Add `BasisKind::BSpline` with explicit `degree` (0..16), `knots` of length
  `size+degree+1`, finite nondecreasing knots, clamped endpoints repeated exactly
  degree+1 times, interior multiplicity at most degree+1, and positive domain
  `[knots[degree],knots[size]]`. Size is at least degree+1. Outside this domain
  values/derivatives are zero. At interior knots use the right-hand convention;
  at the upper endpoint use the inward left-hand value/derivative. Degree-zero
  derivatives are zero. Cox-de Boor zero denominators contribute zero.
- Add `BasisKind::MexicanHat` with `centers` (translations) and `scales`, each
  exactly size, finite translations and finite positive scales. Each term is
  `2/(sqrt(3)*pi^(1/4)*sqrt(scale)) * (1-q*q)*exp(-q*q/2)`,
  `q=(x-center)/scale`, with analytic input derivatives. This is a smooth,
  L2-normalized continuous wavelet family; no discrete transform is implied.
- Gaussian keeps its M1 fixed scalar `width` by default. Opt-in `trainable_rbf`
  uses exactly size finite `log_widths`, with finite positive `exp(log_width)`.
  Centers and log widths are shared across all edges of a layer, explicitly
  initialized in its BasisConfig. BasisValues adds `center_derivatives` and
  `log_width_derivatives`, empty for fixed families. LayerGradients adds `centers`
  and `log_widths`, empty for fixed families. VJPs sum over batch and edges.
  Width positivity is preserved by log parameterization; SGD rejects a candidate
  whose exponent is zero/nonfinite before committing any network parameter.
- `Layer::set_rbf_parameters(centers,log_widths)` is atomic and opt-in only.
  `Layer::insert_knot(x)` inserts a strictly interior knot of allowed multiplicity
  using exact coefficient transformation, adding one basis term. `adapt_grid`
  accepts finite sample values, selects the nonempty knot span with most samples
  (ties: lowest span), and inserts its median sample when strictly interior,
  otherwise its midpoint. Empty/outside-only or unrepresentable spans fail.
  Adaptation is explicit, preserves values (within floating tolerance), and
  invalidates previously shaped gradients. Network exposes indexed refinement.
- Regularization is explicit coefficient L2: objective `lambda/2 * sum(c*c)`
  and VJP `lambda*c`, finite nonnegative lambda. Biases/centers/widths are
  unpenalized. CPU Layer/Network return penalty and parameter gradients; resident
  backward accepts optional lambda and adds it on GPU, with unchanged defaults.
- Resident CUDA supports the two new families and trainable RBF VJP/atomic SGD
  using persistent storage. Grid changes are setup operations: download model,
  explicitly refine on CPU, rebuild executor at new topology/capacity. No hidden
  host fallback or device reallocation inside numerical execution is introduced.
  Python exposes the same operations and owned gradient/configuration snapshots.

## Exit gates

1. Retained executable RED then GREEN steps for new basis values, derivatives,
   endpoint/multiplicity/domain validation, nonlinear VJPs, atomic updates,
   exact/data-driven refinement and regularization.
2. Independent closed-form and finite-difference tests, deterministic localized
   training with independent holdout, mixed-family CPU/resident parity, zero batch,
   unchanged device allocation counts, invalid inputs/overflow and update atomicity.
3. Full regression suite, CPU-only Python build, installed C++/Python use,
   AddressSanitizer and actual GPU Compute Sanitizer.
4. Frozen M3 full-call benchmark with matched CPU/GPU outputs and profiler evidence.
   Identify and address a measured kernel bottleneck; accept tuning only with
   matched complete-call timings. Record hardware/tool limitations without claiming
   unmeasured saturation. Existing M2 benchmark workloads remain regressions.
5. Independent implementation and evidence review; resolve blocking findings.
   Commit/push each finished test/implementation/evidence step. Mark M3 DONE and
   hand off M4 only after all gates pass.

## Mathematical references

Original implementations; no external implementation is ported. Recurrence and
partition conventions: [SciPy BSpline mathematical notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html).
Refinement: [Boehm knot insertion references](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.insert_knot.html).
Wavelet normalization: [SciPy Ricker mathematical definition](https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.signal.ricker.html)
(documentation as a mathematical reference, no dependency on its removed API).
