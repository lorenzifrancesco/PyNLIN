# Problems Found While Pinning Down the Analysis

## 0. Reading this list

This file records problems found by comparing the active Lorenzi Fast code,
its plotting scripts, and the surrounding documentation. It is not a list of
all possible enhancements. Each item identifies a broken or incomplete
logical contract.

Severity has the following meaning:

- **Critical:** can invalidate a principal physical result or claimed proof.
- **High:** can invalidate validation, reproducibility, or an important subset
  of results.
- **Medium:** produces ambiguity, a restricted numerical failure, or a false
  supporting statement.
- **Low:** documentation or presentation defect that should still be fixed.

Status is **open** unless explicitly stated otherwise.

### 0.1 Meaning of a linear model in this list

Here **linear** means linear in the normalized in-channel frequency offsets:

$$
u_{\rm lin}(\mathbf x)
=u_{\rm const}+\boldsymbol\kappa\mathbin{\cdot}\mathbf x.
$$

It retains carrier mismatch and group-slowness walk-off but omits local
quadratic and higher-order in-channel dispersion terms. It does not mean that
fiber propagation is linear, that Kerr NLIN is absent, or that the link power
kernel is linear. The observable remains
$F_\tau=\mathbb E[k(u)M_\tau]$, which is nonlinear in the phase mismatch.

## Critical problems

### P1. Production XPM omits the spacing-dependent leading correction

**Evidence**

- `fast_nlin.py::xpm_fast_batch` evaluates only the linear phase (Logical
  Specification, Eq. 8.4).
- `fast_nlin.py::target_fast_sums` discards the $q$ values returned by
  `xpm_pair_variables` and sums `xpm_fast_batch(nu_pairs)`.
- `publication_novelty.md` records that the local-quadratic high-DGD result is
  $C_{N_1}(q)/|\nu|$, with $C_{N_1}(1)=2\ln2$, while the Fast model tends to
  $1/|\nu|$. It reports a 32 percent deficit at $q=1$, $\nu=300$.
- S6 converts this XPM sum directly into physical NLIN and FWM/XPM shares.

**Consequence**

Absolute XPM, total NSR, and the inferred strict-FWM fraction are provisional.
Adjacent pairs are important, so the error cannot be assumed to disappear in
the aggregate.

**Required resolution**

Define production XPM as $F_{\rm XPM}(\nu,q)$ or evaluate the complete
local-quadratic pair integral. At minimum, implement and validate the
$C_{N_1}(q)/|\nu|$ asymptotic branch with a controlled transition to low
walk-off. Regenerate S4--S6 after the change.

### P2. The S3 certificate bounds the linear model, not the advertised quadratic model

**Status: partially resolved in code; documentation and validation remain
open.**

**Evidence**

- A quadratic-model bound requires
  $g=|u_0|-W-P_q$.
- `fast_analytic.py::envelope_bound` implements $g=|u_0|-W$ and accepts no
  explicitly named quadratic padding.
- The current `select_tube` computes $P_q$ and passes $W+P_q$ as the effective
  width. This applies the quadratic confinement padding during selection and
  discarded-bound accumulation, despite older documents saying it is absent.
- Existing certificate tests compare against linear-model tuple values. They
  do not directly test the bound against local-quadratic QMC values.
- The retained-tuple evaluator remains a linear-phase estimator, so the
  resulting retained sum plus discarded certificate is not by itself a
  certified local-quadratic physical result.

**Consequence**

The discarded-set confinement argument now includes the quadratic padding,
but its implementation contract is indirect and lacks a quadratic regression
test. The complete S3 result still mixes a padded selection certificate with
a retained-tuple estimator that omits quadratic phase.

**Required resolution**

Make the padding explicit in the bound API, add local-quadratic QMC regression
tests for discarded tuples, and separately control or label the retained-tuple
model error. Reconcile `lorenzi_fast_method.md`, the logical specification,
and S3 captions with the code. Do not claim a full physical truncation
guarantee until both discarded and retained errors are controlled at the same
phase-model level.

### P3. The claimed two-coordinate factorization uses an invalid marginal mask correction

**Evidence**

- `fast_s0_factorization.py` defines an equal-split synthetic direction and
  applies $A(d)/A(0)$ to each real tuple.
- The equal-split direction is parallel to the mask coordinate, so mismatch
  and mask are maximally correlated.
- `lorenzi_fast_method.md` itself reports that replacing conditional
  acceptance by marginal $A(d)$ is wrong by $3/2$ at $d=0$ and about 250 at
  $|d|=6.41$ for this direction.
- The factorization script logs bin-level agreement only. It does not compute
  the advertised raw per-tuple statistic "median 1.000, 99.7 percent within
  2x".
- Real and predicted masses are separately normalized within each target,
  removing an absolute-normalization test.

**Consequence**

The current evidence does not establish
$F=F_{\rm syn}(x_\nabla,\mu)A(d)/A(0)$ or independence from walk-off direction.
The predictive "population times universal kernel" claim is not pinned down.

**Required resolution**

Retain direction and support as arguments,
$F_{\rm syn}(x_\nabla,\mu,\widehat{\boldsymbol\nu},d)$, or average over a
defined direction distribution using conditional acceptance. Save and plot
raw per-tuple residuals, stratified by direction and $d$. Treat any reduced
two-coordinate law as an empirical surrogate with a stated error distribution.

### P4. The S2 aggregate error calculation is not a valid aggregate estimate

**Evidence**

- S2 selects all high-ranked tuples plus a logarithmically spaced tail. This
  is a nonuniform deterministic sample with no inclusion-probability weights.
- It sets `weight = quad / sum(quad)` and then compares sums of
  `fast * weight`, `lin * weight`, and `quad * weight`. This weights an
  efficiency by another efficiency and is not
  $\sum(F_{\rm model}-F_{\rm ref})/\sum F_{\rm ref}$.
- Stage (c) only reports the sampled top-256 Fast mass fraction. It does not
  execute the current deterministic refinement path or calculate its error.
- The document nevertheless quotes 0.52 percent and 0.07 percent as aggregate
  validation gates.

**Consequence**

The quoted mass-weighted bulk and quadratic error budgets are unsupported by
the current script. They should not be used to justify production accuracy.

**Required resolution**

Use a complete target reference or a probability sample with known inclusion
probabilities and a stratified/Horvitz--Thompson aggregate estimator. Compute
the signed aggregate difference directly and propagate QMC uncertainty. Run
the actual current refinement selection and evaluator. Retire existing gate
numbers until regenerated.

## High-severity problems

### P5. `uniform_sum_density` fails its documented zero-width contract

**Evidence**

The docstring says zero-width legs are dropped. The implementation keeps the
three-dimensional inclusion-exclusion formula and merely floors the
normalization factors. Supported synthetic directions can contain two exactly
zero coefficients, producing NaNs.

**Required resolution**

Implement separate one-, two-, and three-active-uniform formulas per tuple,
including the all-zero limiting distribution where needed. Add exact tests and
near-zero continuity tests.

### P6. The far approximation factorizes correlated expectations

**Evidence**

The exact quantity is

$$
2\,\mathbb E\left[\frac{1-\cos u}{u^2}\right].
$$

`far_model` replaces it by

$$
2\,\widehat{\mathbb E[u^{-2}]}
\left(1-\mathbb E[\cos u]\right).
$$

Both factors are functions of the same random $u$. Mixed terms are omitted,
so the stated $O((W/u_0)^4)$ accuracy does not follow from the displayed
derivation.

**Required resolution**

Describe the current formula as a heuristic factorized asymptote. Derive the
joint expansion including mixed moments, or evaluate the positive density
integral in the transition region. Keep the rigorous envelope separate from
the point estimate.

### P7. `$|u_0|<W$' is only an unmasked-box crossing test

**Evidence**

$W$ describes the image of the complete cube $(-\pi,\pi)^3$. The support mask
restricts that cube to a polytope. Zero can lie in the unmasked mismatch range
while being absent from the admissible masked domain. `fast_s0_territory.py`
states the caveat correctly, while `lorenzi_fast_method.md` and
`publication_novelty.md` sometimes call it the exact phase-matching
classification.

**Required resolution**

Use two names:

- **box crossing:** $|u_0|\le W$, a necessary condition;
- **masked-domain crossing:**
  $\min_{\mathbf x:M_d(\mathbf x)=1}|u(\mathbf x)|=0$.

For the linear model, derive the latter from the support polytope rather than
overstating the box test.

### P8. Sector-asymptotic documents state incompatible laws

**Evidence**

- `direct_sector_mc.md` presents
  $\rho_3\propto\nu^{-0.31}$ and $\rho_4\propto\nu^{0.29}$ as scaling laws and
  says $L/L_W$ is the single coordinate.
- `npc_sector_asymptotics.md`, `publication_novelty.md`, and the later Lorenzi
  Fast text say those are pre-asymptotic fits: every sector scales as
  $1/\nu$, with ratios depending separately on spacing-to-baud ratio $q$.

**Required resolution**

Mark the old section and summary as retracted/pre-asymptotic. Use the
$(\nu,q)$ constant-ratio result as canonical, subject to its stated
single-span and flat-power assumptions.

### P9. The pulse-orthogonality explanation is internally inconsistent

**Evidence**

`theory.md` and `noise_calculations.md` say shifted Nyquist sectors vanish at
large walk-off, so $N_1$, $N_2$, and 2PC merge. Other passages say they do not
merge under high dispersion, and the newer sector asymptotics give nonzero
3PC/4PC constants. Ordinary unitary dispersion preserves inner products, so
"dispersion breaks orthogonality" is not by itself an adequate explanation.

**Required resolution**

State the limiting procedure explicitly: quantities fixed as $\nu\to\infty$,
the role of $q$, span edges, incomplete collisions, masks, and loss. Restrict
the overlap argument to the ideal complete-collision approximation and use the
frequency-domain theorem for the canonical finite-band asymptote.

### P10. Physical power and polarization conventions are not explicit

**Evidence**

- `physical_nlin_spectrum` applies factors 4 and 2 under a scalar SSFM
  convention but names its argument only `launch_power_w`.
- S6 and the Fast document do not state unambiguously whether power and output
  variance are single-polarization or total dual-polarization quantities.
- Other repository documents retain incompatible workflow factors: obsolete
  TD $16/9$ and PCFM $64/27$ versus the audited active single-polarization
  $128/27$ pair expression.

**Required resolution**

Adopt one convention ledger and encode it in API names and tests. At minimum,
state the polarization convention, observation plane, ordered-tuple convention,
and whether $\gamma$ is scalar or Manakov-normalized. Mark older contradictory
formulas as superseded.

### P11. `gamma_grid` depends on the subset passed to it

**Evidence**

`fullband_mc.py::gamma_grid` treats `fiber.gamma` as a reference value at the
mean requested frequency. S6 filters to currently computed targets before
calling it. Therefore $\gamma(f)$ at the same physical frequency can change
with partial checkpoint coverage or target decimation.

**Required resolution**

Compute $\gamma(f)=2\pi f n_2/[cA_{\rm eff}(f)]$ from documented material
parameters, or normalize the configured gamma at one fixed, stored reference
frequency. Never derive the reference from the requested subset.

### P12. S5 checkpoints do not establish physical provenance

**Evidence**

Checkpoint loading checks only the array length and `n_refine`. It does not
check frequencies, $B$, $L$, dispersion arrays, configuration, refinement
cap, code/model version, or XPM model version. A same-sized incompatible run
can be silently resumed and then converted with the current S6 configuration.

**Required resolution**

Store and verify a deterministic hash of all model inputs and a schema/model
version. S6 must verify that the S5 provenance agrees with its current system
before physical conversion.

### P13. Validation plots do not consistently propagate uncertainty

**Evidence**

- Figure 4 plots $Y=F_{\rm model}/F_{\rm QMC}$ but uses
  $\delta F_{\rm QMC}/F_{\rm QMC}$ as an absolute y-error, omitting the factor
  $Y$.
- S4 compares Fast and MC values and attributes deviations to MC noise but
  stores no component standard errors and plots no uncertainty.
- The fullband wrapper discards an FWM stderr calculated at a lower level.

**Required resolution**

Return and save component uncertainties, use independent randomized
replicates, propagate them through every plotted transformation, and report
pulls. Do not attribute a difference to sampling noise without an uncertainty
on that difference.

### P14. The historical Figure 6 is not reproducible by current S5

**Evidence**

The embedded `s5_fullband_dec4.png` describes a 571-channel interferer grid
under obsolete decimation semantics. Current S5 always uses all 2284
interferers and thins targets only. The image and NPZ are useful evidence of a
past artifact, but they are not current S5 output.

**Required resolution**

Label the asset and its producer revision as historical, preserve exact
provenance, and stop describing all theory figures as regenerable by the
current plotting script. Generate a separate current full-grid S5 figure when
the run is complete.

## Medium-severity problems

### P15. S3 status and complexity claims conflict

The theory note calls S3 unassigned, later reports S3 v1 as implemented, and
describes planned $O(N\log N)$ geometric enumeration. The active S3 v1 first
calls `fwm_tuple_variables`, which exhaustively constructs support tuples, and
then prunes them.

**Resolution:** state that v1 is post-enumeration linear-model pruning;
geometric survivor enumeration remains planned.

### P16. The symbol `$N$' denotes both channel count and collision sum

This makes expressions such as $NT^2/L^2$ ambiguous in a document also using
$N$ for 2284 channels.

**Resolution:** use $N_{\rm ch}$ for channel count, $\mathcal N$ for the
dimensional collision sum, and $F=\mathcal NT^2/L^2$ for normalized
efficiency.

### P17. The quadratic-smallness diagnostic omits the `$\pi^2$' range

S0 calls $q_{\rm eff}<0.5$ "quadratics negligible", where
$q_{\rm eff}=\sum|q_j|$. The actual worst-case phase displacement is
$P_q=\pi^2q_{\rm eff}$, so this threshold permits nearly five radians.

**Resolution:** report $P_q$ and define negligibility by a validated error
criterion rather than a bare coefficient threshold.

### P18. Zero-gradient tuples are assigned `$\mu=0$'

`FWMTupleVariables.mu` and S0 divide with an output initialized to zero. If
$x_\nabla=0$, $\mu$ is undefined; if $u_0\ne0$, its limiting magnitude is
infinite. Assigning zero places such tuples in the phase-matched family.

**Resolution:** represent undefined cases explicitly or use a separate
zero-gradient category.

### P19. FWM support fraction is not always `$2/3$'

Some MC documentation says the support fraction is approximately $2/3$ and
identical for FWM and XPM. The common law is $A(d)$; only $d=0$ gives $2/3$.

**Resolution:** use $A(d)$ for FWM and reserve $2/3$ for center-aligned tuples
and the linear XPM geometry.

### P20. Validation model order is not consistently named

The general MC note describes local Taylor dispersion through fourth order,
while Fast FWM and the active S4 fullband reference use lower-order phase
models in important paths.

**Resolution:** label every result `linear`, `local-quadratic`, or
`local-beta4`; add sensitivity checks before calling a lower-order S4 result a
general Dar ground truth.

### P21. Synthetic directional slices are not averages over directions

In `fast_s0_synthetic.py`, heat maps average efficiencies over directions,
while some slices evaluate the efficiency at the arithmetic mean direction.
The mean need not be a unit vector and can approach zero.

**Resolution:** evaluate the slice for every direction and average the
efficiencies, matching the heat-map definition.

### P22. The fixed boundary in Figure 8 is exact only for equal split

The displayed line $|\mu|=\pi\sqrt3$ corresponds to $|u_0|=W$ for the
equal-split synthetic direction. For a real tuple,

$$
\frac{W}{x_\nabla}
=\pi\frac{\|\boldsymbol\nu\|_1}{\|\boldsymbol\nu\|_2}
$$

varies between $\pi$ and $\pi\sqrt3$.

**Resolution:** draw per-tuple/direction-dependent boundaries or label the
line as the equal-split reference only.

### P23. Figure 8 does not test absolute factorization

Real and predicted arrays are normalized independently within every target,
and panels use independent color limits. This can test coarse spatial shape
but not absolute normalization or target-to-target strength.

**Resolution:** add raw-sum ratios, shared color scales, and residual maps.

### P24. S0 source comments disagree with rendering

The module docstring says histograms are Gaussian-smoothed, while the current
plot code and title say raw bins, no smoothing.

**Resolution:** update the docstring and record bin edges and clipping in the
saved dataset.

### P25. Regeneration and status claims are hand-maintained

`plot_fast_theory_figures.py` produces only Figures 1--5 and 9. Figures 7 and
8 have separate producers; Figure 6 is historical. Completion counts also
differ between documentation passages.

**Resolution:** add a figure manifest containing producer, inputs, model
version, and output. Generate run status from checkpoint metadata rather than
editing prose.

## Low-severity problems

### P26. Zonotope uniformity is overstated

A full-rank square linear image of a uniform cube is uniform on a
parallelepiped. A lower-dimensional projection onto a zonotope is generally
not uniform; its density is proportional to fiber volume. The detailed joint
density derivation uses the correct statement, but the glossary does not.

### P27. The documented 9000-node cap is not the effective cap

The requested near quadrature count is capped at 9000 and then rounded to a
power-of-two bucket based at 64, which can produce 16,384 nodes.

### P28. Figure and section cross-references are stale

Examples include the Figure 1 caption pointing to the wrong regime section,
S3 being called reserved after implementation, and internal plot-function
figure numbers differing from document numbers.

## Cross-document contradictions to retire

| Topic | Conflicting statements | Canonical action |
|---|---|---|
| TD workflow factor | `noise_calculations.md` applies $16/9$; `theory.md` says active workflow does not | Re-audit active path and mark old formula superseded |
| PCFM pair prefactor | Theory/noise notes give $64/27$; calculation audit derives active single-pol $128/27$ | State input/output polarization and one audited equation |
| XPM sector asymptote | Old fitted $\nu^{\mp0.3}$ ratios vs constant ratios depending on $(\nu,q)$ | Retract old scaling section |
| Nyquist sectors | Shifted sectors vanish vs nonzero high-DGD constants | State limiting model and use the newer finite-band theorem |
| S3 | Reserved/planned vs v1 implemented | Distinguish pruning v1 from planned direct enumeration |
| S5 progress | Different completed-target counts | Generate from checkpoint metadata |

## Recommended repair order

1. Fix production XPM and regenerate S4--S6.
2. Correct or relabel the S3 certificate.
3. Replace the S2 aggregate validation protocol and add S4 uncertainty.
4. Add complete S5 checkpoint provenance and a fixed-reference gamma model.
5. Rebuild the factorization experiment with conditional acceptance and raw
   residuals.
6. Resolve polarization/power conventions and retire contradictory formulas.
7. Repair zero-width density handling and the far-model derivation.
8. Add a machine-readable figure manifest and regenerate all current figures.

## Structure-only simplification track

This track is intentionally separate from the scientific repairs above. A
structure-only change must preserve equations, tuple populations, dispatch
thresholds, floating-point operation order inside numerical kernels, array
axes, units, and serialized meaning. Characterization tests should compare
stage outputs before and after each move.

1. Extract the repeated system-to-Fast input construction used by S0--S5 into
   one analysis-layer data object with unit-qualified fields. Preserve the
   distinction between full interferer grids, target-only thinning, and work
   grids.
2. Split `fast_nlin.py` internally by concern: link kernel and quadrature,
   mask geometry, tuple variables, estimators, validation estimators, and
   physical conversion. Keep `fast_nlin.py` as the stable import facade while
   callers migrate.
3. Add canonical derived properties to `FWMTupleVariables`, especially signed
   `linear_coeffs`, `box_halfwidth`, `gradient_scale`, and
   `quad_phase_bound`, instead of reconstructing them in each stage.
4. Consolidate duplicated regime classification, quadrature bucketing, and
   exact-acceptance evaluation without merging the distinct Fast and S3
   scientific contracts.
5. Separate the S5 driver into computation, checkpoint persistence, MC probe
   selection, and plotting. Preserve full-grid array placement, NaNs for
   uncomputed targets, interleaved scheduling, and atomic checkpoint writes.
6. Centralize stage persistence metadata and isolate historical cache loaders.
   Do not silently reinterpret legacy `mu`, target-index, power, or
   reference-plane fields.
7. Move QMC ground-truth helpers out of the production numerical module while
   preserving Sobol dimensions, scrambling, seeds, replicate statistics, and
   compatibility imports.

Do not combine a structural move with the open XPM correction, far-model
derivation, zero-width density repair, tuple deduplication, acceptance-model
changes, or serialized-field renaming. Each of those can change the physics or
the interpreted result and requires its own tests and regenerated evidence.
