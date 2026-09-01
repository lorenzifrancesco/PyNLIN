# Calculation Correctness Audit

Date: 2026-07-14

Audited revision: `3397d18` plus the local working-tree changes present on the
audit date. The worktree was already modified before this audit, including
changes to the fullband MC implementation and tests. Findings therefore apply
to the inspected checkout, not necessarily to commit `3397d18` alone.

## Scope

This audit covers the installable code under `src/pynlin` and the active
calculation path under `analysis/config.py`, `analysis/runners`, and
`analysis/methods`. Generated `build/` copies, notebooks, standalone plotting
studies, and legacy MATLAB/Python implementations are not treated as
authoritative. They were used only for comparison where useful.

The audit checks algebra, dimensions, power and polarization conventions,
limiting cases, numerical invariants, and agreement with independently stated
equations. It does not claim experimental validation of the fitted TD model or
the absolute Raman gain calibration.

## Executive Verdict

The codebase has a sound base of local numerical checks. Unit conversions,
pulse normalization, constellation moments, the core XPM/FWM integrals, Xhkm
contractions, effective-dispersion algebra, the PCFM XCI prefactor under a
single-polarization convention, and undepleted Raman formulas are supported by
analytical or independent numerical checks.

The active workflow is not currently reliable as an absolute end-to-end model
in all modes. Two findings can directly invalidate reported powers:

1. The fullband MC conversion is dimensionally incomplete and scales as power
   squared instead of cubed.
2. The Raman gain matrix violates two-wave photon-rate balance and is consumed
   by the active Jiang power-profile workflow.

Several additional defects affect unequal launch powers, wavelength-dependent
loss, Jiang convergence, and collision diagnostics. PCFM itself is
factor-correct only when its input is understood as single-polarization power;
the public API does not state that convention clearly.

## Convention Ledger

| Quantity or path | Implemented convention | Audit result |
|---|---|---|
| Fiber attenuation | CSV: dB/km converted to dB/m; solver: dB/m converted to the power-exponential coefficient with `ln(10)/10` | Conversion correct; CSV dispatch broken for `SMFiber.loss_profile` |
| Dispersion | `beta2` in s2/m, angular-frequency derivatives | Conversion and Eq. 28 algebra verified |
| Pulse amplitude | Unit-energy pulse, amplitude units s^(-1/2) | Verified numerically |
| Workflow launch power | Intended single-polarization W, inferred from the SSFM bridge | Internally used this way, but public names and plot labels are ambiguous |
| TD pair power | `P_CUT * P_INT^2` | Correct in `total_nlin_uwb`; lost in modulation reconstruction |
| PCFM input/output | Single-polarization launch W and single-polarization NLI W | Prefactors verified under this convention |
| PCFM reference plane | Core API is launch-referenced; workflow multiplies by `P_signal(L)/P_signal(0)` | Consistent for one span when the saved profile contains the observation-point gain |
| Xhkm/FWM sums | Prefactor-free squared longitudinal integrals, units m2 | Core dimensions consistent |
| Fullband MC output | Intended NLI W | Current conversion is dimensionless |
| Raman power | W; propagation coordinate m; gain matrix 1/(W m) | ODE dimensions correct; frequency scaling is not photon-balanced |

## Findings

### C1. Fullband MC does not produce power

Severity: critical

Confidence: confirmed analytically and by a strict expected-failure test

The sampled propagator is

$$
\Phi(\Delta\beta)=\int_0^L e^{(i\Delta\beta-\alpha)z}\,dz,
$$

so `diagnostic.xpm`, `diagnostic.fwm`, and `diagnostic.total` contain sums of
$|\Phi|^2$ and have units m2. This is visible in the XPM and FWM accumulation at
`src/pynlin/methods/td/fullband_mc.py:581-668`.

The conversion in `analysis/runners/methods.py:150-166` is

$$
N_j \mathrel{\widehat{=}} \frac{16}{81}\gamma_j^2 P_j P_{\rm avg} S_j.
$$

Its dimensions are

$$
(W^{-2}m^{-2})(W^2)(m^2)=1,
$$

not W. It also scales by four when all launch powers are doubled, whereas Kerr
NLI power must scale by eight. The same expression exists in the older active
workflow path.

For XPM, the required channel-wise power weight is proportional to
$P_jP_b^2$. For strict nondegenerate FWM, it is proportional to $P_aP_bP_c$.
The diagnostic sums kernels before applying powers, so one arithmetic mean
cannot recover the correct result for unequal launch powers. Strict FWM tuples
also exclude the target channel, but the current conversion multiplies their
sum by target power.

Affected outputs: every `FullbandMCResult.nlin_output_w`, derived NSR, and GSNR.

Recommended remediation: retain pair- or tuple-resolved kernel estimates and
apply the correct cubic products before summation. A flat-power-only fallback
could use $P^3$, but it must be explicitly restricted and its polarization and
modulation prefactor independently derived.

### C2. The SM Raman matrix violates photon-rate balance

Severity: critical

Confidence: confirmed from the two-wave equations and by a strict
expected-failure test

`RamanAmplifier.compute_gain_matrix` forms signed Stokes/anti-Stokes gains and
then multiplies every matrix element by $\nu_i/\nu_j$ at
`src/pynlin/raman/solvers.py:587-614`.

For a pump at $\nu_p$ and Stokes signal at $\nu_s<\nu_p$, with $g_R$ defined as
the signal gain coefficient, the lossless pair should satisfy

$$
\frac{dP_s}{dz}=g_RP_pP_s,
\qquad
\frac{dP_p}{dz}=-\frac{\nu_p}{\nu_s}g_RP_pP_s.
$$

This gives

$$
\frac{1}{\nu_s}\frac{dP_s}{dz}
+\frac{1}{\nu_p}\frac{dP_p}{dz}=0.
$$

The current code additionally scales signal gain by $\nu_s/\nu_p$, so this
identity fails. The multimode implementation already uses the physically
consistent `maximum(1, nu_i/nu_j)` form at
`src/pynlin/raman/solvers.py:1213-1222`.

This matrix is consumed by the active Jiang solver through
`src/pynlin/raman/solvers_jiang.py`, so Raman profiles and all profile-dependent
TD/PCFM results are affected.

Recommended remediation: define precisely whether the sampled Raman response
is the Stokes signal-gain coefficient, then apply the photon-frequency factor
only to the depletion side. Add a two-wave Manley-Rowe regression test.

### H1. TD modulation reconstruction is wrong for unequal launch powers

Severity: high

Confidence: confirmed by direct comparison with the primary TD aggregator

The primary implementation correctly computes

$$
P_{{\rm NLI},i}=\frac{P_i}{R_s^2}
\sum_j C_{ij}\gamma_{ij}^2P_j^2
$$

at `src/pynlin/methods/td/estimator.py:433-475`.

The modulation path instead factors out $P_i^3/R_s^2$ and sums collision terms
without $P_j^2/P_i^2$ at `src/pynlin/methods/td/__init__.py:81-110`. It agrees
only for equal powers. `analysis/runners/methods.py:126-141` uses this path for
the reported 16-QAM result.

The implementation-exact theory text at `docs/source/theory.md:72-160` also
describes a common $P_i^3$ factor and is not exact for unequal powers.

Recommended remediation: include interferer power squared in `sum_a` and
`sum_b`, leaving only $P_i/R_s^2$ outside, or reject nonuniform powers.

### H2. Loaded SMF attenuation is bypassed by Raman propagation

Severity: high

Confidence: confirmed by Python method resolution and by a strict
expected-failure test

`SMFiber.loss_profile` correctly dispatches to `_attenuation_profile` at
`src/pynlin/fiber.py:287-293`, but a second definition at
`src/pynlin/fiber.py:417-420` overrides it and always evaluates the fallback
polynomial. `RamanAmplifier.get_linear_losses` calls `loss_profile` at
`src/pynlin/raman/solvers.py:695-699`.

`attenuation_at()` still uses the CSV profile, so different model paths can use
different attenuation for the same fiber object.

Recommended remediation: remove the duplicate method and retain the
profile-aware implementation. Test interpolation through the solver-facing
method, not only through `attenuation_at()`.

### H3. Automatic Jiang convergence is not sufficient

Severity: high

Confidence: confirmed by an independent `solve_ivp` comparison

When there are no pumps, or pump power does not exceed signal power, `ts_db` is
zero and automatic iteration selection chooses exactly one fixed-point update
at `src/pynlin/raman/solvers_jiang.py:512-539`. Multiple signals can still
exchange power through inter-channel Raman scattering, so one update is not a
solution of the coupled nonlinear ODE.

The audit test compares a two-signal, no-pump case with an independently
integrated ODE and reproduces the mismatch.

Additional consistency risks are visible in the same method:

- Every pump profile is initialized and enforced at `z=L`, even though pump
  directions are accepted by the public solver.
- With `ts_db == 0`, the in-loop reference can become the newly computed pump
  endpoint instead of the requested endpoint.
- Early stopping checks only signal changes, not completion of the pump
  recovery schedule, pump boundary residual, or ODE residual.
- A final whole-profile pump rescaling at
  `src/pynlin/raman/solvers_jiang.py:628-635` does not recompute the signal
  profile, so the returned fields need not jointly satisfy the equations.

Current all-counterpropagating configurations avoid the co-pump boundary
interpretation, but not the convergence and final-rescaling concerns.

Recommended remediation: require boundary and ODE residuals in convergence,
iterate no-pump ISRS cases to convergence, handle each direction's boundary
explicitly, and recompute coupled fields after any final rescaling.

### H4. FWM time-window validation mixes seconds and metres

Severity: high for diagnostics; the coefficient integral itself is unaffected

Confidence: confirmed dimensionally and by a strict expected-failure test

`_check_fwm_discretization` defines the time window as `z[-1] - z[0]` at
`src/pynlin/methods/td/fwm_kernel.py:236-257`, then divides temporal walk-off
in seconds by that length in metres. The resulting warning/assertion can miss
severe pulse-window truncation. The result metadata later uses the actual pulse
time axis correctly at `src/pynlin/methods/td/fwm_kernel.py:379-403`.

Recommended remediation: pass the pulse time window to the checker and report
walk-off in seconds.

### H5. Public collision classification conflicts with contracted sectors

Severity: high for sector-resolved results; total `n1` and `n2` are unaffected

Confidence: confirmed with a one-entry tensor

`classify_collision` calls `h=k` and `h=m` entries 3PC at
`src/pynlin/methods/td/xpm_kernel.py:480-497`. `compute_xhkm_sums` explicitly
assigns those entries to 4PC unless `h=0` or `k=m` at
`src/pynlin/methods/td/xhkm_sums.py:76-93`.

Recommended remediation: choose one physically documented definition and make
both public APIs use the same masks. Regenerate any cached sector curves whose
meaning changes.

### H6. The direct Raman solver does not implement counter-pump boundaries

Severity: high when the direct solver is used with backward pumps

Confidence: confirmed by control-flow inspection

`RamanAmplifier.solve` accepts `use_power_at_fiber_start`, but the argument is
not read after the signature. It creates one IVP vector at `z=0` and changes
only the derivative sign for a negative pump direction. A specified backward
pump launch power is therefore not enforced at `z=L`. The dedicated BVP path
does implement endpoint boundaries.

The active profile workflow currently uses the Jiang solver, so this does not
compound the primary workflow finding. It affects direct API callers and the
wideband wrapper paths that request backward pumping.

Recommended remediation: route mixed-boundary problems to the BVP/shooting
solver or reject unsupported boundary semantics.

### M1. PCFM power factors are correct only under an implicit convention

Severity: medium in the active workflow, high for external API misuse

Confidence: verified against Poggiolini et al., Eqs. 47 and 52

The PCFM paper defines the rectangular channel PSD $G_i$. If
`launch_powers_w` is single-polarization power $P_i$, total dual-polarization
PSD is $G_i=2P_i/B$. The code uses that relation at
`src/pynlin/methods/pcfm/gn.py:667`, applies $16/27$ for SCI and $32/27$ for
XCI, integrates over bandwidth, and divides the result by two at
`src/pynlin/methods/pcfm/gn.py:686-749`.

For one XCI interferer, the resulting single-polarization power is

$$
P_{{\rm XCI},ij}^{\rm pol}
=\frac{128}{27}\frac{P_iP_j^2}{B^2}\gamma_{ij}^2K_{ij}.
$$

The independent audit test verifies this exact expression. The SSFM bridge
also explicitly calls workflow launch power `single-pol` at
`analysis/methods/ssfm_interface.py:154-163`.

However, the core docstring says only “per-channel launch powers”, and plots
label the value `Pch`. If a caller supplies conventional total
dual-polarization channel power, the returned value is four times the correct
total-DP NLI power, or eight times the corresponding per-polarization result.

The core function intentionally omits $p_{\rm CUT}(L)$ and returns a
launch-referenced quantity. The active workflow's single multiplication by
`P_signal,out/P_signal,launch` at `analysis/methods/io.py:142-168` correctly
supplies that endpoint factor for one span. A separate lumped gain is included
only if represented by the saved profile endpoint.

Recommended remediation: rename or document inputs as
`launch_powers_single_pol_w`, state output polarization and reference plane,
and add a total-DP adapter if needed.

### M2. PCFM switches imply unsupported behavior

Severity: medium

Confidence: confirmed by control-flow inspection

`use_numeric_sci=False` executes the same numerical SCI function as `True` at
`src/pynlin/methods/pcfm/gn.py:691-699`. `include_mci=True` emits a warning and
is ignored at `src/pynlin/methods/pcfm/gn.py:632-635`. The module header does
state “SCI+XCI, no MCI”, but configuration and reports expose both switches.

The closed-form XCI branch floors $|\beta_2|$ without warning about the
stretched-island model's low-dispersion validity. The Eq. 18 implementation in
`analysis/methods/analytics.py` has no independent absolute-value regression
oracle in the current suite.

Recommended remediation: remove unsupported switches or fail clearly, and add
paper-derived Eq. 18 values and validity warnings.

### M3. Raman ODE controls and optimizer contain execution defects

Severity: medium

Confidence: confirmed by inspection

`RamanAmplifier.solve` derives or accepts `max_step`, then unconditionally
overwrites it with `0.1` m at `src/pynlin/raman/solvers.py:247-257`. This made
the existing full-span Raman tests exceed the audit's two-minute test budget.
It also prevents callers from controlling numerical cost.

The PyTorch gain optimizer reads `flatness` before its first assignment at
`src/pynlin/raman/pytorch/gain_optimizer.py:103-119`, so its first epoch raises
`UnboundLocalError`.

The ASE ODE retains an interactive `breakpoint()` in exception handling at
`src/pynlin/raman/solvers.py:676-677`.

These are execution and testability defects rather than equation errors, but
they block validation of the affected calculations.

### M4. Documentation is not implementation-exact

Severity: medium

Confidence: confirmed by cross-reference

`docs/source/theory.md` describes the TD power factor as $P_i^3$ and therefore
misses unequal interferer powers. Its PCFM pairwise expression around lines
594-600 gives `64/27`; the active single-polarization implementation gives
`128/27` for one interferer after the final divide by two. Older
`noise_calculations.md` and `pcfm_td_scientific_spec.md` also retain a
workflow-level `16/9` TD factor that active code no longer applies.

Recommended remediation: define one convention table and generate all
derivations from it. Mark historical specifications as superseded.

## Verified Calculations

The following checks pass independently in
`tests/test_calculation_correctness_audit.py`:

| Calculation | Independent check | Result |
|---|---|---|
| Wavelength/frequency | Round trip with SciPy physical constants | Pass |
| dBm/W | Round trip over -30, 0, and 10 dBm | Pass |
| dB/m attenuation | `ln(10)/10` power conversion | Pass |
| `beta2`/dispersion | Algebraic round trip | Pass |
| Gaussian and Nyquist pulse energy | Numerical time integration | Pass |
| QPSK, 16-QAM, Gaussian moments | Closed forms `1`, `33/25`, and `2` | Pass |
| PCFM flat-profile XCI | Paper prefactor and closed-form kernel derived outside production helpers | Pass for single-pol input/output |
| Undepleted co-pump Raman | Independent adaptive ODE integration, including zero pump loss | Pass |
| Undepleted counter-pump Raman | Independent adaptive ODE integration and endpoint check | Pass |

Existing tests add useful evidence for:

| Calculation | Existing evidence | Strength |
|---|---|---|
| XPM kernels | Direct quadrature versus FFT correlation | Strong algorithmic cross-check |
| Generic FWM | Reduction to XPM, detuning and phase suppression | Strong local invariants |
| Xhkm contractions | Hand-constructed tensors and exact partitions | Strong, except classification conflict |
| Dar-style MC | Fixed-sample equivalence and sector identities | Strong local equivalence, no experimental oracle |
| Effective dispersion | Direct implementation of PCFM Eq. 28 | Strong algebraic check |
| TD fitted estimator | Scaling and cache tests | Moderate; no committed publication-quality oracle |
| Fullband MC | Support pruning and reproducibility | Structural only; physical conversion invalid |
| Raman coupled solver | Mostly execution-only tests | Weak |

## Validation Test Status

Focused command:

```text
.venv/bin/python -m pytest tests/test_calculation_correctness_audit.py tests/nlin tests/test_analysis_runners.py -rxX
```

Result on the audited checkout:

```text
63 passed, 7 xfailed
```

The seven strict expected failures correspond to:

1. TD unequal-power modulation reconstruction.
2. SMF attenuation-profile dispatch.
3. Fullband cubic launch-power scaling.
4. Collision-sector classification consistency.
5. FWM time-window validation.
6. Raman two-wave photon-rate balance.
7. Jiang no-pump ISRS convergence.

These use `xfail(strict=True)`: a production change that accidentally bypasses
the test still fails as an unexpected pass until the finding is reviewed and
the marker is removed.

The remaining non-Raman tests produced `9 passed, 2 failed`. Both failures are
stale configuration expectations in `tests/test_system.py`: the tests expect
600 S/C/L channels, while the active ignored `input/studies.toml` defines
1,200 O/E/S/C/L/U channels. They do not indicate a calculation failure, but
they demonstrate that the current test suite is not synchronized with the
active configuration and depends on ignored local files.

An initial run of the pre-existing focused scientific suite passed 49 tests
before entering the legacy Raman tests and exceeding 120 seconds. The timeout
is consistent with the hard-coded 0.1 m maximum step over 50 km spans. A
separate full-suite run is required after fixing or isolating those slow
tests.

The Sphinx build could not start because the environment lacks the configured
`myst_parser` extension. This is an environment/development-dependency blocker;
the new Python test module passes compilation and the repository patch passes
whitespace checks.

## Residual Uncertainty

The audit does not establish absolute experimental accuracy for the fitted TD
surrogate, the empirical Raman response magnitude, the Jiang homotopy method,
or Eq. 18. The repository has no committed publication-quality numerical
oracle for those paths. Local input papers and result files are ignored by Git,
and some existing tests depend on ignored configuration/reference data.

The highest-value next validation step is a small versioned benchmark set with
provenance: one flat-loss GN/PCFM case, one direct collision case, one
two-wave Raman case, one multi-signal ISRS case, and one converged SSFM case.
Each should record polarization convention, reference plane, units, numerical
resolution, source equation or software version, and justified tolerance.

## References

- P. Poggiolini, Y. Jiang, Y. Gao, and F. Forghieri, “Polynomial Closed Form
  Model for Ultra-Wideband Transmission Systems,” arXiv:2508.21563v1, 2025;
  later JLT DOI `10.1109/JLT.2026.3678322`.
- Y. Gao, Y. Jiang, and P. Poggiolini, “The Coherent Polynomials Closed-Form
  Model for Evaluating Nonlinear Interference in Any Island,”
  arXiv:2602.03860v2, 2026.
- R. Dar et al., “Accumulation of nonlinear interference noise in fiber-optic
  systems,” Optics Express 22, 14199-14211, 2014,
  DOI `10.1364/OE.22.014199`.
- K. Rottwitt et al., “Scaling of the Raman gain coefficient: applications to
  germanosilicate fibers,” JLT 21, 1652-1662, 2003,
  DOI `10.1109/JLT.2003.814386`.
