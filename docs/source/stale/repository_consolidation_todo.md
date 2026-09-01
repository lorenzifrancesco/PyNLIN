# Repository Consolidation TODO

## Purpose

Simplify the repository by giving repeated operations one canonical owner while
preserving numerical behavior, scientific conventions, and supported legacy
interfaces. This is a staged consolidation, not a module-layout rewrite.

The active code and [logical analysis](../logical_analysis/index.md) define what
the repository currently calculates. The
[calculation correctness audit](calculation_correctness_audit.md) and
[open logical problems](PROBLEMS.md) constrain this work.
Known-invalid implementations must be corrected or quarantined before they are
made canonical.

## Classification rule

Before combining two implementations, classify their relationship as one of:

- an exact duplicate;
- the same operation implemented by different numerical methods;
- the same name applied to different scientific definitions;
- a supported legacy compatibility adapter;
- a known-invalid historical implementation.

Only the first two categories are direct consolidation candidates. Different
scientific definitions must receive distinct names and contracts. Legacy
adapters require a canonical replacement, a warning, and a removal plan.

## Safety gates

- [ ] Commit minimal test-only system configurations and fiber/profile fixtures.
- [ ] Remove default-test dependence on ignored local `input/` state.
- [ ] Preserve every strict correctness-audit xfail until its defect is fixed by
  an independently checked implementation.
- [ ] Add equivalence tests before replacing duplicate implementations.
- [ ] Add invariant tests for units, axes, power basis, reference plane, index
  spaces, tuple ordering, and power scaling.
- [ ] Establish the supported Python version and enforce it in CI.
- [ ] Record a test and coverage baseline before structural moves.

## Consolidation sequence

### 1. Shared link operations

- [ ] Consolidate `kernel_abs2`, plotting-only link-kernel implementations,
  `_propagator`, and `_propagator_abs2`.
- [ ] Provide separately named dimensionless amplitude, dimensionless power,
  dimensional span-amplitude, and dimensional span-power operations.
- [ ] Test the zero-phase limit, evenness, non-negativity, dimensions, and
  attenuated and lossless forms.

Suggested owner: `pynlin.methods.common.link`.

### 2. FWM interaction enumeration

- [ ] Unify strict ordered FWM tuple construction currently split between
  `fast_nlin.py` and `fullband_mc.py`.
- [ ] Make strictness, ordered `(a, b)` enumeration, carrier residual, support
  shift, and full-grid channel indices explicit.
- [ ] Keep tuple selection and sampling policies separate from tuple definition.
- [ ] Characterize tuple counts and ordering before changing either caller.

Suggested owner: `pynlin.methods.common.interactions`.

### 3. Power-profile contracts

- [ ] Introduce one versioned profile schema with canonical axes
  `(z, channel, mode)`.
- [ ] Record units, channel frequencies, power basis, reference plane, and axis
  metadata.
- [ ] Reject ambiguous array layouts rather than inferring them from shape.
- [ ] Never sum a polarization axis implicitly.
- [ ] Isolate historical pickled payloads and key aliases in legacy readers.
- [ ] Move file loading out of core PCFM and TD numerical functions.

Suggested owner: a shared profile schema in `pynlin`, with persistence policy in
`analysis.runtime`.

### 4. Fiber nonlinearity model

- [ ] Give target gamma and pairwise gamma separate, unit-qualified APIs.
- [ ] Define whether gamma comes from a fixed-reference configured scalar or
  from material nonlinear index and effective area.
- [ ] Remove normalization relative to the mean of a caller-selected frequency
  subset.
- [ ] Make Manakov or polarization factors part of an explicit convention.

Suggested owner: the fiber domain model.

### 5. Physical NLIN aggregation

- [ ] Preserve pair-resolved XPM weights proportional to
  `P_target * P_interferer**2`.
- [ ] Preserve tuple-resolved strict-FWM weights proportional to
  `P_a * P_b * P_c`.
- [ ] State the single-polarization power basis and launch or output reference
  plane in arguments and results.
- [ ] Keep a separately named flat-power adapter where useful.
- [ ] Remove physical conversion formulas from analysis runners and workflows.

Suggested owner: the method-level aggregation layer that still has pair and
tuple information.

### 6. Persistence and caches

- [ ] Introduce a shared schema validator and atomic writer.
- [ ] Require schema version, units, axes, phase model, power basis, reference
  plane, tuple ordering, frequency-grid hash, symbol rate, fiber length, gamma
  model, and code or model version.
- [ ] Test cache rejection when any scientific input changes without a shape
  change.
- [ ] Add explicit readers or migrations for each supported historical schema;
  do not infer old `mu`, `target_indices`, or power semantics from field names.

Suggested owner: an analysis persistence layer shared by S0--S6 and method
results.

### 7. Analysis orchestration

- [ ] Standardize execution on `studies -> context -> runners -> pynlin`.
- [ ] Migrate full-system studies away from `analysis/methods/workflow.py`.
- [ ] Build Fast system inputs once, retaining separate full-grid indices, work
  grid positions, target stride, and interferer stride.
- [ ] Move plotting and repository-path policy out of the installable package.
- [ ] Remove the library-to-analysis dependency from
  `pynlin.methods.td.noise`.

### 8. Compatibility and cleanup

- [ ] Keep `pynlin.nlin` as a leaf compatibility facade; canonical code must not
  import it.
- [ ] Add import and numerical-equivalence tests for supported compatibility
  paths.
- [ ] Add warnings naming the canonical replacement and removal plan.
- [ ] Merge exact low-risk duplicates such as `ideal_fits.py` and
  `ideal_fits_uwb.py` only after characterization.
- [ ] Consolidate repeated analysis-only plotting and signal-generation helpers
  after the scientific core is stable.

## Correct before sharing

The following known inconsistencies must not become shared abstractions in
their current form:

- [ ] Fullband physical conversion with dimensionally incomplete power scaling.
- [ ] The duplicate `SMFiber.loss_profile` definition that overrides
  profile-aware behavior.
- [ ] Unequal-power TD modulation reconstruction.
- [ ] Collision classification that conflates projection sectors with literal
  index coincidence.
- [ ] Raman gain-matrix frequency scaling and photon-rate balance.
- [ ] Fast zero-gradient detuning represented as zero instead of undefined.
- [ ] Fast production XPM and model-level certificate problems listed in
  `logical_analysis/PROBLEMS.md`.

## Completion criteria

- [ ] Every shared scientific operation has one documented canonical owner.
- [ ] Alternative numerical estimators consume the same definitions and are
  cross-checked on common fixtures.
- [ ] Public arrays and persisted arrays state axes and units.
- [ ] Public powers state polarization basis and reference plane.
- [ ] Compatibility imports are isolated, tested, and warning-emitting.
- [ ] Core numerical code has no dependency on analysis code or repository file
  locations.
- [ ] All resolved correctness-audit xfails have become ordinary regression
  tests.
