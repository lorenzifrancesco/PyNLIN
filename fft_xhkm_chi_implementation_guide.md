# Implementation Guide: FFT-Based \(\chi_1/\chi_2\) Reference Curves from \(X_{hkm}\)

## 1. Objective

Add an **opt-in extension** to the existing time-domain validation pipeline that:

1. computes the generic pulse-collision tensor \(X_{hkm}\) with the already implemented FFT backend;
2. contracts that tensor into the normalized collision sums associated with \(\chi_1\) and \(\chi_2\);
3. sweeps the differential group delay (DGD);
4. stores and plots the resulting curves versus
   \[
   \frac{L}{L_W}=L\,|\Delta\beta_1|\,R_s;
   \]
5. reuses the existing normalization
   \[
   \mathcal N\,\frac{T^2}{L^2};
   \]
6. preserves all existing \(X_{0mm}\), S1-reference-curve, cache, and plotting behavior.

This feature must be built **on top of** the current implementation. Existing public and internal workflows must continue to produce the same files, numerical values, and plots unless the new generic-\(X_{hkm}\) functionality is explicitly requested.

---

## 2. Existing behavior to preserve

The current S1 workflow already provides the correct scaling, DGD sweep, normalization, caching, and plotting conventions.

### 2.1 Numeric \(X_{0mm}\) sweep

`src/pynlin/methods/td/validation.py::compute_numeric_nlin(...)` currently:

- constructs a logarithmic DGD grid;
- converts DGD to
  \[
  \frac{L}{L_W}=\mathrm{DGD}\,L\,R_s;
  \]
- computes all \(X_{0mm}\) coefficients;
- forms
  \[
  \mathcal N_{0mm}=\sum_m |X_{0mm}|^2;
  \]
- optionally evaluates different amplification profiles;
- saves the resulting reference curves.

This behavior must remain unchanged.

### 2.2 Reference-curve normalization

`src/pynlin/methods/td/reference_curves.py::save_s1_ref_nlin_curve(...)` stores

\[
x_{\mathrm{norm}}=L R_s=\frac{L}{T}
\]

and applies

\[
\mathcal N_{\mathrm{normalized}}
=
\mathcal N_{\mathrm{raw}}\,x_{\mathrm{norm}}^{-2}
=
\mathcal N_{\mathrm{raw}}\frac{T^2}{L^2}.
\]

The new curves must use the same convention.

### 2.3 Existing backends

The existing `compute_numeric_nlin(...)` behavior and the current backends

```text
direct
x0mm_fft
```

must remain valid and unchanged.

The new generic collision calculation must not silently replace either backend.

### 2.4 Existing cache files

Existing S1 cache files and their schema must continue to load correctly.

Do not add mandatory keys to the existing S1 `.npz` format. If new metadata is useful, introduce a separate cache schema for the generic-\(X_{hkm}\) curves.

### 2.5 Existing plots

The established axes and scaling must remain:

```text
x-axis: L/L_W
y-axis: N T^2/L^2
```

Existing figures generated from \(X_{0mm}\) must not change.

---

## 3. Existing backend to reuse

The generic FFT collision backend already exists in:

```text
src/pynlin/methods/td/xpm_kernel.py
```

The relevant function is:

```python
compute_xpm_kernel_fft(...)
```

Its tensor convention is:

\[
X[i_h,i_r,i_m]
=
X_{h,m+r,m}.
\]

Equivalently,

\[
k=m+r.
\]

The returned index arrays must always be used when locating \(h=0\) or \(r=0\). Do not assume that the zero index is located at the center of the array.

The current standalone numerical example

```text
analysis/standalone_numerical/plot_n_pc_xhkm_spacing.py
```

already demonstrates:

- calling `compute_xpm_kernel_fft`;
- extracting selected \(X_{hkm}\);
- comparing FFT and legacy \(X_{0mm}\);
- classifying 2PC, 3PC, and 4PC terms;
- sweeping DGD through the channel spacing.

The new implementation should reuse this backend rather than introduce another propagation or correlation implementation.

---

## 4. New physical quantities

For the stored tensor

\[
X[h,r,m]=X_{h,m+r,m},
\]

define the raw collision sums

\[
\mathcal N_1
=
\sum_{h,r,m}|X[h,r,m]|^2,
\]

and

\[
\mathcal N_2
=
\sum_{h,m}|X[h,0,m]|^2.
\]

The \(r=0\) sector corresponds to \(k=m\).

Also retain the previously computed two-pulse contribution

\[
\mathcal N_{\mathrm{2PC}}
=
\sum_m |X[0,0,m]|^2
=
\sum_m |X_{0mm}|^2.
\]

Useful derived components are

\[
\mathcal N_{\mathrm{3PC},\,k=m}
=
\mathcal N_2-\mathcal N_{\mathrm{2PC}},
\]

and

\[
\mathcal N_{k\neq m}
=
\mathcal N_1-\mathcal N_2.
\]

The corresponding normalized curves are

\[
\widetilde{\mathcal N}_j
=
\mathcal N_j\frac{T^2}{L^2}.
\]

For diagnostic purposes also store

\[
R_{21}
=
\frac{\mathcal N_2}{\mathcal N_1}.
\]

The conversion to physical \(\chi_1,\chi_2\) prefactors should be kept separate from the collision-sum calculation. The first implementation should treat the normalized \(\mathcal N_1,\mathcal N_2\) curves as the primary output.

---

## 5. Proposed code structure

### 5.1 Add a contraction helper

Add a small module:

```text
src/pynlin/methods/td/xhkm_sums.py
```

Suggested API:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class XhkmSums:
    n1: float
    n2: float
    n_2pc: float
    n_3pc_k_eq_m: float
    n_k_neq_m: float

    @property
    def n2_over_n1(self) -> float:
        if self.n1 <= 0.0:
            return float("nan")
        return self.n2 / self.n1


def compute_xhkm_sums(
    X: np.ndarray,
    h_values: np.ndarray,
    r_values: np.ndarray,
) -> XhkmSums:
    """Contract X[h, r, m] = X_{h,m+r,m} into generic collision sums."""

    X = np.asarray(X)
    h_values = np.asarray(h_values)
    r_values = np.asarray(r_values)

    if X.ndim != 3:
        raise ValueError(f"Expected a rank-3 X tensor, got shape {X.shape}")

    if X.shape[0] != h_values.size:
        raise ValueError("X h-axis does not match h_values")

    if X.shape[1] != r_values.size:
        raise ValueError("X r-axis does not match r_values")

    h0 = np.flatnonzero(h_values == 0)
    r0 = np.flatnonzero(r_values == 0)

    if h0.size != 1:
        raise ValueError("h_values must contain exactly one zero")

    if r0.size != 1:
        raise ValueError("r_values must contain exactly one zero")

    ih0 = int(h0[0])
    ir0 = int(r0[0])

    abs2 = np.abs(X) ** 2

    n1 = float(np.sum(abs2))
    n2 = float(np.sum(abs2[:, ir0, :]))
    n_2pc = float(np.sum(abs2[ih0, ir0, :]))

    return XhkmSums(
        n1=n1,
        n2=n2,
        n_2pc=n_2pc,
        n_3pc_k_eq_m=n2 - n_2pc,
        n_k_neq_m=n1 - n2,
    )
```

This helper must:

- contain no plotting;
- contain no file I/O;
- contain no system-specific constants;
- operate only on the FFT result and its explicit index arrays;
- be independently testable.

### 5.2 Add a new reference-curve schema

Do not overload the existing S1 dataset.

Add a separate module or extend `reference_curves.py` with clearly separated functions:

```python
save_xhkm_sum_reference_curves(...)
load_xhkm_sum_reference_curves(...)
```

Suggested saved arrays:

```text
llw_grid
raw_n1
raw_n2
raw_n_2pc
raw_n_3pc_k_eq_m
raw_n_k_neq_m
ref_n1
ref_n2
ref_n_2pc
ref_n_3pc_k_eq_m
ref_n_k_neq_m
n2_over_n1
```

Suggested metadata:

```text
fiber_length
baud_rate
x_norm
pulse_shape
mode
gvda
gvdb
h_values
r_values
m_margin
n_samples_numeric
time_integral_backend = "xhkm_fft"
schema_version
```

Normalization:

```python
x_norm = fiber_length * baud_rate
normalization = x_norm ** (-2)

ref_n1 = raw_n1 * normalization
ref_n2 = raw_n2 * normalization
ref_n_2pc = raw_n_2pc * normalization
```

Use an independent required-key set and an independent loader validator.

### 5.3 Add a new numeric sweep function

Add a new function to:

```text
src/pynlin/methods/td/validation.py
```

Suggested name:

```python
compute_numeric_xhkm_sum_curves(...)
```

Do not modify the semantics of `compute_numeric_nlin(...)`.

Suggested signature:

```python
def compute_numeric_xhkm_sum_curves(
    gvda: float,
    gvdb: float,
    ipulse: int,
    *,
    h_values: np.ndarray,
    r_values: np.ndarray,
    recompute: bool = False,
    perfect_only: bool = True,
    partial_collisions_margin: int = 5,
) -> Path:
    ...
```

The first version should support only:

```text
time_integral_backend = "xhkm_fft"
```

A later extension may add other generic-\(X_{hkm}\) backends without affecting the old S1 workflow.

---

## 6. Sweep algorithm

The new function should reuse the existing setup from `compute_numeric_nlin(...)`.

### 6.1 Reuse unchanged

Reuse:

- configuration loading;
- pulse construction;
- DGD limits;
- logarithmic DGD sampling;
- conversion to `llw_numeric`;
- GVD arguments;
- amplification-profile handling where supported;
- cache-path conventions;
- logging style.

### 6.2 Per-DGD calculation

For each DGD:

1. determine the required \(m\)-index range, including the configured partial-collision margin;
2. call `compute_xpm_kernel_fft(...)`;
3. contract the result with `compute_xhkm_sums(...)`;
4. store the five raw sums;
5. continue to the next DGD.

Pseudocode:

```python
raw_n1 = np.empty(n_samples_numeric)
raw_n2 = np.empty(n_samples_numeric)
raw_n_2pc = np.empty(n_samples_numeric)
raw_n_3pc = np.empty(n_samples_numeric)
raw_n_k_neq_m = np.empty(n_samples_numeric)

for idx, dgd in enumerate(dgds_numeric):
    m_values = get_m_values(
        fiber=fiber,
        fiber_length=cf.fiber_length,
        dgd=dgd,
        T=1.0 / cf.baud_rate,
        partial_collisions_margin=partial_collisions_margin,
    )

    result = compute_xpm_kernel_fft(
        pulse,
        z,
        h_values,
        r_values,
        m_values,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )

    sums = compute_xhkm_sums(
        result.X,
        result.h_values,
        result.r_values,
    )

    raw_n1[idx] = sums.n1
    raw_n2[idx] = sums.n2
    raw_n_2pc[idx] = sums.n_2pc
    raw_n_3pc[idx] = sums.n_3pc_k_eq_m
    raw_n_k_neq_m[idx] = sums.n_k_neq_m
```

Use the actual argument names and return object of `compute_xpm_kernel_fft(...)`; the pseudocode above only specifies the control flow.

---

## 7. Amplification profiles

The first implementation should prioritize the perfect distributed-amplification case:

\[
f(z)=1.
\]

This is the cleanest environment for validating the generic collision sums and their asymptotic behavior.

If `compute_xpm_kernel_fft(...)` already accepts an amplification profile, use it directly.

If it currently returns the local kernel before the \(z\)-integration, apply the existing space-integration backend rather than creating a new integration routine.

Do not implement a second Raman-profile interpolation or space quadrature.

If generic \(X_{hkm}\) currently supports only \(f(z)=1\), expose that restriction explicitly:

```python
if not perfect_only:
    raise NotImplementedError(
        "Generic Xhkm sum curves currently support only the flat profile."
    )
```

Do not silently ignore a requested profile.

---

## 8. Index-range and convergence policy

Unlike the \(X_{0mm}\) sum, the generic result depends on finite \(h\)- and \(r\)-windows.

The first implementation must make these windows explicit inputs and write them into the saved metadata.

Recommended initial values for development:

```python
h_values = np.arange(-4, 5)
r_values = np.arange(-4, 5)
```

These are development defaults, not universal physical convergence guarantees.

Add a convergence utility later that compares:

\[
\mathcal N_j(H,R)
\]

with

\[
\mathcal N_j(H-\Delta H,R-\Delta R).
\]

For the first feature, the output filename and metadata must include enough information to prevent curves computed with different index windows from being confused.

Do not overwrite a cache produced with different `h_values` or `r_values`.

---

## 9. Plotting

Add a new standalone analysis entry point:

```text
analysis/standalone_numerical/plot_xhkm_sum_curves.py
```

It should load the new reference dataset and reproduce the established plotting style.

### 9.1 Primary plot

Plot against

\[
x=\frac{L}{L_W}
\]

the normalized curves

\[
\mathcal N_1\frac{T^2}{L^2},
\qquad
\mathcal N_2\frac{T^2}{L^2},
\qquad
\mathcal N_{\mathrm{2PC}}\frac{T^2}{L^2}.
\]

Axes:

```python
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L/L_W$")
ax.set_ylabel(r"$\mathcal{N}\,T^2/L^2$")
```

Title:

\[
\frac{L}{L_{D,a}}
=
L\,\beta_{2,a}R_s^2,
\qquad
\frac{L}{L_{D,b}}
=
L\,\beta_{2,b}R_s^2.
\]

### 9.2 Ratio plot

Plot

\[
\frac{\mathcal N_2}{\mathcal N_1}
\]

versus \(L/L_W\).

This should be a separate figure rather than a secondary axis on the main log-log plot.

### 9.3 Optional decomposition plot

Optionally plot:

\[
\mathcal N_{\mathrm{2PC}},
\qquad
\mathcal N_{\mathrm{3PC},\,k=m},
\qquad
\mathcal N_{k\neq m}.
\]

All terms must use the same \(T^2/L^2\) normalization.

### 9.4 Existing plots remain unchanged

Do not alter `simple_plot_threshold(...)`, existing S1 plotters, or their output names as part of this feature.

---

## 10. Cache naming

Add a separate cache-name helper, for example:

```python
xhkm_sum_ref_curve_path(...)
```

The cache name should encode or hash:

- pulse shape;
- profile mode;
- `gvda`;
- `gvdb`;
- `h_values`;
- `r_values`;
- partial-collision margin;
- backend name;
- schema version.

The new cache must not collide with:

```text
s1_ref_nlin_curve_path(...)
```

The old cache path remains reserved for \(X_{0mm}\)-based S1 curves.

---

## 11. Tests

### 11.1 Contraction-unit tests

Add:

```text
tests/nlin/test_xhkm_sums.py
```

Test with synthetic tensors where the expected sums can be computed by inspection.

Required cases:

1. tensor containing only \(h=0,r=0\):
   \[
   \mathcal N_1=\mathcal N_2=\mathcal N_{\mathrm{2PC}};
   \]

2. tensor containing \(r=0\), \(h\neq0\):
   \[
   \mathcal N_1=\mathcal N_2>\mathcal N_{\mathrm{2PC}};
   \]

3. tensor containing \(r\neq0\):
   \[
   \mathcal N_1>\mathcal N_2;
   \]

4. missing or duplicated zero in `h_values` or `r_values` raises an error;

5. mismatched tensor/index shapes raise an error.

### 11.2 Backward-compatibility tests

Existing tests for:

- `compute_numeric_nlin(...)`;
- S1 cache loading;
- `direct`;
- `x0mm_fft`;
- current analysis runners;

must continue to pass without modification to their expected values.

Add a regression test confirming that enabling the new module does not alter the old S1 file.

### 11.3 \(X_{0mm}\) consistency

For the same pulse, DGD, GVD, \(z\)-grid, and \(m\)-range, verify

\[
\sum_m|X[0,0,m]|^2
\]

from `compute_xpm_kernel_fft(...)` against the existing \(X_{0mm}\) FFT backend.

Use both:

- the individual coefficients;
- the summed \(\mathcal N_{\mathrm{2PC}}\).

### 11.4 Reference-curve schema tests

Test:

- save/load round trip;
- normalization consistency;
- metadata consistency;
- rejection of incompatible `h_values`/`r_values`;
- old S1 files still load through the old loader.

### 11.5 Small smoke sweep

Use a very small DGD grid and small index windows:

```python
h_values = np.arange(-1, 2)
r_values = np.arange(-1, 2)
```

Check:

```text
N1 >= N2 >= N_2pc >= 0
```

within numerical tolerance.

Do not put a publication-quality sweep in CI.

---

## 12. Documentation

Update:

```text
docs/source/theory.md
```

with:

- the tensor convention \(X[h,r,m]=X_{h,m+r,m}\);
- definitions of \(\mathcal N_1,\mathcal N_2,\mathcal N_{\mathrm{2PC}}\);
- the normalization \(T^2/L^2\);
- the distinction between the old S1 \(X_{0mm}\) curve and the new generic-\(X_{hkm}\) curves.

Update:

```text
docs/source/noise_calculations.md
```

with the new calculation flow.

Update:

```text
docs/source/analysis_api/scripts.md
```

with the new plotting script.

The documentation must explicitly state that:

- the original S1 workflow is unchanged;
- the new feature is opt-in;
- the new curves reuse the same DGD scaling and normalization;
- convergence with respect to \(h\) and \(r\) must be checked.

---

## 13. Recommended implementation sequence

### Phase 1: pure contraction

Implement and test:

```text
xhkm_sums.py
```

No sweep, plotting, or cache work yet.

### Phase 2: one-point integration

For one DGD and one GVD pair:

- call `compute_xpm_kernel_fft(...)`;
- compute all five sums;
- compare \(\mathcal N_{\mathrm{2PC}}\) with the existing \(X_{0mm}\) backend.

### Phase 3: DGD sweep

Add:

```python
compute_numeric_xhkm_sum_curves(...)
```

and reuse the current DGD setup.

### Phase 4: new cache schema

Store raw and normalized curves with complete metadata.

### Phase 5: plotting

Add the standalone plotter using the existing \(L/L_W\) and \(T^2/L^2\) conventions.

### Phase 6: convergence studies

Study the dependence on:

- \(h_{\max}\);
- \(r_{\max}\);
- partial-collision margin;
- time resolution;
- \(z\)-resolution.

Only after this step should the curves be treated as reference-quality data.

---

## 14. Acceptance criteria

The feature is complete when:

1. all existing tests pass unchanged;
2. existing S1 files and plots are unchanged;
3. the new contraction helper is independently tested;
4. the \(h=0,r=0\) sector reproduces the existing FFT \(X_{0mm}\) sum;
5. a small generic-\(X_{hkm}\) DGD sweep can be generated;
6. the output uses
   \[
   x=L/L_W
   \]
   and
   \[
   y=\mathcal N T^2/L^2;
   \]
7. the saved dataset records all truncation parameters;
8. the main and ratio plots are generated from the saved dataset;
9. unsupported amplification-profile behavior raises an explicit error;
10. no existing backend is replaced or reinterpreted.

---

## 15. Non-goals of the first implementation

Do not include, in the first version:

- a replacement for the Dar frequency-domain Monte Carlo code;
- automatic fitting of the new curves;
- a new propagation backend;
- a second FFT implementation;
- polarization-multiplexed contractions;
- automatic infinite-index extrapolation;
- integration into the general TOML studies runner;
- changes to the semantics of `pynlin.methods.mc.compute_chi1_chi2(...)`;
- removal or renaming of the existing S1 workflow.

Those can be addressed after the generic collision sums have been numerically validated.
