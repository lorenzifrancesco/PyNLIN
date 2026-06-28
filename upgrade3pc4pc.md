
# Implementation note: collision-resolved computation of (X_{h,k,m}) in PyNLIN

## Goal

PyNLIN currently computes the dominant two-pulse-collision contribution (X_{0,m,m}) using a direct time-domain overlap integral. The next target is to generalize this into a fast collision-resolved evaluator for the full XPM coefficients

[
X_{h,k,m}
=========

\int_0^L dz, f(z)
\int dt,
g_z^*(t)
g_z(t-hT)
g_z^*(t-kT-\beta_2\Omega z)
g_z(t-mT-\beta_2\Omega z),
]

where (g_z(t)) is the linearly dispersed pulse, (T) is the symbol period, (\Omega) is the WDM channel spacing, and (f(z)) contains the loss/gain profile.

The purpose is not only to compute total NLIN variance, but to preserve individual collision identities. This enables separate numerical analysis of 2PC, 3PC, and 4PC contributions, especially for shaped or temporally correlated symbol sequences.

## Why the current direct method is limited

The direct method computes one coefficient at a time:

[
X_{h,k,m}
\approx
\sum_j \Delta z_j f(z_j)
\sum_n \Delta t,
g_j^*(t_n)
g_j(t_n-hT)
g_j^*(t_n-kT-\beta_2\Omega z_j)
g_j(t_n-mT-\beta_2\Omega z_j).
]

This is simple and useful for validation, but if many triples ((h,k,m)) are needed, the cost scales roughly as

[
N_h N_k N_m N_z N_t.
]

This becomes expensive because many different ((k,m)) choices share the same relative delay structure.

## Reindexing

Introduce

[
r = k-m,
]

so that

[
k=m+r,
\qquad
X_{h,k,m}=X_{h,m+r,m}.
]

Then

[
X_{h,m+r,m}
===========

\int_0^L dz, f(z)
\int dt,
g_z^*(t)
g_z(t-hT)
g_z^*(t-(m+r)T-\beta_2\Omega z)
g_z(t-mT-\beta_2\Omega z).
]

For fixed (z), (h), and (r), define

[
C_h^{(z)}(t)
============

g_z^*(t)g_z(t-hT),
]

[
D_r^{(z)}(t)
============

g_z^*(t-rT)g_z(t).
]

Then

[
X_{h,m+r,m}
===========second-order estimate below the \Delta\beta_{\mathrm{disp}} definition in both documents:
If only second-order dispersion is retained, and $\omega_1,\omega_2$ are offsets from the channel centers, then $\eta_j(\omega,0)\simeq \beta_{2j}\omega$, so that $\Delta\be

\int_0^L dz, f(z)
\left[
C_h^{(z)} \star D_r^{(z)}
\right](mT+\beta_2\Omega z).
]

Therefore, for fixed ((z,h,r)), one cross-correlation gives all (m)-values at once.

## Main computational advantage

The direct method evaluates every ((h,k,m)) independently.

The correlation method evaluates one correlation for each ((z,h,r)), then samples it at all delays

[
\tau_m = mT+\beta_2\Omega z.
]

The approximate cost becomes

[
N_z N_h N_r N_t\log N_t
+
N_z N_h N_r N_m,
]

instead of

[
N_h N_k N_m N_z N_t.
]

The gain is strongest when (N_m) is large but (r=k-m) only needs a modest memory window.

## Special case: 2PC

For 2PC,

[
h=0,
\qquad
r=0,
\qquad
k=m.
]

Then

[
C_0^{(z)}(t)=D_0^{(z)}(t)=|g_z(t)|^2,
]

and

[
X_{0,m,m}
=========

\int_0^L dz, f(z)
\left[
|g_z|^2 \star |g_z|^2
\right](mT+\beta_2\Omega z).
]

This should be implemented first because it is the simplest FFT-correlation version and can be directly compared against the current trivial 2PC method.

## Proposed implementation plan

### 1. Add a kernel API

Create a function with an interface of the form

```python
compute_xpm_kernel(
    pulse,
    fiber,
    grid,
    h_values,
    r_values,
    m_values,
    method="fft-correlation",
)
```

The returned object should store

```python
X[h_index, r_index, m_index]
```

representing

[
X_{h,m+r,m}.
]

It should also store metadata:

```python
h_values
r_values
m_values
T
beta2
Omega
z_grid
t_grid
pulse_shape
normalization
```

### 2. Keep the direct method for validation

Retain or add

```python
compute_xpm_coefficient_direct(h, k, m, ...)
```

This should evaluate the double integral directly for a single triple ((h,k,m)).

Use it only for debugging and regression tests.

Validation test:

```python
for random triples (h, r, m):
    k = m + r
    X_direct = compute_xpm_coefficient_direct(h, k, m, ...)
    X_fft = X_table[h, r, m]
    assert relative_error(X_direct, X_fft) < tolerance
```

The tolerance will depend on (t)-grid resolution, (z)-grid resolution, interpolation, and FFT padding.

### 3. Implement FFT-correlation method

For each longitudinal point (z_j):

1. Compute the dispersed pulse

[
g_j(t)=\mathcal F^{-1}
\left[
G_0(\omega)
\exp\left(i\frac{\beta_2 z_j}{2}\omega^2\right)
\right].
]

2. For each (h), build

[
C_h^{(j)}(t)=g_j^*(t)g_j(t-hT).
]

3. For each (r), build

[
D_r^{(j)}(t)=g_j^*(t-rT)g_j(t).
]

4. Compute the correlation

[
Q_{h,r}^{(j)}(\tau)
===================

C_h^{(j)} \star D_r^{(j)}.
]

5. Sample

[
Q_{h,r}^{(j)}(\tau_m),
\qquad
\tau_m=mT+\beta_2\Omega z_j.
]

6. Accumulate

[
X_{h,m+r,m}
\mathrel{+}=
w_j f(z_j)
Q_{h,r}^{(j)}(mT+\beta_2\Omega z_j).
]

Here (w_j) is the longitudinal quadrature weight.

### 4. Use interpolation for non-grid delays

The sampling point

[
mT+\beta_2\Omega z_j
]

will generally not lie exactly on the discrete delay grid. Therefore, the correlation array should be interpolated.

Start with linear interpolation. Later, if needed, add cubic interpolation.

The interpolation convention must be tested carefully against the direct method.

### 5. Use zero padding

The FFT correlation should be computed with enough zero padding to avoid circular wrap-around.

If the time grid has (N_t) points, use at least

[
N_{\rm fft} \ge 2N_t
]

or the next fast FFT length above (2N_t).

### 6. Define collision masks

After computing (X[h,r,m]), define masks for collision classes.

A minimal convention is:

[
\text{2PC}: h=0,\ r=0.
]

[
\text{3PC}: \text{exactly one nontrivial coincidence among the relevant symbol indices}.
]

[
\text{4PC}: \text{fully off-diagonal terms}.
]

In implementation, the exact classifier should be encoded as a function:

```python
classify_collision(h, r, m) -> {"2pc", "3pc", "4pc"}
```

This avoids hard-coding masks in several places.

### 7. Compute collision-resolved perturbations

Given symbol arrays (a_i) and (b_i), compute

[
\Delta a_i^{\rm XPM}
====================

i2\gamma
\sum_{h,r,m}
a_{i+h}
b_{i+m+r}^*
b_{i+m}
X_{h,m+r,m}.
]

Then separate

[
\Delta a_i^{(2{\rm PC})},
\qquad
\Delta a_i^{(3{\rm PC})},
\qquad
\Delta a_i^{(4{\rm PC})}
]

by applying the collision masks.

Compute observables:

[
P_2 = \mathbb E|\Delta a_i^{(2{\rm PC})}|^2,
]

[
P_3 = \mathbb E|\Delta a_i^{(3{\rm PC})}|^2,
]

[
P_4 = \mathbb E|\Delta a_i^{(4{\rm PC})}|^2.
]

Also compute cross terms:

[
P_{23}
======

2\operatorname{Re}
\mathbb E[
\Delta a_i^{(2{\rm PC})}
\Delta a_i^{(3{\rm PC})*}
],
]

and similarly for (P_{24}) and (P_{34}). Do not assume that the collision classes add incoherently before checking.

## First milestone

The first milestone should be the FFT-correlation reproduction of the existing 2PC calculation:

[
X_{0,m,m}^{\rm direct}
\quad
\text{versus}
\quad
X_{0,m,m}^{\rm fft}.
]

Expected result:

[
\max_m
\frac{
|X_{0,m,m}^{\rm direct}-X_{0,m,m}^{\rm fft}|
}{
\max_m |X_{0,m,m}^{\rm direct}|
}
\ll 1.
]

Once this works, the same infrastructure can be generalized to all ((h,r)).

## Second milestone

Compute the full table

[
X[h,r,m]
]

for small windows, for example

[
h\in[-5,5],
\qquad
r\in[-5,5],
\qquad
m\in[m_{\min},m_{\max}].
]

Compare random entries with direct quadrature.

Then increase the windows until the neglected kernel norm is small:

[
\sum_{\text{outside window}} |X_{h,m+r,m}|^2
\ll
\sum_{\text{inside window}} |X_{h,m+r,m}|^2.
]

## Third milestone

Use the table to compare two ensembles with the same marginal amplitude distribution:

1. i.i.d. shaped symbols;
2. CCDM/PAS-like finite-block correlated symbols.

Measure

[
P_2,\quad P_3,\quad P_4,
]

and the cross terms. This directly tests whether temporal shaping correlations mainly suppress the 2PC phase-noise part or also modify the 3PC/4PC residuals.

## Recommended module structure

Possible files:

```text
pynlin/
    kernels/
        xpm_time_domain.py
        collision_masks.py
    statistics/
        collision_observables.py
    tests/
        test_xpm_2pc_fft_vs_direct.py
        test_xpm_random_entries_fft_vs_direct.py
```

Suggested core functions:

```python
linear_pulse_at_z(pulse, z, grid, beta2)
build_C(g_z, h, T, grid)
build_D(g_z, r, T, grid)
fft_correlate(C, D, grid)
sample_correlation(Q, tau_values, grid)
compute_xpm_kernel_fft(...)
compute_xpm_coefficient_direct(...)
classify_collision(h, r, m)
compute_collision_observables(...)
```

## Numerical cautions

1. Check the sign convention for correlation versus convolution.
2. Check whether the desired delay is (mT+\beta_2\Omega z) or its negative under the actual array convention.
3. Use direct quadrature as the reference.
4. Use zero padding to avoid circular correlation artifacts.
5. Track pulse normalization carefully.
6. Track whether (\beta_2) is expressed in SI, ps(^2)/km, or normalized units.
7. Track whether (\Omega) is angular frequency or ordinary frequency. The formula above assumes angular frequency if (\Omega) appears in (\beta_2\Omega z).
8. Store metadata in the output kernel file to avoid mixing incompatible conventions.
9. Start with real Nyquist pulses and no pre-dispersion, then add pre-dispersion later.
10. Benchmark against the old 2PC routine before trusting the generalized result.

## Summary

The proposed implementation does not change the mathematical definition of (X_{h,k,m}). It reorganizes the computation by using

[
r=k-m
]

and evaluating

[
X_{h,m+r,m}
===========

\int_0^L dz, f(z)
\left[
C_h^{(z)}\star D_r^{(z)}
\right](mT+\beta_2\Omega z).
]

This turns many separate overlap integrals into reusable FFT correlations. The result is a collision-resolved kernel table (X[h,r,m]), suitable for 2PC/3PC/4PC decomposition and for studying temporally correlated shaped symbol sequences.