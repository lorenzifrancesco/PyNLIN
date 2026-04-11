# PCFM/TD Scientific Spec Sheet

## Scope

This note summarizes the scientific content behind the current TD and PCFM
calculations used in the `PyNLIN` UWB analysis workflow, with emphasis on:

- the time-domain (TD) collision-coefficient model already present in `pynlin`,
- the PCFM XCI closed form used in the SMF/UWB analysis scripts,
- the idealized flat-profile and constant-dispersion asymptotics used to
  interpret baud-rate, spacing, and length sweeps,
- the current implementation-level coherence limits between TD and PCFM.

It is a scientific specification of what is being computed, not a software API
reference.

## Physical Setting

The current analysis targets single-span, Raman-assisted or flat-profile,
single-mode ultra-wideband systems with:

- channel grid `f_i`,
- baud rate `B`,
- fiber length `L`,
- effective area `A_eff(f)`,
- dispersion `beta2(f)` or `beta2_eff(f_i, f_j)`,
- launch powers `P_i`.

The main idealized comparison regime used in the recent checks is:

- flat signal power profiles: `p_i(z) = 1`,
- constant `beta2`,
- constant `A_eff`,
- uniform channel spacing,
- XCI-only comparison,
- equal launch powers.

## Core Quantities

### Kerr coefficient

For the CUT channel `i`,

```{math}
\gamma_i = \frac{2 \pi f_i}{c} \frac{n_2}{A_{\mathrm{eff},i}},
\qquad
n_2 = 2.6 \times 10^{-20}\ \mathrm{m^2/W}.
```

For PCFM XCI between channels `i` and `j`,

```{math}
\gamma_{ij} = \frac{2 \pi f_i}{c} \frac{2 n_2}{A_{\mathrm{eff},i}+A_{\mathrm{eff},j}}.
```

### Walk-off variable

The natural dimensionless walk-off variable is

```{math}
x_{ij} = L B\,\mathrm{DGD}_{ij}.
```

Under constant `beta2`,

```{math}
\mathrm{DGD}_{ij} \simeq 2 \pi |\beta_2|\,|f_j-f_i|,
```

so

```{math}
x_{ij} = 2 \pi |\beta_2| L B |f_j-f_i|.
```

For uniform spacing `Delta f = r B`,

```{math}
x_{ij} = 2 \pi |\beta_2| L B^2 r.
```

## TD Model

### Collision coefficients

The TD solver computes channel-pair collision coefficients

```{math}
\mathcal{N}_{m_A,\nu_A,m_B,\nu_B},
```

then converts them back to SI units using

```{math}
y_{\mathrm{norm}} = \frac{1}{(L B)^2},
\qquad
\mathcal{N}^{(\mathrm{SI})} = \frac{\mathcal{N}}{y_{\mathrm{norm}}}.
```

### TD aggregation

The final TD NLIN power currently used in the workflow has the structure

```{math}
P^{\mathrm{TD}}_{m_A,\nu_A}
=
\frac{16}{9}
\frac{P_{\nu_A}^3 \gamma_{\nu_A}^2}{B^2}
\sum_{m_B,\nu_B}
\mathcal{N}^{(\mathrm{SI})}_{m_A,\nu_A,m_B,\nu_B}
\kappa_{m_A,m_B}^2
\Theta(m_A,m_B).
```

For SMF, the mode structure collapses and the modulation prefactor becomes

```{math}
\Theta(\mu_0) = 5 \mu_0 - 4,
```

where

```{math}
\mu_0 = \frac{\langle |b|^4 \rangle}{\langle |b|^2 \rangle^2}.
```

Examples:

- Gaussian: `mu0 = 2`,
- 64-QAM: `mu0 ≈ 1.38095`,
- 16-QAM: `mu0 = 1.32`.

### Modulation-affine decomposition

The `analysis/pcfm/td.py` layer does not recompute TD collisions. It rewrites
the TD aggregation as

```{math}
P^{\mathrm{TD}}(\mu_0)
=
\frac{16}{9}\,\mathcal{P}\,(\mu_0\,S_a + S_b),
```

where `S_a` and `S_b` are modulation-independent sums built from the already
computed collision tensor. This allows the same TD tensor to generate Gaussian,
16-QAM, 64-QAM, and 256-QAM curves without recomputing collisions.

Scientifically, this is not a new solver; it is a reparameterization of the
final TD aggregation.

## TD Asymptotics

The stage-S1 reference curves define the normalized TD behavior

```{math}
\mathcal{N}^{(\mathrm{SI})}_{ij} = (L B)^2 F_{\mathrm{S1}}(x_{ij}).
```

The large-walk-off asymptotic is

```{math}
F_{\mathrm{S1}}(x) \sim \frac{c_{\mathrm{pulse}}}{x},
\qquad x \gg 1.
```

For the currently fitted Nyquist perfect-amplification curve, the constant is
numerically close to

```{math}
c_{\mathrm{Nyquist}} \approx 1.
```

Therefore, in the idealized regime,

```{math}
P^{\mathrm{TD}}_{ij}(\mu_0)
\sim
\frac{16}{9}
\frac{P^3 \gamma^2}{B^2}
(5\mu_0-4)
(L B)^2
\frac{c_{\mathrm{pulse}}}{x_{ij}}.
```

Under constant `beta2`,

```{math}
P^{\mathrm{TD}}_{ij}(\mu_0)
\sim
\frac{16}{9}
\frac{P^3 \gamma^2}{B^2}
(5\mu_0-4)
\frac{c_{\mathrm{pulse}} L B}{2 \pi |\beta_2| |f_j-f_i|}.
```

Hence, in the asymptotic XCI regime:

- fixed spacing: `P_TD ~ 1/B`,
- fixed baud rate: `P_TD ~ 1/|Delta f|`,
- fixed powers and dispersion: `P_TD ~ L`.

## PCFM XCI Model

### Exact closed-form kernel used now

For the current flat-profile XCI closed form,

```{math}
K_{\mathrm{XCI},ij}
=
\frac{L}{2\pi |\beta_{2,\mathrm{eff}}|}
\log\!\left(
\frac{|f_j-f_i| + B/2}{|f_j-f_i| - B/2}
\right)
S_j.
```

For a flat profile,

```{math}
S_j = 1.
```

### Current implementation normalization

The present `compute_pcfm_nlin` implementation uses the following effective
normalization:

- input PSD-like quantity: `g_i = 2 P_i / B`,
- XCI PSD prefactor: `(32/27) g_i g_j^2 gamma_ij^2 K_xci`,
- extra internal `/2` in the XCI PSD term,
- final output conversion `P = G B / 2`.

After these steps, the pairwise XCI power under equal launch powers is

```{math}
P^{\mathrm{PCFM,XCI}}_{ij}
=
\frac{64}{27}
\frac{P^3 \gamma_{ij}^2}{B^2}
K_{\mathrm{XCI},ij}.
```

This is the normalization actually used by the current PCFM path.

## PCFM XCI Asymptotics

### Low `B / |Delta f|` expansion

The exact log kernel admits the expansion

```{math}
\log\!\left(
\frac{|Delta f| + B/2}{|Delta f| - B/2}
\right)
=
\frac{B}{|Delta f|}
+ \frac{B^3}{12 |Delta f|^3}
+ \frac{B^5}{80 |Delta f|^5}
+ \cdots
```

The current analytic helper for the scaling scripts supports:

- exact log kernel,
- first-order asymptotic,
- third-order asymptotic,
- fifth-order asymptotic.

The scaling scripts currently use the first-order form:

```{math}
\log\!\left(
\frac{|Delta f| + B/2}{|Delta f| - B/2}
\right)
\approx
\frac{B}{|Delta f|}.
```

Thus,

```{math}
K_{\mathrm{XCI},ij}
\sim
\frac{L B}{2\pi |\beta_{2,\mathrm{eff}}|\,|Delta f|}.
```

and

```{math}
P^{\mathrm{PCFM,XCI}}_{ij}
\sim
\frac{64}{27}
\frac{P^3 \gamma_{ij}^2 L}{2\pi |\beta_{2,\mathrm{eff}}|\,B\,|Delta f|}.
```

Therefore the present PCFM-XCI asymptotics are:

- fixed spacing: `P_PCFM,XCI ~ 1/B`,
- fixed baud rate: `P_PCFM,XCI ~ 1/|Delta f|`,
- fixed powers and dispersion: `P_PCFM,XCI ~ L`.

## Matched TD vs PCFM Comparison in the Idealized Regime

Under:

- SMF,
- flat profile,
- constant `beta2`,
- constant `A_eff`,
- equal launch powers,
- XCI-only comparison,

the pairwise ratio is

```{math}
\frac{P^{\mathrm{PCFM,XCI}}_{ij}}{P^{\mathrm{TD}}_{ij}(\mu_0)}
=
\frac{4}{3(5\mu_0-4)}
\frac{
r_{ij}
\log\!\left(\frac{r_{ij}+1/2}{r_{ij}-1/2}\right)
}{
x_{ij} F_{\mathrm{S1}}(x_{ij})
},
\qquad
r_{ij} = \frac{|Delta f_{ij}|}{B}.
```

Using the large-walk-off TD asymptotic

```{math}
x F_{\mathrm{S1}}(x) \to c_{\mathrm{pulse}},
```

the ratio tends to

```{math}
\frac{P^{\mathrm{PCFM,XCI}}}{P^{\mathrm{TD}}}
\to
\frac{4}{3(5\mu_0-4)c_{\mathrm{pulse}}}
r \log\!\left(\frac{r+1/2}{r-1/2}\right).
```

Since

```{math}
r \log\!\left(\frac{r+1/2}{r-1/2}\right) \to 1
\qquad (r \to \infty),
```

the asymptotic limit is

```{math}
\frac{P^{\mathrm{PCFM,XCI}}}{P^{\mathrm{TD}}}
\to
\frac{4}{3(5\mu_0-4)c_{\mathrm{pulse}}}.
```

With the current Nyquist fit `c_pulse ≈ 1`, this gives approximately:

- Gaussian: `PCFM / TD ≈ 0.22`,
- 64-QAM: `PCFM / TD ≈ 0.46`.

So, under the present implementation and in the idealized XCI asymptotic
regime, the trend is:

```text
TD > PCFM-XCI
```

not the reverse.

## Current Scientific Caveats

### 1. TD and PCFM are not compared on perfectly symmetric normalizations

The present PCFM path and GN reference paths do not use exactly the same
normalization conventions. This is the main remaining coherence issue in the
scientific interpretation of absolute ratios.

### 2. `analysis/pcfm/td.py` is a postprocessing layer, not a second TD solver

Its scientific role is to expose the modulation-affine decomposition of the
TD output. That functionality is useful, but scientifically it belongs to the
same TD model already implemented in `pynlin`.

### 3. TD/PCFM gamma conventions must stay aligned

The active TD branch should use the same pairwise Kerr coefficient convention
as PCFM XCI,

```{math}
\gamma_{ij} = \frac{2 \pi f_i}{c} \frac{2 n_2}{A_{\mathrm{eff},i}+A_{\mathrm{eff},j}},
```

which reduces to the usual CUT coefficient on the diagonal. If TD and PCFM
diverge on this point, off-diagonal comparisons become biased as soon as
`A_eff(f)` is not constant.

## Recommended Scientific Interpretation

The cleanest way to interpret the present calculations is:

- TD provides the reference collision-based nonlinear estimate.
- PCFM provides a semi-analytical XCI/SCI model with exact or asymptotic
  closed-form kernels.
- In the flat-profile, constant-dispersion regime, TD and PCFM-XCI share the
  same leading scaling exponents:
  - `~ 1/B`,
  - `~ 1/|Delta f|`,
  - `~ L`.
- The difference is in the prefactor, not in the leading exponent.
- Under the current implementation conventions, that prefactor favors TD over
  PCFM-XCI in the asymptotic regime.

## Minimal Refactor Target

For maximum scientific coherence, the preferred future state is:

- keep a single TD aggregation implementation in `pynlin`,
- expose the modulation-affine TD decomposition there,
- keep a single explicit polarization/PSD normalization convention shared by
  PCFM, GN-numeric, and GN-direct,
- compare TD and PCFM only after matching:
  - XCI-only vs XCI-only,
  - full NLIN vs full NLIN,
  - identical launch-power normalization,
  - identical Kerr-coefficient convention.
