# Theory Notes

This page sketches the main mathematical models used across `pynlin` and
the accompanying analysis scripts. It is intentionally light on
implementation details; see the API docs for signatures and defaults.

## 1. Raman model

We model Raman gain with longitudinal gain/loss profiles and overlap
integrals:

$$\frac{\mathrm{d}P_s(z)}{\mathrm{d}z}
= -\alpha_s P_s(z) + g_R \frac{\rho}{A_\mathrm{eff}} P_p(z) P_s(z),$$

where $\alpha_s$ is the signal attenuation, $g_R$ the Raman gain
coefficient, $\rho$ a polarization factor, and $A_\mathrm{eff}$ the
effective area. Counter-propagating pumps follow

$$P_p(z) = P_{p,\mathrm{in}} \exp[-\alpha_p (L - z)],$$

yielding closed-form solutions for $P_s(z)$ under undepleted pump
assumptions (see `analysis/undepleted_fB.py`).

Effective parameters governing the undepleted-pump model are:

- $g_{\mathrm{eff}} = \rho g_R / A_\mathrm{eff}$ (effective Raman gain
  coefficient)
- $\alpha_s, \alpha_p$ (signal and pump attenuation, in Np/m)
- $L$ (fiber length), $P_{p,\mathrm{in}}$ (pump launch power), $P_s(0)$
  (signal launch power)

Under the undepleted-pump approximation the pump is decoupled,

$$\frac{\mathrm{d}P_p}{\mathrm{d}z} = -\alpha_p P_p,$$

and the signal has the closed-form solution

$$P_s(z) = P_s(0)\exp\left(-\alpha_s z + g_{\mathrm{eff}}\int_0^z P_p(z')\,\mathrm{d}z'\right).$$

For a co-propagating pump, $P_p(z)=P_{p,\mathrm{in}}\exp(-\alpha_p z)$,

$$P_s(z)=P_s(0)\exp\left(-\alpha_s z + \frac{g_{\mathrm{eff}}P_{p,\mathrm{in}}}{\alpha_p}\left(1-e^{-\alpha_p z}\right)\right).$$

For a counter-propagating pump launched at $z=L$,

$$P_p(z)=P_{p,\mathrm{in}}\exp[-\alpha_p(L-z)],$$

$$P_s(z)=P_s(0)\exp\left(-\alpha_s z + \frac{g_{\mathrm{eff}}P_{p,\mathrm{in}}e^{-\alpha_p L}}{\alpha_p}\left(e^{\alpha_p z}-1\right)\right).$$

## 2. Time-domain NLI model (implementation-exact)

This section documents the equations currently executed by the reusable TD/MC
method layer (`pynlin.methods.td`, `pynlin.methods.mc`) and orchestrated by the
PCFM workflow (`analysis/methods/workflow.py`).

The TD formulas below are implementation-exact for this repository. They
do not map one-to-one onto a single closed-form derivation in
`input/pcfm.pdf`. The explicit paper-equation references in these notes
therefore concentrate on the PCFM section below, where the
correspondence is direct and verifiable.

### Collision-coefficient path and units

The workflow computes collision coefficients via
`collision_coeffs_system_uwb` and then converts them to SI in
`total_nlin_uwb` using

$$y_\mathrm{norm} = \frac{1}{(L R_s)^2}, \qquad
\mathcal{N}^{(\mathrm{SI})}_{m_A,\nu_A,m_B,\nu_B}
= \frac{\mathcal{N}_{m_A,\nu_A,m_B,\nu_B}}{y_\mathrm{norm}}.$$

With per-channel launch powers $P_{m_A,\nu_A}$ (broadcast across modes
when needed), TD uses the cubic prefactor

$$\mathcal{C}_{m_A,\nu_A}=\frac{P_{m_A,\nu_A}^3}{R_s^2},$$

and applies Kerr scaling inside the interferer sum with the pairwise
coefficient

$$\gamma_{\nu_A,\nu_B}
= \frac{2\pi f_{\nu_A}}{c}\frac{2n_2}{A_{\mathrm{eff},\nu_A}+A_{\mathrm{eff},\nu_B}}.$$

The implementation constant is
$n_2 = 2.6\times 10^{-20}\,\mathrm{m^2/W}$.

### Multiplicity and mode-coupling factors

For each pair $(m_A,m_B)$, the mode-coupling weight is
$\kappa^2_{m_A,m_B}$ from `get_kappa2_matrix_uwb`. In the current
workflow call, `use_kappa=True` and `use_x_mode=True`, so
$\kappa^2_{m_A,m_B}$ is loaded from `input/fiber_data/kappa_uwb.csv`.
In SMF this reduces to the validated CSV value
$\kappa_{0,0}\approx 8/9$, hence $\kappa^2_{0,0}\approx (8/9)^2$.

The multiplicity prefactor is

$$\begin{aligned}
\Theta(m_A,m_B) =
\begin{cases}
\mu_0\left(2S_{m_A}+3\right)-4, & m_A=m_B,\\[4pt]
2S_{m_B}(\mu_0-1), & m_A\neq m_B,
\end{cases}
\end{aligned}$$

where `SPATIAL_MODES = [1,2,2,1]` and $\mu_0$ is the constellation
kurtosis factor. In `total_nlin_uwb` this $\mu_0$ is fixed to 64-QAM
(`MU0 = qam_mu0(64)`).

### TD NLIN expression actually used in workflow

The per-mode/per-channel TD NLIN computed in `total_nlin_uwb` is

$$P^{\mathrm{TD}}_{m_A,\nu_A}
=
\mathcal{C}_{m_A,\nu_A}
\sum_{m_B,\nu_B}
\mathcal{N}^{(\mathrm{SI})}_{m_A,\nu_A,m_B,\nu_B}
\kappa^2_{m_A,m_B}
\Theta(m_A,m_B)
\gamma_{\nu_A,\nu_B}^2.$$

No additional workflow-level $16/9$ scaling is applied.

### Modulation sweep path (TD decomposition)

For modulation sweeps, the workflow computes
`(constant_prefactor, sum_a, sum_b)` in `analysis/pcfm/td.py`:

$$\mathrm{sum\_a}_{m_A,\nu_A}
=
\sum_{m_B}
\kappa^2_{m_A,m_B}
\left(\sum_{\nu_B}\mathcal{N}^{(\mathrm{SI})}_{m_A,\nu_A,m_B,\nu_B}\gamma_{\nu_A,\nu_B}^2\right)
a(m_A,m_B),$$

$$\mathrm{sum\_b}_{m_A,\nu_A}
=
\sum_{m_B}
\kappa^2_{m_A,m_B}
\left(\sum_{\nu_B}\mathcal{N}^{(\mathrm{SI})}_{m_A,\nu_A,m_B,\nu_B}\gamma_{\nu_A,\nu_B}^2\right)
b(m_A,m_B),$$

with

$$\begin{aligned}
a(m_A,m_B), b(m_A,m_B)
=
\begin{cases}
\left(2S_{m_A}+3,\ -4\right), & m_A=m_B,\\[4pt]
\left(2S_{m_B},\ -2S_{m_B}\right), & m_A\neq m_B.
\end{cases}
\end{aligned}$$

For a target modulation with kurtosis $\mu_0$, the TD estimate is

$$P^{\mathrm{TD}}_{m_A,\nu_A}(\mu_0)
=
\mathcal{C}_{m_A,\nu_A}
\left(\mu_0\,\mathrm{sum\_a}_{m_A,\nu_A}
+ \mathrm{sum\_b}_{m_A,\nu_A}\right).$$

The workflow evaluates this for 16/64/256-QAM plus Gaussian ($\mu_0=2$
in `constellation_stats.gaussian_mu0`).

### TD normalization used in exports/plots

The signal-power denominator for all GSNR/NLI-normalized outputs is
$P_{\mathrm{sig},i}(L)$ from `_resolve_signal_power`:

1.  If profile is available and valid, $P_{\mathrm{sig},i}(L)$ is the
    last-z profile sample.
2.  Otherwise it falls back to launch powers.

Then

$$\mathrm{GSNR}^{\mathrm{TD}}_i
= 10\log_{10}\!\left(\frac{P_{\mathrm{sig},i}(L)}
{\max(P^{\mathrm{TD}}_{\mathrm{NLI},i}, 10^{-18})}\right).$$

## 3. PCFM/GN model (implementation-exact)

Primary paper reference for the PCFM-specific equations in this section:
P. Poggiolini, Y. Jiang, Y. Gao, and F. Forghieri,
`Polynomial Closed Form Model for Ultra-Wideband Transmission Systems`,
local copy: `input/pcfm.pdf`. Whenever a paper equation number is quoted
below, it refers to that PDF.

### Runtime flow and power-profile handling

The workflow supports these profile modes: `flat`, `cached`,
`recompute`, `cached_no_profile_launch`, `recompute_no_profile_launch`.

In flat mode, it writes a synthetic profile

$$P_i(z_k) = P_{i,\mathrm{launch}}, \quad \forall k,$$

thus $p_i(z)=1$.

When using real Raman profiles, the code validates them by checking
finite values and requiring

$$\max_{i,z} P_i(z) \le 10\ \mathrm{W}.$$

### Launch-power resolution and validation

Per-channel launch powers are resolved with priority profile -\> CSV -\>
TOML, depending on run mode.

All dBm-to-W conversions use

$$P[\mathrm{W}] = 10^{(P_{\mathrm{dBm}}-30)/10}.$$

When a CSV is used, powers are interpolated in dBm over frequency. When
profile launch powers are used, they are checked against current
settings with tolerance

$$\max_i |P^{\mathrm{profile}}_{i,\mathrm{dBm}} - P^{\mathrm{expected}}_{i,\mathrm{dBm}}|
\le 0.1\ \mathrm{dB}.$$

### Signal profile loading and normalization

`load_signal_profiles` returns channel profiles $P_i(z)$. For 3D stored
tensors, the implementation sums one axis to obtain channel powers (this
may aggregate polarization/mode components, depending on file layout).

The profiles are normalized directly to the launch sample:

$$p_i(z)=\frac{P_i(z)}{P_i(0)}.$$

This normalized SPP is the implementation counterpart of the paper
definition in Poggiolini et al., Eq. (33).

The normalized profile is clipped to `[0, MAX_SPP]` with `MAX_SPP=1e3`.

### Polynomial profile representation

For PCFM and GN-numeric, the profile is fit in normalized distance
$u=(z-z_0)/L \in [0,1]$:

$$p_i(u)\approx \sum_{n=0}^{N} a_{i,n}u^n.$$

This is the same polynomial SPP representation introduced in Poggiolini
et al., Eq. (49), with a normalized-distance variable in the
implementation.

The closed-form XCI helper term is

$$S_i = \sum_{n,k}\frac{a_{i,n}a_{i,k}}{n+k+1},$$

implemented via coefficient convolution (`poly_sum`). This is the
coefficient sum that appears in the XCI closed form of Poggiolini et
al., Eq. (50).

### Dispersion model used in phase terms

If `use_beta2_eff=True` and beta-spline derivatives are available, the
code uses Poggiolini Eq. (28):

$$\beta_{2,\mathrm{eff}}(f_m,f_k)
=
\beta_2
+ \pi\beta_3(f_m+f_k-2f_c)
+ \frac{2}{3}\pi^2\beta_4
\left[(f_m-f_c)^2 + (f_m-f_c)(f_k-f_c) + (f_k-f_c)^2\right].$$

Otherwise it falls back to sampled channel $\beta_2$.

### Numerical SCI/XCI kernels

The phase model is

$$\phi(f_1,f_2,z)=C_\phi\,\beta_{2,\mathrm{eff}}\,f_1f_2z,$$

with $C_\phi=4\pi^2$ by default.

The combination of the $4\pi^2$ phase factor, the effective-dispersion
approximation above, and the per-island loss flattening mirrors the
paper path from Eq. (29) to Eq. (38). The code also uses the same
channel-flat loss approximation as Poggiolini et al., Eq. (30).

SCI kernel:

$$K_{\mathrm{SCI},i}
=
\int_{-B/2}^{B/2}\!\!\int_{-B/2}^{B/2}
\left|
\int_0^L p_i(z)e^{j\phi(f_1,f_2,z)}\,dz
\right|^2
df_2\,df_1.$$

XCI kernel:

$$K_{\mathrm{XCI},ij}
=
\int_{-B/2}^{B/2}\!\!\int_{\Delta f_{ij}-B/2}^{\Delta f_{ij}+B/2}
\left|
\int_0^L p_j(z)e^{j\phi(f_1,f_2,z)}\,dz
\right|^2
df_2\,df_1.$$

At the paper level, these per-island kernel constructions correspond to
the combined SPP definition of Poggiolini et al., Eq. (36), and the
generic kernel definition of Eq. (38). The implementation specializes
those formulas to the SCI/XCI cases shown above.

All integrals are evaluated numerically with trapezoidal integration on
uniform grids. The `direct` GN path uses sampled $p(z)$ directly instead
of polynomial fits.

### PCFM and GN PSD equations in code

Let

$$g_i = \frac{2P_{i,\mathrm{launch}}}{B},$$

$$\gamma_i = \frac{2\pi f_i}{c}\frac{n_2}{A_{\mathrm{eff},i}}, \qquad
\gamma_{ij} = \frac{2\pi f_i}{c}\frac{2n_2}{A_{\mathrm{eff},i}+A_{\mathrm{eff},j}}.$$

The XCI nonlinear coefficient $\gamma_{ij}$ matches Poggiolini et al.,
Eq. (42). The SCI coefficient $\gamma_i$ is the single-channel Kerr
coefficient of Eq. (19).

Then for every CUT $i$:

$$G_{\mathrm{SCI},i}
= \frac{16}{27}g_i^3\gamma_i^2K_{\mathrm{SCI},i},$$

$$G_{\mathrm{XCI},ij}
= \frac{32}{27}g_ig_j^2\gamma_{ij}^2K_{\mathrm{XCI},ij},$$

$$G_{\mathrm{NLI},i}
= G_{\mathrm{SCI},i} + \sum_{j\neq i}G_{\mathrm{XCI},ij}.$$

These expressions are the implementation counterparts of the paper-level
PCFM SCI/XCI PSD structure in Poggiolini et al., Eqs. (47), (48), and
(52). The code keeps the same prefactor structure, but see the note
below on the endpoint factor $p_i(L)$.

Only interferers satisfying $|\Delta f_{ij}| > B/2$ are included. For
PCFM closed-form XCI (when `use_numeric_xci=False`):

$$K_{\mathrm{XCI},ij}
\approx
\frac{L}{2\pi\max(|\beta_{2,\mathrm{eff}}|,10^{-30})}
\ln\!\left(\frac{|\Delta f_{ij}|+B/2}{|\Delta f_{ij}|-B/2}\right)S_j.$$

This is the code-level specialization of Poggiolini et al., Eq. (50),
with the implementation guard `max(|\beta_{2,\mathrm{eff}}|,10^{-30})`
added for numerical robustness.

The endpoint factor $p_i(L)$ is not applied inside the PCFM/GN PSD
expressions. Those producers return launch-referenced NLIN powers, and
the workflow applies the endpoint conversion once with
$P_{\mathrm{sig},i}(L)/P_{\mathrm{sig},i}(0)$ before export and plotting.

### Per-polarization output correction (current implementation)

PCFM, GN-numeric, and GN-direct all apply a final output correction:

$$P^{\mathrm{model}}_{\mathrm{NLI},i}
=
\frac{G_{\mathrm{NLI},i}B}{2}.$$

The same $1/2$ factor is also applied to returned SCI and XCI power
components.

Code-level logic is explicit:

- the PSD prefactors use channel-level launch normalization
  ($g_i=2P_i/B$ via `_to_per_channel_power`),
- kernel accumulation builds channel-level NLIN PSD,
- the returned NLIN power is converted to per-polarization values with
  `_to_per_polarization_power(...)=.../2`.

So the divide-by-2 is a final representation choice (per-polarization
reporting), not a change to the kernel physics.

### Idealized flat-profile SMF comparison

For the idealized case (historically `input/pcfm_struct.toml`):

- single spatial mode,
- constant $\beta_2$,
- constant $A_{\mathrm{eff}}$,
- flat signal-power profiles $p(z)\equiv 1$,
- XCI-only comparison (same CUT/interferer subset in both models),
- equal launch powers $P_i=P_j=P$,

the implementation formulas simplify enough to compare TD and PCFM
directly.

Write $B$ for the baud rate, $\Delta f_{ij}=r_{ij}B$ with $r_{ij}>1/2$,
and

$$x_{ij}=L B\,\mathrm{DGD}_{ij}
=2\pi |\beta_2| L B |\Delta f_{ij}|
=2\pi |\beta_2| L B^2 r_{ij}.$$

In the flat-profile TD path, `fit_nlin` returns the
perfect-amplification Nyquist/Gaussian stage-S1 collision curve as a
function of $x$ (`softplus(dgd * L * br, *ps)`), so the per-pair SI
collision coefficient can be written as

$$\mathcal{N}^{(\mathrm{SI})}_{ij}
= (L B)^2 F_{\mathrm{S1}}(x_{ij}),$$

where $F_{\mathrm{S1}}$ is the normalized reference curve loaded from
the S1 cache.

The fitted S1 reference used by the TD path is parameterized as

$$F_{\mathrm{S1}}(x)
=
a\left(1+\left(\frac{x}{\Lambda}\right)^{1/\eta}\right)^{-\eta},$$

with code-level correspondence

$$
(a,\Lambda,\eta) \equiv (\texttt{ps[0]}, \texttt{ps[1]}, \texttt{ps[2]}).
$$

Equivalently, this is the `softplus(x, a, b, c)` fit used in
`ideal_fits.py` and `ideal_fits_uwb.py`, with

$$
b \equiv \Lambda, \qquad c \equiv \eta.
$$

For the current cached perfect-S1 references
($\mathrm{gvd}_a=\mathrm{gvd}_b=0$, no Raman), the fitted parameters are
approximately:

- Gaussian pulse (`ipulse=0`)

  $$
  (a,\Lambda,\eta) \approx (0.28221327,\; 3.28726640,\; 0.49259348).
  $$

- Nyquist pulse (`ipulse=1`)

  $$
  (a,\Lambda,\eta) \approx (0.47125353,\; 2.11578009,\; 0.92291662).
  $$

### LO and HI correction scaling in the TD fit

The Raman/GVD-aware TD path does not refit the whole S1 curve. Instead,
it starts from the ideal coefficients $(a,\Lambda,\eta)$ and applies two
separate parameter corrections:

- an LO correction that changes the plateau value,
- an HI correction that shifts the horizontal turning-point scale.

Write the ideal curve as

$$
\mathcal{N}_{\mathrm{ideal}}(x)
=
a\left(1+\left(\frac{x}{\Lambda}\right)^{1/\eta}\right)^{-\eta}.
$$

Let $a_{\mathrm{LO}}$ denote the corrected LO plateau value extracted
from the Raman/GVD lookup, and let $h_{\mathrm{HI}}$ denote the HI
correction factor from the Raman integral. The implementation applies
the following transformations.

#### LO correction

The LO correction replaces the plateau amplitude and rescales
$\Lambda$ so that the product $a\Lambda$ is preserved:

$$
a \longrightarrow a_{\mathrm{LO}},
$$

$$
\Lambda \longrightarrow \Lambda_{\mathrm{LO}}
=
\Lambda \frac{a}{a_{\mathrm{LO}}},
$$

$$
\eta \longrightarrow \eta.
$$

Therefore

$$
\mathcal{N}_{\mathrm{LO}}(x)
=
a_{\mathrm{LO}}
\left(
1+\left(
\frac{x}{\Lambda (a/a_{\mathrm{LO}})}
\right)^{1/\eta}
\right)^{-\eta}.
$$

Equivalently,

$$
a_{\mathrm{LO}}\Lambda_{\mathrm{LO}} = a\Lambda.
$$

This is exactly the scaling implemented by
`apply_plateau_correction`, which sets

$$
\texttt{ps[0]} \leftarrow a_{\mathrm{LO}}, \qquad
\texttt{ps[1]} \leftarrow \texttt{ps[1]}\frac{\texttt{old\_lo\_value}}{\texttt{lo\_value}}.
$$

#### HI correction

The HI correction leaves the plateau untouched and multiplies the
horizontal scale:

$$
a \longrightarrow a,
$$

$$
\Lambda \longrightarrow \Lambda_{\mathrm{HI}}
=
\Lambda h_{\mathrm{HI}},
$$

$$
\eta \longrightarrow \eta.
$$

Hence

$$
\mathcal{N}_{\mathrm{HI}}(x)
=
a\left(1+\left(\frac{x}{\Lambda h_{\mathrm{HI}}}\right)^{1/\eta}\right)^{-\eta}.
$$

This is the scaling implemented by `apply_turning_point_correction`,
which applies

$$
\texttt{ps[1]} \leftarrow \texttt{ps[1]} \, h_{\mathrm{HI}}.
$$

#### Combined LO + HI correction

When both corrections are active, the final corrected fit is

$$
\mathcal{N}_{\mathrm{corr}}(x)
=
a_{\mathrm{LO}}
\left(
1+\left(
\frac{x}{\Lambda_{\mathrm{corr}}}
\right)^{1/\eta}
\right)^{-\eta},
$$

with

$$
\Lambda_{\mathrm{corr}}
=
\Lambda \frac{a}{a_{\mathrm{LO}}} h_{\mathrm{HI}}.
$$

So the final parameter map is

$$
a_{\mathrm{corr}} = a_{\mathrm{LO}}, \qquad
\Lambda_{\mathrm{corr}} = \Lambda \frac{a}{a_{\mathrm{LO}}} h_{\mathrm{HI}}, \qquad
\eta_{\mathrm{corr}} = \eta.
$$

Two practical consequences follow immediately:

- the LO correction preserves the large-walkoff tail prefactor
  $a\Lambda$,
- the HI correction directly scales that tail prefactor by
  $h_{\mathrm{HI}}$.

In the current flat-$f_B$ branch, only the LO correction is applied; the
HI correction is skipped.

For SMF, the TD modulation coefficient is $5\mu_0-4$ (because $S_0=1$),
hence the current workflow gives

$$P^{\mathrm{TD}}_{ij}(\mu_0)
= \frac{P^3\gamma^2}{B^2}(5\mu_0-4)\,(L B)^2F_{\mathrm{S1}}(x_{ij}).$$

For PCFM-XCI, the current implementation uses `g_ch = 2P/B`, the
closed-form kernel with $S_j=1$ for $p(z)\equiv 1$, and then the final
per-polarization divide-by-2. With constant $A_{\mathrm{eff}}$,
$\gamma_{ij}=\gamma_i=\gamma$, so

$$P^{\mathrm{PCFM,XCI}}_{ij}
= \frac{64}{27}\frac{P^3\gamma^2}{B^2}K_{\mathrm{XCI},ij},$$

$$K_{\mathrm{XCI},ij}
= \frac{L}{2\pi |\beta_2|}
\ln\!\left(\frac{r_{ij}+1/2}{r_{ij}-1/2}\right)
= (L B)^2 \frac{r_{ij}}{x_{ij}}
\ln\!\left(\frac{r_{ij}+1/2}{r_{ij}-1/2}\right).$$

Therefore the matched pairwise ratio is

$$\frac{P^{\mathrm{PCFM,XCI}}_{ij}}{P^{\mathrm{TD}}_{ij}(\mu_0)}
=
\frac{64}{27(5\mu_0-4)}
\frac{
r_{ij}\ln\!\left(\frac{r_{ij}+1/2}{r_{ij}-1/2}\right)
}{
x_{ij}F_{\mathrm{S1}}(x_{ij})
}.$$

#### Script-only Eq. (18) XCI alternative

The standalone script
`analysis/standalone_analytical/analytical_n_vs_kxci.py` also plots an
alternative flat-Raman XCI curve labelled
$K_{\mathrm{XCI}}^{\mathrm{Eq.18}}$. In the script, this is implemented
in normalized variables as

$$
K_{\mathrm{XCI}}^{\mathrm{Eq.18}}(x)\,T^2L^{-2}
=
\frac{1}{2\pi}
\cdot
\frac{2}{\pi (L/L_{\mathrm{eff}})}
\left[
g\!\left(\pi \frac{L}{L_{\mathrm{eff}}}\left(x+\frac12\right)\right)
-
g\!\left(\pi \frac{L}{L_{\mathrm{eff}}}\left(x-\frac12\right)\right)
\right],
$$

or equivalently

$$
K_{\mathrm{XCI}}^{\mathrm{Eq.18}}(x)\,T^2L^{-2}
=
\frac{1}{\pi^2 (L/L_{\mathrm{eff}})}
\left[
g\!\left(\pi \frac{L}{L_{\mathrm{eff}}}\left(x+\frac12\right)\right)
-
g\!\left(\pi \frac{L}{L_{\mathrm{eff}}}\left(x-\frac12\right)\right)
\right].
$$

Here

$$
g(z)=J(z)-\operatorname{Si}(z)+\frac{1-\cos z}{z},
$$

with

$$
J(z)=\int_0^z \frac{\operatorname{Si}(t)}{t}\,dt.
$$

The three terms in $g(z)$ are all odd functions of $z$, so $g$ itself
is odd. Therefore, if one analytically extends the formula to small $x$,
then at $x=0$ one gets

$$
K_{\mathrm{XCI}}^{\mathrm{Eq.18}}(0)\,T^2L^{-2}
=
\frac{2}{\pi^2 (L/L_{\mathrm{eff}})}
g\!\left(\pi \frac{L}{L_{\mathrm{eff}}}\frac12\right),
$$

which is finite. So, unlike the closed-form logarithmic XCI kernel, the
Eq. (18) alternative does not develop a singularity as $x\to 0$.

In the current script, however, this alternative curve is still masked
for $x\le 1/2$ so that all plotted XCI curves share the same
non-overlapping-domain convention as the closed-form
$K_{\mathrm{XCI}}$.

This expression is the cleanest implementation-level comparison
available in the idealized regime. It also shows that there is no
general proof in the current code that `PCFM >= TD`.

For Nyquist pulses, the stored S1 perfect-curve fit is well approximated
at large walk-off by

$$F_{\mathrm{S1}}(x)\sim \frac{c_{\mathrm{N}}}{x},$$

with $c_{\mathrm{N}}\approx 1$ for the current cached fit. Then

$$\frac{P^{\mathrm{PCFM,XCI}}_{ij}}{P^{\mathrm{TD}}_{ij}(\mu_0)}
\sim
\frac{64}{27(5\mu_0-4)c_{\mathrm{N}}}
r_{ij}\ln\!\left(\frac{r_{ij}+1/2}{r_{ij}-1/2}\right).$$

Since $r\ln((r+1/2)/(r-1/2))\to 1$ as $r\to\infty$, the asymptotic limit
becomes

$$\frac{P^{\mathrm{PCFM,XCI}}}{P^{\mathrm{TD}}}
\to
\frac{64}{27(5\mu_0-4)c_{\mathrm{N}}}.$$

For Gaussian modulation ($\mu_0=2$), this tends to about `0.40` when
$c_{\mathrm{N}}\approx 1$; for 64-QAM ($\mu_0\approx 1.381$), it is
about `0.82`. So under the present implementation, the idealized
asymptotic trend is the opposite inequality: TD remains larger than
PCFM-XCI.

### GSNR and normalized outputs in workflow

For each model output vector $P_{\mathrm{NLI},i}$, the workflow computes

$$\mathrm{GSNR}_i
=
10\log_{10}\!\left(
\frac{P_{\mathrm{sig},i}(L)}
{\max(P_{\mathrm{NLI},i},10^{-18})}
\right).$$

TD, PCFM, and GN producers use normalized profiles and return
launch-referenced NLIN powers. Before this ratio is formed, the workflow
converts them to output powers with
$P_{\mathrm{sig},i}(L)/P_{\mathrm{sig},i}(0)$.

The denominator $P_{\mathrm{sig},i}(L)$ is obtained from
`_resolve_signal_power` and is not additionally divided by 2 in this
path.

The optional GN-direct plotting branch also stores

$$\eta^{\mathrm{GNdir}}_i = \frac{P^{\mathrm{GNdir}}_{\mathrm{NLI},i}}{P_{\mathrm{sig},i}(L)}$$

for convenience in ratio plots.

### TD-vs-PCFM(XCI) diagnostic currently computed

The workflow prints (Gaussian-modulation TD against PCFM-XCI):

$$\Delta_i = P^{\mathrm{TD,Gauss}}_{\mathrm{NLI},i} - P^{\mathrm{PCFM,XCI}}_{\mathrm{NLI},i},
\qquad
r_i = \frac{P^{\mathrm{TD,Gauss}}_{\mathrm{NLI},i}}
{\max(P^{\mathrm{PCFM,XCI}}_{\mathrm{NLI},i},10^{-30})},$$

plus dB-domain differences:

$$\Delta^{\mathrm{dB}}_i =
10\log_{10}\!\big(\max(P^{\mathrm{TD,Gauss}}_{\mathrm{NLI},i},10^{-30})\big)
-
10\log_{10}\!\big(\max(P^{\mathrm{PCFM,XCI}}_{\mathrm{NLI},i},10^{-30})\big).$$

### Supporting models and data

#### Four-wave mixing (FWM)

Phase matching is assessed via polynomial fits of $\beta_0(\omega)$ per
mode and the plane

$$a \omega_1 + b \omega_2 + c \omega_3 + d = 0,$$

constructed from mode tuples and permutation signs $p$. Intersections of
this plane with the frequency cube identify FWM-relevant combinations
(see `analysis/phase_matching.py` and `analysis/fwm_efficiency.py`).

#### Fiber properties

**Overlap integrals.** Spatial overlap integrals (OI) between modes are
precomputed from numerical field solutions and fitted with low-order
polynomials in wavelength. The tensor $\mathrm{OI}(\lambda_1,\lambda_2)$
feeds NLIN and Raman estimators (see `analysis/oi_fit.py`).

**Dispersion and group delay.** Group delay and dispersion are
represented by mode-wise polynomials in angular frequency:

$$\beta_1(\omega) \approx p_1 \omega^2 + p_2 \omega + p_3, \qquad
\beta_2(\omega) = \frac{\mathrm{d}\beta_1}{\mathrm{d}\omega},$$

with coefficients derived from MATLAB fits and converted to SI units

**Optimization workflows.** Pump optimization solves for pump
wavelengths/powers that flatten on--off gain across modes/channels. A
typical loop:

1.  Load config (WDM grid, fiber, target gain).
2.  Initialize pumps, run gradient-based optimizer (PyTorch) or reuse
    cached solutions.
3.  Repropagate with a NumPy solver for verification.
4.  Plot signal/pump profiles and flatness metrics.

See `analysis/optimize.py` and plotting helpers in
`analysis/components/plot_optimization.py`.

## 4. Direct Monte-Carlo TD integration (Dar NLIN)

The direct Monte-Carlo time-domain (TD) integration used for Dar NLIN is
implemented in `src/darnlin/nlin.py` (NumPy port of the original
MATLAB). It estimates the frequency-domain integrals by random sampling
of phase variables and returns NLIN variance for a single interferer.

Implementation mapping (Dar NLIN):

- Monte-Carlo sampling draws $R \sim \mathcal{U}(-\pi,\pi)$ in 4 or 5
  dimensions, depending on the term.
- Inter-channel variance uses `calc_interChannel` which computes
  $\chi_1$ and $\chi_2$ and combines them as
  $\sigma^2 = \chi_1 + (\mu_0-2)\chi_2$ (plus a
  polarization-multiplexing correction when `pol_mux=1`). The function
  returns an error estimate from sample variance.
- Additional inter-channel terms are evaluated in
  `calc_interChannel_addTerms` ($X_{21}\ldots X_{24}$).
- Intra-channel terms are evaluated in `calc_intraChannel`
  ($X_1, X_0, X_2, X_{21}, X_3$) and combined in `_intra_var`.
- Normalization in `main` converts physical inputs to normalized units
  using $T = 1/R_s$ and scales $\beta_2$, $\alpha$, and $\Delta f$
  accordingly before calling the Monte-Carlo kernels.
- The geometric sum across spans is modeled by the factor
  $(1-e^{i n_{span} \beta_2 \Delta z})/(1-e^{i \beta_2 \Delta z})$
  inside each sampled term.

### Prefactor-free generic-$X_{hkm}$ sums

The optional FFT generic-collision workflow stores the tensor convention

$$X[h,r,m]=X_{h,m+r,m}, \qquad r=k-m.$$

It contracts this tensor into

$$N_1=\sum_{h,r,m}|X[h,r,m]|^2,$$

and

$$N_2=\sum_{h,m}|X[h,0,m]|^2.$$

These quantities are named $N_1$ and $N_2$ because they correspond to the
collision-sum parts of Dar-style $\chi_1$ and $\chi_2$ without the physical
prefactors. Stored curves use the S1 normalization $T^2/L^2$ and the
walk-off axis $L/L_W$. The existing $X_{0mm}$ S1 workflow remains unchanged.
The saved diagnostics also separate 2PC, 3PCa ($h=0,k\ne m$), 3PCb
($h\ne0,k=m$), residual 3PC edge coincidences, and 4PC sectors.

In the high-walk-off limit the generic coefficients approach a complete-
collision factorization of the form

$$X_{h,m+r,m}
\propto
\frac{1}{|\Delta\beta_1|}
\langle g(t),g(t-hT)\rangle
\langle g(t-rT),g(t)\rangle.$$

Therefore the high-walk-off class decomposition depends on pulse
orthogonality. For Nyquist sinc pulses, symbol-spaced shifts are
orthogonal, so the shifted overlaps vanish for $h\ne0$ or $r\ne0$ and

$$N_1 \simeq N_2 \simeq N_{2PC}$$

at large $L/L_W$. Gaussian pulses are not orthogonal under symbol-spaced
shifts, so 3PCa, 3PCb, and 4PC sectors retain finite high-walk-off
prefactors. Their curves then share the same large-walk-off scaling as the
2PC term but do not merge with it.

With strong dispersion the generic-collision sums become much more sensitive
to the finite collision support used in the numerical contraction. The
horizontal axis $L/L_W$ uses the input symbol time $T$, but the relevant
collision overlap is set by the dispersed pulse width,

$$T_{\mathrm{eff}}(z) \sim T\sqrt{1 + (z/L_D)^2}.$$

For $L/L_D \gg 1$, pulses broaden substantially during the span. Terms whose
nominal collision centers lie outside the fiber can still contribute through
their dispersive tails and through partial collisions near the fiber edges.
Therefore the support in $m$ must include enough outside-centered partial
collisions, and the support in $h$ and $r=k-m$ must include enough neighboring
symbol offsets. Aggressively truncated sums, for example using only
$h,r\in[-1,1]$ with a small $m$ margin, can show an apparent broad maximum
around

$$L/L_W = O(L/L_D),$$

because the retained center-inside collisions and the omitted edge/outside
partial collisions scale differently with walk-off. In the higher-support
calculations using $h,r\in[-5,5]$ and an $m$ margin of 10, this upward bump is
removed from the Nyquist high-dispersion curves, showing that the previous
feature was primarily a finite-support artifact rather than a robust physical
maximum. The high-dispersion regime should therefore be reported together
with the truncation metadata, and convergence in $h$, $r$, and the partial-
collision margin should be checked before interpreting local extrema.

An important consequence of high dispersion is that the $N_1$ and $N_2$ curves
do **not** merge in the large-walk-off limit, even for Nyquist pulses. The
dispersive broadening breaks the symbol-spaced orthogonality of the
ideal sinc pulse: the dispersed pulse $\tilde g(z,t)$ has non-zero overlap
$\langle\tilde g(z,t),\tilde g(z,t-hT)\rangle$ for $h\neq0$, and similarly
for the $r=k-m$ index. This keeps the $h\neq0$ or $r\neq0$ sectors
(3PCa, 3PCb, 4PC) finite at arbitrarily large $L/L_W$, so $N_2$ (which only
retains $r=0$) permanently carries a smaller fraction of the total collision
power than $N_1$. For $L/L_D=10$ at $L/L_W=40$ the ratio is already
$N_2/N_1\approx0.55$, well within the numerically reliable regime.

When extending the walk-off range much beyond $L/L_W\sim80$ with the
direct FFT kernel, the finite $z$ resolution becomes a concern. The
collision width $w=T/\Delta\beta_1$ shrinks as $w=L/(L/L_W)$, so at
$L/L_W=300$ the width is $1.3$~m. If the $z$ grid in the numerical
integration is coarser than this width, the collision integrals are severely
undersampled and the curves show artificial splits or flattening.

The kernel includes an **auto-refinement** mode (`auto_refine=True` in
`compute_xpm_kernel_fft`) that automatically densifies the $z$ grid to
maintain at least `min_pts_per_collision` points per collision width.
It caps at `max_z_points=500` to keep computation practical. For
$L/L_W\lesssim100$ this ensures adequate resolution ($\gtrsim3$ pts/coll.).
For higher $L/L_W$ the cap is reached and warnings fire; the extended-range
plots mark this region as under-resolved.  The $z$-axis warning can be
upgraded to an `AssertionError` via `discretization_action="assert"`.

A separate **time-window** check warns when the total walk-off
$|dgd|\times L$ exceeds the pulse time window, indicating that the FFT
wrap-around may miss physical collisions. This check is non-fatal by default
because the computational cost of a fully adequate time window (many
thousands of symbols at high $L/L_W$) is prohibitive with the uniform-grid
FFT approach.
