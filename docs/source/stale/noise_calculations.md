# Noise Calculation Note

This note derives the noise quantities used in the current TD and PCFM
workflows in the same sequence in which the code computes them. The main
goal is not to introduce a new model, but to make the chain from
intermediate quantities to final noise power explicit, especially for
the TD versus PCFM comparison.

The derivation below is implementation-aware. It follows the formulas
currently used in `analysis/pcfm/workflow.py`,
`pynlin.methods.td`, `pynlin.methods.td.estimator`, and
`pynlin.methods.pcfm.gn`.

## 1. Shared Physical Quantities

Let

- $f_i$ be the WDM channel frequencies,
- $B$ the baud rate,
- $L$ the fiber length,
- $P_i$ the launch power of channel $i$,
- $A_{\mathrm{eff},i}$ the effective area seen by channel $i$.

The Kerr coefficient used throughout the repository is

$$\gamma_i = \frac{2\pi f_i}{c}\frac{n_2}{A_{\mathrm{eff},i}},$$

with

$$n_2 = 2.6\times 10^{-20}\ \mathrm{m^2/W}.$$

For the PCFM XCI path the mixed Kerr coefficient is

$$\gamma_{ij}
= \frac{2\pi f_i}{c}\frac{2n_2}{A_{\mathrm{eff},i}+A_{\mathrm{eff},j}}.$$

The quantity that controls TD pairwise interaction strength is the
group-delay difference

$$\mathrm{DGD}_{ij} = |\beta_{1,i}-\beta_{1,j}|,$$

and the associated dimensionless walk-off variable is

$$x_{ij} = L B\,\mathrm{DGD}_{ij}.$$

In the constant-dispersion single-mode limit this becomes

$$x_{ij}=2\pi |\beta_2| L B |f_j-f_i|.$$

This variable will reappear repeatedly, because the TD collision model
is fundamentally a function of normalized walk-off.

## 2. TD Noise Calculation

### Generic Xhkm prefactor-free sums

The optional generic-collision workflow uses the FFT evaluator
`compute_xpm_kernel_fft` to compute

$$X[h,r,m] = X_{h,m+r,m},$$

where $r=k-m$. The first contractions built on this tensor are named
$N_1$ and $N_2$ to make clear that physical prefactors have not yet been
applied:

$$N_1 = \sum_{h,r,m} |X[h,r,m]|^2,$$

$$N_2 = \sum_{h,m} |X[h,0,m]|^2.$$

These are the prefactor-free collision-sum parts corresponding to the
Dar-style $\chi_1$ and $\chi_2$ terms. The saved reference curves use the
same normalization as the S1 curves,

$$\widetilde N_j = N_j\frac{T^2}{L^2},$$

and the same walk-off axis $L/L_W=L|\Delta\beta_1|R_s$. The calculation is
opt-in and does not alter the existing S1 $X_{0mm}$ workflow. Because the
generic tensor is truncated in $h$, $r$, and $m$, convergence with respect
to those finite windows must be checked before treating a curve as
reference-quality.

The companion prefactor-free Monte-Carlo diagnostic in
`pynlin.methods.td.xhkm_mc` estimates the same two fundamental quantities as
$N_1^{MC}$ and $N_2^{MC}$ directly from the reduced Dar inter-channel
integrands. It deliberately omits physical constants such as nonlinear
coefficients, launch powers, modulation cumulants, and polarization factors.
Those constants belong to later NLIN assembly, not to the collision-sum
definitions. The MC diagnostic also rejects non-flat signal power profiles,
because otherwise the sampled Dar integrands would not correspond to the same
flat-profile sums computed from the deterministic $X_{hkm}$ tensor.

The validation script `analysis/standalone_numerical/compare_mc_xhkm_sums.py`
uses the `[mc_validation]` TOML section to compute both the MC estimates and
finite-window deterministic sums from `compute_xpm_kernel_fft` plus
`compute_xhkm_sums`. It saves raw values and basic comparison plots under
`media/mc-validation`. The first diagnostic plots include a single fitted
scale factor for the MC curves, because this reduced MC path and the finite
time-domain tensor still expose different absolute normalization conventions;
the shape and ratio trends are therefore the primary comparison until that
remaining normalization bridge is fixed.

This caveat is especially important at high dispersion. Dispersive pulse
broadening makes partial collisions with nominal centers outside the fiber
contribute through their tails and through the span edges. If the $m$ margin
or the $h,r$ windows are too small, the curve can show artificial local
features, such as an upward bump around $L/L_W=O(L/L_D)$. The higher-support
plots therefore record their finite support explicitly, e.g. $h,r\in[-5,5]$
and $m$ margin 10.

For diagnostics the tensor is also decomposed into collision classes:

$$N_{2PC}=\sum_m |X[0,0,m]|^2,$$

$$N_{3PCa}=\sum_{r\ne0,m}|X[0,r,m]|^2,$$

$$N_{3PCb}=\sum_{h\ne0,m}|X[h,0,m]|^2.$$

A separate 3PC-other bucket contains the remaining $h=k$ or $h=m$ edge
coincidences, and the 4PC bucket contains the sector where all symbol
indices are distinct.

At large walk-off, complete collisions approximately factorize into
symbol-spaced pulse overlaps,

$$X_{h,m+r,m}\propto
|\Delta\beta_1|^{-1}
\langle g(t),g(t-hT)\rangle
\langle g(t-rT),g(t)\rangle.$$

Consequently, Nyquist pulses drive all shifted-overlap sectors to zero and
make $N_1$, $N_2$, and $N_{2PC}$ merge. Gaussian pulses are not
symbol-spaced orthogonal, so non-2PC sectors keep finite asymptotic
prefactors and the decomposed curves remain separated.

### Passage 1: build collision coefficients

The TD path starts by computing collision coefficients

$$\mathcal{N}_{m_A,\nu_A,m_B,\nu_B},$$

through `collision_coeffs_system_uwb`. These coefficients are not yet
noise powers. They are dimensionless collision strengths computed from
fitted stage-S1 collision curves and then corrected for the actual Raman
profile and dispersion used by the channel pair.

Schematically,

$$\mathcal{N}_{m_A,\nu_A,m_B,\nu_B}
=
F_{\mathrm{TD}}\!\left(
L B |\beta_1(m_A,\nu_A)-\beta_1(m_B,\nu_B)|;
\text{profile corrections},\beta_{2,A},\beta_{2,B}
\right).$$

In the flat-profile limit, the profile-dependent corrections collapse
and the pair coefficient reduces to the stored stage-S1 reference curve

$$\mathcal{N}_{ij}=F_{\mathrm{S1}}(x_{ij}).$$

### Passage 2: convert the collision tensor back to SI units

The TD solver internally works in normalized units. The conversion used
in `total_nlin_uwb` is

$$y_\mathrm{norm} = \frac{1}{(L B)^2}, \qquad
\mathcal{N}^{(\mathrm{SI})}_{m_A,\nu_A,m_B,\nu_B}
= \frac{\mathcal{N}_{m_A,\nu_A,m_B,\nu_B}}{y_\mathrm{norm}}
= (L B)^2 \mathcal{N}_{m_A,\nu_A,m_B,\nu_B}.$$

This step is important because it isolates the physical span-length and
baud-rate scaling from the normalized collision law itself.

### Passage 3: apply launch-power and Kerr scaling

Once the SI collision coefficients are available, the code multiplies
them by the cubic power/Kerr prefactor

$$\mathcal{P}_{\nu_A} = \frac{P_{\nu_A}^3 \gamma_{\nu_A}^2}{B^2}.$$

This factor explains why both TD and PCFM scale cubically with launch
power in the matched equal-power comparison.

### Passage 4: apply modulation and mode multiplicities

The TD sum is not only geometric. It is also weighted by a
modulation-dependent multiplicity factor

$$\Theta(m_A,m_B) =
\begin{cases}
\mu_0(2S_{m_A}+3)-4, & m_A=m_B,\\[4pt]
2S_{m_B}(\mu_0-1), & m_A\neq m_B,
\end{cases}$$

where

$$\mu_0=\frac{\langle |b|^4\rangle}{\langle |b|^2\rangle^2}$$

is the constellation kurtosis factor and
`SPATIAL_MODES = [1,2,2,1]`.

In the direct `total_nlin_uwb` path, the implementation fixes $\mu_0$
to 64-QAM. In the modulation-sweep path, the same collision tensor is
reused for several values of $\mu_0$.

### Passage 5: sum over interferers

With coupling weights $\kappa^2_{m_A,m_B}$, the raw TD NLIN power is

$$P^{\mathrm{TD,raw}}_{m_A,\nu_A}
=
\mathcal{P}_{\nu_A}
\sum_{m_B,\nu_B}
\mathcal{N}^{(\mathrm{SI})}_{m_A,\nu_A,m_B,\nu_B}
\kappa^2_{m_A,m_B}
\Theta(m_A,m_B).$$

In the current TD versus PCFM workflow, `use_kappa=False` and
`use_x_mode=True`, so

$$\kappa^2_{m_A,m_B}=1.$$

The workflow then multiplies the raw TD result by the extra factor

$$P^{\mathrm{TD}}_{m_A,\nu_A}
= \frac{16}{9}P^{\mathrm{TD,raw}}_{m_A,\nu_A}.$$

This is the TD quantity actually plotted and compared against PCFM.

### Passage 6: why the TD modulation sweep is affine in $\mu_0$

The helper `analysis/pcfm/td.py` rewrites the mode prefactor as

$$\Theta(m_A,m_B)=\mu_0\,a(m_A,m_B)+b(m_A,m_B),$$

with

$$\begin{aligned}
a(m_A,m_B), b(m_A,m_B)
=
\begin{cases}
\left(2S_{m_A}+3,\ -4\right), & m_A=m_B,\\[4pt]
\left(2S_{m_B},\ -2S_{m_B}\right), & m_A\neq m_B.
\end{cases}
\end{aligned}$$

After summing the collision tensor once, the TD result becomes

$$P^{\mathrm{TD}}_{m_A,\nu_A}(\mu_0)
=
\frac{16}{9}\,
\mathcal{P}_{\nu_A}
\left(\mu_0\,\mathrm{sum\_a}_{m_A,\nu_A}
+ \mathrm{sum\_b}_{m_A,\nu_A}\right).$$

This matters because it clarifies that the modulation sweep is not a
second TD model. It is only a refactoring of the final aggregation.

### Passage 7: idealized TD asymptote

In the flat-profile constant-dispersion SMF limit,

$$\mathcal{N}^{(\mathrm{SI})}_{ij}=(L B)^2F_{\mathrm{S1}}(x_{ij}),$$

and since $S_0=1$, the SMF modulation factor is

$$\Theta(\mu_0)=5\mu_0-4.$$

The pairwise TD contribution is then

$$P^{\mathrm{TD}}_{ij}(\mu_0)
=
\frac{16}{9}\frac{P^3\gamma^2}{B^2}(5\mu_0-4)(L B)^2F_{\mathrm{S1}}(x_{ij}).$$

For large walk-off, the stored Nyquist fit behaves like

$$F_{\mathrm{S1}}(x)\sim \frac{c_{\mathrm{pulse}}}{x},$$

so

$$P^{\mathrm{TD}}_{ij}(\mu_0)
\sim
\frac{16}{9}
\frac{P^3\gamma^2}{B^2}
(5\mu_0-4)
\frac{c_{\mathrm{pulse}} L B}{2\pi |\beta_2|\,|f_j-f_i|}.$$

This immediately yields the asymptotic TD scalings:

- fixed spacing: $P^{\mathrm{TD}} \sim 1/B$,
- fixed baud rate: $P^{\mathrm{TD}} \sim 1/|\Delta f|$,
- fixed powers and dispersion: $P^{\mathrm{TD}} \sim L$.

## 3. PCFM Noise Calculation

### Passage 1: start from longitudinal power profiles

The PCFM path starts from channel power profiles $P_i(z)$ rather than
from collision coefficients. In flat-profile mode,

$$P_i(z)=P_{i,\mathrm{launch}},$$

so the normalized signal-power profile is simply

$$p_i(z)=1.$$

With Raman profiles, the normalized profile used in the PCFM kernels is

$$p_i(z)=\frac{P_i(z)}{P_i(0)}.$$

This is the central PCFM input because the model tracks how nonlinear
mixing accumulates along the fiber.

### Passage 2: fit the normalized profile with a polynomial

The implementation approximates the normalized profile on
$u=z/L\in[0,1]$ as

$$p_i(u)\approx \sum_{n=0}^{N} a_{i,n}u^n.$$

This polynomial fit is used to compute the helper term

$$S_i = \sum_{n,k}\frac{a_{i,n}a_{i,k}}{n+k+1}.$$

For a flat profile, $S_i=1$.

The endpoint value $p_i(L)$ is also computed, but the current
implementation does not multiply it into the final SCI/XCI PSD formula.

### Passage 3: build the phase kernel

PCFM turns the profile into nonlinear noise through a dispersive phase
integral

$$\phi(f_1,f_2,z)=4\pi^2\,\beta_{2,\mathrm{eff}}\,f_1f_2z.$$

If the higher-order dispersion data are available, the effective
dispersion is

$$\beta_{2,\mathrm{eff}}(f_m,f_k)
=
\beta_2
+ \pi\beta_3(f_m+f_k-2f_c)
+ \frac{2}{3}\pi^2\beta_4
\left[(f_m-f_c)^2 + (f_m-f_c)(f_k-f_c) + (f_k-f_c)^2\right].$$

Otherwise the code falls back to sampled channel $\beta_2$ values.

This is the main conceptual split from TD:

- TD is organized around walk-off and collision curves,
- PCFM is organized around longitudinal profiles and phase kernels.

### Passage 4: evaluate SCI and XCI kernels

The numerical SCI kernel for the CUT $i$ is

$$K_{\mathrm{SCI},i}
=
\int_{-B/2}^{B/2}\!\!\int_{-B/2}^{B/2}
\left|
\int_0^L p_i(z)e^{j\phi(f_1,f_2,z)}\,dz
\right|^2
df_2\,df_1.$$

The numerical XCI kernel between CUT $i$ and interferer $j$ is

$$K_{\mathrm{XCI},ij}
=
\int_{-B/2}^{B/2}\!\!\int_{\Delta f_{ij}-B/2}^{\Delta f_{ij}+B/2}
\left|
\int_0^L p_j(z)e^{j\phi(f_1,f_2,z)}\,dz
\right|^2
df_2\,df_1.$$

In the closed-form XCI branch this is replaced by

$$K_{\mathrm{XCI},ij}
\approx
\frac{L}{2\pi\max(|\beta_{2,\mathrm{eff}}|,10^{-30})}
\ln\!\left(\frac{|\Delta f_{ij}|+B/2}{|\Delta f_{ij}|-B/2}\right)S_j.$$

Only interferers with $|\Delta f_{ij}|>B/2$ enter this XCI sum.

### Passage 5: convert kernels into PSD

The current implementation uses the launch-power normalization

$$g_i = \frac{2P_i}{B},$$

which is a repository-specific convention in the present PCFM path.

The PSD contributions are then

$$G_{\mathrm{SCI},i}
= \frac{16}{27}g_i^3\gamma_i^2K_{\mathrm{SCI},i},$$

$$G_{\mathrm{XCI},ij}
= \frac{32}{27}g_ig_j^2\gamma_{ij}^2K_{\mathrm{XCI},ij},$$

$$G_{\mathrm{NLI},i}
= G_{\mathrm{SCI},i} + \sum_{j\neq i}G_{\mathrm{XCI},ij}.$$

The code then applies an internal `/2` in the XCI PSD term and the
final output conversion

$$P^{\mathrm{model}}_{\mathrm{NLI},i}
= \frac{G_{\mathrm{NLI},i}B}{2}.$$

Combining those conventions, the flat-profile pairwise XCI power under
equal launch powers is

$$P^{\mathrm{PCFM,XCI}}_{ij}
= \frac{64}{27}\frac{P^3\gamma_{ij}^2}{B^2}K_{\mathrm{XCI},ij}.$$

### Passage 6: idealized PCFM asymptote

In the flat-profile constant-dispersion limit,

$$K_{\mathrm{XCI},ij}
=
\frac{L}{2\pi |\beta_2|}
\ln\!\left(
\frac{|f_j-f_i|+B/2}{|f_j-f_i|-B/2}
\right).$$

For $|\Delta f_{ij}|\gg B/2$,

$$\ln\!\left(
\frac{|\Delta f|+B/2}{|\Delta f|-B/2}
\right)
=
\frac{B}{|\Delta f|}
+ \frac{B^3}{12|\Delta f|^3}
+ \frac{B^5}{80|\Delta f|^5}
+ \cdots,$$

so to first order

$$K_{\mathrm{XCI},ij}
\sim
\frac{L B}{2\pi |\beta_2|\,|\Delta f_{ij}|},$$

and

$$P^{\mathrm{PCFM,XCI}}_{ij}
\sim
\frac{64}{27}
\frac{P^3\gamma^2 L}{2\pi |\beta_2|\,B\,|\Delta f_{ij}|}.$$

Therefore PCFM-XCI has the same leading exponents as TD in the
large-walk-off regime:

- fixed spacing: $P^{\mathrm{PCFM,XCI}} \sim 1/B$,
- fixed baud rate: $P^{\mathrm{PCFM,XCI}} \sim 1/|\Delta f|$,
- fixed powers and dispersion: $P^{\mathrm{PCFM,XCI}} \sim L$.

## 4. TD Versus PCFM

### Passage 1: put both models in the same idealized regime

To compare the two models cleanly, impose:

- single spatial mode,
- constant $\beta_2$,
- constant $A_{\mathrm{eff}}$,
- flat profile $p(z)\equiv 1$,
- equal launch powers $P_i=P_j=P$,
- XCI-only comparison.

Write

$$\Delta f_{ij}=r_{ij}B, \qquad r_{ij}>\frac{1}{2},$$

so that

$$x_{ij}=2\pi |\beta_2| L B^2 r_{ij}.$$

Under these assumptions the TD pair contribution is

$$P^{\mathrm{TD}}_{ij}(\mu_0)
=
\frac{16}{9}\frac{P^3\gamma^2}{B^2}(5\mu_0-4)(L B)^2F_{\mathrm{S1}}(x_{ij}),$$

while the PCFM pair contribution is

$$P^{\mathrm{PCFM,XCI}}_{ij}
=
\frac{64}{27}\frac{P^3\gamma^2}{B^2}(L B)^2
\frac{r_{ij}}{x_{ij}}
\ln\!\left(\frac{r_{ij}+1/2}{r_{ij}-1/2}\right).$$

### Passage 2: form the ratio

Taking the ratio eliminates the common cubic power dependence, the
common Kerr scale, and the common dimensional prefactor:

$$\frac{P^{\mathrm{PCFM,XCI}}_{ij}}{P^{\mathrm{TD}}_{ij}(\mu_0)}
=
\frac{4}{3(5\mu_0-4)}
\frac{
r_{ij}\ln\!\left(\frac{r_{ij}+1/2}{r_{ij}-1/2}\right)
}{
x_{ij}F_{\mathrm{S1}}(x_{ij})
}.$$

This is the cleanest implementation-level comparison because it makes
the agreement and disagreement explicit:

- both methods have the same leading dependence on $P$, $L$, $B$, and
  $|\Delta f|$,
- the difference is in the dimensionless prefactor,
- TD keeps an explicit modulation factor through $\mu_0$,
- the current PCFM-XCI branch does not.

### Passage 3: asymptotic comparison

If the TD stage-S1 fit satisfies

$$F_{\mathrm{S1}}(x)\sim \frac{c_{\mathrm{pulse}}}{x},$$

then

$$\frac{P^{\mathrm{PCFM,XCI}}_{ij}}{P^{\mathrm{TD}}_{ij}(\mu_0)}
\sim
\frac{4}{3(5\mu_0-4)c_{\mathrm{pulse}}}
r_{ij}\ln\!\left(\frac{r_{ij}+1/2}{r_{ij}-1/2}\right).$$

As $r\to\infty$,

$$r\ln\!\left(\frac{r+1/2}{r-1/2}\right)\to 1,$$

so

$$\frac{P^{\mathrm{PCFM,XCI}}}{P^{\mathrm{TD}}}
\to
\frac{4}{3(5\mu_0-4)c_{\mathrm{pulse}}}.$$

With the current Nyquist fit $c_{\mathrm{pulse}}\approx 1$, this gives
approximately

- Gaussian: `PCFM / TD ≈ 0.22`,
- 64-QAM: `PCFM / TD ≈ 0.46`.

So under the present implementation, the asymptotic idealized trend is

$$\mathrm{TD} > \mathrm{PCFM\text{-}XCI}.$$

### Passage 4: why care is still needed outside the idealized limit

That ratio is only fully clean in the matched idealized regime. Outside
it, the current implementation contains several asymmetries:

- TD uses the CUT-channel Kerr coefficient in the current aggregation,
  while PCFM-XCI uses the mixed coefficient $\gamma_{ij}$.
- TD may be run with `exclude_self_channel=True`, which turns the result
  into an XCI-like diagnostic rather than full TD NLIN.
- PCFM computes $p_i(L)$ but does not currently include it in the final
  PSD formula.
- PCFM, GN-numeric, and GN-direct express their polarization
  normalization through a final divide-by-2 conversion that is not
  written the same way in the TD path.

Because of these choices, absolute ratios should only be interpreted
after the compared quantities have been normalized in the same way.

## 5. Final Reported Quantities

The models above return nonlinear noise powers. The workflow then forms
GSNR-like observables using the resolved signal power at the output:

$$\mathrm{GSNR}_i
=
10\log_{10}\!\left(
\frac{P_{\mathrm{sig},i}(L)}
{\max(P_{\mathrm{NLI},i},10^{-18})}
\right).$$

CSV exports also include the corresponding noise-to-signal ratio,
$\mathrm{NSR}_{\mathrm{NLI},i}=10\log_{10}(P_{\mathrm{NLI},i}/P_{\mathrm{sig},i}(L))$,
so the GSNR column is the sign-reversed dB quantity.

For the specific TD-versus-PCFM(XCI) diagnostic, the workflow reports

$$\Delta_i = P^{\mathrm{TD,Gauss}}_{\mathrm{NLI},i}
- P^{\mathrm{PCFM,XCI}}_{\mathrm{NLI},i},$$

$$r_i = \frac{P^{\mathrm{TD,Gauss}}_{\mathrm{NLI},i}}
{\max(P^{\mathrm{PCFM,XCI}}_{\mathrm{NLI},i},10^{-30})},$$

and

$$\Delta_i^{\mathrm{dB}}
=
10\log_{10}\!\big(\max(P^{\mathrm{TD,Gauss}}_{\mathrm{NLI},i},10^{-30})\big)
-
10\log_{10}\!\big(\max(P^{\mathrm{PCFM,XCI}}_{\mathrm{NLI},i},10^{-30})\big).$$

These quantities do not define new physics. They are summary diagnostics
built on top of the TD and PCFM derivations above.
