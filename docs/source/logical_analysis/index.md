# Logical Specification of the Lorenzi Fast Analysis

```{toctree}
:hidden:

PROBLEMS
```

## 0. Purpose, scope, and authority

This document pins down the logical chain implemented by the Lorenzi Fast
S0--S6 analysis. Its purpose is not to advertise the method or to collect all
historical observations. Its purpose is to make every reported quantity
traceable to a minimal set of definitions and to state which equations are
actually evaluated by each stage and figure.

The present scope is deliberately narrow:

- the scripts `analysis/fwm/fast_s0_*.py` through `fast_s6_physical.py`;
- the numerical core in `pynlin.methods.td.fast_nlin` and
  `pynlin.methods.td.fast_analytic`;
- the nine figures embedded in `lorenzi_fast_method.md`;
- the flat-power, single-span, rectangular-spectrum model used by those
  scripts.

This is an implementation specification, not a claim of experimental
validity. Known failures of the logical or numerical contract are collected in
[PROBLEMS](PROBLEMS.md). If this specification and an older narrative document
disagree about the active code, this specification plus the cited source code
is the intended description. A disagreement must still be resolved rather
than hidden.

### 0.1 Epistemic labels

Every load-bearing relation has one of the following statuses.

| Label | Meaning |
|---|---|
| **Definition** | Introduces notation; true by convention. |
| **Identity** | Follows exactly from earlier definitions. |
| **Bound** | A proved one-sided inequality under stated assumptions. |
| **Approximation** | Replaces a defined quantity and requires a validity condition. |
| **Estimator** | A numerical random or deterministic estimate of a defined quantity. |
| **Empirical result** | A result measured on a finite dataset; not a theorem. |
| **Provisional** | Used by the current pipeline although a known problem remains. |

An expression must not move between these categories without an explicit
argument.

## 1. Causal spine

The entire calculation is organized by the following dependency chain:

$$
\boxed{
\text{fiber and channel plan}
\longrightarrow
\text{admissible tuple}
\longrightarrow
\Delta\beta(\mathbf x)
\longrightarrow
K(L\Delta\beta)M(\mathbf x)
\longrightarrow
F_\tau
\longrightarrow
S_t
\longrightarrow
\sigma_{\mathrm{NLIN},t}^2
}
\tag{1.1}
$$

Here $\tau$ denotes one XPM pair or one strict-FWM tuple. Each arrow is
defined below. The central object is the dimensionless per-interaction
efficiency $F_\tau$. Fast quadrature, closed forms, QMC, MC, refinement, and
the S3 branches are different ways of estimating the same $F_\tau$ at
different model levels. They are not different physical observables.

## 2. Primitive definitions

The minimal input set is

$$
\mathcal D_0=
\left\{
L,\ B,\ \{f_j\}_{j=1}^{N_{\rm ch}},\ \beta(\omega),\
\gamma(f),\ \{P_j\},\ \{\mathcal S_j\}
\right\}.
\tag{2.1}
$$

**Definition.** The primitives are:

| Symbol | Meaning | Units |
|---|---|---|
| $L$ | span length | m |
| $B$ | symbol rate and rectangular channel bandwidth | Hz |
| $N_{\rm ch}$ | number of WDM channels | 1 |
| $f_j$ | center frequency of channel $j$ | Hz |
| $\beta(\omega)$ | propagation constant | m$^{-1}$ |
| $\gamma(f)$ | scalar nonlinear coefficient used by the physical layer | W$^{-1}$m$^{-1}$ |
| $P_j$ | launch power of channel $j$ under the scalar convention | W |
| $\mathcal S_j$ | in-channel spectral support | rad s$^{-1}$ |

The active Fast analysis assumes

$$
\mathcal S_j=\{\omega_j+B x:-\pi<x<\pi\},
\qquad \omega_j=2\pi f_j,
\qquad T=B^{-1}.
\tag{2.2}
$$

The normalized offset $x$ is dimensionless. Equation (2.2) is a rectangular,
Nyquist-width spectral model; it is not a statement about arbitrary pulse
shapes.

The local dispersion coefficients are derived, not primitive:

$$
\beta_k^{(j)}
\equiv
\left.\frac{d^k\beta}{d\omega^k}\right|_{\omega_j}.
\tag{2.3}
$$

## 3. Interaction sets and support

### 3.1 Strict FWM tuples

For target channel $t$, define the ordered strict-FWM set

$$
\mathcal T_t^{\rm FWM}
=
\left\{(a,b,c):a,b,c,t\ \text{are pairwise distinct and } |d_{abc\to t}|<4\pi\right\},
\tag{3.1}
$$

with support shift

$$
d_{abc\to t}
\equiv
\frac{2\pi(f_a+f_b-f_c-f_t)}{B}.
\tag{3.2}
$$

The code enumerates ordered $(a,b)$ pairs. This fact later changes the
physical FWM multiplicity.

For one tuple, sample independent in-channel coordinates

$$
x_a,x_b,x_c\overset{\rm iid}{\sim}\mathcal U(-\pi,\pi),
\qquad
x_d=x_a+x_b-x_c+d.
\tag{3.3}
$$

The exact output-support indicator is

$$
M_d(\mathbf x)
\equiv
\mathbf 1_{\{|x_d|<\pi\}}.
\tag{3.4}
$$

The unconditional acceptance is therefore

$$
A(d)
\equiv
\mathbb E[M_d]
=\Phi_3(\pi-d)-\Phi_3(-\pi-d),
\tag{3.5}
$$

where $\Phi_3$ is the CDF of a sum of three independent
$\mathcal U(-\pi,\pi)$ variables. Exact consequences are

$$
A(0)=\frac23,
\qquad
0\le A(d)\le\frac23,
\qquad
A(d)=0\quad\text{for }|d|\ge4\pi.
\tag{3.6}
$$

Thus the support cut in (3.1) is lossless for this spectral model.

### 3.2 XPM pairs

The XPM interaction set is

$$
\mathcal T_t^{\rm XPM}=\{(t,b):b\ne t\}.
\tag{3.7}
$$

Its support mask is introduced explicitly in Section 8. It is not obtained by
calling a strict-FWM tuple an XPM tuple.

## 4. Phase mismatch and model levels

The exact accumulated phase mismatch of a tuple is

$$
u(\mathbf x)
\equiv
L\left[
\beta_a(\omega_a+Bx_a)
+\beta_b(\omega_b+Bx_b)
-\beta_c(\omega_c+Bx_c)
-\beta_t(\omega_t+Bx_d)
\right].
\tag{4.1}
$$

The implementation works in the target group-delay frame. With
$\beta_{1,j}^{(t)}=\beta_1^{(j)}-\beta_1^{(t)}$, define

$$
u_0=L\Delta\beta_{\rm center}^{(t)},
\qquad
\nu_j=LB\beta_{1,j}^{(t)},
\qquad
q_j=\frac12 LB^2\beta_2^{(j)}.
\tag{4.2}
$$

The target-frame expression for $u_0$ is the one evaluated by
`fwm_tuple_variables`; it subtracts the target linear phase from the sampled
global propagation constants.

The active code distinguishes two relevant model levels.

**Definition, linear model:**

$$
u_{\rm lin}(\mathbf x)
=u_0+\nu_a x_a+\nu_bx_b-\nu_cx_c.
\tag{4.3}
$$

**Definition, local-quadratic model:**

$$
u_{\rm quad}(\mathbf x)
=u_{\rm lin}(\mathbf x)
+q_ax_a^2+q_bx_b^2-q_cx_c^2-q_tx_d^2.
\tag{4.4}
$$

The production Fast FWM estimator primarily evaluates (4.3). QMC validation
can evaluate either (4.3) or (4.4). Neither equation includes arbitrary
higher-order in-channel dispersion.

Auxiliary scales are admitted because they simplify model selection and have
direct physical meaning:

$$
w_j\equiv\pi|\nu_j|,
\qquad
W\equiv w_a+w_b+w_c,
\qquad
x_\nabla\equiv\sqrt{\nu_a^2+\nu_b^2+\nu_c^2},
\qquad
\mu\equiv\frac{u_0}{x_\nabla}.
\tag{4.5}
$$

$W$ is the half-width of the linear mismatch range over the unmasked cube:

$$
u_{\rm lin}(\mathbf x)\in[u_0-W,u_0+W].
\tag{4.6}
$$

$x_\nabla$ is the Euclidean walk-off scale, while $\mu$ separates center
detuning from that scale. If $x_\nabla=0$, $\mu$ is mathematically undefined;
the current code's zero assignment is recorded as a problem.

The maximum magnitude of the omitted quadratic correction under the mask is
bounded by

$$
P_q
\equiv
\pi^2\left(|q_a|+|q_b|+|q_c|+|q_t|\right).
\tag{4.7}
$$

## 5. Master observable

The lossless normalized link kernel is

$$
K(u)
\equiv
\left|\frac1L\int_0^L e^{iu z/L}\,dz\right|^2
=\frac{4\sin^2(u/2)}{u^2},
\qquad K(0)=1.
\tag{5.1}
$$

It obeys

$$
0\le K(u)\le \min\left(1,\frac4{u^2}\right).
\tag{5.2}
$$

**Definition.** At model level $r\in\{\mathrm{lin},\mathrm{quad}\}$, the
dimensionless efficiency of FWM tuple $\tau$ is

$$
F_\tau^{(r)}
\equiv
\mathbb E_{\mathbf x}
\left[K\!\left(u_r(\mathbf x)\right)M_d(\mathbf x)\right].
\tag{5.3}
$$

This is the normalized collision sum previously written as
$\mathcal N T^2/L^2$. To prevent collision with the channel count, this
document uses

$$
F\equiv\frac{\mathcal N T^2}{L^2},
\qquad N_{\rm ch}\equiv\text{channel count}.
\tag{5.4}
$$

Direct consequences are

$$
0\le F_\tau^{(r)}\le A(d)\le\frac23.
\tag{5.5}
$$

The dimensional, prefactor-free quantity saved by S4/S5 is

$$
\mathcal K_\tau^{(r)}\equiv L^2F_\tau^{(r)},
\tag{5.6}
$$

with units m$^2$.

## 6. Exact linear representation and mask dependence

Let

$$
v=\nu_ax_a+\nu_bx_b-\nu_cx_c,
\qquad
\rho_{\mathbf w}(v)=\text{density of }v.
\tag{6.1}
$$

The density is the inclusion-exclusion density of a sum of three centered
uniform variables. Without the support mask,

$$
\mathbb E[K(u_0+v)]
=\int_{-W}^{W}K(u_0+v)\rho_{\mathbf w}(v)\,dv
=2\int_0^1(1-t)\cos(u_0t)
\prod_j\operatorname{sinc}(w_jt)\,dt.
\tag{6.2}
$$

The mask and mismatch are functions of the same $\mathbf x$. Therefore the
masked quantity is, exactly,

$$
F_\tau^{(\rm lin)}
=\int_{-W}^{W}
K(u_0+v)\rho_{\mathbf w}(v)A(v;d,\boldsymbol\nu)\,dv,
\tag{6.3}
$$

where

$$
A(v;d,\boldsymbol\nu)
\equiv
P\!\left(M_d=1\mid
\nu_ax_a+\nu_bx_b-\nu_cx_c=v\right).
\tag{6.4}
$$

Replacing (6.4) by the marginal $A(d)$ is an approximation unless independence
has been established. `exact_conditional_acceptance` evaluates (6.4) through
the exact joint density of mismatch and mask coordinate. The bulk model
`pointwise_conditional_acceptance` is an empirical shape approximation.

## 7. Fast FWM estimators

### 7.1 Regime dispatch

`linear_tuple_estimate` partitions tuples as

$$
\begin{aligned}
\mathcal R_{\rm far}&:\ |u_0|>3W+3000,\\
\mathcal R_{\rm wide}&:\ W>3000\ \text{and not far},\\
\mathcal R_{\rm near}&:\ \text{otherwise}.
\end{aligned}
\tag{7.1}
$$

These thresholds are numerical dispatch choices, not physical phase
boundaries. The separate line $|u_0|=W$ is only the boundary for zero to be
reachable over the unmasked linear cube.

Near tuples evaluate (6.3) by non-negative $v$-space quadrature. The bulk pass
uses approximate conditional acceptance; selected tuples are re-evaluated
with exact conditional acceptance.

Wide tuples split the integral at $U=48\pi$:

$$
F_{\rm wide}\approx
\int_{|u|<U}K(u)\rho(u-u_0)A(u-u_0)\,du
+\int_{|u|\ge U}\frac{2}{u^2}\rho(u-u_0)A(d)\,du.
\tag{7.2}
$$

The first term resolves the kernel; the second replaces its oscillations by
their mean and uses marginal acceptance.

The current far estimator is

$$
\widehat F_{\rm far}
=2A(d)\,\widehat{\mathbb E[u^{-2}]}
\left(1-\mathbb E[\cos u]\right),
\tag{7.3}
$$

with

$$
\widehat{\mathbb E[u^{-2}]}
=\frac1{u_0^2}\left(1+\frac{3\sigma^2}{u_0^2}\right),
\quad
\sigma^2=\frac13\sum_jw_j^2,
\quad
\mathbb E[\cos u]=\cos u_0\prod_j\operatorname{sinc}(w_j).
\tag{7.4}
$$

Equation (7.3) is an approximation. It factorizes two correlated functions of
$u$ and must not be described as the exact second-order expansion of the full
expectation; see Problem P6.

### 7.2 Refinement

`target_fast_sums` first computes a bulk value for every support-surviving
tuple. It then replaces selected values by deterministic exact-acceptance
linear-model quadrature. The selected set contains:

- up to 16,384 top-ranked near tuples;
- up to `n_refine` additional top-ranked tuples with $W\le300$.

This removes acceptance-model error for the selected tuples. It does not add
the quadratic phase in (4.4), and ranking by the approximate value is not a
certificate that every important tuple was selected.

### 7.3 Envelope bound and S3

For the linear model, if

$$
g_{\rm lin}=|u_0|-W>0,
\tag{7.5}
$$

then every point in the cube has $|u|\ge g_{\rm lin}$ and

$$
F_\tau^{(\rm lin)}
\le A(d)\min\left(1,\frac4{g_{\rm lin}^2}\right).
\tag{7.6}
$$

For the local-quadratic model, the corresponding safe gap is

$$
g_{\rm quad}=|u_0|-W-P_q.
\tag{7.7}
$$

The current `fast_analytic.envelope_bound` has a linear-width API, but
`select_tube` passes $W+P_q$ as its effective width. Thus the discarded-set
bound includes the quadratic confinement padding. Existing tests exercise the
discarded bound against linear tuple values, and the retained-tuple evaluator
still uses the linear phase model; a complete local-quadratic result is not yet
certified. S3 v1 also performs exhaustive support enumeration before pruning;
it is post-enumeration value pruning, not yet direct geometric tube
enumeration.

For retained wide, crossing tuples, the S3 sheet approximation is

$$
\widehat F_{\rm sheet}
=2\pi\rho_{\mathbf w}(-u_0)\,
A(-u_0;d,\boldsymbol\nu),
\tag{7.8}
$$

valid when the density and conditional acceptance vary slowly over the kernel
core. In the code this branch additionally requires $W>2000$, a distance of
at least 200 rad from the unmasked support edge, and conditional acceptance at
the phase-matched point of at least 0.05.

## 8. XPM specialization

For XPM pair $(t,b)$ define

$$
\nu_{tb}=LB\left(\beta_1^{(b)}-\beta_1^{(t)}\right).
\tag{8.1}
$$

In the active linear XPM model, let $y=x_1-x_2$. Integrating over the target
input coordinate gives the masked measure

$$
p_M(y)=\frac{(2\pi-|y|)^2}{(2\pi)^3},
\qquad |y|<2\pi.
\tag{8.2}
$$

Its cosine transform is

$$
H(\theta)
=\int_{-2\pi}^{2\pi}p_M(y)\cos(\theta y)\,dy
=\frac1{\pi^2\theta^2}
-\frac{\sin(2\pi\theta)}{2\pi^3\theta^3},
\qquad H(0)=\frac23.
\tag{8.3}
$$

Therefore the active Fast XPM efficiency is exactly, at the linear model
level,

$$
F_{tb}^{(\rm XPM,lin)}
=2\int_0^1(1-t)H(\nu_{tb}t)\,dt,
\tag{8.4}
$$

and

$$
F_{tb}^{(\rm XPM,lin)}\sim\frac1{|\nu_{tb}|}
\quad\text{as }|\nu_{tb}|\to\infty.
\tag{8.5}
$$

The local-quadratic XPM phase used by QMC is

$$
u_{\rm XPM,quad}
=\nu(x_1-x_2)
+q_t(x_{\rm out}^2-x_{\rm in}^2)
+q_b(x_1^2-x_2^2).
\tag{8.6}
$$

At fixed spacing-to-baud ratio $q=|f_b-f_t|/B$, the quadratic terms can alter
the leading asymptotic constant. The current production calculation uses
(8.4), so physical XPM results remain provisional; see Problem P1.

## 9. Per-target sums and physical conversion

Define dimensionless per-target sums

$$
S_t^{\rm XPM}=\sum_{b\ne t}F_{tb}^{\rm XPM},
\qquad
S_t^{\rm FWM}=\sum_{(a,b,c)\in\mathcal T_t^{\rm FWM}}F_{abc\to t}.
\tag{9.1}
$$

S5 stores

$$
K_t^{\rm XPM}=L^2S_t^{\rm XPM},
\qquad
K_t^{\rm FWM}=L^2S_t^{\rm FWM},
\tag{9.2}
$$

in m$^2$.

Under the current S6 assumptions of equal scalar launch power $P$, Gaussian
symbols, flat longitudinal power, and the scalar SSFM coefficient convention,
the code applies

$$
\sigma_{\rm XPM,t}^2
=4\gamma_t^2P^3K_t^{\rm XPM},
\qquad
\sigma_{\rm FWM,t}^2
=2\gamma_t^2P^3K_t^{\rm FWM}.
\tag{9.3}
$$

The FWM factor is 2 because the tuple sum contains both ordered $(a,b)$ and
$(b,a)$ entries. The total and noise-to-signal ratio are

$$
\sigma_{\rm NLIN,t}^2
=\sigma_{\rm XPM,t}^2+\sigma_{\rm FWM,t}^2,
\qquad
\mathrm{NSR}_t
=\frac{\sigma_{\rm NLIN,t}^2}{P},
\qquad
\mathrm{NSR}_{t,\rm dB}=10\log_{10}\mathrm{NSR}_t.
\tag{9.4}
$$

Hence a necessary invariant is

$$
\sigma_{\rm NLIN}^2\propto P^3,
\qquad
\mathrm{NSR}\propto P^2.
\tag{9.5}
$$

The present API names do not state whether $P$ and $\sigma^2$ are
single-polarization quantities. Until that convention is made explicit and
tested, cross-model absolute comparisons require care.

## 10. Stage contracts

| Stage | Defined input | Calculated output | Model status |
|---|---|---|---|
| S0 | Full channel grid and selected targets | Tuple coordinates, bulk $F^{(\rm lin)}$, weighted histograms | Census of the approximate bulk model |
| S1 | Selected real tuples | Plain-MC and randomized-Sobol estimates of the same integrand | Implementation cross-check |
| S2 | Nonuniform tuple sample | Bulk, linear-QMC, quadratic-QMC comparisons | Diagnostic; current aggregate weighting is not valid |
| S3 v1 | Exhaustively enumerated tuples, $\varepsilon$ | Retained linear-model sum and discarded linear bound | Post-enumeration linear certificate |
| S4 | Probe targets | Fast and MC sums in m$^2$ | No propagated MC uncertainty in current output |
| S5 | Full interferer grid, possibly thinned targets | $K_t^{\rm XPM}$, $K_t^{\rm FWM}$ | Production prefactor-free spectrum; checkpoint provenance incomplete |
| S6 | S5 sums, $\gamma_t$, equal $P$ | Physical variances and NSR | Provisional because of XPM and convention issues |

Target decimation in the current S5 script thins only the reported target list.
Interferers remain on the full grid. Historical S5 files produced by
interferer decimation describe a different physical channel plan and must not
be combined with current outputs.

## 11. Figure ledger

This section defines every panel embedded in `lorenzi_fast_method.md`.
Rendering choices are included where they affect interpretation.

### Figure 1: mismatch density and dispatch

![Mismatch density and numerical regime dispatch](../_static/lorenzi-fast/density_regimes.png)

Panel (a) plots $\rho_{\mathbf w}(v)$ from (6.1) for three fixed width
vectors. Panel (b) plots the indicator regions in (7.1) in the
$(W,|u_0|)$ plane. The dotted $|u_0|=W$ line is the unmasked box-crossing
boundary, not a dispatch boundary. Both axes in panel (b) are logarithmic.

Source: `analysis/fwm/plot_fast_theory_figures.py::fig_density_regimes`.

### Figure 2: unconditional support acceptance

![Unconditional output-support acceptance](../_static/lorenzi-fast/support_acceptance.png)

The curve is exactly $A(d)$ from (3.5). Vertical lines mark $|d|=4\pi$.
There is no numerical estimator or uncertainty in this figure.

Source: `plot_fast_theory_figures.py::fig_support_acceptance`.

### Figure 3: conditional acceptance

![Exact and approximate conditional acceptance](../_static/lorenzi-fast/acceptance_exact_vs_approx.png)

Both panels plot (6.4) against $v/W$ at $d=0.4$. Blue is the zonotope-based
conditional law; orange is `pointwise_conditional_acceptance`. The panels do
not plot the efficiency error quoted in the caption. Those numbers come from
separate evaluations of (6.3), including a randomized-Sobol reference.

Source: `plot_fast_theory_figures.py::fig_acceptance`.

### Figure 4: fast model divided by QMC

![Fast linear-model estimates divided by randomized-Sobol references](../_static/lorenzi-fast/fast_vs_qmc.png)

For each retained synthetic tuple $i$,

$$
X_i=F_{i,\rm QMC}^{(\rm lin)},
\qquad
Y_i=\frac{\widehat F_i^{(\rm lin)}}{F_{i,\rm QMC}^{(\rm lin)}}.
\tag{11.1}
$$

The dataset contains 60 seeded synthetic tuples and retains only references
with relative QMC standard error below 2 percent. The x-axis is logarithmic.
The currently drawn absolute y-error is

$$
\delta Y_{i,\rm drawn}=\frac{\delta F_{i,\rm QMC}}{F_{i,\rm QMC}},
\tag{11.2}
$$

whereas first-order propagation through (11.1) gives

$$
\delta Y_i\approx
Y_i\frac{\delta F_{i,\rm QMC}}{F_{i,\rm QMC}}.
\tag{11.3}
$$

The missing factor is Problem P13.

Source: `plot_fast_theory_figures.py::fig_fast_vs_qmc`.

### Figure 5: kernel envelope

![Lossless link kernel, envelope, and an example reachable mismatch set](../_static/lorenzi-fast/kernel_envelope.png)

The blue curve is $K(u)$ from (5.1); the dashed curve is its envelope (5.2).
The shaded example uses $u_0=38$, $W=12$, hence $g=26$. It displays the
kernel-level ceiling $4/g^2$. The full tuple bound also contains $A(d)$ and
is not itself plotted.

Source: `plot_fast_theory_figures.py::fig_kernel_envelope`.

### Figure 6: historical decimation warning

![Historical S5 output produced by obsolete interferer-decimation semantics](../_static/lorenzi-fast/s5_fullband_dec4.png)

The top panel plots historical arrays

$$
K_t^{\rm XPM},\qquad K_t^{\rm FWM},\qquad
K_t^{\rm XPM}+K_t^{\rm FWM}
\tag{11.4}
$$

on a logarithmic axis. The lower panel plots fast/MC ratios for each
component. The file was generated with the obsolete rule that decimated the
interferer grid to 571 channels. It is a historical counterexample showing
that interferer decimation changes the physics; the current S5 script cannot
regenerate it under current semantics.

Source data: `media/lorenzi-fast/s5_fullband_dec4.npz`.

### Figure 7: S0 territory

![S0 strict-FWM tuple territory by count and per-target-normalized bulk mass](../_static/lorenzi-fast/s0_territory.png)

For logarithmic bins $I_p,J_q$, define the count histogram

$$
C_{pq}=\sum_i
\mathbf 1_{\{X_i\in I_p\}}
\mathbf 1_{\{Y_i\in J_q\}}.
\tag{11.5}
$$

For target $t$, define the normalized bulk weight

$$
\bar F_i
=\frac{\widehat F_i^{(\rm lin,bulk)}}
{\sum_{k:\,t(k)=t}\widehat F_k^{(\rm lin,bulk)}}.
\tag{11.6}
$$

The mass histogram is

$$
M_{pq}=\sum_i\bar F_i
\mathbf 1_{\{X_i\in I_p\}}
\mathbf 1_{\{Y_i\in J_q\}}.
\tag{11.7}
$$

Top panels use $(X,Y)=(|u_0|,W)$; bottom panels use
$(X,Y)=(x_\nabla,|\mu|)$. Displayed color is
$\log_{10}\max(C_{pq},10^{-300})$ or
$\log_{10}\max(M_{pq},10^{-300})$. Values below $10^{-6}$ on an axis are
clipped into its first bin. The current source uses raw bins without
smoothing. Per-target normalization removes absolute target-to-target
loudness.

Source: `analysis/fwm/fast_s0_territory.py`.

### Figure 8: S0 factorization diagnostic

![S0 population, synthetic kernel, real mass, and predicted mass](../_static/lorenzi-fast/s0_factorization.png)

The synthetic kernel is

$$
F_{\rm syn}(x,\mu)
=\widehat F^{(\rm lin)}
\left(u_0=\mu x,
\boldsymbol\nu=\frac{x}{\sqrt3}(1,1,-1),
d=0\right).
\tag{11.8}
$$

For each real tuple, the script constructs

$$
F_{i,\rm pred}
=F_{\rm syn}(x_{\nabla,i},|\mu_i|)
\frac{A(d_i)}{A(0)},
\tag{11.9}
$$

normalizes (11.9) separately within each target, and then forms (11.7).
Panel (d) is therefore not a pixelwise multiplication of panels (a) and (b):
the acceptance correction and normalization are applied per tuple before
binning. The real and predicted panels also use independent six-decade color
ranges. The fixed dashed line $|\mu|=\pi\sqrt3$ is the $|u_0|=W$ identity only
for the equal-split synthetic direction, not for every real tuple.

Equation (11.9) is an empirical diagnostic, not an exact factorization; its
use of marginal acceptance conflicts with known conditional-acceptance
effects. See Problem P3.

Source: `analysis/fwm/fast_s0_factorization.py`.

### Figure 9: XPM reduction

![Exact linear-model XPM transform and one-dimensional efficiency](../_static/lorenzi-fast/xpm_reduction.png)

Panel (a) plots $H(\theta)$ from (8.3) and the guide
$1/(\pi\theta)^2$. Panel (b) plots (8.4) and the linear-model asymptote (8.5).
Both are deterministic one-dimensional calculations. The word "exact" in
this context means exact within the linear XPM phase model, not exact relative
to (8.6).

Source: `plot_fast_theory_figures.py::fig_xpm`.

## 12. Additional generated plots not embedded in the theory note

The stage scripts also generate plots whose equations follow directly from
the stage contracts:

- `s2_collapse.png` plots $\widehat F/F_{\rm lin}^{\rm QMC}$,
  $|\widehat F-F_{\rm lin}^{\rm QMC}|/F_{\rm lin}^{\rm QMC}$, and
  $|F_{\rm quad}^{\rm QMC}-F_{\rm lin}^{\rm QMC}|/F_{\rm lin}^{\rm QMC}$.
- `s3_tube_gate.png` plots retained/reference FWM sum, survivor fraction, and
  discarded linear bound divided by retained sum versus $\varepsilon$.
- `s4_targets.png` plots (9.2) for Fast and MC and their ratios. It currently
  has no MC error bars.
- current `s5_fullband.png` plots (9.2) and Fast/MC probe ratios on the full
  interferer grid, with target-only thinning if requested.
- `s6_physical.png` plots (9.3) and (9.4).

These definitions should be copied into their captions if those figures are
promoted into the canonical theory document.

## 13. Required invariants and proof obligations

The following checks should hold for every run at the stated model level:

$$
0\le F_\tau\le A(d)\le\frac23,
\tag{13.1}
$$

$$
K_t^{\rm XPM}\ge0,
\qquad
K_t^{\rm FWM}\ge0,
\tag{13.2}
$$

$$
\sigma_{\rm NLIN,t}^2
=\sigma_{\rm XPM,t}^2+\sigma_{\rm FWM,t}^2,
\tag{13.3}
$$

$$
\sigma^2(\lambda P)=\lambda^3\sigma^2(P)
\quad\text{under the equal-power model},
\tag{13.4}
$$

and, for a certified discarded set $D$ at the same model level,

$$
\sum_{\tau\in D}F_\tau
\le\sum_{\tau\in D}B_\tau.
\tag{13.5}
$$

Equation (13.5) is meaningful only when $B_\tau$ bounds the same phase model
used to define $F_\tau$. A linear-model bound does not automatically certify a
quadratic or physical calculation.

## 14. Interpretation checklist

Before drawing a physical conclusion from a Fast result, answer all of the
following:

1. Is the quantity dimensionless $F$, dimensional prefactor-free $K=L^2F$,
   or physical variance $\sigma^2$?
2. Is the phase model linear, local-quadratic, or higher order?
3. Was the output-support mask treated conditionally or by marginal $A(d)$?
4. Were interferers kept on the full physical grid?
5. Is the result bulk, refined, S3-pruned, MC, or QMC?
6. Does a quoted certificate bound the same model as the reported result?
7. Are powers scalar, single-polarization, or total dual-polarization?
8. Is the cache fingerprint sufficient to establish provenance?
9. Are uncertainty bars defined for the transformed plotted quantity?
10. Is a stated trend exact, asymptotic, fitted, or merely observed on a
    finite sweep?

The unresolved cases are indexed in [PROBLEMS](PROBLEMS.md).
