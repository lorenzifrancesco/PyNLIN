# High-DGD asymptotics of the nPC collision sectors

*2026-08-24. Derivation and numerical verification of the asymptotic law for
the XPM collision sectors at large walk-off: **every sector scales as
DGD$^{-1}$**, with sector ratios tending to constants that depend on the pair
geometry through the spacing-to-baud ratio. This is an independent
reconstruction validating the analytic result proved by F. Lorenzi (private
notes; to be merged with this note when transcribed); it supersedes the
fitted $\nu^{\mp 0.3}$ power laws of the original sector-scaling sweep
([`direct_sector_mc.md`](direct_sector_mc.md)), which are shown here to be
pre-asymptotic transients. Companion claim record:
[`publication_novelty.md`](stale/publication_novelty.md), Claim A.*

## 1. Setting and notation

We work with the direct CRN sector estimator
[`estimate_xhkm_sectors_direct_mc`](../../src/pynlin/methods/td/xhkm_mc.py)
in its scalar-$\beta_2$ normalization: one masked link kernel per draw,

$$
K(x; a, u) \;=\; \mathbf 1_{|x+a|\le\pi}\,\mathbf 1_{|u-x|\le\pi}\;
\hat\Lambda\big(\Delta\beta L\big),
\qquad
\hat\Lambda(\delta) = \frac{e^{i\delta}-1}{i\delta}
= \int_0^1 e^{i\delta\sigma}\,d\sigma ,
$$

with target-in frequency $a$, interferer frequency $u$ (both uniform on
$(-\pi,\pi)$), shared outer frequency $x$ (the target in/out difference,
uniform on $(-2\pi,2\pi)$), and phase

$$
\Delta\beta L \;=\; -\,\nu\, x\, g(u - x - a),
\qquad
g(y) = 1 + \frac{y}{\omega},
\qquad
\omega = 2\pi q,
$$

where $q = \Delta f/B$ is the pair's channel-spacing-to-baud ratio and
$\nu = L/L_W = 2\pi q\,\beta_2 L$ (library normalization,
$L$-normalized $\hat\Lambda$) is the Dar collision count — the DGD
accumulated over the span in symbol slots.

**The sectors are an ANOVA decomposition.** Conditional on the shared $x$,
the four sectors are exactly the Sobol–Hoeffding variance components of
$K(x;\cdot,\cdot)$ over the independent inner variables $(a,u)$:

$$
\begin{aligned}
2\mathrm{PC} &= 2\,\mathbb E_x\big[\,|\mathbb E_{a,u}K|^2\,\big], &
3\mathrm{PCa} &= 2\,\mathbb E_x\big[\,\mathbb E_u|\mathbb E_a K|^2 - |\mathbb E_{a,u}K|^2\,\big],\\
3\mathrm{PCb} &= 2\,\mathbb E_x\big[\,\mathbb E_a|\mathbb E_u K|^2 - |\mathbb E_{a,u}K|^2\,\big], &
4\mathrm{PC} &= 2\,\mathbb E_x\big[\,\mathbb E_{a,u}|K|^2 - \mathbb E_u|\mathbb E_a K|^2 - \mathbb E_a|\mathbb E_u K|^2 + |\mathbb E_{a,u}K|^2\,\big],
\end{aligned}
$$

with $N_1 = 2\mathrm{PC} + 3\mathrm{PCa} + 3\mathrm{PCb} + 4\mathrm{PC}
= 2\,\mathbb E_x\mathbb E_{a,u}|K|^2$ (the factor 2 is the estimator's
`_DIRECT_SECTOR_CONST`). The CRN paired projections of the estimator are the
Monte Carlo of precisely these components.

## 2. The scaling limit

Substitute $\xi = \nu x$ and let $\nu \to \infty$ at fixed $\xi$. Two things
happen on the phase-matched strip $|x| \lesssim 1/\nu$:

1. **The masks saturate**: $|x + a| \le \pi \to |a| \le \pi$ and
   $|u - x| \le \pi \to |u| \le \pi$, i.e. both indicators $\to 1$ a.s.
2. **The in-band phase modulation survives**: the phase tends to
   $-\xi\,g(u-a)$ with $g(y) = 1 + y/\omega$ of **order one** — it does *not*
   degenerate to $-\xi$. This is the crux: because the $(a,u)$-dependence of
   the kernel persists in the limit, every interaction sector keeps an
   $O(1)$ share of the strip, and hence inherits its $1/\nu$ scaling.

Since $\mathbb E_x[\,\cdot\,] = \frac{1}{4\pi}\int_{-2\pi}^{2\pi}(\cdot)\,dx
= \frac{1}{4\pi\nu}\int_{-2\pi\nu}^{2\pi\nu}(\cdot)\,d\xi$ and the limiting
integrands are even in $\xi$ with integrable tails (shown below), dominated
convergence gives the

**Limit theorem.** For every sector $S \in \{N_1, 2\mathrm{PC},
3\mathrm{PCa}, 3\mathrm{PCb}, 4\mathrm{PC}\}$,

$$
S(\nu) \;\xrightarrow[\nu\to\infty]{}\; \frac{C_S(q)}{\nu},
\qquad
C_S(q) \;=\; \frac{1}{\pi}\int_0^\infty I_S(\xi;q)\,d\xi ,
$$

where $I_S$ is the corresponding ANOVA component of the limiting kernel
$\hat\Lambda(\xi\,g(u-a))$ over $(a,u)$ uniform on $(-\pi,\pi)^2$.
Consequently **all nPC contributions scale as DGD$^{-1}$**, and the sector
ratios tend to the finite, geometry-dependent constants
$\rho_S^\infty(q) = C_S(q)/C_{2\mathrm{PC}}(q)$.

*Tail integrability.* Only $y = u - a$ enters the limit, with triangular
density $T(y) = (2\pi - |y|)/(2\pi)^2$ on $(-2\pi, 2\pi)$, so
$g \in [1 - 1/q,\ 1 + 1/q]$. For $q > 1$, $g$ is bounded away from zero and
$I_{N_1} \sim \langle 2/(\xi g)^2\rangle$ — integrable. At $q = 1$
(adjacent-channel Nyquist), $g$ grazes zero at the corner
$y \to -2\pi$ ($a \to \pi$, $u \to -\pi$: the quasi-degenerate-FWM corner of
the XPM domain), but the triangular density vanishes linearly there,
$T \sim g$, giving $I_{N_1} \sim \ln\xi/\xi^2$ — still integrable. For
$q < 1$ (sub-Nyquist spacing) the second branch of the Dar hyperbola crosses
the domain interior and the analysis changes; $q \ge 1$ is assumed
throughout.

## 3. Explicit forms of the constants

Reduce the inner averages to one dimension. With $T(y)$ as above and
$\hat K = |\hat\Lambda|^2 = 4\sin^2(\delta/2)/\delta^2$:

$$
I_{N_1}(\xi) = \int T(y)\,\hat K(\xi g(y))\,dy,
\qquad
I_{2\mathrm{PC}}(\xi) = \Big|\int T(y)\,\hat\Lambda(\xi g(y))\,dy\Big|^2 .
$$

The one-sided projection uses the exact antiderivative
$F(\tau) = \int_0^\tau \hat\Lambda = \mathrm{Si}(\tau) + i\,\mathrm{Cin}(\tau)$
(with $\mathrm{Cin}(\tau) = \gamma + \ln\tau - \mathrm{Ci}(\tau)$):

$$
\mathbb E_a\hat\Lambda \;=\; h(u)
= \frac{\omega}{2\pi\xi}\Big[F\big(\xi g(u+\pi)\big) - F\big(\xi g(u-\pi)\big)\Big],
\qquad
I_{q3}(\xi) = \frac{1}{2\pi}\int_{-\pi}^{\pi} |h(u)|^2\,du ,
$$

and $I_{3\mathrm{PCa}} = I_{q3} - I_{2\mathrm{PC}}$,
$I_{4\mathrm{PC}} = I_{N_1} - 2 I_{q3} + I_{2\mathrm{PC}}$.

**3PCa = 3PCb exactly in the limit**: the limiting kernel depends on $(a,u)$
only through $y = u - a$, and $(a, u) \mapsto (-u, -a)$ maps the uniform law
to itself while exchanging the two one-sided projections. (This proves in
the asymptotic regime what the estimator measured at the 1% level across the
band.)

**Closed form for the total.** The $\xi$-integral of $\hat K$ is elementary
($\int_0^\infty \hat K(\xi g)\,d\xi = \pi/g$), giving

$$
C_{N_1}(q) \;=\; \int T(y)\,\frac{dy}{g(y)}
\;=\; q\Big[(q+1)\ln(q+1) + (q-1)\ln(q-1) - 2q\ln q\Big],
$$

with $C_{N_1}(1) = 2\ln 2 \approx 1.3863$, $C_{N_1}(2) \approx 1.0465$,
$C_{N_1}(4) \approx 1.0107$, and $C_{N_1} \to 1$ as $q \to \infty$ (the
classic $F_{\mathrm{XPM}} \to 1/|\nu|$ sheet limit of
[`lorenzi_fast_method.md`](lorenzi_fast_method.md) §12, enhanced at close
spacing by the $g$-weighting).

## 4. Numerical verification

Two independent computations agree (scratchpad `sector_constants2.py` for
the limit integrals; `sector_asymptotics.py` for the direct CRN sweep at
$\nu = 10\ldots10^5$, $4\times10^6$ samples × 4 seeds per point; raw sweep
preserved at `~/tmp/sector_asymptotics_sweep_2026-08-24.txt`):

| $q$ | $C_{N_1}$ | MC | $C_{2\mathrm{PC}}$ | MC | $C_{3\mathrm{PC,tot}}$ | MC | $C_{4\mathrm{PC}}$ | MC |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.3860 | 1.385 | 0.8938 | 0.888 | 0.2119 | 0.216 | 0.2804 | 0.286 |
| 2 | 1.0462 | 1.046 | 0.9172 | 0.915 | 0.0720 | 0.073 | 0.0570 | 0.059 |
| 4 | 1.0104 | 1.012 | 0.9507 | 0.950 | 0.0339 | 0.034 | 0.0258 | 0.027 |

(MC columns: plateaus of $S\cdot\nu$ from the $\nu \ge 10^4$ rows of the
sweep.) The asymptotic sector ratios follow:

| $q$ | $\rho_3^\infty = C_{3,\rm tot}/C_2$ | $\rho_4^\infty = C_4/C_2$ |
|---|---|---|
| 1 | 0.237 | 0.314 |
| 2 | 0.0785 | 0.0622 |
| 4 | 0.0357 | 0.0271 |

Notes:

* At $q = 1$, $4\mathrm{PC} > 3\mathrm{PC}_{\rm tot}$ *asymptotically* —
  the 4PC-over-3PC crossover observed in the sweeps is a real feature of the
  constants at close spacing, not a transient.
* The slow drift of $\rho_4(q{=}1)$ in the MC (still $\sim$3%/decade at
  $\nu = 10^5$) is a finite-$\nu$ transient toward $0.314$; the limit
  integral is convergent (no $\ln\nu$ growth — the would-be $1/\xi$ tail
  coefficient of $I_{N_1}$ vanishes because the triangular density is zero
  at the grazing corner).
* A mask-only factorization of the kernel (dropping the $O(1)$ in-band phase
  modulation $g$) predicts $3\mathrm{PC} \sim \ln\nu/\nu^2$,
  $4\mathrm{PC} \sim \nu^{-2}$ and is refuted by the sweep by $\sim$20× at
  $q = 4$, $\nu = 3\times10^3$ — recorded as a cautionary negative result.

## 5. Consequences

1. **Retraction of the fitted power laws.** The
   $\rho_3 \approx 0.133\,\nu^{-0.31}$ / $\rho_4 \approx 0.019\,\nu^{+0.29}$
   fits (range $\nu \in [5, 300]$) are the crossover between the
   low-walk-off plateau and the constants above; the measured local slopes
   at $\nu = 10\!-\!100$ ($-0.45\ldots-0.2$ and $+0.39\ldots+0.16$
   depending on $q$) bracket the fitted exponents, confirming the
   diagnosis. There are no fractional asymptotic exponents.
2. **Two coordinates, not one.** The sector structure is a function of
   $(\nu, q)$ — collision count *and* spacing-to-baud ratio. The band sweep
   held $q$ fixed, which is why it appeared single-coordinate.
3. **Dar reconciliation.** Dar et al.'s "phase noise $\propto\Omega_s^{-1}$,
   all other terms $\propto\Omega_s^{-2}$" concerns the *spacing* dependence
   at fixed link; here that lives in the decay of
   $C_{3\mathrm{PC}}, C_{4\mathrm{PC}}$ with $q$ (measured: roughly halving
   per octave of $q$ over $q = 1\ldots4$; the large-$q$ law of the constants
   is left open), while the DGD-dependence at fixed geometry is uniformly
   $\nu^{-1}$.
4. **Modulation-format modeling.** Since the ratios converge, the
   asymptotic sector mix — hence the non-Gaussian constellation correction —
   is a *fixed, precomputable* function of $q$ alone: two numbers per pair
   geometry.

Open: closed forms for $C_{2\mathrm{PC}}, C_{3\mathrm{PC}}, C_{4\mathrm{PC}}$
(only $C_{N_1}$ is closed so far); the large-$q$ asymptotics of the
constants; the $q < 1$ (sub-Nyquist) regime; multi-span / lossy profiles
(replace $\hat\Lambda$ by the corresponding link function — the ANOVA and
strip arguments are unchanged); merging with the original analytic proof.
