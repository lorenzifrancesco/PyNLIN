# In-band phase truncation in closed-form GN models

Replaces the three earlier notes `gan_phase_curvature_comparison.md`,
`gan_xpm_smf_reassessment.md` and `cfm_local_phase_comparison.md`, which have
been removed.

Every identity below is proved in place and re-verified symbolically or by
quadrature in
[`analysis/standalone_analytical/verify_inband_phase_truncation.py`](../../analysis/standalone_analytical/verify_inband_phase_truncation.py),
which is also the source of every number in Secs. 5, 6 and 11. Bold tags such
as **(3a)** name the corresponding check in that script's output.

## 1. Scope and notation

Closed-form GN models differ in how many of the in-band frequency coordinates
of a channel tuple they retain inside the phase mismatch. This note states the
exact phase, classifies the published truncations by that criterion, derives
the asymptotic pair efficiency of each class in closed form, and evaluates the
three classes on the O-band parameters of Gan et al.

Angular frequencies are $\omega = 2\pi f$. A tuple is an ordered interaction
$(a,b,c)\to t$ with the third leg conjugated, so that
$\omega_t = \omega_a + \omega_b - \omega_c$ up to a carrier residual. In-band
offsets are normalized as $\Omega_j = B\,x_j$ with $B$ the symbol rate; for a
rectangular Nyquist channel $x_j \in (-\pi,\pi)$, so that the ordinary-frequency
offset spans $(-B/2, B/2)$. The link kernel of a span of length $L$ with
longitudinal interaction profile $A(s)$, $s = z/L$, is

$$\lambda_A(u) = \int_0^1 A(s)\,e^{ius}\,ds ,
\qquad
k_A(u) = \frac{|\lambda_A(u)|^2}{|\lambda_A(0)|^2},
\qquad
k_A(0) = 1 ,$$

with $u = L\,\Delta\beta$ the accumulated mismatch. For $A \equiv 1$ this is
$k(u) = 4\sin^2(u/2)/u^2$.

Standing symbols: $\Delta f$ channel spacing, $r = \Delta f/B$ spacing-to-baud
ratio, $\nu = B L\,|\Delta\beta_1|$ walk-off in symbol slots,
$d = L/L_D = |\beta_2| B^2 L$ accumulated broadening,
$\vartheta_j = \tfrac12 L B^2 \beta_2^{(j)}$.

Acronyms: CUT channel under test; NLI nonlinear interference; XCI, MCI cross-
and multi-channel interference.

## 2. The exact tuple phase

**Proposition 1 (leg cancellation).** For any energy-conserving tuple, the
$\beta_0$ and $\beta_1$ parts of $\Delta\beta$ vanish identically, at every
in-band offset. **(1a)**

*Proof.* With $W_t = W_a + W_b - W_c$,
$(g_0 + g_1 W_a) + (g_0 + g_1 W_b) - (g_0 + g_1 W_c) - (g_0 + g_1 W_t)
= g_1 (W_a + W_b - W_c - W_t) = 0$. $\square$

**Proposition 2 (uniform $\beta_2$).** If $\beta(\omega)$ is quadratic with
constant $\beta_2$, then exactly

$$\Delta\beta = -\beta_2\,(W_a - W_c)(W_b - W_c) . \qquad \textbf{(1b)}$$

*Proof.* By Proposition 1 only the quadratic part survives, giving
$\tfrac12\beta_2[W_a^2 + W_b^2 - W_c^2 - (W_a{+}W_b{-}W_c)^2]$. Expanding the
square gives $-2(W_a - W_c)(W_b - W_c)$ inside the bracket. $\square$

**Proposition 3 (cubic $\beta$).** If $\beta$ is truncated after $\beta_3$,
then exactly

$$\Delta\beta = (W_a - W_c)(W_c - W_b)\,
\beta_2\!\left(\tfrac{W_a + W_b}{2}\right),$$

and each of the three factors is affine in $(x_a, x_b, x_c)$. **(1c)**

*Proof.* Direct expansion of both sides; the difference is the zero
polynomial. Affinity holds because $W_j = \omega_j + B x_j$ and
$\beta_2(\omega) = \beta_2 + \beta_3\omega$. $\square$

**Local expansion.** Writing $x_t = x_a + x_b - x_c + \delta$ with
$\delta = \delta\Omega/B$ the normalized carrier residual, the accumulated
phase splits as

$$u = u_0 + \underbrace{\kappa_a x_a + \kappa_b x_b - \kappa_c x_c}_{\text{gradient}}
+ \underbrace{\vartheta_a x_a^2 + \vartheta_b x_b^2 - \vartheta_c x_c^2 - \vartheta_t x_t^2}_{\Delta u_{\mathrm{quad}}}
+ \cdots ,$$

$$u_0 = L\big[\beta_0^{(a)}+\beta_0^{(b)}-\beta_0^{(c)}-\beta_0^{(t)}\big],
\qquad
\kappa_j = L B\,\big[\beta_1^{(j)} - \beta_1^{(t)}\big] .$$

For $\beta$ truncated after $\beta_3$ the expansion terminates at cubic order,
so $\Delta u_{\mathrm{quad}}$ plus one cubic term is the complete remainder.

**Proposition 4 (unmasked curvature bound).** On the cube
$x_j \in (-\pi,\pi)$,

$$|\Delta u_{\mathrm{quad}}| \le P_q
:= \pi^2\big(|\vartheta_a| + |\vartheta_b| + |\vartheta_c| + |\vartheta_t|\big) .$$

*Proof.* Triangle inequality on the four terms, each bounded by
$\pi^2|\vartheta_j|$. $\square$

$P_q$ bounds a phase displacement, not a kernel error. It is attained only when
the four terms fail to cancel and all four coordinates sit at a band edge, so
it is not tight. Two quantities must be distinguished when it is used: whether
$P_q$ is small compared with $1$ (the width of the coherent plateau of $k$), and
whether $P_q$ is small compared with the distance of the linear phase interval
from $u = 0$. Neither implies the other.

$\mu = u_0/X_\nabla$ with $X_\nabla = \|(\kappa_a,\kappa_b,-\kappa_c)\|_2$ is
undefined at $X_\nabla = 0$; a tuple with $X_\nabla = 0$ is not a $\mu = 0$
tuple.

## 3. The XPM island

Let the CUT be the target channel and the interferer be at spacing $\Delta f$.
Label the four in-band coordinates $x_{\mathrm{in}}$ (CUT input), $x_2$
(interferer, unconjugated), $x_1$ (interferer, conjugated) and
$x_{\mathrm{out}} = x_{\mathrm{in}} + x_2 - x_1$ (CUT output). Set

$$X = x_2 - x_1 , \qquad y = x_1 - x_{\mathrm{in}} .$$

**Proposition 5 (exact XPM reduction).** For uniform $\beta_2$, with
$\nu = 2\pi\beta_2 B \Delta f L$,

$$u = \nu\,X\,g(y), \qquad g(y) = 1 + \frac{y}{2\pi r} ,
\qquad y = x_1 - x_{\mathrm{in}} = x_2 - x_{\mathrm{out}} .
\qquad \textbf{(2a)}, \textbf{(2b)}$$

*Proof.* Proposition 2 with $W_a - W_c = -2\pi\Delta f + B(x_{\mathrm{in}} - x_1)$
and $W_b - W_c = B(x_2 - x_1)$ gives
$\Delta\beta = 2\pi\beta_2 B\Delta f\,X\,[1 + B(x_1 - x_{\mathrm{in}})/(2\pi\Delta f)]$;
multiply by $L$ and use $B/(2\pi\Delta f) = 1/(2\pi r)$. The second equality
follows by substituting $x_{\mathrm{out}}$. $\square$

[`xpm_in_channel_curvature.md`](xpm_in_channel_curvature.md) §3 states the same
identity in the coordinates $(y,s) = (x_2 - x_1,\ x_{\mathrm{in}} - x_1)$ as
$u = y(\nu - 2qs)$, which is Proposition 5 after $y \to X$, $s \to -y$ and
$2q/\nu = 1/(2\pi r)$.

**Corollary (consistency with the four-term form).** With a common $\beta_2$
and $q = \tfrac12\beta_2 B^2 L$,

$$u - \nu X = q\,\big(x_{\mathrm{in}}^2 + x_2^2 - x_1^2 - x_{\mathrm{out}}^2\big)
= 2 q X y ,$$

which is the quadratic phase evaluated by
[`qmc_xpm_ground_truth`](../../src/pynlin/methods/td/fast_nlin.py). **(2c)**

Two consequences are immediate from Proposition 5.

1. $g - 1 = O(1/r)$, not $O(1/\nu)$. Increasing walk-off at fixed spacing does
   not remove it.
2. $y$ is a difference of two independent in-band coordinates, one belonging to
   the CUT and one to the interferer. Under independent uniform $x_j$ its
   density is triangular on $(-2\pi, 2\pi)$, not uniform.

## 4. The three phase-truncation classes

A single-span phase model for the XPM island is fixed by which of
$x_{\mathrm{in}}$ and $x_2$ it keeps inside $g$:

| class | phase | induced law of $y$ |
|---|---|---|
| **linear** | $u = \nu X$ | point mass at $y = 0$ |
| **frozen-CUT** | $u = \nu X\,(1 + x_2/2\pi r)$ | uniform on $(-\pi,\pi)$ |
| **local-quadratic** | $u = \nu X\,(1 + (x_2 - x_{\mathrm{out}})/2\pi r)$ | triangular on $(-2\pi,2\pi)$ |

The frozen-CUT class fixes the CUT frequency at band centre inside the phase and
retains the interferer coordinate; the local-quadratic class retains both.

## 5. Asymptotic pair efficiency

Define the masked pair efficiency
$F = \mathbb E\big[k_A(u)\,\mathbf 1\{|x_{\mathrm{out}}| < \pi\}\big]$ over
independent uniform $x_{\mathrm{in}}, x_1, x_2$.

**Theorem 1.** For a flat profile and $r > 1$,

$$\lim_{|\nu|\to\infty} |\nu|\,F = C := \mathbb E\!\left[\frac{1}{|g(y)|}\right],$$

the expectation being taken under the law of $y$ induced by the phase model.

*Proof.* Change variables $(x_{\mathrm{in}}, x_1, x_2) \to (x_{\mathrm{in}}, y, X)$,
of unit Jacobian. The constraints are $|x_{\mathrm{in}}| < \pi$,
$|x_{\mathrm{in}} + y| < \pi$, $|x_{\mathrm{in}} + y + X| < \pi$ and the mask
$|x_{\mathrm{in}} + X| < \pi$. For fixed $(x_{\mathrm{in}}, y)$ with
$g(y) \ne 0$, the point $X = 0$ is interior to the admissible $X$-interval, and

$$\int k(\nu X g)\,dX \;\xrightarrow[|\nu|\to\infty]{}\;
\frac{1}{|\nu g|}\int_{-\infty}^{\infty} k(s)\,ds = \frac{2\pi}{|\nu g|} ,$$

using $\int 4\sin^2(s/2)/s^2\,ds = 2\pi$. The two $X$-dependent constraints
collapse onto the two remaining ones. Hence

$$|\nu| F \to \frac{1}{(2\pi)^2}\int_{-2\pi}^{2\pi}
\frac{\big|\{x_{\mathrm{in}} : |x_{\mathrm{in}}| < \pi,\ |x_{\mathrm{in}}+y| < \pi\}\big|}
{|g(y)|}\,dy
= \int_{-2\pi}^{2\pi} T(y)\,\frac{dy}{|g(y)|} ,$$

with $T(y) = (2\pi - |y|)/(2\pi)^2$, the triangular density. Replacing $T$ by
the law of $y$ that the model actually imposes gives the general statement.
$\square$

**Corollary (the three constants).** With $t = y/2\pi$, **(3a)**–**(3c)**

$$C_{\mathrm{lin}}(r) = 1 ,$$

$$C_{\mathrm{froz}}(r) = \int_{-1/2}^{1/2}\frac{dt}{1 + t/r}
= r\,\ln\frac{r + 1/2}{r - 1/2}
\qquad (r > 1/2),$$

$$C_{\mathrm{exact}}(r) = \int_{-1}^{1}\frac{(1-|t|)\,dt}{1 + t/r}
= r\big[(r{+}1)\ln(r{+}1) + (r{-}1)\ln(r{-}1) - 2r\ln r\big]
\qquad (r \ge 1).$$

*Proof of the last line.* Substitute $s = r + t$ and split at $t = 0$:

$$\int_0^1\frac{(1-t)\,dt}{1+t/r} = r\!\int_r^{r+1}\frac{(1{+}r{-}s)}{s}ds
= r\big[(1{+}r)\ln\tfrac{r+1}{r} - 1\big],$$

$$\int_{-1}^0\frac{(1+t)\,dt}{1+t/r} = r\!\int_{r-1}^{r}\frac{(1{-}r{+}s)}{s}ds
= r\big[(1{-}r)\ln\tfrac{r}{r-1} + 1\big].$$

Adding and collecting logarithms gives the stated form. $\square$

$C_{\mathrm{exact}} = C_{N_1}$ of
[`npc_sector_asymptotics.md`](npc_sector_asymptotics.md) §3 and $C(r_{\rm eff})$
of [`xpm_in_channel_curvature.md`](xpm_in_channel_curvature.md) §6(b), derived
there by the same $y$-average. Their large-$r$ expansions are **(3f)**, **(3g)**

$$C_{\mathrm{froz}} = 1 + \frac{1}{12 r^2} + \frac{1}{80 r^4} + O(r^{-6}),
\qquad
C_{\mathrm{exact}} = 1 + \frac{1}{6 r^2} + \frac{1}{15 r^4} + O(r^{-6}),$$

so the frozen-CUT phase recovers half of the leading correction.

At $r = 100/96 = 1.041667$:

```text
C_lin   = 1.00000  =  0     dB
C_froz  = 1.08955  = +0.372 dB
C_exact = 1.29147  = +1.111 dB
```

The constants depend on $r$ alone; $\nu$ has cancelled. Their separation
decreases with $r$:

| $r$ | $C_{\mathrm{froz}}$ [dB] | $C_{\mathrm{exact}}$ [dB] | ratio [dB] |
|---:|---:|---:|---:|
| 1.042 | 0.372 | 1.111 | 0.738 |
| 1.25 | 0.249 | 0.608 | 0.358 |
| 1.5 | 0.169 | 0.381 | 0.212 |
| 2.0 | 0.093 | 0.197 | 0.104 |
| 4.0 | 0.023 | 0.046 | 0.023 |
| 8.0 | 0.006 | 0.011 | 0.006 |

**Approach to the limit.** The $X$-integral of the flat kernel is elementary,
$\int_0^s k = 2\,\mathrm{Si}(s) - 4\sin^2(s/2)/s$, so $F$ reduces to a
two-dimensional quadrature with no sampling error. At $r = 1.041667$ **(6a)**,
**(6b)**:

| $\nu$ | linear | frozen-CUT | local-quadratic |
|---:|---:|---:|---:|
| 300 | 0.99452 | 1.08138 | 1.26082 |
| 1000 | 0.99811 | 1.08669 | 1.27925 |
| 3000 | 0.99930 | 1.08847 | 1.28645 |
| 10000 | 0.99976 | 1.08919 | 1.28965 |
| 30000 | 0.99991 | 1.08942 | 1.29077 |
| $\infty$ | 1.00000 | 1.08955 | 1.29147 |

At $r = 1$ the limit is $C_{\mathrm{exact}}(1) = 2\ln 2 = 1.3863$; the linear
model is then below the local-quadratic model by $24.4\%$ at $\nu = 300$,
$26.7\%$ at $\nu = 3000$, and $27.9\%$ in the limit.

## 6. Band-resolved profile and the locally-white assumption

$C_{\mathrm{froz}}$ and $C_{\mathrm{exact}}$ differ only in the treatment of
$x_{\mathrm{out}}$. That difference is a statement about the shape of
$G_{\mathrm{NLI}}(f)$ across the CUT band and is testable on the kernel itself.

**Proposition 6.** Conditioning on the CUT output offset and taking
$|\nu| \to \infty$, **(3d)**, **(6d)**

$$C(x_{\mathrm{out}})
= r\,\ln\frac{r + 1/2 - x_{\mathrm{out}}/2\pi}{r - 1/2 - x_{\mathrm{out}}/2\pi} .$$

*Proof.* At fixed $x_{\mathrm{out}}$ the free coordinates are
$x_{\mathrm{in}}, x_2$, with $x_1 = x_{\mathrm{in}} + x_2 - x_{\mathrm{out}}$
and $X = x_{\mathrm{out}} - x_{\mathrm{in}}$. The $x_{\mathrm{in}}$-integral
concentrates at $x_{\mathrm{in}} = x_{\mathrm{out}}$ exactly as in Theorem 1,
where the mask is inactive, leaving
$C = (2\pi)^{-1}\!\int_{-\pi}^{\pi} dx_2/|g(x_2 - x_{\mathrm{out}})|$; substitute
$t = (x_2 - x_{\mathrm{out}})/2\pi$. $\square$

**Corollary.** $C(0) = C_{\mathrm{froz}}$, and the band average of
$C(x_{\mathrm{out}})$ over $x_{\mathrm{out}} \sim \mathcal U(-\pi,\pi)$ is
$C_{\mathrm{exact}}$, because $x_2$ and $x_{\mathrm{out}}$ are independent and
uniform, making $y$ triangular. **(3e)**

Quadrature of the masked kernel at $\nu = 1000$, $r = 1.041667$
($4\times 2^{20}$ Sobol points):

| $x_{\mathrm{out}}/\pi$ | masked kernel $\times\,\nu$ | $C(x_{\mathrm{out}})$ | vs band centre [dB] |
|---:|---:|---:|---:|
| $-0.90$ | 0.72523 | 0.72640 | $-1.761$ |
| $-0.50$ | 0.84986 | 0.85079 | $-1.074$ |
| $0.00$ | 1.08756 | 1.08955 | $0$ |
| $+0.50$ | 1.54514 | 1.55008 | $+1.531$ |
| $+0.90$ | 2.54018 | 2.58052 | $+3.745$ |
| band average | — | 1.29147 | $+0.738$ |

In this limit the adjacent-channel XPM contribution to $G_{\mathrm{NLI}}(f)$
increases monotonically toward the interferer, by $5.51$ dB over
$|x_{\mathrm{out}}| \le 0.9\pi$ and $6.80$ dB over the full band. Its band
average exceeds its centre value by $0.738$ dB. Locally white NLI is an
assumption about the *total* NLI PSD; the tilts of interferers on opposite
sides of the CUT have opposite signs and decrease with $r$, so flatness of the
total is compatible with a nonzero per-pair tilt. Freezing $x_{\mathrm{out}}$
inside the phase of a single pair is the step separating $C_{\mathrm{froz}}$
from $C_{\mathrm{exact}}$ and is not implied by flatness of the total.

## 7. Classification of the published closed forms

The following table records what each source does with the two in-band
coordinates. Entries in the "phase used" column are the source's own
expressions; the last column is the class constant that follows from Sec. 5 for
that phase. Statements about the sources are attributions, not results derived
here.

| model | phase used | mixed Hessian in $(f_1,f_2)$ | CUT frequency | island geometry | class constant |
|---|---|---|---|---|---|
| Gan et al., arXiv:2510.11867, via Buglia et al., JLT **41**, 3577 (2023) App. B | $-4\pi^2 f_1'(f_2'{+}\Delta f)\beta_2$, then $f_2'{+}\Delta f \to \Delta f$ | removed | fixed at $f_{\mathrm{CUT}}$ | rectangular | $C_{\mathrm{lin}} = 1$ |
| PCFM1, arXiv:2508.21563, Eqs. 33–34 | $-4\pi^2 f_1' f_2'\,\beta_{2,\mathrm{eff}}$ | retained | fixed at $f_{\mathrm{CUT}}$ | lozenge $\to$ inscribed rectangle; XCI stretched along the CUT dimension | $C_{\mathrm{froz}}$ |
| PCFM2 / any-island, arXiv:2602.03860 | same bilinear form | retained | fixed at $f_{\mathrm{CUT}}$ | arbitrary rectangle inside an island, MCI included, no stretching | $C_{\mathrm{froz}}$ for the centre-frequency rectangle |
| CFM6, Eq. 17 then asinh reduction | $4\pi^2 (f_1{-}f)(f_2{-}f)[\beta_2 + \pi\beta_3(f_1{+}f_2{-}2f_{\mathrm{ref}})] + \beta_4$ terms | retained | fixed at $f_{c,\mathrm{CUT}}$ | rectangular | $C_{\mathrm{froz}}$ in the large-argument limit |
| local-quadratic (this note; PyNLIN reference paths) | $u = \nu X g(y)$ of Prop. 5 | retained | integrated over the CUT band | masked exact island | $C_{\mathrm{exact}}$ |

The structural difference between the first row and the rest is the interferer
in-band coordinate: Buglia's Appendix B replaces $f_2 + \Delta f$ by $\Delta f$,
which sets $y = 0$; PCFM Eq. 33 keeps $f_1' f_2'$ with $f_2'$ running over the
interferer band, which leaves $y$ uniform. No listed source integrates the CUT
output frequency inside the phase, which is what makes $y$ triangular.

Each source states where its approximation degrades. Buglia et al. write that
"the channels most impacted by this approximation are those near the COI".
PCFM1 calls locally white NLI "an accepted approximation" and defers its removal;
PCFM2 states that the white-NLI and strictly-rectangular assumptions are "no
longer required" once the integration rectangle is decoupled from the island.

### 7.1 $C_{\mathrm{froz}}$ appears in PCFM's own XCI kernel

PCFM1 Eq. 50 gives, for the stretched XCI island with span length $L_s$ and
polynomial power-profile coefficients $p_n$,

$$K_{\mathrm{XCI},n} = \frac{L_s}{2\pi|\beta_{2,\mathrm{eff},n}|}\,
\log\frac{f_n - f_{\mathrm{CUT}} + B_n/2}{f_n - f_{\mathrm{CUT}} - B_n/2}\,
\sum_{m,k} \frac{p_m p_k L_s^{m+k}}{m+k+1} .$$

For a flat power profile only the $m = k = 0$ term survives. Under the linear
phase the log factor is replaced by its wide-spacing limit $B_n/\Delta f = 1/r$.
Taking the ratio of the two,

$$\frac{K_{\mathrm{XCI}}}{K_{\mathrm{linear}}}
= r\,\ln\frac{r+1/2}{r-1/2} = C_{\mathrm{froz}}(r) ,$$

which is the constant obtained in Sec. 5 by leaving $y$ uniform. Two
independent routes give the same number. This holds for equal channel
bandwidths; for unequal bandwidths the log factor persists but $r$ must be read
as the ratio of the interferer band edges to the interferer bandwidth.

Stretching the XCI island to $(-\infty,\infty)$ along the CUT-bandwidth
dimension does not change $C_{\mathrm{froz}}$: that limit acts on the
$X$-integral, whose integrand is already concentrated on $|X| \lesssim 1/\nu$,
while the $y$-average is untouched.

### 7.2 CFM6's asinh reduction

CFM6's XCI closed form is a difference of $\operatorname{asinh}$ evaluated at
the two interferer band edges, with argument proportional to
$\beta_{2,\mathrm{eff}} B_{\mathrm{CUT}}(f_{\mathrm{edge}} - f_{c,\mathrm{CUT}})$
divided by the span loss coefficient. When both arguments greatly exceed $1$,
$\operatorname{asinh}(Ax) = \ln(2Ax) + O((Ax)^{-2})$ and their difference
reduces to $\ln[(r{+}1/2)/(r{-}1/2)]$, placing CFM6 in the frozen-CUT class.
When the arguments are of order $1$ or smaller — low $\beta_{2,\mathrm{eff}}$,
low baud rate, or high loss — $\operatorname{asinh}$ is not logarithmic, the
loss coefficient does not cancel between the two terms, and the reduction does
not apply. The Taylor order of CFM6's reduction from its Eq. 17 $\chi$, which
retains $\beta_4$, to an island-centred effective $\beta_2$ has not been traced
here.

## 8. The Gan two-variable slice

Gan et al. evaluate the GN integrand at the CUT centre frequency. With
in-band ordinary-frequency offsets $p$ in channel $j$ and $q$ in channel $k$,
the conjugated leg is at $p + q$ in channel $m$ and the observed target offset
is zero:

$$\Delta\beta_{\mathrm G}(p,q) = \beta(\omega_j{+}2\pi p) + \beta(\omega_k{+}2\pi q)
- \beta(\omega_m{+}2\pi[p{+}q]) - \beta(\omega_i) .$$

Its Taylor coefficients about $p = q = 0$ are

$$\varphi_1 = 2\pi\big[\beta_1^{(j)} - \beta_1^{(m)}\big],
\qquad
\varphi_2 = 2\pi\big[\beta_1^{(k)} - \beta_1^{(m)}\big],$$

$$H_{11} = (2\pi)^2\big[\beta_2^{(j)} - \beta_2^{(m)}\big],
\quad
H_{22} = (2\pi)^2\big[\beta_2^{(k)} - \beta_2^{(m)}\big],
\quad
H_{12} = -(2\pi)^2\beta_2^{(m)} .$$

Gan's Appendix B keeps $\varphi_0 + \varphi_1 p + \varphi_2 q$ and drops
$H$ and all higher in-band derivatives. Its $\varphi_0, \varphi_1, \varphi_2$
contain $\beta_2, \beta_3, \beta_4$ because those determine $\beta$ and its
first derivatives at the tuple centre. Including $\beta_4$ in the gradient
coefficients is therefore not the same as retaining the in-band Hessian.

**Proposition 7 (slice correspondence).** Setting $a = j$, $b = k$, $c = m$,
$t = i$, $x_t = 0$, $x_c = x_a + x_b$, $\delta = 0$ in the local expansion of
Sec. 2 reproduces $L(\varphi_0 + \varphi_1 p + \varphi_2 q)$ exactly, and the
omitted quadratic phase is exactly the Gan Hessian: **(1d)**

$$\Delta u_{\mathrm{quad}}\big|_{\text{slice}}
= (\vartheta_j - \vartheta_m)x_a^2 + (\vartheta_k - \vartheta_m)x_b^2
- 2\vartheta_m x_a x_b
\;\longleftrightarrow\;
L\big[\tfrac12 H_{11}p^2 + H_{12}pq + \tfrac12 H_{22}q^2\big] .$$

*Proof.* Substitute $x_c = x_a + x_b$ into
$\vartheta_j x_a^2 + \vartheta_k x_b^2 - \vartheta_m x_c^2$ and expand; use
$\Omega = B x = 2\pi f_{\mathrm{offset}}$ and
$\vartheta_j = \tfrac12 L B^2\beta_2^{(j)}$ for the dictionary. $\square$

The two truncations are therefore the same phase model on the Gan slice. The
surrounding integrals differ: Gan neglects the rectangular-spectrum factor
$\Pi[(p{+}q)/B_m]$, replaces the polygonal domain by a circumscribed rectangle,
and adjusts the bandwidth prefactor with $\max(B_j, B_k, B_m)$; the PyNLIN Fast
formulation retains the output mask $\mathbf 1\{|x_t| < \pi\}$ and admits
$\delta \ne 0$ families explicitly.

### 8.1 Relative error of the linear phase for strict FWM

By Proposition 3, for cubic $\beta$ the phase is a product of three affine
factors, $u = L\,P\,Q\,G$. Writing $P = P_0(1{+}\varepsilon_P)$,
$Q = Q_0(1{+}\varepsilon_Q)$, $G = G_0(1{+}\varepsilon_G)$,

$$\frac{u}{u_0} = 1 + \varepsilon_P + \varepsilon_Q + \varepsilon_G
+ \big[\varepsilon_P\varepsilon_Q + \varepsilon_P\varepsilon_G
+ \varepsilon_Q\varepsilon_G + \varepsilon_P\varepsilon_Q\varepsilon_G\big],$$

the first line being the linear model and the bracket the exact omitted
remainder.

**Proposition 8.** For $\beta_3 = 0$ and independent
$x_j \sim \mathcal U(-\pi,\pi)$, with $\Delta n_{ac}$, $\Delta n_{cb}$ the
integer channel-index separations, the relative phase error
$\varepsilon = \varepsilon_P\varepsilon_Q$ satisfies **(4a)**–**(4c)**

$$\mathbb E[\varepsilon] = -\frac{1}{12\,r^2\,\Delta n_{ac}\,\Delta n_{cb}},
\qquad
\sqrt{\mathbb E[\varepsilon^2]} = \frac{\sqrt{1/30}}{r^2|\Delta n_{ac}\Delta n_{cb}|},
\qquad
\sqrt{\operatorname{Var}\varepsilon}
= \frac{\sqrt{19/45}}{4\,r^2|\Delta n_{ac}\Delta n_{cb}|} .$$

*Proof.* $\varepsilon_P = (x_a - x_c)/(2\pi r\Delta n_{ac})$ and
$\varepsilon_Q = (x_c - x_b)/(2\pi r\Delta n_{cb})$. With $m_2 = \pi^2/3$ and
$m_4 = \pi^4/5$,
$\mathbb E[(x_a{-}x_c)(x_c{-}x_b)] = -m_2$ and
$\mathbb E[(x_a{-}x_c)^2(x_c{-}x_b)^2] = 3m_2^2 + m_4 = 8\pi^4/15$, the cross
terms vanishing by oddness. Dividing by $(2\pi r\Delta n_{ac})(2\pi r\Delta n_{cb})$
and by its square gives the first two lines, since
$(8\pi^4/15)/(2\pi)^4 = 1/30$. For the third,

$$\operatorname{Var}\varepsilon = \mathbb E[\varepsilon^2] - (\mathbb E\varepsilon)^2
= \frac{8\pi^4/15 - \pi^4/9}{(2\pi)^4 r^4 \Delta n_{ac}^2\Delta n_{cb}^2}
= \frac{19/45}{16\,r^4\Delta n_{ac}^2\Delta n_{cb}^2}. \qquad\square$$

These laws hold where the denominator does not vanish. They do not apply to a
phase-matched degeneracy, and they do not apply to XPM, for which
$\Delta n_{cb} = 0$: that case is Sec. 3, where the omitted term multiplies the
linear phase rather than adding to it, with relative size $O(1/r)$ that persists
at arbitrarily large $\nu$.

## 9. Multi-span accumulation

**Proposition 9.** For $N_s$ identical spans of length $L_s$ each producing the
same complex amplitude $A_{\mathrm{span}}(\Delta\beta)$, **(5a)**

$$|A_{\mathrm{total}}|^2 = |A_{\mathrm{span}}|^2\chi_{N_s},
\qquad
\chi_{N_s} = \left|\sum_{n=0}^{N_s-1}e^{in\Delta\beta L_s}\right|^2
= \frac{\sin^2(N_s\Delta\beta L_s/2)}{\sin^2(\Delta\beta L_s/2)}
= N_s + 2\sum_{n=1}^{N_s-1}(N_s - n)\cos(n\Delta\beta L_s).$$

*Proof.* Geometric sum for the middle expression; expanding
$|\sum e^{in\varphi}|^2 = \sum_{n,m}e^{i(n-m)\varphi}$ and counting the
$N_s - |n{-}m|$ pairs at each lag gives the right expression. $\square$

If the linearized mismatch carries a per-unit-length error
$\varepsilon_\beta = \Delta\beta_{\mathrm{exact}} - \Delta\beta_{\mathrm{lin}}$,
the $n$-th cross-span cosine argument carries $n L_s\varepsilon_\beta$, i.e.
$n$ times the per-span phase error. A coherent multi-span use of the linear
phase therefore requires control of $N_s P_{\mathrm{curv,span}}$, not only of
$P_{\mathrm{curv,span}}$.

Gan et al. do not apply one accumulation rule to all terms. Their integral GN
reference contains $\chi$ in full with $\beta$ through $\beta_4$. Their
closed-form general FWM assumes incoherent accumulation,
$\eta_{\mathrm{FWM,total}} \approx N_s\eta_{\mathrm{FWM,span}}$, justified
numerically by an inferred FWM coherence factor below approximately $0.03$ in
the reported $16.1$ THz O-band system. Under that assumption the relevant
integration length for one FWM tuple is $L_s$, not $N_s L_s$. SPM and XPM
receive explicit coherent corrections derived from the cosine sum. Their
Appendix D recommends including the higher-order phase term in the single-span
contribution but not in the coherent multi-span correction.

Consequently, for a fixed physical span and tuple, $u_0$, $\boldsymbol\kappa$,
$\vartheta$ and $P_q$ all scale linearly in $L$, and three regimes must be
distinguished:

- $|u_0|$ dominant: $P_q/|u_0|$ is independent of $L$, and increasing $L$
  suppresses the tuple without changing the relative curvature correction;
- linear phase crossing $u = 0$ inside the support: gradient and curvature both
  scale with $L$, and a scalar $P_q$ condition is conservative but does not
  determine the efficiency error;
- $u_0 \approx 0$ and $X_\nabla \approx 0$ with $P_q \gtrsim 1$: the quadratic
  phase is leading order, and the linear model predicts a coherent plateau that
  does not exist.

A total distance of $800$ km therefore denotes two different problems: one
uniform $800$ km interaction, in which curvature enters one kernel with
$L = 800$ km; or ten incoherently accumulated $80$ km spans, in which curvature
enters ten kernels with $L_s = 80$ km and the powers are added. Gan's
closed-form FWM uses the second.

## 10. Coordinates are not interchangeable

For a locally uniform-$\beta_2$ adjacent pair,

$$d = \frac{L}{L_D} = |\beta_2| B^2 L,
\qquad
\nu = \frac{L}{L_W} = B L|\Delta\beta_1|,
\qquad
\frac{\nu}{d} = 2\pi r ,
\qquad
\frac{L_W}{L_D} = \frac{1}{2\pi r} ,$$

the last two following from $\Delta\beta_1 = 2\pi\beta_2\Delta f$ for the
adjacent pair. At $r = 1.041667$, $\nu = 6.545\,d$ and $L_W/L_D = 0.153$. High
$\nu$ means many completed collisions; high $d$ means appreciable accumulated
broadening. In a smooth single-mode fiber with an adjacent pair the two are
locked by $\nu = 2\pi r d$, so low DGD implies low local GVD. In a fiber where
group-velocity matching can make $\Delta\beta_1$ small while $\beta_2$ remains
large — a few-mode fiber, or a nonlocal pair whose integrated $\beta_2$ cancels
across the zero-dispersion wavelength — they are independent, and the last case
requires the full $\beta_3/\beta_4$ phase rather than a local uniform-$\beta_2$
estimate.

## 11. Numerical results at the Gan O-band parameters

Source parameters (Gan et al., Table I and Sec. III-A): reference wavelength
$1302.3$ nm at $D = 0$; slope $0.087$ ps/(nm$^2$ km); curvature
$-9.714\times10^{-5}$ ps/(nm$^3$ km); $B = 96$ GBd; $\Delta f = 100$ GHz;
$L = 80$ km; $161$ channels; attenuation approximately $0.28$–$0.40$ dB/km
(read from Fig. 3). The dispersion profile is reconstructed as

$$D(\lambda) = S(\lambda - \lambda_0) + \tfrac12\dot S(\lambda - \lambda_0)^2,
\qquad
\beta_2(\lambda) = -D(\lambda)\lambda^2/(2\pi c),$$

and $|\nu|$ is obtained by integrating $\beta_2$ across the pair,
$\nu = BL\,|\int_{\omega_t}^{\omega_i}\beta_2\,d\omega|$.

The lossy columns use $A(s) = e^{-\alpha L s}$ with the kernel normalized to
$k_A(0) = 1$; that normalization is common to the three models and cancels in
the ratios. At $80$ km:

| $\alpha$ [dB/km] | $\alpha L$ [Np] | $L_{\mathrm{eff}}$ [km] |
|---:|---:|---:|
| 0.28 | 5.158 | 15.42 |
| 0.34 | 6.263 | 12.75 |
| 0.40 | 7.368 | 10.85 |

### 11.1 Adjacent pair

Randomized Sobol quadrature, $4\times 2^{18}$ points per entry, common random
numbers across the three models. Largest relative standard error in the table:
$9.3\times10^{-5}$, i.e. $0.0004$ dB. Columns 5–7 are the flat $80$ km profile;
columns 8–10 the same span at $0.34$ dB/km. "froz" and "exact" are relative to
the linear phase; "residual" is exact $-$ froz, the part the frozen-CUT phase
does not recover.

| $\lambda$ [nm] | $D$ [ps/(nm km)] | $L/L_D$ | $\|\nu\|$ | froz [dB] | exact [dB] | resid [dB] | froz [dB] | exact [dB] | resid [dB] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1349.19 | $+3.973$ | 2.830 | 18.40 | 0.298 | 0.678 | 0.380 | 0.206 | 0.358 | 0.152 |
| 1337.15 | $+2.973$ | 2.081 | 13.50 | 0.281 | 0.608 | 0.327 | 0.182 | 0.285 | 0.103 |
| 1325.33 | $+1.978$ | 1.360 | 8.78 | 0.252 | 0.500 | 0.248 | 0.148 | 0.190 | 0.042 |
| 1313.71 | $+0.986$ | 0.666 | 4.25 | 0.192 | 0.299 | 0.107 | 0.087 | 0.064 | $-0.024$ |
| 1307.98 | $+0.493$ | 0.330 | 2.05 | 0.131 | 0.100 | $-0.031$ | 0.027 | 0.000 | $-0.027$ |
| 1302.30 | 0.000 | 0.000 | 0.11 | $-0.001$ | $-0.001$ | 0.001 | 0.000 | 0.000 | 0.000 |
| 1291.08 | $-0.982$ | 0.641 | 4.30 | 0.193 | 0.302 | 0.109 | 0.088 | 0.065 | $-0.023$ |
| 1280.06 | $-1.959$ | 1.256 | 8.32 | 0.248 | 0.486 | 0.238 | 0.144 | 0.179 | 0.036 |
| 1258.56 | $-3.898$ | 2.417 | 15.91 | 0.290 | 0.646 | 0.356 | 0.195 | 0.324 | 0.129 |

At the O-band edges the frozen-CUT phase accounts for $0.298$ dB of the
$0.678$ dB flat-span error and $0.206$ dB of the $0.358$ dB lossy-span error,
leaving $0.380$ dB and $0.152$ dB. For $0.3 \le L/L_D \le 0.7$ the frozen-CUT
model exceeds the local-quadratic result, by at most $0.03$ dB. No entry reaches
the $1.111$ dB flat-profile asymptote of Sec. 5, because $|\nu| \le 18.4$ and,
under loss, only $16\%$ of the nominal span length contributes.

### 11.2 Sum over 80 interferer spacings

At locally uniform $\beta_2$, so that spacing dilution is isolated from
zero-dispersion crossings and $\beta_3/\beta_4$: interferer $m$ has
$r_m = m\,r$ and $\nu_m = m\,\nu_1$.

| $\lambda$ [nm] | $L/L_D$ | loss [dB/km] | froz [dB] | exact [dB] | resid [dB] |
|---:|---:|---:|---:|---:|---:|
| 1349.19 | 2.830 | 0 | 0.072 | 0.167 | 0.095 |
| 1349.19 | 2.830 | 0.34 | 0.047 | 0.084 | 0.037 |
| 1325.33 | 1.360 | 0 | 0.060 | 0.122 | 0.061 |
| 1325.33 | 1.360 | 0.34 | 0.032 | 0.045 | 0.013 |
| 1258.56 | 2.417 | 0 | 0.070 | 0.159 | 0.089 |
| 1258.56 | 2.417 | 0.34 | 0.044 | 0.076 | 0.032 |

On the $80$ km, $0.34$ dB/km span, the aggregate shift of the local-quadratic
phase relative to the linear phase is $0.04$–$0.08$ dB and the residual left by
the frozen-CUT phase is $0.01$–$0.04$ dB. The distinction between
$C_{\mathrm{froz}}$ and $C_{\mathrm{exact}}$ is a per-pair statement at dense
spacing; the aggregate is smaller by roughly an order of magnitude, and this
table is a local-uniform-$\beta_2$ calculation, not a reproduction of any
published aggregate.

## 12. Implementation status in PyNLIN

Verified against the current source:

- [`fwm_tuple_variables`](../../src/pynlin/methods/td/fast_nlin.py) computes
  $u_0$, $\kappa_a,\kappa_b,\kappa_c$, $q_a,q_b,q_c,q_t$ (with
  $q_j = \vartheta_j$) and $\delta$ for each tuple.
- Curvature enters tuple *selection* through a conservative $P_q$ padding, which
  prevents declaring a tuple safely dephased when curvature could move it toward
  $u = 0$.
- Retained-tuple FWM efficiencies are evaluated with the linear phase.
- The XPM production sum
  ([`xpm_pair_variables`](../../src/pynlin/methods/td/fast_nlin.py),
  [`xpm_fast_batch`](../../src/pynlin/methods/td/fast_nlin.py)) computes $q$ but
  passes only $\nu$ to the estimator, whose kernel is $u = \nu(x_1 - x_2)$; it
  is therefore in the linear class, $C_{\mathrm{lin}} = 1$.
- [`qmc_xpm_ground_truth`](../../src/pynlin/methods/td/fast_nlin.py) and
  [`qmc_tuple_ground_truth`](../../src/pynlin/methods/td/fast_nlin.py) carry the
  full quadratic phase; by the corollary to Proposition 5 the XPM ground truth
  is the local-quadratic class. These are reference paths, not the production
  estimator.

## 13. Assumptions and open items

- The constants of Sec. 5 are flat-profile, constant-$\beta_2$,
  $|\nu| \to \infty$ limits. Sec. 11 gives the finite-$\nu$, lossy values.
- $C_{\mathrm{froz}}$ requires $r > 1/2$ and $C_{\mathrm{exact}}$ requires
  $r \ge 1$. For $r < 1$ the second branch of the walk-off hyperbola crosses the
  island interior and the $y$-average of Theorem 1 is not the complete
  asymptotic form.
- Whether a single island-centred $\beta_{2,\mathrm{eff}}$ reproduces both the
  centre and the gradient of the effective dispersion across a wide island under
  $\beta_3$ and $\beta_4$ is not assessed here. It does not affect the constants
  of Sec. 5, which assume constant $\beta_2$.
- Decoupling the integration rectangle from the island, as PCFM2 does, admits
  integration over the CUT frequency dimension and hence $C_{\mathrm{exact}}$.
  Whether the published PCFM2 formulas are evaluated that way has not been
  checked.
- The regime $u_0 \approx 0$, $X_\nabla \approx 0$, $P_q \gtrsim 1$ is covered by
  none of the closed forms in Sec. 7; it requires a curvature-retaining tuple
  integral, and a curvature-retaining phased-array treatment if spans are
  mutually coherent.
- Gan's reported validation covers $1\times80$ km to $10\times80$ km, $96$ GBd
  channels, and $4.1$–$16.1$ THz optical bandwidth, as an aggregate NLI
  comparison. It does not isolate the error due to omitted per-tuple curvature.

## 14. Reproduction

```text
python analysis/standalone_analytical/verify_inband_phase_truncation.py
```

Symbolic checks use `sympy`; the asymptotic constants are integrated in closed
form and cross-checked numerically at five values of $r$; the flat-profile
efficiencies use the exact $X$-integration of Sec. 5, converged to $10^{-6}$ in
the node count; the lossy tables use randomized Sobol quadrature with common
random numbers. The script reports 23 checks.

## 15. Source map

**Gan et al., arXiv:2510.11867** — Eq. (2) integral GN model; Eqs. (3)–(5) link
function, phase mismatch, phased-array factor; Eqs. (7)–(10) incoherently
accumulated closed-form FWM; Eq. (20) per-tuple FWM coefficient;
Appendix B Eqs. (37)–(40) first-order two-dimensional Taylor approximation;
Eqs. (23), (26), (27) coherent SPM/XPM corrections; Appendix D accumulated
higher-order phase error.

**Buglia et al., JLT 41, 3577 (2023)**, DOI 10.1109/JLT.2023.3256185 —
Appendix B, the $f_2 + \Delta f \to \Delta f$ step.

**PCFM1** arXiv:2508.21563 Eqs. 33–34, 50; **PCFM2** arXiv:2602.03860.

**Lorenzi et al.**, DOI 10.1109/JLT.2026.3680666 — collision coordinates
$L_D = T^2/|\beta_2|$, $L_W = T/|\Delta\beta_1|$, $M = L/L_W$.

**PyNLIN** — [`src/pynlin/methods/td/fast_nlin.py`](../../src/pynlin/methods/td/fast_nlin.py);
[`xpm_in_channel_curvature.md`](xpm_in_channel_curvature.md) for the
two-variable XPM law at finite walk-off and its plateau and zero-walk-off
limits;
[`npc_sector_asymptotics.md`](npc_sector_asymptotics.md) §3 for $C_{N_1}$;
[`lorenzi_fast_method.md`](lorenzi_fast_method.md) §12 for the sheet limit;
[`analysis/standalone_analytical/verify_linear_phase_validity.py`](../../analysis/standalone_analytical/verify_linear_phase_validity.py)
for the strict-FWM factorization.
