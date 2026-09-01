# Global dispersion, local channels, and four-channel phase mismatch

This note separates three objects that are easy to mix up:

1. The **global propagation curve** of the fiber, $\beta(\omega)$.
2. The **local Taylor expansion** of that curve inside one channel.
3. The **four-channel mismatch function** $\Delta\beta$ associated with one
   FWM tuple $(d,a,b,c)$.

The first object belongs to the fiber, the second belongs to one channel, and
the third belongs to a combination of four channels. Quantities such as
$\mu$, $x_\nabla$, and the FWM efficiency belong to the third level.

## 1. One global dispersion curve for the fiber

A physical fiber has one propagation constant

$$
\beta(\omega),
$$

as a function of physical angular frequency $\omega$. Its derivatives are

$$
\beta_1=\frac{d\beta}{d\omega},\qquad
\beta_2=\frac{d^2\beta}{d\omega^2},\qquad
\beta_3=\frac{d^3\beta}{d\omega^3},\qquad
\beta_4=\frac{d^4\beta}{d\omega^4}.
$$

Their units are

$$
\begin{array}{c|c}
\text{quantity} & \text{units}\\
\hline
\beta & \mathrm{m}^{-1}\\
\beta_1 & \mathrm{s\,m}^{-1}\\
\beta_2 & \mathrm{s}^2\mathrm{m}^{-1}\\
\beta_3 & \mathrm{s}^3\mathrm{m}^{-1}\\
\beta_4 & \mathrm{s}^4\mathrm{m}^{-1}.
\end{array}
$$

Around one global reference frequency $\omega_r$, a fourth-order model is

$$
\beta(\omega_r+\Omega)
=\beta_{0,r}+\beta_{1,r}\Omega
+\frac{\beta_{2,r}}{2}\Omega^2
+\frac{\beta_{3,r}}{6}\Omega^3
+\frac{\beta_{4,r}}{24}\Omega^4.
\tag{1}
$$

The coefficients in (1) describe the global fiber curve near $\omega_r$.
They do not describe a particular FWM tuple.

In a measured system, the global object need not be a polynomial. It can be a
spline fitted to sampled $\beta(\omega)$, $\beta_1(\omega)$, or
$\beta_2(\omega)$ data. A polynomial is simply a convenient controlled model.

### Zero-dispersion wavelength

The zero-dispersion frequency is defined globally by

$$
\boxed{\beta_2(\omega_{\mathrm{ZDW}})=0.}
$$

For a cubic model with constant $\beta_3$,

$$
\beta_2(\omega_r+\Omega)=\beta_{2,r}+\beta_{3,r}\Omega,
$$

and therefore

$$
\Omega_{\mathrm{ZDW}}=-\frac{\beta_{2,r}}{\beta_{3,r}}.
\tag{2}
$$

The ZDW is a property of the global fiber curve. It is not itself a
four-channel phase-matching condition.

## 2. A local expansion inside each channel

Let channel $j$ have carrier frequency $\omega_j$. A frequency inside that
channel is written as

$$
\widetilde\omega_j=\omega_j+\nu_j,
$$

where $\nu_j$ is a local offset from the channel center. In the Dar estimator,
the normalized local coordinate lies in

$$
\frac{\nu_j}{B}\in[-\pi,\pi],
$$

with $B$ the baud rate.

The local Taylor expansion is

$$
\beta(\omega_j+\nu_j)
=\beta_{0,j}+\beta_{1,j}\nu_j
+\frac{\beta_{2,j}}{2}\nu_j^2
+\frac{\beta_{3,j}}{6}\nu_j^3
+\frac{\beta_{4,j}}{24}\nu_j^4+\cdots,
\tag{3}
$$

where

$$
\boxed{\beta_{n,j}=\left.\frac{d^n\beta}{d\omega^n}\right|_{\omega_j}.}
$$

These are not independent fiber parameters. They are the global curve and
its derivatives evaluated at channel $j$.

For the fourth-order global model (1), translating the expansion from
$\omega_r$ to $\omega_j$ gives

$$
\boxed{
\beta_{m,j}
=\sum_{n=m}^{4}
\frac{\beta_{n,r}}{(n-m)!}
(\omega_j-\omega_r)^{n-m}.}
\tag{4}
$$

For example,

$$
\begin{aligned}
\beta_{2,j}
&=\beta_{2,r}
+\beta_{3,r}(\omega_j-\omega_r)
+\frac{\beta_{4,r}}{2}(\omega_j-\omega_r)^2,\\
\beta_{3,j}
&=\beta_{3,r}+\beta_{4,r}(\omega_j-\omega_r),\\
\beta_{4,j}&=\beta_{4,r}.
\end{aligned}
$$

Thus a global cubic or quartic curve naturally produces different local
$\beta_1$, $\beta_2$, and $\beta_3$ values in different channels.

### What a local model answers

The local expansion answers questions such as:

- How much group delay changes across this channel.
- How much a pulse broadens inside this channel.
- Whether a quadratic channel-local approximation is accurate over its
  bandwidth.

It does not yet answer whether four selected frequencies are phase matched.
For that, the four local evaluations must be combined.

## 3. The four-channel FWM tuple

For the FWM process

$$
a+b-c\longrightarrow d,
$$

the physical frequencies obey energy conservation:

$$
\widetilde\omega_a+
\widetilde\omega_b-
\widetilde\omega_c-
\widetilde\omega_d=0.
\tag{5}
$$

The propagation-constant mismatch is

$$
\boxed{
\Delta\beta
=\beta(\widetilde\omega_a)
+\beta(\widetilde\omega_b)
-\beta(\widetilde\omega_c)
-\beta(\widetilde\omega_d).}
\tag{6}
$$

Equation (6) is a **tuple-level quantity**. It is built from four evaluations
of the same global fiber curve.

Choose $\nu_a$, $\nu_b$, and $\nu_c$ as independent local coordinates. Then
(5) fixes

$$
\nu_d
=\nu_a+\nu_b-\nu_c
+\delta\omega_{0},
\tag{7}
$$

where

$$
\delta\omega_0
=\omega_a+
\omega_b-
\omega_c-
\omega_d
$$

is the carrier-frequency residual. The tuple mismatch is therefore a scalar
function of three variables:

$$
\Delta\beta=\Delta\beta(\nu_a,\nu_b,\nu_c).
$$

The admissible domain is the part of the three-dimensional channel cube for
which all four local frequencies, including (7), remain inside their
passbands.

## 4. What “center mismatch” means

The reference used by the controlled single-tuple calculation sets

$$
\nu_a=\nu_b=\nu_c=0.
$$

Energy conservation then requires

$$
\nu_d=\delta\omega_0.
$$

Consequently,

$$
\boxed{
\Delta\beta_{\mathrm{center}}
=\beta(\omega_a)+\beta(\omega_b)-\beta(\omega_c)
-\beta(\omega_d+\delta\omega_0).}
\tag{8}
$$

If the channel carriers already conserve energy,
$\delta\omega_0=0$, all four frequencies in (8) are carrier centers.
Otherwise, “center” means that $a$, $b$, and $c$ are at their centers and the
generated $d$ frequency is shifted as required by conservation.

This is why the raw combination
$\beta_{0,a}+\beta_{0,b}-\beta_{0,c}-\beta_{0,d}$ is not always the physical
center mismatch: channel $d$ may need to be evaluated away from its center.

## 5. Expanding the tuple mismatch around its reference

Near the reference point, the tuple-level function can be expanded as

$$
\Delta\beta(\boldsymbol\nu)
=\Delta\beta_{\mathrm{center}}
+\boldsymbol g\cdot\boldsymbol\nu
+\frac12\boldsymbol\nu^T H\boldsymbol\nu
+\frac{1}{6}\mathcal T[\boldsymbol\nu,
\boldsymbol\nu,
\boldsymbol\nu]+\cdots,
\tag{9}
$$

where

$$
\boldsymbol\nu=(\nu_a,\nu_b,\nu_c).
$$

The objects in (9) have different meanings:

- $\Delta\beta_{\mathrm{center}}$ is one scalar for the tuple.
- $\boldsymbol g=\nabla_{\boldsymbol\nu}\Delta\beta$ is the tuple mismatch
  gradient.
- $H$ is the tuple mismatch Hessian.
- $\mathcal T$ contains tuple-level third derivatives.

They are derived from the local fiber coefficients but are not equal to one
channel's $\beta_1$, $\beta_2$, or $\beta_3$.

For example, let

$$
\omega_{d,*}=\omega_d+\delta\omega_0
=\omega_a+\omega_b-\omega_c.
$$

Then the three gradient components are

$$
\boxed{
\begin{aligned}
g_a&=\beta_1(\omega_a)-\beta_1(\omega_{d,*}),\\
g_b&=\beta_1(\omega_b)-\beta_1(\omega_{d,*}),\\
g_c&=-\beta_1(\omega_c)+\beta_1(\omega_{d,*}).
\end{aligned}}
\tag{10}
$$

Thus the tuple gradient measures differences of group delay among the four
participating frequencies. The Hessian similarly combines their local
$\beta_2$ values.

## 6. The meanings of x and mu

For a tuple with nonzero gradient, define the mismatch variation over one baud
scale as

$$
S_\nabla=B\|\boldsymbol g\|_2.
\tag{11}
$$

It has units of inverse length. The dimensionless gradient scale is

$$
\boxed{x_\nabla=LS_\nabla
=LB\|\nabla_{\boldsymbol\nu}\Delta\beta\|_2.}
\tag{12}
$$

The normalized center detuning is

$$
\boxed{
\mu=\frac{\Delta\beta_{\mathrm{center}}}{S_\nabla}
=\frac{\Delta\beta_{\mathrm{center}}}
{B\|\nabla_{\boldsymbol\nu}\Delta\beta\|_2}.}
\tag{13}
$$

Therefore

$$
\boxed{L\Delta\beta_{\mathrm{center}}=\mu x_\nabla.}
\tag{14}
$$

In the linearized model, the shortest frequency displacement from the
reference point to $\Delta\beta=0$ is

$$
d_\nu
=\frac{|\Delta\beta_{\mathrm{center}}|}{\|\boldsymbol g\|_2},
$$

so

$$
\boxed{|\mu|=d_\nu/B.}
$$

This gives the elementary interpretation:

> $|\mu|$ is approximately the distance to the tuple's phase-matching surface,
> measured in baud-rate units along the normal direction.

The sign of $\mu$ says on which side of that surface the reference point lies.

### Zero-gradient tuples

If $\boldsymbol g=0$, equation (13) cannot be used. The mismatch begins at
quadratic or higher order. The controlled script then uses the convention

$$
x_{\mathrm{curv}}=LB^2|\beta_2(\omega_d)|,
$$

and

$$
\mu_{\mathrm{curv}}
=\frac{\Delta\beta_{\mathrm{center}}}
{B^2|\beta_2(\omega_d)|}.
$$

This compares center mismatch with a characteristic quadratic variation over
one bandwidth. It is no longer a linearized geometric distance. Other
curvature normalizations are possible; this one is an explicit convention of
the controlled study.

## 7. Natural parameters versus controlled parameters

There are two conceptually different ways to obtain $(x,\mu)$.

### Fixed physical fiber

Start from one fixed global $\beta(\omega)$. For every tuple:

1. Evaluate its local channel coefficients.
2. Build its $\Delta\beta$ function.
3. Calculate $x$ and $\mu$ from (12)--(13).

In this case, $x$ and $\mu$ are derived tuple-specific quantities. Changing
the tuple does not change the fiber.

### Controlled single-tuple sweep

The scaling script instead prescribes $x$ and $\mu$:

1. It holds the carrier-index tuple, baud rate, spacing-to-baud ratio, and
   length fixed.
2. It rescales the dispersion coefficients to obtain the requested $x$.
3. It shifts $\beta_{0,a}$ by a constant to obtain the requested $\mu$.

Thus the controlled script does **not** set channel spacing and bandwidth from
$x$ and $\mu$.  The core Dar MC estimator does not do so either: it receives an
already constructed `FWMChannels` object and a baud rate.  Inverting prescribed
$(x,\mu)$ into $(B,\Delta f)$ is a different problem and, for a measured
dispersion profile, generally requires a constrained numerical inversion.

The value `natural` means no independent $\beta_{0,a}$ shift: the mismatch is
the one generated by the selected global polynomial model.

An explicit numerical $\mu$ is a diagnostic construction. Shifting only
$\beta_{0,a}$ generally means that the four local channel models no longer
come from one unmodified global $\beta(\omega)$ curve. This is useful for
isolating the effect of detuning, but it should not be confused with changing
the ZDW or another physical fiber coefficient.

Similarly, equal numerical $(x,\mu)$ values for two different tuples need not
represent the same physical fiber, because the script rescales dispersion
using each tuple's own gradient.

### Validation on the configured physical system

`analysis/standalone_numerical/validate_fwm_mc_real_tuples.py` performs the
complementary physical check.  It loads the SMF-28 profile, WDM grid, baud rate,
spacing, and length from TOML and never modifies them.  For exact
carrier-conserving tuples it evaluates the MC coefficient and derives

$$
x=L B\|\nabla\Delta\beta\|_2,
\qquad
\mu=\frac{\Delta\beta_{\rm center}}
{B\|\nabla\Delta\beta\|_2}
$$

from the same local channel models used by the estimator.  The generated
translation and span plots therefore show which $(x,\mu)$ values the real
system actually realizes, rather than imposing those values by changing the
fiber.

The plotted ordinate is $N_{dabc}T^2/L^2$, the prefactor-free contribution of
one tuple.  It is a nonlinear-noise coefficient/proxy, not received noise power:
launch powers, nonlinear coefficients, amplification profiles, and modulation
prefactors are intentionally absent.  A typical invocation is

```bash
python analysis/standalone_numerical/validate_fwm_mc_real_tuples.py \
  --config input/studies.toml --n-samples 20000
```

The configured O--U carriers cover approximately 179--215 THz, while the
SMF-28 profile has its ZDW near 228 THz.  Consequently, the script also builds
a virtual 25 GHz grid around the profile ZDW, using the TOML baud rate, spacing,
and length.  With the default FWM shift $s_f=3$, it scans the degenerate-pump family

$$
(d,a,b,c)=(d,d+3,d+3,d+6).
$$

Writing the repeated-pump frequency as $\omega_0$ and
$\delta=3\Delta\omega_{\rm ch}$, its center mismatch is

$$
2\beta(\omega_0)-\beta(\omega_0-\delta)-\beta(\omega_0+\delta)
=-\beta_2(\omega_0)\delta^2
-\frac{\beta_4(\omega_0)}{12}\delta^4+O(\delta^6).
$$

Odd orders cancel, so this is specifically a $\beta_4$-sensitive ZDW
diagnostic.  It is not included in the strict three-distinct-interferer
full-band FWM total, where modulation-moment and permutation weights would
need separate treatment.  The ZDW figure compares globally consistent Taylor
models of orders two, three, and four about the repeated-pump frequency.  A
fourth curve uses exact measured-profile values at the channel centers and a
local fourth-order expansion inside each passband; this separates global model
truncation from local finite-bandwidth truncation.

With the default XPM shift $s_x=3$, the ordinary XPM choice $i=d+s_x$ has the
generic-FWM embedding

$$
(d,a,b,c)=(d,d,i,i)=(d,d,d+s_x,d+s_x).
$$

After exchanging the two dummy interferer-frequency variables, its all-index
generic-FWM integral is exactly the XPM $N_1$ integral, without a factor of two,
$2\pi$, or $T^2$.  This implementation identity is enforced by the transformed-
domain regression test `test_xpm_n1_equals_repeated_channel_fwm_mc`; it is not a
physical-system scan observable and is therefore not included in the plots.

This XPM tuple is distinct from the symmetric degenerate-pump ZDW diagnostic
$(d,d+s_f,d+s_f,d+2s_f)$, so equality is checked only between the two
implementations of $(d,d,d+s_x,d+s_x)$.

### Full measured-spectrum comparison

The validation script also sweeps the full frequency domain covered by the
SMF-28 CSV, restricted so that every configured shifted channel remains inside
the measured profile.  At every point it computes

- dedicated XPM MC for target $d$ and interferer $d+s_x$;
- generic FWM MC for $(d,d+s_f,d+s_f,d+2s_f)$;
- the linear-phase Fast efficiency for that same explicit repeated-pump tuple,
  both with resolved fringes and with the upper-envelope policy.

Both calculations use the TOML baud rate, 25 GHz channel grid, fiber length,
attenuation setting, and local measured-profile coefficients through
$\beta_4$.  The target-$d$ frequency is the common plot coordinate.  The
result, ratio, FWM $x$, and local $\beta_2$ are saved in

`media/fwm/real-tuples/real_tuple_full_spectrum_xpm_fwm.pdf`.

Both plotted MC values are divided by $L^2$.  The dedicated XPM estimator
returns $N_1$ in $\mathrm{m}^2$, while the generic estimator returns
$N_{dabc}T^2$ in the same units, so the plotted dimensionless curves are

$$
\frac{N_{1,\mathrm{XPM}}}{L^2},
\qquad
\frac{N_{dabc}T^2}{L^2}.
$$

The Fast curves already return the latter dimensionless efficiency and need
no further normalization. They are evaluated through
`fwm_tuple_variables_for_indices`, which deliberately permits repeated legs;
they are therefore per-tuple diagnostics and are not taken from the strict-FWM
channel sum, whose tuple population is pairwise distinct. The resolved Fast
curve retains the physical linear-model fringes. The upper-envelope curve
replaces the link power kernel by

$$
k_{\rm env}(u)=\min\!\left(1,\frac{4}{u^2}\right)
$$

and is intended as a smooth envelope estimate. The MC curve retains local
in-channel dispersion through $\beta_4$, so disagreement with either Fast
curve also includes phase-model order, not only numerical error.

This normalization removes the trivial coherent $L^2$ scale and makes a
perfectly phase-matched, lossless value approach the passband-support fraction
(about $2/3$ for these exact center-matched tuples).  These are prefactor-free
mixing coefficients, not received noise powers: $\gamma$, launch powers,
modulation moments, Raman profiles, and physical NLIN multiplicities are not
included.

The Dar Xhkm MC code retains a compatibility mode based on the secant value

$$
\beta_{2,\mathrm{eff}}
=\frac{\beta_1(f_i)-\beta_1(f_d)}{2\pi(f_i-f_d)},
$$

and the normalized Dar argument is $\beta_{2,\mathrm{eff}}B^2$.  The physical
spectrum diagnostic instead evaluates the same channel-local Taylor mismatch
through $\beta_4$ as the direct XPM $N_1$ and FWM estimators.  This mismatch is
used consistently in the $N_1$, $N_2$, 2PC, and $(2PC+3PCa)$ aggregate
integrals.  Under constant quadratic dispersion it reduces sample by sample to
the scalar Dar arguments.

The 3PCa, 3PCb, and 4PC values remain differences of aggregate integrals.
Finite-sample residuals can therefore be negative through cancellation
variance, even though the exact collision sectors are disjoint nonnegative
sums of $|X_{h,r,m}|^2$.

The spectrum diagnostic treats these as statistical residual estimates.  It
processes equal-size batches, estimates the aggregate covariance from the batch
means, and propagates it through the linear sector transformation.  Sampling
stops after a minimum budget when every sector has a positive lower confidence
bound and satisfies either a relative-error target or an absolute fallback
scaled by $N_1$.  In symbols, a sector is resolved when

$$
\overline N-k\,\mathrm{SE}>0,
\qquad
\frac{\mathrm{SE}}{\overline N}<r_{\max}
\quad\text{or}\quad
\frac{\mathrm{SE}}{N_1}<a_{\max}.
$$

Resolved positive sectors are displayed with $k\sigma$ error bars.  Positive
points that exhaust the maximum budget without resolving are marked separately;
negative residuals are not clipped or treated as zero.  The saved data include
the actual sample count and stopping reason.  This keeps the logarithmic panel
readable but does not turn the residual construction into a direct nonnegative
sector estimator.

The defaults are read from TOML:

```toml
[fwm.real_tuple_spectrum]
xpm_shift_channels = 2
fwm_shift_channels = 1
spectrum_step = 2
sector_min_samples = 200000
sector_max_samples = 2400000
sector_batch_size = 25000
sector_step = 10
sector_max_relative_error = 0.25
sector_max_stderr_over_n1 = 0.0025
sector_sigma_threshold = 3.0
```

The two shifts are independent signed channel offsets; the FWM outer channel
is always $d+2s_f$, preserving carrier-center conservation.  CLI options
`--xpm-shift`, `--fwm-shift`, and `--spectrum-step` override TOML.  The default
step of two evaluates every 50 GHz.  This is a comparison of two different physical
mixing processes, not the implementation-equivalence test above, so their
curves are not expected to coincide.

The lower-left of the $\beta_2$ panel includes the same fixed-$d$
passband-island inset used by the single-tuple collapse figure.  It highlights the XPM island
$(0,s_x,s_x)$ and FWM island $(s_f,s_f,2s_f)$ together, using the corresponding
curve colors.  The panel above it shows the FWM $x$ and $\mu$ coordinates
together, with separate left and right axes.

The inset also overlays each process's $\eta=e^{-1}$ contour evaluated at the
SMF-28 ZDW.  For each process, the shifted/repeated channel is placed at
$f_{\rm ZDW}$, hence $f_d=f_{\rm ZDW}-s\Delta f$.  The contour uses the global
fourth-order Taylor model about the measured-profile ZDW and
$\eta=\operatorname{sinc}^2(L\Delta\beta/2)$ over the same fixed-$d$ plane as
the collapse inset.  The window spans at least ten channel periods so the
$e^{-1}$ level remains visible even when the highlighted islands themselves
are almost perfectly phase matched.  XPM is solid and FWM is dashed; when the
two configured shifts are equal, their ZDW contours coincide mathematically.

## 8. The fixed-d inset plane

The complete tuple domain is three-dimensional, while the inset is
two-dimensional. The inset uses the section

$$
\nu_d=0,
$$

so energy conservation fixes

$$
\widetilde\omega_c
=\widetilde\omega_a+
\widetilde\omega_b-
\omega_d.
$$

Its axes are

$$
X=\frac{f_a-f_d}{\Delta f},
\qquad
Y=\frac{f_b-f_d}{\Delta f}.
$$

The displayed lozenges and small triangles are exact intersections of the
finite $a$, $b$, and $c$ passbands on this fixed-$d$ plane. They are support
geometry, not contours of $\Delta\beta$.

On the same plane, the tuple phase is

$$
\Delta\beta_d(X,Y)
=\beta(\widetilde\omega_a)
+\beta(\widetilde\omega_b)
-\beta(\widetilde\omega_a+
\widetilde\omega_b-
\omega_d)
-\beta(\omega_d).
\tag{15}
$$

The phase contours should be evaluated from (15), without maximizing over a
hidden frequency, if they are to describe the same plane as the polygons.

## 9. Cubic and quartic phase geometry

Choose $\omega_d$ as the global expansion origin and write the fixed-$d$
frequency offsets simply as $\Omega_a$ and $\Omega_b$. Then
$\Omega_c=\Omega_a+\Omega_b$.

For a cubic global model,

$$
\boxed{
\Delta\beta
=-\Omega_a\Omega_b
\left[
\beta_2+\frac{\beta_3}{2}(\Omega_a+\Omega_b)
\right].}
\tag{16}
$$

The phase-matched branches are

$$
\Omega_a=0,
\qquad
\Omega_b=0,
\qquad
\Omega_a+\Omega_b=-\frac{2\beta_2}{\beta_3}.
$$

The additional diagonal branch is related to the ZDW but is not located at
the ZDW itself. From (2), it lies at twice the ZDW offset.

Including a global $\beta_4$ gives

$$
\boxed{
\Delta\beta
=-\Omega_a\Omega_b
\left[
\beta_2
+\frac{\beta_3}{2}(\Omega_a+\Omega_b)
+\frac{\beta_4}{12}
(2\Omega_a^2+3\Omega_a\Omega_b+2\Omega_b^2)
\right].}
\tag{17}
$$

The quartic term bends the additional phase-matching branch. This is a global
dispersion effect that appears in the tuple-level phase only after the four
global evaluations are combined.

## 10. Efficiency and the e^-1 contours

For a flat lossless span, the normalized complex longitudinal amplitude is

$$
A(\Delta\beta)
=\frac{1}{L}\int_0^L e^{i\Delta\beta z}\,dz
=e^{iL\Delta\beta/2}
\operatorname{sinc}\left(\frac{L\Delta\beta}{2}\right).
$$

The positive power efficiency used by the MC integrand is

$$
\boxed{
\eta=|A|^2
=\operatorname{sinc}^2\left(\frac{L\Delta\beta}{2}\right).}
\tag{18}
$$

Thus the inset contours labeled

$$
\eta=|A|^2=e^{-1}
$$

are power-efficiency contours. They are equivalently
$|A|=e^{-1/2}$ contours. They are not contours of the complex quantity $A$.

The principal $e^{-1}$ boundary satisfies

$$
|L\Delta\beta|=3.28854546.
$$

These contours show the finite-$x$ width of the coherent region. The central
phase-matching condition remains

$$
\Delta\beta=0.
$$

## 11. A compact hierarchy

The complete chain is

$$
\boxed{
\text{global }\beta(\omega)
\longrightarrow
\text{local }\{\beta_{n,j}\}
\longrightarrow
\Delta\beta_{dabc}(\boldsymbol\nu)
\longrightarrow
\{\Delta\beta_{\mathrm{center}},\boldsymbol g,H,\mu,x\}
\longrightarrow
\eta.}
$$

In words:

- The fiber supplies one global curve.
- Each channel samples that curve locally.
- A four-channel tuple combines four samples into one mismatch function.
- $x$ and $\mu$ summarize the scale and offset of that tuple-level function.
- The longitudinal propagator converts the mismatch into FWM efficiency.

Keeping these levels separate avoids treating a global coefficient such as
$\beta_3$, a local coefficient such as $\beta_{2,j}$, and a tuple quantity
such as $\nabla\Delta\beta$ as if they were interchangeable.

## 12. Current implementation boundary

The current code has several related but distinct paths:

- `System` can use sampled global dispersion profiles and splines.
- Full-band FWM builds channel-local $\beta_0$, $\beta_1$, and $\beta_2$ from
  those profiles; its vectorized production path remains quadratic.
- The controlled single-tuple Dar path supports global quadratic and cubic
  models and evaluates local terms through $\beta_3$.
- Generic Dar `FWMChannels` and the frequency-domain MC estimator carry local
  Taylor terms through $\beta_4$.
- The real-tuple validation obtains $\beta_0,\ldots,\beta_4$ as derivatives of
  one quintic spline of the measured global $\beta(\omega)$ profile.
- The direct time-domain pulse propagator remains quadratic and rejects
  nonzero $\beta_3$ or $\beta_4$ rather than silently omitting them.
- PCFM utilities also extract global coefficients through $\beta_4$.

Thus local fourth-order support is available for the Dar estimator and the ZDW
validation, but is not yet uniform across every numerical backend.
