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

1. It rescales the dispersion coefficients to obtain the requested $x$.
2. It shifts $\beta_{0,a}$ by a constant to obtain the requested $\mu$.

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
  those profiles.
- The controlled single-tuple Dar path supports global quadratic and cubic
  models and evaluates local terms through $\beta_3$.
- The direct time-domain pulse propagator remains quadratic and rejects
  nonzero $\beta_3$.
- PCFM utilities can extract global coefficients through $\beta_4$, but the
  generic Dar `FWMChannels` model does not yet carry a local $\beta_4$ term.

Therefore a fourth-order global clarification is mathematically useful, but
full fourth-order support is not yet uniform across every numerical backend.
