# High-detuning FWM oscillations at intermediate scale

Single-tuple FWM curves can show apparently isolated jumps at intermediate
values of the mismatch scale $x$, especially for the largest values of the
controlled detuning $|\mu|$. These points are generally samples of coherent
sinc lobes, not a change in the asymptotic power law or an instability of the
longitudinal propagator.

## Exact starting point

Using the notation of
[Single-tuple FWM scaling](fwm_single_tuple_scaling.md), write the
dimensionless mismatch as

$$
L\Delta\beta_x(\boldsymbol\Omega)
=x\left[\mu+\psi(\boldsymbol\Omega)\right],
$$

where $\mu$ is the normalized center detuning and $\psi$ is the variation over
the admissible four-band spectral domain $D$. The exact lossless propagator is

$$
A_x(\boldsymbol\Omega)
=\frac{e^{ix(\mu+\psi)}-1}{ix(\mu+\psi)}.
$$

Therefore

$$
\boxed{
\mathcal N(x)
=\frac{4}{x^2}
\int_D\rho(\boldsymbol\Omega)
\frac{\sin^2\!\left(
x[\mu+\psi(\boldsymbol\Omega)]/2
\right)}
{[\mu+\psi(\boldsymbol\Omega)]^2}
\,d^3\boldsymbol\Omega.}
\tag{1}
$$

No phase averaging has been applied in (1). The oscillations are already
present in this exact equation.

## Separation of the common and differential phases

Use

$$
4\sin^2(u/2)=2-2\cos(u)
$$

in (1). Defining

$$
w_\mu(\boldsymbol\Omega)
=\frac{\rho(\boldsymbol\Omega)}
{[\mu+\psi(\boldsymbol\Omega)]^2},
$$

gives the exact representation

$$
\mathcal N(x)
=\frac{2}{x^2}
\left[
W_\mu-
\operatorname{Re}\left(
e^{ix\mu}F_\mu(x)
\right)
\right],
\tag{2}
$$

where

$$
W_\mu=\int_Dw_\mu(\boldsymbol\Omega)\,d^3\boldsymbol\Omega
$$

and

$$
F_\mu(x)
=\int_Dw_\mu(\boldsymbol\Omega)
e^{ix\psi(\boldsymbol\Omega)}\,d^3\boldsymbol\Omega.
\tag{3}
$$

Equation (2) separates two effects:

- $e^{ix\mu}$ is the common center-detuning phase.
- $F_\mu(x)$ describes dephasing across the finite spectral domain.

The second term can alternately subtract from or add to the smooth envelope
$2W_\mu/x^2$. This produces minima and secondary lobes.

## High-detuning approximation

When

$$
|\mu|\gg\sup_{\boldsymbol\Omega\in D}|\psi(\boldsymbol\Omega)|,
$$

the denominator varies weakly over the passbands:

$$
w_\mu(\boldsymbol\Omega)
=\frac{\rho(\boldsymbol\Omega)}{\mu^2}
\left[1+O\left(\frac{\psi}{\mu}\right)\right].
$$

To leading order,

$$
\mathcal N(x)
\simeq
\frac{2}{\mu^2x^2}
\left[
M_0-
\operatorname{Re}\left(
e^{ix\mu}\widehat\rho_\psi(x)
\right)
\right],
\tag{4}
$$

with

$$
M_0=\int_D\rho(\boldsymbol\Omega)\,d^3\boldsymbol\Omega,
$$

and

$$
\widehat\rho_\psi(x)
=\int_D\rho(\boldsymbol\Omega)
e^{ix\psi(\boldsymbol\Omega)}\,d^3\boldsymbol\Omega.
$$

The rapid common phase has approximate period

$$
\boxed{\Delta x\simeq\frac{2\pi}{|\mu|}.}
\tag{5}
$$

Increasing $|\mu|$ therefore places more oscillations inside the same
intermediate-$x$ interval.

## Why the behavior is restricted to intermediate x

At sufficiently small $x$, the differential phase
$x\psi(\boldsymbol\Omega)$ is small across the domain. Spectral samples then
remain coherent, and the curve resembles a single sinc-squared propagator
with pronounced minima and lobes.

At sufficiently large $x$, the differential phase varies rapidly over $D$.
Under the regularity assumptions used in the scaling note, the
Riemann--Lebesgue lemma gives

$$
F_\mu(x)\longrightarrow0.
$$

Equation (2) then reduces to the smooth nonresonant envelope

$$
\boxed{
\mathcal N(x)\sim\frac{2W_\mu}{x^2}.}
\tag{6}
$$

Thus the intermediate oscillations and the eventual $x^{-2}$ law are two
regimes of the same exact equation. The oscillations do not imply a new
asymptotic exponent.

## Why logarithmic sampling looks like jumping points

The script uses a geometrically spaced grid in $x$. Its local spacing grows
approximately in proportion to $x$, while the common-phase period (5) is
approximately constant. Eventually the grid spacing becomes comparable to or
larger than one lobe:

$$
\Delta x_{\mathrm{grid}}\gtrsim\frac{2\pi}{|\mu|}.
$$

Neighboring plotted points can then land near a sinc minimum and a secondary
maximum, respectively. Connecting those sparse samples makes a continuous
oscillation look like one or more isolated jumps. This is an aliasing or
under-resolution effect in the plotted $x$ grid, even when every individual
integral value is correct.

A practical resolution condition is

$$
\boxed{
\Delta x_{\mathrm{grid}}
\ll\frac{2\pi}{|\mu_{\max}|}.}
\tag{7}
$$

A locally linear or adaptively refined grid is more efficient than increasing
the number of logarithmic points uniformly over the complete range.

## Physical lobes versus Monte Carlo noise

The Monte Carlo estimator adds sampling fluctuations to the physical
oscillations. The script deliberately reuses the same random frequencies for
all $x$ values within a seed. This common-random-number construction makes
neighboring errors correlated and improves comparisons, but residual
correlated wiggles can remain when the contribution is very small.

The two effects can be distinguished as follows:

- Increase `--n-x`: physical lobes resolve into smooth oscillations.
- Increase `--n-samples`: Monte Carlo scatter and error bars decrease, while
  physical minima remain at the same $x$.
- Increase `--n-seeds`: seed-dependent fluctuations decrease.
- Change `--seed`: sampling artifacts move, while resolved physical lobes
  remain fixed.
- Compare the jump size with `normalized_stderr`: a displacement much larger
  than the reported uncertainty is unlikely to be sampling noise alone.

For very deep minima, relative Monte Carlo errors can become large even when
the absolute error is small. Logarithmic plotting makes these minima visually
prominent.

## Recommended diagnostic sweep

For a suspected high-$\mu$ lobe, first identify a narrow interval
$[x_1,x_2]$ around the jump and evaluate it on a linear grid satisfying (7).
Then repeat with at least two sample counts and seeds. The interpretation is:

$$
\begin{array}{c|c}
\text{Feature remains at fixed }x\text{ as sampling increases}
& \text{coherent propagator lobe},\\
\text{Feature moves or shrinks with samples/seeds}
& \text{Monte Carlo fluctuation},\\
\text{Feature disappears only when }x\text{ is sampled densely}
& \text{coarse-grid aliasing}.
\end{array}
$$

The most common outcome at high $|\mu|$ is a physical sinc-lobe structure
that is visually exaggerated by an under-resolved logarithmic grid, with a
smaller Monte Carlo contribution superimposed.
