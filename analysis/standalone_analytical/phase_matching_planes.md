# Phase-matching planes in the 3D FWM frequency space

This note connects two representations that so far lived apart:

- `the-domain-in-freq-space-{single,multiple,01}.py`, which draw, in the 3D space
  $(\nu_1,\nu_2,\nu_3)$, the **frequency**-matching support of FWM (the cube
  $|\nu_j|\le B/2$ clipped by the slab $|\nu_1-\nu_2+\nu_3|\le B/2$, replicated on
  the WDM grid);
- `analysis/fwm/fwm-efficiency/fwm_and_dispersion.ipynb`, which draws, in a 2D map,
  the **phase**-matching contours $\Delta\beta=0$.

The question this note answers: *is the 2D contour set the projection of a set of
planes in the 3D space?* The answer is yes, with one qualification — for a
dispersion law truncated at $\beta_3$ the phase-matching locus is **exactly three
planes**; $\beta_4$ leaves two of them exactly planar and bends the third into a
quadric.

The companion script is `phase-matching-planes.py`.

## 1. Coordinates

Keep the sign convention of the sibling scripts,

$$\omega_1-\omega_2+\omega_3=\omega_4,\qquad \nu_j=\omega_j-\omega_{\rm COI},$$

so $\nu_4=\nu_1-\nu_2+\nu_3$ is eliminated and $(\nu_1,\nu_2,\nu_3)$ is a complete,
unconstrained coordinate system: every FWM tuple is one point of $\mathbb{R}^3$.
Energy conservation is already built into the coordinates — this is why the picture
is 3D and not 4D.

The dispersion law is the notebook's Taylor expansion about a fixed reference
$\omega_r$, and we write $\delta=\omega_{\rm COI}-\omega_r$ for the COI offset from
that reference, so that $\omega_j = \omega_r + \delta + \nu_j$. The phase mismatch
of the tuple is

$$\Delta\beta=\beta(\omega_1)+\beta(\omega_3)-\beta(\omega_2)-\beta(\omega_4).$$

## 2. The exact factorisation

Write $X_j=\omega_j-\omega_r=\delta+\nu_j$. Energy conservation says the two pairs
$\{X_1,X_3\}$ and $\{X_2,X_4\}$ have the **same mean**. Introduce that common mean
and the two half-splittings,

$$S=\frac{X_1+X_3}{2}=\frac{X_2+X_4}{2},\qquad
p=\frac{X_1-X_3}{2},\qquad q=\frac{X_2-X_4}{2}.$$

Substituting $X_1=S+p$, $X_3=S-p$, $X_2=S+q$, $X_4=S-q$ into
$\beta=\sum_{n\le4}\beta_n X^n/n!$, every odd power cancels pairwise and one is left
with

$$\boxed{\;\Delta\beta=\bigl(p^{2}-q^{2}\bigr)\left[\;\bar\beta_2(S)+\frac{\beta_4}{12}\bigl(p^{2}+q^{2}\bigr)\right]\;}$$

where

$$\bar\beta_2(S)=\beta_2+\beta_3 S+\tfrac12\beta_4 S^{2}$$

is simply **the local GVD evaluated at the common mean frequency**. Note that
$\beta_0$ and $\beta_1$ drop out identically — $\beta_1$ because the two pairs share
a mean, which is the coordinate-free statement of the notebook's remark that
$\beta_1$ cancels from the maps.

In the script coordinates the splittings are

$$p-q=\nu_1-\nu_2,\qquad p+q=\nu_2-\nu_3,\qquad
S=\delta+\frac{\nu_1+\nu_3}{2},$$

so $p^2-q^2=(\nu_1-\nu_2)(\nu_2-\nu_3)$ and
$p^2+q^2=\tfrac12\bigl[(\nu_1-\nu_2)^2+(\nu_2-\nu_3)^2\bigr]$, giving the form used
in the code:

$$\Delta\beta=(\nu_1-\nu_2)(\nu_2-\nu_3)\left[\bar\beta_2\!\left(\delta+\tfrac{\nu_1+\nu_3}{2}\right)
+\frac{\beta_4}{24}\Bigl((\nu_1-\nu_2)^2+(\nu_2-\nu_3)^2\Bigr)\right].$$

`phase-matching-planes.py` checks this against a term-by-term evaluation on 2000
random tuples; the residual is at the $10^{-13}\,\mathrm{m^{-1}}$ level against
mismatches of order $10^{2}\,\mathrm{m^{-1}}$, i.e. exact to rounding.

## 3. The three components of $\Delta\beta=0$

The zero set is the union of three surfaces.

**$P_1:\ \nu_1=\nu_2$** (equivalently $p=q$, i.e. $\omega_1=\omega_2$ *and*
$\omega_3=\omega_4$). A plane with normal $(1,-1,0)$, exact at every dispersion
order. This is the degenerate limit in which one "pump" coincides with one output:
the XPM sector, not genuine FWM.

**$P_2:\ \nu_2=\nu_3$** (equivalently $p=-q$, i.e. $\omega_3=\omega_2$ and
$\omega_1=\omega_4$). Normal $(0,1,-1)$, the other XPM sector.

Both are pure consequences of energy conservation plus the pairwise structure of
$\Delta\beta$: they hold for *any* $\beta(\omega)$, not just a quartic. They are the
3D origin of the notebook's remark that the zero-offset axis lines are always phase
matched.

**$Q$:** the genuinely nondegenerate branch,

$$\bar\beta_2(S)+\frac{\beta_4}{12}\bigl(p^{2}+q^{2}\bigr)=0 .$$

Two regimes:

- **$\beta_4=0$.** The condition reduces to $\bar\beta_2(S)=0$, i.e.
  $S=S_{\rm ZDF}=-\beta_2/\beta_3$, independent of $p$ and $q$. In $\nu$ coordinates
  this is the **plane**

  $$\nu_1+\nu_3=2\nu_{\rm ZDF},\qquad \nu_{\rm ZDF}=\omega_{\rm ZDF}-\omega_{\rm COI},$$

  with normal $(1,0,1)$. Physically: *the arithmetic mean of the two pumps sits at
  the zero-dispersion frequency*. So through $\beta_3$ the phase-matching locus is
  exactly **three planes**, with normals $(1,-1,0)$, $(0,1,-1)$, $(1,0,1)$ — and
  this is the affirmative answer to the original question.

- **$\beta_4\ne0$.** $Q$ becomes a quadric. It is most cleanly parametrised by
  $(p,q)$ rather than by $(\nu_1,\nu_3)$, because the condition then determines $S$
  alone:

  $$\frac{\beta_4}{2}S^{2}+\beta_3 S+\beta_2+\frac{\beta_4}{12}\bigl(p^{2}+q^{2}\bigr)=0 .$$

  Taking the root continuous with the $\beta_4\to0$ limit gives a single smooth
  sheet, which is the tangent plane $\nu_1+\nu_3=2\nu_{\rm ZDF}$ bent by a
  curvature $\propto\beta_4$. The bending is quadratic in the *splittings*, not in
  the absolute offsets: it is invisible for closely spaced tuples and grows over
  full-band ones. For the SMF-28 coefficients and a $\pm15$ THz box the script
  measures a maximum departure of $\approx1.2$ THz — small enough that "three
  planes" remains a good working picture, large enough to matter for UWB tuples.

  A caveat on $\bar\beta_2$: because it is quadratic in $S$, it has a second root
  (at $\approx80$ THz from the COI in the script's configuration). That root is an
  artefact of truncating the Taylor expansion, not a physical second ZDF, and the
  script deliberately follows only the branch anchored at the first root.

## 4. The notebook map is the slice $\nu_4=0$

The notebook fixes one of the four waves at the COI and sweeps the other two:
$\omega_1=\omega_{\rm COI}+\Omega_1$, $\omega_2=\omega_{\rm COI}+\Omega_2$,
$\omega_3=\omega_{\rm COI}+\Omega_1+\Omega_2$. In the coordinates above this is

$$\nu_1=\Omega_1,\qquad \nu_3=\Omega_2,\qquad \nu_2=\Omega_1+\Omega_2,\qquad \nu_4=0,$$

i.e. **the plane $\nu_1-\nu_2+\nu_3=0$** in the 3D space — exactly the central slab
face of the frequency-matching region drawn by the sibling scripts. So the 2D map is
not a projection but a *section*, and the phase-matching lines on it are the traces
of $P_1$, $P_2$, $Q$:

| 3D surface | trace on $\nu_4=0$ |
|---|---|
| $P_1:\ \nu_1=\nu_2$ | $\Omega_2=0$ (horizontal axis line) |
| $P_2:\ \nu_2=\nu_3$ | $\Omega_1=0$ (vertical axis line) |
| $Q$ ($\beta_4=0$) | $\Omega_1+\Omega_2=2\nu_{\rm ZDF}$ (anti-diagonal line) |
| $Q$ ($\beta_4\ne0$) | $\bar\beta_2\!\left(\delta+\frac{\Omega_1+\Omega_2}{2}\right)+\frac{\beta_4}{24}\bigl(\Omega_1^2+\Omega_2^2\bigr)=0$ (conic) |

Restricting the boxed formula to this slice gives $p^2-q^2=-\Omega_1\Omega_2$
directly, so $\Delta\beta=-\Omega_1\Omega_2\,[\cdots]$ — three lines, two of them the
axes. In the pure-$\beta_2$ case the bracket is the constant $\beta_2$ and one
recovers $\Delta\beta=-\beta_2\Omega_1\Omega_2$, the textbook result.

This also explains why the notebook's *default* map shows only two lines: with
COI = reference = 193.4 THz the anti-diagonal sits at
$\Omega_1+\Omega_2=2\nu_{\rm ZDF}\approx69$ THz, far outside the $\pm5$ THz default
span. Moving the COI toward the ZDF (the script uses 220 THz) brings the third line
into view. The right panel of the figure reproduces the notebook map from the
factorised formula and shows all three traces.

## 5. Channel placement: which cubes each plane crosses

Overlay the WDM grid of the sibling scripts: channel $n$ occupies
$\nu\in n\Omega_0\pm B/2$, so tuple $(n_1,n_2,n_3)$ owns the cube of side $B$
centred at $\Omega_0(n_1,n_2,n_3)$. A plane with unit-integer normal crosses that
cube iff the plane's linear form, whose range over the cube is the centre value
$\pm B$, changes sign. This turns each surface into a **channel selection rule**
(writing $B/\Omega_0=R_s/\Delta f$):

- $P_1$ crosses $(n_1,n_2,n_3)$ iff $|n_1-n_2|\le R_s/\Delta f$;
- $P_2$ iff $|n_2-n_3|\le R_s/\Delta f$;
- $Q$ (at $\beta_4=0$) iff $\bigl|n_1+n_3-2\nu_{\rm ZDF}/\Omega_0\bigr|\le R_s/\Delta f$.

For any non-oversubscribed grid, $\Delta f\ge R_s$, so the two XPM planes collapse to
the single-index conditions $n_1=n_2$ and $n_2=n_3$ — they touch only the degenerate
tuples, as they should. The third plane instead selects an **anti-diagonal band of
pump pairs**, $n_1+n_3\approx 2\nu_{\rm ZDF}/\Omega_0$, with $n_2$ free:
a one-parameter family of channel triples straddling the zero-dispersion frequency,
and the only place where genuinely nondegenerate FWM is phase matched. This is the
geometric statement behind the usual "keep the grid away from the ZDF" rule, and it
is what the script's cube wireframes mark.

The $\beta_4$ bending widens that band slightly and tilts it; the script does not use
the linearised rule but tests the exact sign change of the bracket over the eight
corners of each cube.

## 6. What the script draws

`phase-matching-planes.py` writes `media/standalone_analytical/phase_matching_planes.pdf`
(and `.png`). Left panel, in the 3D offset space: the two XPM planes, the tangent
plane $\nu_1+\nu_3=2\nu_{\rm ZDF}$, the exact $\beta_4$-bent sheet on top of it, the
dashed notebook slice $\nu_4=0$, and the channel cubes the sheet crosses. Right
panel: the notebook's 2D map recomputed on that slice, with the $\Delta\beta=0$
contour showing the same three curves.

It also prints three diagnostics: the factorisation residual, the zero-dispersion
offsets, and the maximum in-box bending of the sheet away from its tangent plane.

For a rotatable version, `phase-matching-planes-interactive.py` reuses the same
geometry helpers and writes
`media/standalone_analytical/phase_matching_planes.html`: a self-contained page with
an orthographic canvas renderer (drag to rotate, wheel to zoom, shift-drag to pan),
per-layer toggles and the four axis presets of the sibling scripts. Both scripts use
only the project's declared dependencies -- unlike the plotly siblings, which need a
package that is not in `pyproject.toml`.

## 7. Where this is used

Section 10.4 of [`lorenzi_fast_method.md`](../../docs/source/lorenzi_fast_method.md)
applies the factorisation to the walk-off orientation of the Fast estimator. The
link is that the per-leg walk-off vector $\mathbf c=(\nu_a,\nu_b,-\nu_c)$ of that
note is the gradient $\nabla\Delta\beta$ of the object factorised here, so §2 gives
it in closed form. The consequence is that the walk-off direction is pinned to a
single orientation class exactly on the surfaces above — on $Q$ the two
unconjugated legs share a group velocity and the third is frozen to the target.
The claims are machine-checked in
[`verify_walkoff_orientation.py`](verify_walkoff_orientation.py).

Section 10.5 of the same note uses the factorisation a second way. Applied to
the *in-band-shifted* frequencies it is exact, so the accumulated phase is a
product of three factors each affine in the in-band offsets — which is what
controls when the estimator's linear phase model stops being adequate, and why
its dominant error vanishes on the sheet $Q$. Checked in
[`verify_linear_phase_validity.py`](verify_linear_phase_validity.py).

## 8. Open points

- The Taylor model is anchored at a fixed $\omega_r$ and is stretched at the
  $\pm15$ THz offsets used here. For quantitative UWB work the surfaces should be
  recomputed from the tabulated `input/fiber_data/smf28.csv` $\beta(\omega)$; the
  factorisation of §2 does *not* survive that (it is specific to a quartic), but
  $P_1$ and $P_2$ do, since they follow from the pair structure alone.
- Only the $\Delta\beta=0$ locus is drawn. The efficiency-weighted picture wants the
  slab $|\Delta\beta|\lesssim 2\pi/L_{\rm eff}$ around each surface, whose thickness
  is set by $\partial\Delta\beta/\partial\nu$ and is strongly anisotropic — that
  thickness, not the surface itself, is what controls how many tuples actually
  contribute.
