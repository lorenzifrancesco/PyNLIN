# Publishable results and literature novelty assessment

*Compiled 2026-08-24. Inventory of the verifiable, quantitative claims supported
by this repository, a literature check of each against the state of the art,
and a bundling/priority plan. Companion documents:
[`lorenzi_fast_method.md`](../lorenzi_fast_method.md),
[`direct_sector_mc.md`](../direct_sector_mc.md),
[`fwm_single_tuple_scaling.md`](../fwm_single_tuple_scaling.md).*

## 1. Candidate publishable claims

### Claim A — nPC sector structure of XPM NLIN collapses to universal power laws

**Statement (corrected 2026-08-24; supersedes the fitted power laws).** At
high DGD, **all four collision sectors scale as DGD$^{-1}$**: the sector
ratios $\rho_3 = 3\mathrm{PC}/2\mathrm{PC}$ and
$\rho_4 = 4\mathrm{PC}/2\mathrm{PC}$ tend to nonzero constants that depend on
the pair geometry through the spacing-to-baud ratio $q = \Delta f/B$. Proven
analytically (F. Lorenzi, private notes; an independent in-repo derivation
with explicit formulas for the asymptotic constants is
[`npc_sector_asymptotics.md`](../npc_sector_asymptotics.md)) and verified with
the direct CRN estimator far beyond the original sweep range,
$\nu = L/L_W$ up to $10^5$ at $q \in \{1, 2, 4\}$. The limit-theorem
constants match the MC plateaus to $<1\%$ on all four sectors and all three
geometries; the total is closed-form,
$C_{N_1}(q) = q[(q{+}1)\ln(q{+}1) + (q{-}1)\ln(q{-}1) - 2q\ln q]$
($= 2\ln 2$ at $q{=}1$; MC $1.385$), and the asymptotic ratios are
$\rho_3^\infty = 0.237 / 0.0785 / 0.0357$,
$\rho_4^\infty = 0.314 / 0.0622 / 0.0271$ at $q = 1/2/4$ (single span,
$\alpha = 0$; the slow $\rho_4$ drift at $q{=}1$ is a finite-$\nu$ transient
toward $0.314$ — the limit integral is convergent).

Mechanism (sketch, consistent with the numerics): conditional on the shared
outer frequency $x$, the sectors are the ANOVA (Sobol–Hoeffding) components
of the masked link kernel over the inner frequencies $(a, u)$. The kernel
concentrates on the phase-matched strip $|x| \lesssim 1/\nu$, where the phase
is $-\nu x\,(1 + (u - x - a)/\omega)$, $\omega = 2\pi q$: the in-band
modulation $(u-x-a)/\omega$ remains **order one** as $\nu \to \infty$, so
every interaction sector inherits the strip's $1/\nu$ scaling with a
$q$-dependent constant (at $q = 1$ the second branch of the Dar hyperbola
additionally grazes the domain corner $a = \pi$, $u - x = -\pi$). A naive
mask-only factorization — which would predict
$3\mathrm{PC} \sim \ln\nu/\nu^2$, $4\mathrm{PC} \sim \nu^{-2}$ — is
quantitatively refuted by the same numerics (it under-predicts $\rho_3$ at
$q{=}4$, $\nu{=}3{\times}10^3$ by $\sim$20×).

**Retraction.** The previously fitted power laws
$\rho_3 \approx 0.133\,\nu^{-0.31}$, $\rho_4 \approx 0.019\,\nu^{+0.29}$
(band sweep, fit range $\nu \in [5, 300]$) are **pre-asymptotic transients**
of the crossover between the low-walk-off plateau
($\rho_3 \to \approx 0.21$, $\rho_4 \to \approx 0.11$ for $\nu \lesssim 1$)
and the asymptotic constants — the fit range was too short, and the exponents
$\mp 1/3$ are not physical. The 4PC-over-3PC crossover survives as a
transient/geometry feature, not an asymptotic law. Likewise the
"single-coordinate $L/L_W$" reduction must be amended: the ratios depend on
$(L/L_W,\ q)$ separately (the band sweep held $q$ fixed, hiding the second
coordinate).

Supporting verifiable facts: 3PCa = 3PCb to 1% across the band (median
$|{\rm ratio}-1| = 0.0105$), and **exactly** in the scaling limit (symmetry
proof in [`npc_sector_asymptotics.md`](../npc_sector_asymptotics.md) §3); the
natural guess $4\mathrm{PC} = \rho^2\,2\mathrm{PC}$ is falsified by 13×; at
$q = 1$ the 4PC-over-3PC crossover is a genuine asymptotic feature
($C_{4\mathrm{PC}} > C_{3\mathrm{PC,tot}}$).

**Novelty verdict: novel, with one reconciliation obligation.** The prior art
is [Dar, Feder, Mecozzi & Shtaif, *Pulse Collision Picture*, JLT
2016](https://opg.optica.org/jlt/abstract.cfm?uri=jlt-34-2-593) (and
[*Accumulation of NLIN*](https://arxiv.org/pdf/1310.6137)): qualitative
2/3/4-pulse classification, complete/incomplete collisions, 2PC =
format-dependent phase noise vs 4PC = format-independent circular noise. Their
only quantitative scaling: phase noise $\propto \Omega_s^{-1}$, **"all other
terms $\propto \Omega_s^{-2}$"** — a statement about *channel separation* at
fixed link, not about the DGD asymptotics at fixed pair geometry. Not found
anywhere: the all-sectors DGD$^{-1}$ law with its geometry-dependent ratio
constants $\rho^\infty(q)$, the two-coordinate $(L/L_W, q)$ structure, the
transient (V-shape / crossover) phenomenology, the 3PCa≡3PCb identity, the
rank-1 falsification.

**Referee obligation (was "risk", now resolvable).** Dar's $\Omega_s^{-2}$
lives in the $q$-dependence of the asymptotic constants (which decay with
spacing), not in the $\nu$-dependence — the paper should state this
reconciliation explicitly and derive the $\rho^\infty(q)$ law. Secondary
obligations: multi-span / lossy generalization (the constants above are
single-span flat-power), and pinning down the slow residual drift of
$\rho_4$ at $q = 1$ (still $\sim$3%/decade at $\nu = 10^5$ — logarithmic
correction or unconverged constant).

**Verification record (2026-08-24).** Direct CRN estimator, $4{\times}10^6$
samples × 4 seeds per point, $\nu = 10 \dots 10^5$ (13 log-spaced points),
$q \in \{1,2,4\}$, single span, $\alpha = 0$: $2\mathrm{PC}\cdot\nu$,
$3\mathrm{PC}\cdot\nu$, $4\mathrm{PC}\cdot\nu$ all tend to constants; local
log-slopes of the ratios decay to $\approx 0$ (within seed noise) above
$\nu \sim 10^3$. The early-range local slopes ($-0.45\dots{-0.2}$ for
$\rho_3$, $+0.39\dots{+0.16}$ for $\rho_4$ at $\nu = 10\!-\!100$) bracket the
previously fitted $\mp 0.3$, confirming the fit-range diagnosis. Script:
scratchpad `sector_asymptotics.py` (to be promoted to `analysis/`).

### Claim B — Mechanism-resolved full-band NLIN: strict FWM can dominate XPM over OESCLU

**Statement (pending confirmation).** At full spectral resolution over the
OESCLU bands (2284 × 25 GHz channels, ZDW inside the O band), strict
inter-channel FWM exceeds XPM NLIN over most of the band, reaching ~70% of the
total, with deep notches at inter-band guard gaps; NSR −24.6…−10.7 dB at
−5 dBm.

**Status.** NOT yet citable: the 72% figure is from the 1200-channel run
predating the band-alignment fix; the corrected full-resolution S5 run is
checkpointed at 735/2284 targets. Blockers: finish S5; one end-to-end SSFM
spot-check at a few channels.

**Novelty verdict: the *message* is taken, the *result* is not.** The field
converged in 2024–2026 on "FWM/MCI must be included near the ZDW":

- [UCL O-to-U ISRS-GN numerical integral model
  (arXiv 2401.18022, 2024)](https://arxiv.org/html/2401.18022): FWM/MCI by
  numerical integration, 1260–1675 nm — but 589 channels at 100 GHz,
  Gaussian-only, 4×V100 GPUs (3.6 s/eval), 0.18–0.79 dB mean error at the ZDW,
  **no FWM-vs-XPM decomposition**.
- [Closed-form GN with FWM efficiency for O-band
  (arXiv 2510.11867, JLT 2026)](https://arxiv.org/abs/2510.11867): closed-form
  FWM term, ≤161 channels / 16.1 THz — no decomposition, narrower band.
- [Poggiolini group, PCFM2 "any island" closed-form
  (arXiv 2602.03860, Feb 2026)](https://arxiv.org/html/2602.03860): polynomial
  closed forms on arbitrary GN islands incl. $\beta_2 \to 0$ — SCI/XCI/MCI
  evaluated *collectively*, frequency-domain only, no mechanism breakdown.

What nobody has published, i.e. the reframed contribution:

1. a **mechanism-resolved decomposition** (how much of NLIN is strict FWM,
   per channel, across the band) — GN island integrals structurally cannot
   produce this;
2. **full 25 GHz / 2284-channel resolution** — 4–14× finer than any competitor;
   our decimation result (Claim E2) proves resolution matters for FWM, which
   retroactively questions coarse-grid MCI claims;
3. a **time-domain / pulse-collision route across the ZDW** at all — the
   collision literature is entirely C-band;
4. a path to **modulation-format dependence near the ZDW** (via Claim A) —
   every competitor is explicitly Gaussian-only.

**Referee risks.** (i) **ISRS**: all three competitors include it; we are
$\alpha = 0$ flat-power, single span, Gaussian symbols. Either implement via
the documented $\hat K = |\mathcal F[\rho(z)]|^2$ extension point or scope the
claim explicitly ("dispersion-limited NLIN at flat power") and expect the
objection. (ii) **Urgency**: with PCFM2 dated Feb 2026, the window is months,
not years.

### Claim C — Two-coordinate universality and factorization of the FWM census

**Statement.** Any strict-FWM tuple's efficiency is a universal function of
two dimensionless coordinates — $x_\nabla = LB\lVert\nabla\Delta\beta\rVert_2$
and the dimensionless detuning $\mu = u_0/x_\nabla$ ($u_0 = L\Delta\beta_{\rm center}$) — times a quantized grid-acceptance factor:
verified median ratio 1.000 with 99.7% of 400k tuples within 2×. The
full-band mass map factorizes as (channel-plan population histogram) ×
(universal interaction kernel) × (acceptance) — bin-level median deviation
−0.031 dex (raw bins) — making the NLIN census of any future channel plan predictable
without evaluating a single tuple. The kernel carries the parameter-free
dichotomy $|u_0| < W$: surface-crossing tuples scale as $x_\nabla^{-1}$,
gapped as $x_\nabla^{-2}$ (fitted exponents −0.999, −2.000); on the full grid
the boundary separates 99.1% of the *mass* from 99.1% of the *tuples* (near
regime: 99.95% of mass in 0.5% of tuples; 50% of mass in the top 380 of
69.5M).

**Novelty verdict: nothing comparable found.** Closest structural relative is
the GN island decomposition itself ([Poggiolini
framework](https://arxiv.org/pdf/1704.06461), PCFM2), which evaluates islands
but makes no universality claim and has no predict-from-population corollary.
The $x^{-2}$ law has classical echoes (CW FWM efficiency
$\eta \propto 1/\Delta\beta^2$, Forghieri/Inoue 1990s; Dar's $\Omega^{-2}$),
but the classification with a parameter-free boundary, the $x^{-1}$
surface-crossing class, and the verified factorization appear new.

**Referee risk.** Framing, not prior art: must read as physics (a universality
/ master-curve result), not internal validation of our estimator.

### Claim D — Certified fast estimator: first NLIN computation with a truncation guarantee

**Statement.** A proved per-tuple bound
$N\,T^2\!/L^2 \le A(d)\min(1, 4/g^2)$, $g = |u_0| - W - P_q$ (with $N\,T^2\!/L^2$ the normalized per-tuple collision sum — the notation of `direct_sector_mc.md`, replacing the earlier ad hoc symbol $F$), permits pruning ~$10^7$
tuples/target to thousands while reporting a rigorous per-target certificate on
the discarded mass, with the exhaustive computation as the
$\varepsilon \to 0$ limit; survivors are evaluated by regime-dispatched
analytics (exact XPM 1D reduction with sheet limit $F \to 1/|\nu|$; exact
zonotope conditional acceptance) at sub-percent measured error, seconds per
target at full resolution. Unifying spine: the sheet-limit density formula
$\hat F_{\rm sheet} = 2\pi\,\rho_{\rm joint}(u{=}0)$ covers both XPM pairs
(reproducing $1/|\nu|$ exactly) and wide FWM tuples (3-digit agreement).

**Novelty verdict: no prior art found.** Nothing in the GN/EGN or collision
literature offers a proved inequality on neglected contributions; MCI terms
are historically dropped or kept by heuristic distance cuts. "First NLIN
computation with a rigorous truncation certificate" appears safe. The exact
zonotope conditional-acceptance law is a self-contained lemma with no analog
found (GN handles support exactly via island geometry; ours is the time-domain
counterpart).

### Claim E — Methodological results (each independently citable)

1. **Direct CRN sector estimator** (`estimate_xhkm_sectors_direct_mc`):
   sector-resolved NLIN MC via common-random-number paired projections of the
   Golani tensor; unbiased (<1% vs high-N oracle), 7×/9×/19× seed-scatter
   reduction (3PCa/3PCb/4PC at N=10⁵), 40× tighter 4PC SEM at high N —
   turned previously unresolvable quantities (4PC at high walk-off: 52%
   scatter at max budget) into 2–3%-resolved at 1/10 the samples. *No
   domain-specific variance reduction for NLIN sector MC found in the
   literature* (EGN MC is used at brute-force sample counts, e.g.
   [distribution-matching work](https://arxiv.org/pdf/1907.02846)). Novel but
   methods-note level; best framed as the method section of Claim A ("the
   estimator that made the physics measurable").
2. **Interferer-decimation artifact**: decimating interferers (not just
   targets) is a physics change for FWM, not a sampling approximation —
   5-order-of-magnitude spectral artifacts at decimation 4, absent at full
   resolution. Not documented anywhere found; a short, sharp negative result
   other UWB groups would cite. Belongs inside paper B.

## 2. Bundling and priority

- **Paper A (sector scaling laws)** = Claim A + Claim E1 as method section.
  Safest and most distinctive: prior art is a single well-cited 2016 paper it
  visibly extends. Strongest with (i) the analytic DGD$^{-1}$ proof
  transcribed, plus a derivation of the asymptotic ratio constants
  $\rho^\infty(q)$, (ii) the explicit statement that Dar's $\Omega^{-2}$
  lives in the $q$-dependence of those constants, (iii) multi-span / lossy
  generalization or explicit scoping.
- **Paper B (full-band mechanism-resolved NLIN)** = Claims B + C + D + E2.
  Reposition from "FWM matters" (lost to 2024–2026 GN literature) to "first
  mechanism-resolved, full-resolution, certificate-carrying decomposition of
  UWB NLIN, on a formalism that extends to arbitrary formats". Gated on:
  finishing the S5 full-resolution run (735/2284), an SSFM spot-check, and an
  answer to the ISRS objection. Time-sensitive (PCFM2: Feb 2026).
- The internal gate numbers (0.52% bulk error, 0.07% quadratic omission,
  0.94–1.04 MC ratios, Sobol sweep medians) are the *validation section* of
  paper B, not claims in themselves.

## 3. Open items blocking submission

| Item | Blocks | Status |
|---|---|---|
| Finish S5 full-resolution run (2284 ch, dec 1) | Claim B | checkpointed 835/2284, resumable |
| SSFM end-to-end spot-check at probe channels | Claim B | interface exists (`ssfm_interface.py`) |
| ISRS: implement $\rho(z)$ kernel or scope explicitly | Claim B | extension point documented |
| Merge F. Lorenzi's proof with the in-repo derivation | Claim A | in-repo reconstruction done ([`npc_sector_asymptotics.md`](../npc_sector_asymptotics.md)); private-notes proof to merge |
| Closed forms for $C_{2\mathrm{PC}}, C_{3\mathrm{PC}}, C_{4\mathrm{PC}}$ | Claim A (polish) | limit integrals derived + verified <1%; only $C_{N_1}$ closed so far |
| Large-$q$ asymptotics of the constants (Dar-$\Omega^{-2}$ link) | Claim A | reconciliation stated (§5 of the note); $q$-law derivation open |
| 3PCa = 3PCb symmetry proof | Claim A | **done in the scaling limit** (note §3); finite-$\nu$ case open |
| Multi-span / lossy generalization of sector laws | Claim A (scope) | open |
| Promote high-DGD sweep (`sector_asymptotics.py`) to `analysis/` | Claim A (reproducibility) | scratchpad only |
| S3 tube-construction stage | Claim D (completeness) | v1 implemented + tested (`fast_analytic.py`, 2026-08-24); first gate: mid-C target, $\varepsilon{=}10^{-6}$ keeps 1.7e-3 of tuples at sum ratio 0.9998 with 9e-4 certificate; fitted bridge + geometric enumeration open |
| XPM close-spacing correction: fast pair model omits in-band curvature; true $N_1/L^2 \to C_{N_1}(q)/\nu$ vs fast $1/\nu$ — 32% deficit measured at $q{=}1$, $\nu{=}300$ (asymptote $2\ln 2$); fix via the closed-form $C_{N_1}(q)$ | Claim B accuracy (adjacent pairs dominate XPM) | found 2026-08-24; open |
| Re-measure S2/S4 gates on corrected 2284-ch grid | paper B validation | scheduled |

## 4. Literature checked (2026-08-24)

- [Dar, Feder, Mecozzi, Shtaif — Pulse Collision Picture, JLT 34(2), 2016](https://opg.optica.org/jlt/abstract.cfm?uri=jlt-34-2-593)
- [Dar et al. — Accumulation of NLIN in fiber-optic systems](https://arxiv.org/pdf/1310.6137)
- [UCL — Optimising O-to-U Band Transmission Using Fast ISRS GN Numerical Integral Model (arXiv 2401.18022)](https://arxiv.org/html/2401.18022)
- [Closed-form GN Model Supporting O-Band Transmission (arXiv 2510.11867, JLT 2026)](https://arxiv.org/abs/2510.11867)
- [Gao, Jiang, Poggiolini — PCFM2: Coherent Polynomials Closed-Form Model for Any Island (arXiv 2602.03860)](https://arxiv.org/html/2602.03860)
- [CFM6 — closed-form NLI EGN model, multiband, arbitrary Raman (arXiv 2405.08512)](https://arxiv.org/pdf/2405.08512)
- [General Nonlinear Model for Arbitrary Modulation Formats with ISRS (arXiv 2509.10009)](https://arxiv.org/pdf/2509.10009)
- [Poggiolini et al. — GN model detailed derivation (arXiv 1209.0394)](https://arxiv.org/pdf/1209.0394); [signal-noise interactions (arXiv 1704.06461)](https://arxiv.org/pdf/1704.06461)
- [Distribution matching for the nonlinear fibre channel — EGN MC usage (arXiv 1907.02846)](https://arxiv.org/pdf/1907.02846)

Searches covered: pulse-collision sector scalings; O-band/ZDW GN-with-FWM
models; UWB FWM-vs-XPM decomposition; NLIN MC variance reduction; FWM
enumeration/pruning and GN islands; universality/master-curve framings. No
prior art found for: sector-ratio power laws, census factorization, certified
truncation bounds, CRN sector estimation, the decimation artifact.
