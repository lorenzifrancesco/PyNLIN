"""Figure for the tube-selectivity census in doc §9.3.

For probe targets across the band, enumerates the full support-surviving
strict-FWM population, applies the current mask-aware ``select_tube``
selector, and reports (a) how many tuples survive and (b) which of the three
phase-matching surfaces of §10.4 each survivor lies closest to.

The three surfaces, in the (omega_a, omega_b, omega_c) coordinates of §10.4:

    P1: omega_a = omega_c                    normal (1, 0, -1)/sqrt2
    P2: omega_b = omega_c                    normal (0, 1, -1)/sqrt2
    Q : (omega_a + omega_b)/2 = omega_ZDW    normal (1, 1,  0)/sqrt2

All three normals share the same norm, so the raw distances |wa - wc|,
|wb - wc| and |wa + wb - 2 w_ZDW| are directly comparable and the argmin is a
fair Euclidean nearest-surface test.

Caveat carried into the caption: the nearest-surface label is a *geometric
guide*, not the selection predicate.  Survival is decided by
g_q <= 2 sqrt(A(d)/eps), which depends on the signed projection kappa, the
support shift -kappa*d, ||c_perp||_1, A(d) and the quadratic padding P_q
(§9, §9.2).  This figure says which surface a survivor sits nearest to; it
does not claim that nearness is why it survived.

Output (media/lorenzi-fast and docs/source/_static/lorenzi-fast):
  tube_census.png  -- (a) enumerated vs retained tuples per target,
                      (b) absolute retained mass per nearest surface,
                      (c) the same as a share
  tube_census.npz  -- the per-target census arrays
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fast_analytic import select_tube
from pynlin.methods.td.fast_nlin import (
    fwm_tuple_variables,
    linear_tuple_estimate,
    xpm_fast_batch,
    xpm_pair_variables,
)
from pynlin.methods.td.fullband_mc import (
    _beta0_abs_from_fiber,
    decimated_system,
    estimate_zdw_frequency,
)
from pynlin.system import System

# Categorical palette, validated (lightness band, chroma floor, CVD
# separation, normal-vision floor, contrast vs surface all PASS).
SURFACE_COLORS = ["#e8590c", "#5f3dc4", "#2f9e44"]
SURFACE_LABELS = [
    r"$P_1:\ \omega_a=\omega_c$",
    r"$P_2:\ \omega_b=\omega_c$",
    r"$Q:\ \bar\omega=\omega_{\rm ZDW}$",
]
INK, MUTED, GRID = "#22262b", "#8b929a", "#d8d8d4"
# Population colours, matching plot_scaled_band_spectrum.py so the two figures
# can be read together: blue = XPM pairs, orange = strict FWM.
C_XPM, C_FWM = "#1971c2", "#e8590c"
# SSFM-validated coefficient counting (doc section 13, physical_nlin_spectrum):
# sigma^2_XPM = 4 gamma^2 P^3 C_XPM but sigma^2_FWM = 2 gamma^2 P^3 C_FWM, so
# the prefactor-free sums must be weighted before they can be compared.
XPM_COEFF, FWM_COEFF = 4.0, 2.0


def census(system: System, freqs, beta0, beta1, beta2, target, epsilon, zdw,
           baud_rate):
    """Per-target survivor counts and nearest-surface split."""
    variables = fwm_tuple_variables(
        freqs,
        beta0,
        beta1,
        beta2,
        float(baud_rate),
        float(system.fiber_length),
        int(target),
    )
    if variables.u0.size == 0:
        return None

    keep, certificate = select_tube(variables, epsilon)
    coefficients = np.stack(
        [variables.nu_a, variables.nu_b, -variables.nu_c], axis=-1
    )
    estimate = linear_tuple_estimate(variables.u0, coefficients, variables.d)

    f_a = freqs[variables.a[keep]]
    f_b = freqs[variables.b[keep]]
    f_c = freqs[variables.c[keep]]
    distances = np.stack(
        [np.abs(f_a - f_c), np.abs(f_b - f_c), np.abs(f_a + f_b - 2.0 * zdw)]
    )
    nearest = np.argmin(distances, axis=0)

    weight = estimate.values[keep]
    weight = np.where(np.isfinite(weight), weight, 0.0)

    # The XPM pair sum for the same target.  This census enumerates strict-FWM
    # TRIPLES only; the XPM sector is a disjoint population of PAIRS (t, b),
    # ~N of them against ~N^3/2 triples, and is carried by a different code
    # path (doc §12).  It costs ~10 ms against the ~30 s of the FWM pass, and
    # it is reported here because the P1/P2 labels above are near-degenerate
    # strict FWM, NOT the XPM sector -- a distinction that is easy to lose
    # when only the FWM curves are plotted.
    _, nu_pairs, _ = xpm_pair_variables(beta1, beta2, float(baud_rate),
                                        float(system.fiber_length), target)
    xpm_total = float(np.sum(xpm_fast_batch(nu_pairs)))

    return {
        "n_all": int(variables.u0.size),
        "n_keep": int(keep.size),
        "certificate": float(certificate),
        "count": np.array([int((nearest == i).sum()) for i in range(3)]),
        "mass": np.array([float(weight[nearest == i].sum()) for i in range(3)]),
        "xpm": xpm_total,
        "fwm_retained": float(weight.sum()),
    }


def plot(data, zdw_thz, epsilon, paths, q_threshold_thz=None):
    f = data["f_thz"]
    n_all, n_keep = data["n_all"], data["n_keep"]
    count_fraction = data["count"] / np.maximum(
        data["count"].sum(1, keepdims=True), 1
    )
    mass_fraction = data["mass"] / np.maximum(
        data["mass"].sum(1, keepdims=True), 1e-300
    )

    fig, ax = plt.subplots(1, 4, figsize=(20.5, 4.8), constrained_layout=True)
    for axis in ax:
        axis.axvline(zdw_thz, color=MUTED, ls=":", lw=1.6, zorder=0)
        axis.set_xlabel("target frequency [THz]")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.grid(axis="y", color=GRID, lw=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED)
        axis.xaxis.label.set_color(MUTED)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.set_xlim(f.min() - 1.5, f.max() + 1.5)
        axis.annotate(
            "ZDW", (zdw_thz, 1.0), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=9, color=MUTED,
        )
        if q_threshold_thz is not None and f.min() <= q_threshold_thz <= f.max():
            # Predicted analytically from 2 f_ZDW - f_max, before enumeration.
            axis.axvline(q_threshold_thz, color=MUTED, ls="--", lw=1.2, zorder=0)
            axis.annotate(
                r"$Q$ on", (q_threshold_thz, 0.90),
                xycoords=("data", "axes fraction"),
                ha="right", va="bottom", fontsize=9, color=MUTED,
            )

    ax[0].semilogy(f, n_all, "o-", color=MUTED, lw=2, ms=5,
                   label="enumerated (support-surviving)")
    exponent = int(round(np.log10(epsilon)))
    eps_label = (rf"10^{{{exponent}}}" if np.isclose(epsilon, 10.0**exponent)
                 else rf"{epsilon:g}")
    ax[0].semilogy(f, n_keep, "o-", color=INK, lw=2, ms=5,
                   label=rf"tube survivors, $\varepsilon={eps_label}$")
    ax[0].set_ylim(0.3 * n_keep.min(), 4.0 * n_all.max())
    ax[0].set_ylabel("tuples per target")
    ax[0].yaxis.label.set_color(MUTED)
    ax[0].set_title("How many tuples are considered", color=INK, fontsize=12)
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")

    peak = int(np.argmax(n_keep))
    ax[0].annotate(
        "peak {:.2f}\\% of enumerated".format(100 * n_keep[peak] / n_all[peak]),
        (f[peak], n_keep[peak]), textcoords="offset points", xytext=(-6, 12),
        fontsize=9, color=INK, ha="right",
    )

    # P1 and P2 coincide identically (the a<->b relabelling symmetry of the
    # ordered enumeration), so P2 is dashed to keep both visible.
    styles = ["o-", "o--", "o-"]
    for i in range(3):
        # Zeros are absences, not small values: break the line instead of
        # plunging it to the axis floor on a log scale.
        mass = np.where(data["mass"][:, i] > 0, data["mass"][:, i], np.nan)
        extra = {"dashes": (5, 2)} if i == 1 else {}
        ax[1].semilogy(f, mass, styles[i], color=SURFACE_COLORS[i], lw=2, ms=4,
                       label=SURFACE_LABELS[i], **extra)
    ax[1].set_ylabel("retained mass (prefactor-free)")
    ax[1].yaxis.label.set_color(MUTED)
    ax[1].set_title("Nearest surface, absolute mass", color=INK, fontsize=12)
    floor = data["mass"][data["mass"] > 0].min() if (data["mass"] > 0).any() else 1e-12
    ax[1].set_ylim(0.2 * floor, 5 * data["mass"].max())

    # Panel 4: the two disjoint populations on one axis.  Everything in panels
    # (b) and (c) is strict FWM; the XPM sector never enters this census.
    #
    # The prefactor-free sums are NOT directly comparable: physically
    # sigma^2_XPM = 4 gamma^2 P^3 C_XPM but sigma^2_FWM = 2 gamma^2 P^3 C_FWM
    # (doc section 13, SSFM-validated counting in physical_nlin_spectrum).
    # Plotting the bare sums overstates FWM by exactly 2x and puts the
    # crossover in the wrong place.  The common gamma^2 P^3 cancels in the
    # comparison, so plot 4 C_XPM against 2 C_FWM.
    if "xpm" in data:
        xpm_w = XPM_COEFF * data["xpm"]
        fwm_w = FWM_COEFF * data["fwm_retained"]
        ax[3].semilogy(f, xpm_w, "o-", color=C_XPM, lw=2, ms=4,
                       label=r"XPM pairs, $4\,C^{\rm XPM}$")
        ax[3].semilogy(f, np.where(fwm_w > 0, fwm_w, np.nan),
                       "o-", color=C_FWM, lw=2, ms=4,
                       label=r"strict FWM (tube), $2\,C^{\rm FWM}$")
        ax[3].set_ylabel(r"$\sigma^2/(\gamma^2P^3)$, physical weighting")
        ax[3].yaxis.label.set_color(MUTED)
        ax[3].set_title("The two populations, physically weighted",
                        color=INK, fontsize=12)
        ax[3].legend(frameon=False, fontsize=9, loc="upper left")
        ratio = xpm_w / np.maximum(fwm_w, 1e-300)
        lo = int(np.argmax(ratio))
        ax[3].annotate(
            rf"XPM/FWM $\simeq{ratio[lo]:.0f}\times$",
            (f[lo], xpm_w[lo]), textcoords="offset points",
            xytext=(6, -4), fontsize=9, color=MUTED, ha="left",
        )
        crossing = fwm_w > xpm_w
        if crossing.any():
            ax[3].axvspan(f[crossing].min(), f[crossing].max(),
                          color=C_FWM, alpha=0.10, lw=0, zorder=0)
            ax[3].annotate(
                "FWM > XPM",
                (0.5 * (f[crossing].min() + f[crossing].max()), 0.03),
                xycoords=("data", "axes fraction"), ha="center",
                fontsize=9, color=C_FWM,
            )

    for panel, fraction, title, label_at in (
        (ax[2], mass_fraction, "Nearest surface, mass share",
         int(np.argmin(np.abs(f - (zdw_thz - 2.0))))),
    ):
        panel.stackplot(f, *(100 * fraction.T), colors=SURFACE_COLORS,
                        edgecolor="white", lw=1.4)
        panel.set_ylim(0, 100)
        panel.set_yticks([0, 25, 50, 75, 100])
        panel.ticklabel_format(axis="y", style="plain", useOffset=False)
        panel.set_ylabel("share of survivors [\\%]")
        panel.yaxis.label.set_color(MUTED)
        panel.set_title(title, color=INK, fontsize=12)

        cumulative = np.cumsum(100 * fraction, axis=1)
        middle = cumulative - 50 * fraction
        for i in range(3):
            if fraction[label_at, i] > 0.12:
                panel.text(f[label_at], middle[label_at, i], SURFACE_LABELS[i],
                           ha="center", va="center", fontsize=9,
                           color="white", weight="bold")

    ax[1].legend(frameon=False, fontsize=9, loc="upper left")

    for path in paths:
        fig.savefig(path, dpi=200, facecolor="white")
        lg.info(f"wrote {path}")
    plt.close(fig)


def census_landmarks(freqs, beta2, zdw, length, spacing):
    """Closed-form structure of the census, computed before any enumeration.

    Everything here costs a polyfit: it is a statement about the fiber and the
    channel plan, not about the tuples.  Two features of the nearest-surface
    census are analytically located, and both are narrow enough that a uniform
    target sweep aliases them (this is why the sweep below is not uniform).

    1. Q THRESHOLD.  Energy conservation gives wbar = (w_a+w_b)/2 = (w_c+w_t)/2,
       so the sheet condition wbar = w_ZDW fixes the conjugated leg outright,

           f_c* = 2 f_ZDW - f_t .

       Q is reachable only while f_c* lies in the band, i.e. only for

           2 f_ZDW - f_max  <=  f_t  <=  2 f_ZDW - f_min ,

       intersected with the band itself.  Below that edge NO tuple of the target
       can put its pump mean on the ZDW and the Q population is exactly empty --
       a hard band-edge fact, not a sampling accident.

    2. ZDW RESONANCE of the near-degenerate (P1/P2) sector.  With beta2 locally
       linear, beta2(wbar) = beta3 (wbar - w_ZDW), the accumulated mismatch of a
       near-degenerate tuple is

           u0 = L (w_a - w_c)(w_c - w_b) beta3 (wbar - w_ZDW),

       so demanding |u0| <~ pi bounds the detuning of a family whose two leg
       separations are m and n channels by

           |f_bar - f_ZDW|  <=  half_width_1ch / (m n),
           half_width_1ch = pi / (2 pi L (2 pi df)^2 |beta3|) .

       The observed resonance is the ENVELOPE over families, so this is a
       scaling law rather than an exact width: the dominant m n = 1..10 gives
       0.2-2 THz, bracketing the 0.75 THz FWHM measured by direct scan.  The
       window and step returned below are derived from it conservatively.
    """
    f_min, f_max = float(np.min(freqs)), float(np.max(freqs))
    beta3 = float(np.polyfit(2.0 * np.pi * (freqs - zdw), beta2, 1)[0])

    q_lo = max(2.0 * zdw - f_max, f_min)
    q_hi = min(2.0 * zdw - f_min, f_max)
    # The nearest-surface LABEL turns on slightly earlier than exact
    # reachability: argmin calls a tuple Q as soon as |w_a + w_b - 2 w_ZDW|
    # beats the P1/P2 distance, and the latter is floored at one channel
    # spacing because w_a = w_c is forbidden for strict FWM.  Using
    # w_a + w_b = w_c + w_t, that reads |w_c + w_t - 2 w_ZDW| < spacing, i.e.
    # the onset moves down by exactly one spacing -- a grid-dependent edge,
    # unlike q_lo itself.  Verified at k = 1 (predicted 219.632, observed
    # 219.65) and k = 8 (predicted 219.456, observed 219.43), both within half
    # a channel.
    q_label_lo = max(q_lo - spacing, f_min)
    half_1ch = np.pi / (2.0 * np.pi * length * (2.0 * np.pi * spacing) ** 2 * abs(beta3))

    return {
        "zdw": zdw,
        "beta3": beta3,
        "q_threshold_lo": q_lo,
        "q_threshold_hi": q_hi,
        "q_label_onset": q_label_lo,
        "resonance_center": zdw,
        # widest dominant family (m n ~ 1) sets how far the structure extends
        "resonance_half_width_1ch": half_1ch,
        # m n ~ 10 is the width-weighted dominant product; step to resolve it
        "resonance_step": half_1ch / 10.0,
    }


def landmark_report(mark) -> list[str]:
    """One line per analytic fact, for the log and the npz."""
    return [
        f"ZDW                       {mark['zdw'] / 1e12:10.3f} THz",
        f"beta3                     {mark['beta3']:10.4e} s^3/m",
        f"Q reachable for f_t in   [{mark['q_threshold_lo'] / 1e12:.3f}, "
        f"{mark['q_threshold_hi'] / 1e12:.3f}] THz  (f_c* = 2 f_ZDW - f_t in band)",
        f"Q nearest-surface onset   {mark['q_label_onset'] / 1e12:10.3f} THz"
        f"  (one spacing lower; grid dependent)",
        f"ZDW resonance centred at  {mark['resonance_center'] / 1e12:10.3f} THz",
        f"  half-width (m n = 1)    {mark['resonance_half_width_1ch'] / 1e12:10.3f} THz",
        f"  suggested target step   {mark['resonance_step'] / 1e12:10.3f} THz",
    ]


def coverage_guard(f_thz, max_gap_thz, landmarks=()) -> list[str]:
    """Flag intervals with no samples at all.

    The rate test below cannot see these: if two targets straddling a wide gap
    happen to have similar values, the line drawn between them reads as a
    plateau, and a flat segment is indistinguishable from an unsampled one.
    (This is exactly how a 5.05 THz hole between 221.2 and 226.3 THz once read
    as a genuine plateau in the P1/P2 mass.)  Absence of data and absence of
    structure look identical on a line plot, so the sampling has to be checked
    separately from the values.


    A wide gap is not automatically a defect: away from the landmarks the curves
    are smooth and coarse sampling is adequate.  So gaps are reported as facts,
    and only escalated when one swallows a predicted landmark -- which would
    mean the refinement failed to place a target where structure is known to be.
    """
    f = np.asarray(f_thz, dtype=float)
    out = []
    for i in range(len(f) - 1):
        span = f[i + 1] - f[i]
        if span <= max_gap_thz:
            continue
        swallowed = [m for m in landmarks if f[i] < m < f[i + 1]]
        note = (f" -- CONTAINS predicted landmark(s) "
                f"{', '.join(f'{m:.2f}' for m in swallowed)} THz"
                if swallowed else
                " -- interpolated, not measured")
        out.append(
            f"coverage: {span:.2f} THz gap between {f[i]:.2f} and "
            f"{f[i + 1]:.2f} THz{note}"
        )
    return out


def aliasing_guard(f_thz, series, name, rate=2.0, landmarks=(), tol_thz=0.0
                   ) -> list[str]:
    """Post-hoc detector: adjacent samples of a resolved curve stay close.

    A fully resolved sweep varies smoothly between neighbours; a jump far larger
    than the local trend means the sampling stepped over structure.  This needs
    no prior knowledge of where the features are, so it also catches anything
    the closed forms above do not predict.

    The test is on the RATE of change, decades per THz, not on the bare ratio:
    a 6x change spread over 5 THz is a smoothly sampled trend, whereas the same
    6x across 0.2 THz means the sweep stepped over something.  On the reference
    0.25 THz scan the rate never exceeds 1.3 dec/THz, so the default 2.0 leaves
    headroom while still catching the aliased features (7.7 dec/THz).

    A step that straddles a predicted landmark is reported as EXPECTED: the Q
    threshold is a true discontinuity (the population is identically zero on one
    side), so bisecting it forever would not remove the jump.  ``tol_thz``
    widens the landmark by the grid quantization, since a target can only sit on
    an existing channel.  Only the UNEXPECTED lines are calls to sample finer.
    """
    msgs = []
    values = np.asarray(series, dtype=float)

    def _straddles(f_lo, f_hi):
        return any(f_lo - tol_thz <= m <= f_hi + tol_thz for m in landmarks)

    for i in range(len(values) - 1):
        lo, hi = values[i], values[i + 1]
        f_lo, f_hi = float(f_thz[i]), float(f_thz[i + 1])
        span = max(f_hi - f_lo, 1e-9)
        if min(lo, hi) <= 0.0:
            if max(lo, hi) <= 0.0:
                continue
            kind = "EXPECTED" if _straddles(f_lo, f_hi) else "UNEXPECTED"
            msgs.append(
                f"{name}: on/off transition between {f_lo:.2f} and {f_hi:.2f} "
                f"THz -- {kind}"
            )
            continue
        r = max(lo, hi) / min(lo, hi)
        dec_per_thz = np.log10(r) / span
        if dec_per_thz > rate:
            kind = ("EXPECTED (predicted landmark in this interval)"
                    if _straddles(f_lo, f_hi)
                    else "UNEXPECTED -- likely unresolved, bisect here")
            msgs.append(
                f"{name}: {r:.1f}x over {span:.2f} THz "
                f"({dec_per_thz:.1f} dec/THz) between {f_lo:.2f} and "
                f"{f_hi:.2f} THz -- {kind}"
            )
    return msgs


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument(
        "--docs-dir", type=Path, default=Path("docs/source/_static/lorenzi-fast")
    )
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--n-targets", type=int, default=12)
    parser.add_argument(
        "--n-transition", type=int, default=16,
        help="Extra targets concentrated on the ZDW transition, where the "
             "nearest-surface composition changes fastest.",
    )
    args = parser.parse_args()

    system = System.from_toml(args.config)
    # Decimation at constant power density and filling factor: delta_f -> k
    # delta_f forces B -> k B and P -> k P, which also leaves d (and hence the
    # mask geometry) invariant.  See fullband_mc.DecimatedSystem.
    decimated = decimated_system(system, args.decimation)
    freqs = decimated.freqs
    baud_rate = decimated.baud_rate
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta0 = _beta0_abs_from_fiber(system, freqs, beta1, beta2)
    zdw = float(estimate_zdw_frequency(system))

    lg.info(
        f"decimation {decimated.factor}: {freqs.size} channels, "
        f"spacing {decimated.channel_spacing / 1e9:.2f} GHz, "
        f"B {baud_rate / 1e9:.2f} GBd, P {decimated.launch_power_dbm:+.2f} dBm, "
        f"filling factor {decimated.filling_factor:.4f}"
    )
    lg.info(f"ZDW {zdw / 1e12:.2f} THz, eps={args.epsilon:g}")

    # Cheap closed-form structure, before any tuple is enumerated.  The two
    # features it locates are narrower than a uniform sweep can resolve, so the
    # target list is built around them rather than spread evenly.
    mark = census_landmarks(freqs, beta2, zdw, float(system.fiber_length),
                            decimated.channel_spacing)
    lg.info("analytic landmarks (no enumeration):")
    for line in landmark_report(mark):
        lg.info(f"  {line}")

    def _nearest(f_hz):
        return int(np.argmin(np.abs(freqs - f_hz)))

    targets = list(np.linspace(8, freqs.size - 9, args.n_targets).astype(int))
    step = max(mark["resonance_step"], decimated.channel_spacing)
    if args.n_transition:
        # (a) the ZDW resonance: span the widest dominant family, step fine
        # enough to resolve the envelope.
        span = mark["resonance_half_width_1ch"]
        edge = np.arange(mark["resonance_center"] - span,
                         mark["resonance_center"] + span + 0.5 * step, step)
        # (b) the Q threshold: the population turns on discontinuously there.
        for edge_f in (mark["q_threshold_lo"], mark["q_threshold_hi"]):
            edge = np.concatenate([edge, edge_f + np.arange(-4, 5) * step])
        edge = edge[(edge >= freqs.min()) & (edge <= freqs.max())]
        targets += [_nearest(f) for f in edge]
    targets = sorted(set(int(t) for t in targets))
    lg.info(f"{len(targets)} targets "
            f"(uniform {args.n_targets} + landmark-driven refinement)")

    rows = []
    for i, target in enumerate(targets):
        result = census(system, freqs, beta0, beta1, beta2, target,
                        args.epsilon, zdw, baud_rate)
        if result is None:
            continue
        result["f_thz"] = float(freqs[target] / 1e12)
        rows.append(result)

        mass = result["mass"] / max(result["mass"].sum(), 1e-300)
        lg.info(
            f"[{i + 1}/{len(targets)}] {result['f_thz']:7.2f} THz  "
            f"all={result['n_all']:9d} keep={result['n_keep']:7d} "
            f"({result['n_keep'] / result['n_all']:6.3%})  "
            f"mass P1/P2/Q = {np.round(100 * mass, 1)}"
        )

    rows.sort(key=lambda r: r["f_thz"])
    data = {
        "f_thz": np.array([r["f_thz"] for r in rows]),
        "n_all": np.array([r["n_all"] for r in rows], dtype=float),
        "n_keep": np.array([r["n_keep"] for r in rows], dtype=float),
        "certificate": np.array([r["certificate"] for r in rows]),
        "count": np.stack([r["count"] for r in rows]).astype(float),
        "mass": np.stack([r["mass"] for r in rows]),
        "xpm": np.array([r["xpm"] for r in rows]),
        "fwm_retained": np.array([r["fwm_retained"] for r in rows]),
    }

    # Post-hoc aliasing check on the quantities the figure actually shows.
    known = (mark["q_threshold_lo"] / 1e12, mark["q_threshold_hi"] / 1e12,
             mark["q_label_onset"] / 1e12)
    # A target can only sit on an existing channel, so the predicted threshold
    # is fuzzy by half a channel spacing.
    tol = 0.5 * decimated.channel_spacing / 1e12
    warnings = (
        aliasing_guard(data["f_thz"], data["mass"][:, 0] + data["mass"][:, 1],
                       "P1+P2", landmarks=known, tol_thz=tol)
        + aliasing_guard(data["f_thz"], data["mass"][:, 2], "Q",
                         landmarks=known, tol_thz=tol)
    )
    # Sampling coverage is a separate question from value continuity.
    gaps = coverage_guard(data["f_thz"], 4.0 * mark["resonance_step"] / 1e12,
                          landmarks=known)
    blind = [g for g in gaps if "CONTAINS" in g]
    for g in gaps:
        (lg.warning if "CONTAINS" in g else lg.info)(f"aliasing guard: {g}")
    if gaps and not blind:
        lg.info(f"aliasing guard: {len(gaps)} coarse interval(s), none hiding a "
                "predicted landmark")
    unexpected = [w for w in warnings if "UNEXPECTED" in w]
    for w in warnings:
        (lg.warning if "UNEXPECTED" in w else lg.info)(f"aliasing guard: {w}")
    if unexpected or blind:
        lg.warning(f"aliasing guard: {len(unexpected)} unexplained step(s), "
                   f"{len(blind)} landmark(s) inside a gap; refine there")
    else:
        lg.info("aliasing guard: every large step is a predicted landmark")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / f"tube_census_dec{decimated.factor}.npz",
             zdw_thz=zdw / 1e12, epsilon=args.epsilon,
             q_threshold_lo_thz=mark["q_threshold_lo"] / 1e12,
             q_label_onset_thz=mark["q_label_onset"] / 1e12,
             q_threshold_hi_thz=mark["q_threshold_hi"] / 1e12,
             beta3=mark["beta3"],
             resonance_half_width_1ch_thz=mark["resonance_half_width_1ch"] / 1e12,
             decimation=decimated.factor, baud_rate=baud_rate,
             channel_spacing=decimated.channel_spacing,
             filling_factor=decimated.filling_factor,
             launch_power_dbm=decimated.launch_power_dbm, **data)
    suffix = "" if decimated.factor == 1 else f"_dec{decimated.factor}"
    plot(data, zdw / 1e12, args.epsilon,
         [args.out_dir / f"tube_census{suffix}.png",
          args.docs_dir / f"tube_census{suffix}.png"],
         q_threshold_thz=mark["q_threshold_lo"] / 1e12)
    lg.success("tube census saved")


if __name__ == "__main__":
    main()
