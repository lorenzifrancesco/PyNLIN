"""Lorenzi Fast S1+S2: per-tuple error gate on real-system tuples.

S1: validates the randomized-Sobol ground truth against plain Monte Carlo on
the identical integrand (implementation cross-check).

S2: for tuples drawn from the S0 territory (stratified by fast-pass mass),
separates the three error sources of the fast pass:
  (a) analytic linear-model estimate vs exact-mask linear QMC   -> mask/regime model error
  (b) linear QMC vs full quadratic QMC                          -> q (in-channel beta2) error
  (c) production value (fast + top-K QMC refinement) vs (b)     -> end-to-end per-tuple error
Reports mass-weighted aggregate errors, which is what the band sums inherit.

Decimation note: the fast estimate and the QMC ground truth are evaluated on
the SAME tuples, so the per-tuple comparison is valid at any decimation --
but at decimation > 1 the sampled tuple population (and hence the
mass-weighted aggregates) is biased toward near-exact zero-sum combinations
and does not represent the full-grid physics. Use --decimation 1 whenever
the aggregate numbers are to be quoted.
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

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fast_nlin import (
    fwm_tuple_variables,
    kernel_abs2,
    linear_tuple_estimate,
    qmc_tuple_ground_truth,
    xpm_fast_batch,
    xpm_pair_variables,
    qmc_xpm_ground_truth,
)
from pynlin.methods.td.fullband_mc import (
    _beta0_abs_from_fiber,
    decimated_frequency_grid,
)
from pynlin.system import System


def plain_mc_tuple(u0, nu, q, d, n=4_000_000, seed=5, include_quadratic=True):
    rng = np.random.default_rng(seed)
    x = 2.0 * np.pi * (rng.random((3, n)) - 0.5)
    x_d = x[0] + x[1] - x[2] + d
    mask = np.abs(x_d) < np.pi
    u = u0 + nu[0] * x[0] + nu[1] * x[1] - nu[2] * x[2]
    if include_quadratic:
        u = u + q[0] * x[0] ** 2 + q[1] * x[1] ** 2 - q[2] * x[2] ** 2 - q[3] * x_d**2
    vals = kernel_abs2(u) * mask
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(n))


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument("--decimation", type=int, default=8)
    parser.add_argument("--n-targets", type=int, default=5)
    parser.add_argument("--tuples-per-target", type=int, default=60)
    parser.add_argument("--qmc-points", type=int, default=1 << 17)
    args = parser.parse_args()

    system = System.from_toml(args.config)
    _, freqs = decimated_frequency_grid(system, args.decimation)
    baud_rate = float(system.pulse.baud_rate)
    length = float(system.fiber_length)
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta0_abs = _beta0_abs_from_fiber(system, freqs, beta1)
    n = freqs.size
    targets = np.unique(np.linspace(0, n - 1, args.n_targets).astype(int))

    # ---------------- S1: QMC vs plain MC on a handful of real tuples -------
    lg.info("S1: QMC ground-truth implementation cross-check")
    v0 = fwm_tuple_variables(
        freqs, beta0_abs, beta1, beta2, baud_rate, length, int(targets[len(targets) // 2])
    )
    coeffs0 = np.stack([v0.nu_a, v0.nu_b, -v0.nu_c], axis=-1)
    est0 = linear_tuple_estimate(v0.u0, coeffs0, v0.d)
    pick = np.argsort(est0.values)[::-1][:4]
    for i in pick:
        nu_i = (v0.nu_a[i], v0.nu_b[i], v0.nu_c[i])
        q_i = (v0.q_a[i], v0.q_b[i], v0.q_c[i], v0.q_t)
        qmc_val, qmc_err = qmc_tuple_ground_truth(
            u0=v0.u0[i], nu=nu_i, q=q_i, d=v0.d[i],
            n_points=args.qmc_points, n_replicates=8, seed=11,
        )
        mc_val, mc_err = plain_mc_tuple(v0.u0[i], nu_i, q_i, v0.d[i])
        pull = abs(qmc_val - mc_val) / np.hypot(qmc_err, mc_err)
        lg.info(
            f"  tuple (a,b,c)=({v0.a[i]},{v0.b[i]},{v0.c[i]}): "
            f"QMC={qmc_val:.5e}+-{qmc_err:.1e}  MC={mc_val:.5e}+-{mc_err:.1e}  pull={pull:.2f}"
        )

    # ---------------- S2: stratified per-tuple error decomposition ----------
    lg.info("S2: error decomposition on mass-stratified tuples")
    rows = []
    for t in targets:
        v = fwm_tuple_variables(freqs, beta0_abs, beta1, beta2, baud_rate, length, int(t))
        if v.u0.size == 0:
            continue
        coeffs = np.stack([v.nu_a, v.nu_b, -v.nu_c], axis=-1)
        est = linear_tuple_estimate(v.u0, coeffs, v.d)
        order = np.argsort(est.values)[::-1]
        k = args.tuples_per_target
        # Stratified pick: all top-mass tuples plus a log-spaced tail sample.
        head = order[: k // 2]
        tail_pool = order[k // 2 :]
        tail = tail_pool[
            np.unique(np.geomspace(1, tail_pool.size, k - head.size).astype(int) - 1)
        ]
        for i in np.concatenate([head, tail]):
            nu_i = (v.nu_a[i], v.nu_b[i], v.nu_c[i])
            q_i = (v.q_a[i], v.q_b[i], v.q_c[i], v.q_t)
            lin, lin_err = qmc_tuple_ground_truth(
                u0=v.u0[i], nu=nu_i, q=q_i, d=v.d[i],
                n_points=args.qmc_points, n_replicates=6, seed=23,
                include_quadratic=False,
            )
            quad, quad_err = qmc_tuple_ground_truth(
                u0=v.u0[i], nu=nu_i, q=q_i, d=v.d[i],
                n_points=args.qmc_points, n_replicates=6, seed=29,
                include_quadratic=True,
            )
            rows.append((
                int(t), int(v.a[i]), int(v.b[i]), int(v.c[i]),
                v.u0[i], np.pi * (abs(nu_i[0]) + abs(nu_i[1]) + abs(nu_i[2])),
                est.values[i], est.regime[i], lin, lin_err, quad, quad_err,
            ))
    arr = np.array(rows, dtype=float)
    fast, regime = arr[:, 6], arr[:, 7].astype(int)
    lin, lin_err, quad, quad_err = arr[:, 8], arr[:, 9], arr[:, 10], arr[:, 11]

    good = lin > 0
    mask_err = np.abs(fast - lin) / np.maximum(lin, 1e-300)
    q_shift = np.abs(quad - lin) / np.maximum(lin, 1e-300)
    weight = quad / np.sum(quad[good])

    def wq(x, sel, qs=(0.5, 0.9, 0.99)):
        if not np.any(sel):
            return "n/a"
        return ", ".join(f"p{int(100*q)}={np.quantile(x[sel], q):.1%}" for q in qs)

    lg.info(f"(a) mask/regime error |fast-lin|/lin: {wq(mask_err, good)}")
    for rid, name in ((0, "near"), (1, "far"), (2, "wide")):
        sel = good & (regime == rid)
        lg.info(f"      {name}: {wq(mask_err, sel)}  (n={np.sum(sel)})")
    lg.info(f"(b) quadratic-model shift |quad-lin|/lin: {wq(q_shift, good)}")
    agg_a = abs(np.sum(fast[good] * weight[good]) / np.sum(lin[good] * weight[good]) - 1)
    agg_b = abs(np.sum(quad[good] * weight[good]) / np.sum(lin[good] * weight[good]) - 1)
    lg.info(f"    mass-weighted aggregate: mask/regime bias={agg_a:.2%}, quadratic shift={agg_b:.2%}")

    # (c) production path: top-mass tuples are QMC-refined, so their error is
    # the QMC noise; unrefined remainder carries the analytic error.
    order = np.argsort(fast)[::-1]
    cum = np.cumsum(fast[order]) / np.sum(fast)
    refined_share = cum[min(len(cum) - 1, 255)]
    resid = 1.0 - refined_share
    lg.info(
        f"(c) production: refined share (top-256 here) ~{refined_share:.1%}; "
        f"unrefined remainder {resid:.1%} carrying analytic error above"
    )

    # XPM: fast vs QMC with and without quadratics, across the band.
    lg.info("XPM pair check (target = mid-band)")
    t = int(targets[len(targets) // 2])
    others, nu_pairs, q_pairs = xpm_pair_variables(beta1, beta2, baud_rate, length, t)
    q_t = 0.5 * beta2[t] * baud_rate**2 * length
    sel = np.unique(np.geomspace(1, others.size, 8).astype(int) - 1)
    fast_xpm = xpm_fast_batch(nu_pairs[sel])
    xpm_rows = []
    for j, i in enumerate(sel):
        gt_lin, e1 = qmc_xpm_ground_truth(
            nu=nu_pairs[i], q_b=q_pairs[i], q_t=q_t,
            n_points=1 << 16, n_replicates=6, include_quadratic=False,
        )
        gt_quad, e2 = qmc_xpm_ground_truth(
            nu=nu_pairs[i], q_b=q_pairs[i], q_t=q_t,
            n_points=1 << 16, n_replicates=6, include_quadratic=True,
        )
        xpm_rows.append((nu_pairs[i], fast_xpm[j], gt_lin, e1, gt_quad, e2))
        lg.info(
            f"  pair b={others[i]} nu={nu_pairs[i]:.2e}: fast={fast_xpm[j]:.4e} "
            f"lin={gt_lin:.4e}+-{e1:.0e} quad={gt_quad:.4e}+-{e2:.0e} "
            f"err_lin={abs(fast_xpm[j]-gt_lin)/max(gt_lin,1e-300):.2%} "
            f"q_shift={abs(gt_quad-gt_lin)/max(gt_lin,1e-300):.2%}"
        )
    xpm_arr = np.array(xpm_rows, dtype=float)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "s2_collapse.npz", rows=arr, xpm_rows=xpm_arr)
    plot_s2_collapse(arr, xpm_arr, args.out_dir)
    lg.success(f"S2 saved to {args.out_dir}")


_REGIME_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
_REGIME_NAMES = {0: "near", 1: "far", 2: "wide"}


def _scatter_by_regime(ax, xs, ys, regime, *, ms=14):
    for rid, color in _REGIME_COLORS.items():
        sel = regime == rid
        if not np.any(sel):
            continue
        ax.plot(
            xs[sel], ys[sel], "o", ms=ms * 0.35, mfc=color, mec="none", alpha=0.7,
            label=_REGIME_NAMES[rid],
        )


def plot_s2_collapse(
    arr: np.ndarray,
    xpm_arr: np.ndarray | None,
    out_dir: Path,
) -> None:
    """Render the per-tuple error-decomposition panels.

    ``arr`` columns: target, a, b, c, u0, W, fast, regime, lin, lin_err,
    quad, quad_err.  ``xpm_arr`` (optional) columns: nu, fast, lin, lin_err,
    quad, quad_err.  Works standalone on a saved ``s2_collapse.npz``.
    """
    arr = np.asarray(arr, dtype=float)
    fast, regime = arr[:, 6], arr[:, 7].astype(int)
    lin, lin_err, quad, quad_err = arr[:, 8], arr[:, 9], arr[:, 10], arr[:, 11]
    u0, W = arr[:, 4], arr[:, 5]
    good = lin > 0
    mask_err = np.abs(fast - lin) / np.maximum(lin, 1e-300)
    q_shift = np.abs(quad - lin) / np.maximum(lin, 1e-300)
    noise_a = lin_err / np.maximum(lin, 1e-300)
    noise_b = np.hypot(lin_err, quad_err) / np.maximum(lin, 1e-300)

    has_xpm = xpm_arr is not None and np.asarray(xpm_arr).size > 0
    n_rows = 3 if has_xpm else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(9.5, 3.4 * n_rows + 0.8))
    axes = np.atleast_2d(axes)

    ax = axes[0, 0]
    lims = [
        max(0.3 * np.min(lin[good]), 1e-300),
        3.0 * np.max(lin[good]),
    ]
    ax.plot(lims, lims, "-", color="gray", lw=0.8, zorder=0)
    _scatter_by_regime(ax, lin[good], fast[good], regime[good])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"QMC linear ground truth $F_{\rm lin}$")
    ax.set_ylabel(r"fast analytic $F$")
    ax.set_title("(0) fast vs QMC ground truth (y=x reference)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[0, 1]
    _scatter_by_regime(ax, fast[good], mask_err[good], regime[good])
    ax.axhline(np.median(noise_a[good]), color="black", ls="--", lw=0.9)
    ax.annotate(
        "QMC noise", xy=(0.98, np.median(noise_a[good])),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"fast analytic $F$")
    ax.set_ylabel(r"$|F-F_{\rm lin}|/F_{\rm lin}$")
    ax.set_title("(a) mask/regime model error vs tuple mass")

    ax = axes[1, 0]
    _scatter_by_regime(ax, fast[good], q_shift[good], regime[good])
    ax.axhline(np.median(noise_b[good]), color="black", ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"fast analytic $F$")
    ax.set_ylabel(r"$|F_{\rm quad}-F_{\rm lin}|/F_{\rm lin}$")
    ax.set_title("(b) quadratic (in-channel beta2) shift vs tuple mass")

    ax = axes[1, 1]
    ratio = np.abs(u0[good]) / np.maximum(W[good], 1e-300)
    _scatter_by_regime(ax, ratio, mask_err[good], regime[good])
    ax.axvline(1.0, color="black", ls=":", lw=1.0)
    ax.annotate(
        r"$|u_0|=W$", xy=(1.0, 0.95), xycoords=("data", "axes fraction"),
        xytext=(4, 0), textcoords="offset points", fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|u_0|/W$ (phase-matching proximity)")
    ax.set_ylabel(r"$|F-F_{\rm lin}|/F_{\rm lin}$")
    ax.set_title("(a) mask error vs proximity to $|u_0|=W$")

    if has_xpm:
        xpm = np.asarray(xpm_arr, dtype=float)
        nu_x, fast_x = xpm[:, 0], xpm[:, 1]
        lin_x, lin_err_x, quad_x, quad_err_x = xpm[:, 2], xpm[:, 3], xpm[:, 4], xpm[:, 5]
        good_x = lin_x > 0

        ax = axes[2, 0]
        lims = [max(0.3 * np.min(lin_x[good_x]), 1e-300), 3.0 * np.max(lin_x[good_x])]
        ax.plot(lims, lims, "-", color="gray", lw=0.8, zorder=0)
        ax.errorbar(
            lin_x[good_x], fast_x[good_x],
            xerr=lin_err_x[good_x], fmt="o", ms=4, color="tab:purple", alpha=0.8,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"QMC linear ground truth (XPM pair)")
        ax.set_ylabel(r"fast analytic $F_{\rm XPM}$")
        ax.set_title("XPM: fast vs QMC ground truth (y=x, x-error = QMC noise)")

        ax = axes[2, 1]
        q_shift_x = np.abs(quad_x - lin_x) / np.maximum(lin_x, 1e-300)
        ax.plot(
            np.abs(nu_x[good_x]), q_shift_x[good_x], "o", ms=4,
            color="tab:purple", alpha=0.8,
        )
        ax.axhline(
            np.median((np.hypot(lin_err_x, quad_err_x) / np.maximum(lin_x, 1e-300))[good_x]),
            color="black", ls="--", lw=0.9,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$|\nu|$ XPM walk-off [rad]")
        ax.set_ylabel(r"$|F_{\rm quad}-F_{\rm lin}|/F_{\rm lin}$")
        ax.set_title("XPM: quadratic shift vs walk-off")

    fig.suptitle(
        "Lorenzi Fast S2: per-tuple error decomposition\n"
        "(dashed lines: median QMC noise floor of each comparison)"
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "s2_collapse.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
