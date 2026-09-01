"""Consolidated figures for the fast-vs-MC validation campaign.

Reads the ``campaign_fast_vs_mc.npz`` files written by
``analysis/fwm/fast_mc_validation_campaign.py`` (one per pass) and produces
the summary figures:

1. ``campaign_summary_fwm.png`` -- per-tuple strict-FWM error against the
   reduced variables, with the dispatcher's *far margin*
   ``rho = |u0| / (FAR_MARGIN_FACTOR * W + FAR_MARGIN_OFFSET)`` as the
   leading coordinate: ``rho = 1`` is exactly where the analytic dispatcher
   switches to the far closed form, and the campaign shows the error is
   organized by it.
2. ``campaign_summary_fwm_map.png`` -- the (x_grad, |mu|) phase-diagram map
   with the dispatch boundary drawn, and the error against |d|.
3. ``campaign_summary_xpm.png`` -- the XPM (nu, L/L_D) error surface, the
   n-PC sector composition, and the physical-comb pairs.

Usage::

    python analysis/fwm/plot_campaign_summary.py \
        --fwm media/lorenzi-fast/campaign_wide/campaign_fast_vs_mc.npz \
        --fwm media/lorenzi-fast/campaign_nyquist/campaign_fast_vs_mc.npz \
        --xpm media/lorenzi-fast/campaign_xpm/campaign_fast_vs_mc.npz \
        --out-dir media/lorenzi-fast
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
from pynlin.methods.td.fast_nlin import FAR_MARGIN_FACTOR, FAR_MARGIN_OFFSET

BRANCH_NAMES = ("sheet", "far", "fallback")
PASS_MARKERS = ("o", "^", "s")
PASS_COLORS = ("#2b7bba", "#d95f02", "#1b9e77")


def load_table(path: Path, key: str) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    if f"{key}_fields" not in d:
        return {}
    fields = [str(f) for f in d[f"{key}_fields"]]
    rows = d[key]
    out: dict[str, np.ndarray] = {}
    for i, f in enumerate(fields):
        col = np.array([r[i] for r in rows], dtype=object)
        try:
            out[f] = col.astype(float)
        except (TypeError, ValueError):
            out[f] = col.astype(str)
    return out


def derived_fwm(t: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    W = t["widths_sum"]
    rho = np.abs(t["u0"]) / (FAR_MARGIN_FACTOR * W + FAR_MARGIN_OFFSET)
    return {
        "rho": np.maximum(rho, 1e-3),
        "mu": np.maximum(np.abs(t["mu"]), 1e-4),
        "d": np.abs(t["d"]),
        "zeta": np.maximum(t["zeta"], 1e-7),
        "x": np.maximum(t["x_grad"], 1e-2),
        "q": np.maximum(t["q_sum"], 1e-3),
        "err": (t["fast"] - t["qmc_full"]) / t["qmc_full"],
        "err_model": (t["fast"] - t["qmc_lin"]) / t["qmc_lin"],
        "err_quad": (t["qmc_lin"] - t["qmc_full"]) / t["qmc_full"],
        "branch": t["branch"].astype(int),
    }


def plot_fwm(tables: list[tuple[str, dict]], out_dir: Path) -> None:
    specs = (
        ("rho", r"far margin $\rho = |u_0| / (3W + 3000)$", True),
        ("mu", r"detuning $|\mu| = |u_0| / x_\nabla$", True),
        ("d", r"support offset $|d|$", False),
        ("x", r"loudness $x_\nabla$", True),
        ("zeta", r"zero-line proximity $\zeta$", True),
        ("q", r"quadratic budget $\sum_j |q_j|$", True),
    )
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    for ax, (key, label, logx) in zip(axs.ravel(), specs):
        for k, (name, D) in enumerate(tables):
            ax.scatter(
                D[key], 100.0 * np.abs(D["err"]), s=8, alpha=0.35,
                marker=PASS_MARKERS[k % 3], color=PASS_COLORS[k % 3],
                linewidths=0, label=name,
            )
        if key == "rho":
            ax.axvline(1.0, color="k", lw=1.0, ls="--")
            ax.text(1.05, 0.02, "far dispatch", rotation=90, fontsize=7,
                    transform=ax.get_xaxis_transform(), va="bottom")
        for lvl in (1.0, 10.0):
            ax.axhline(lvl, color="0.65", lw=0.6, ls=":")
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 2e3)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$|$fast$/$QMC $-\,1|$ [\%]")
        ax.grid(alpha=0.22)
    axs[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Per-tuple strict-FWM error of the fast path against QMC ground truth "
        "(dotted: 1\\% and 10\\%)"
    )
    fig.savefig(out_dir / "campaign_summary_fwm.png", dpi=170)
    plt.close(fig)

    # Error decomposition and the map.
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for k, (name, D) in enumerate(tables):
        axs[0].scatter(D["rho"], 100 * np.abs(D["err_model"]), s=8, alpha=0.35,
                       marker=PASS_MARKERS[k % 3], color=PASS_COLORS[k % 3],
                       linewidths=0, label=name)
        axs[1].scatter(D["rho"], 100 * np.abs(D["err_quad"]), s=8, alpha=0.35,
                       marker=PASS_MARKERS[k % 3], color=PASS_COLORS[k % 3],
                       linewidths=0)
    for ax, title in zip(
        axs[:2],
        ("model error: fast vs QMC(linear phase)",
         "quadratic error: QMC(linear) vs QMC(full)"),
    ):
        ax.axvline(1.0, color="k", lw=1.0, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 2e3)
        ax.set_xlabel(r"far margin $\rho$")
        ax.set_ylabel(r"relative error [\%]")
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.22)
    axs[0].legend(fontsize=8)

    D = tables[0][1]
    for _, T in tables[1:]:
        D = {k: np.concatenate([D[k], T[k]]) for k in D}
    sc = axs[2].scatter(
        D["x"], D["mu"], c=np.clip(100 * np.abs(D["err"]), 1e-2, 1e2),
        norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1e2), s=10, cmap="magma_r",
    )
    xs = np.geomspace(max(D["x"].min(), 1e-2), D["x"].max(), 200)
    # rho = 1 in (x, |mu|) coordinates, bracketed by the two extreme
    # width-to-norm ratios W/x in [pi, pi*sqrt(3)].
    for ratio, style in ((np.pi, "--"), (np.pi * np.sqrt(3.0), ":")):
        axs[2].plot(
            xs, FAR_MARGIN_FACTOR * ratio + FAR_MARGIN_OFFSET / xs,
            color="k", lw=1.0, ls=style,
        )
    axs[2].set_xscale("log"); axs[2].set_yscale("log")
    axs[2].set_xlabel(r"$x_\nabla$"); axs[2].set_ylabel(r"$|\mu|$")
    axs[2].set_title(r"error map with the far-dispatch band $\rho = 1$", fontsize=10)
    fig.colorbar(sc, ax=axs[2], label=r"$|$error$|$ [\%]")
    fig.savefig(out_dir / "campaign_summary_fwm_map.png", dpi=170)
    plt.close(fig)


def plot_xpm(sweep: dict, pairs: dict, out_dir: Path) -> None:
    nu, lld = sweep["nu"], sweep["lld"]
    fast, qf, ql, n1 = sweep["fast"], sweep["qmc_full"], sweep["qmc_lin"], sweep["n1"]
    llds = np.unique(lld)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, v in enumerate(llds):
        m = lld == v
        o = np.argsort(nu[m])
        col = cmap(i / max(len(llds) - 1, 1))
        axs[0].plot(nu[m][o], 100 * np.abs(fast[m][o] / qf[m][o] - 1.0),
                    marker="o", ms=3, lw=1.2, color=col, label=f"{v:g}")
        axs[1].plot(nu[m][o], 100 * np.abs(fast[m][o] / ql[m][o] - 1.0),
                    marker="o", ms=3, lw=1.2, color=col)
    for ax, title in zip(
        axs[:2],
        (r"fast vs QMC(full): total error",
         r"fast vs QMC(linear): model error only"),
    ):
        ax.axhline(1.0, color="0.65", lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 1e3)
        ax.set_xlabel(r"$\nu = L/L_W$ (pair walk-off)")
        ax.set_ylabel(r"$|$fast$/$MC $-\,1|$ [\%]")
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.22)
    axs[0].legend(title=r"$L/L_D$", fontsize=7, title_fontsize=8, ncol=2)

    # n-PC composition at the extreme L/L_D values.
    for i, v in enumerate((llds[0], llds[-1])):
        m = lld == v
        o = np.argsort(nu[m])
        for key, lab, col in (
            ("n_2pc", "2PC", "#1b9e77"), ("n_3pca", "3PCa", "#d95f02"),
            ("n_3pcb", "3PCb", "#7570b3"), ("n_4pc", "4PC", "#e7298a"),
        ):
            axs[2].plot(nu[m][o], sweep[key][m][o] / n1[m][o],
                        marker="o", ms=2.5, lw=1.2, color=col,
                        ls="-" if i == 0 else "--",
                        label=f"{lab} ($L/L_D$={v:g})")
    axs[2].set_xscale("log"); axs[2].set_xlabel(r"$\nu = L/L_W$")
    axs[2].set_ylabel(r"sector $/\ N_1$")
    axs[2].set_title("n-PC composition of the same pair efficiency", fontsize=10)
    axs[2].grid(alpha=0.22); axs[2].legend(fontsize=6, ncol=2)
    fig.suptitle(
        "XPM pair efficiency: the fast closed form is exact in the linear "
        "model; all error is the in-channel quadratic term"
    )
    fig.savefig(out_dir / "campaign_summary_xpm.png", dpi=170)
    plt.close(fig)

    if pairs:
        keep = pairs.get("converged", np.ones_like(pairs["nu"])) > 0.5
        e = 100 * np.abs(pairs["fast"][keep] / pairs["qmc_full"][keep] - 1.0)
        anu = np.abs(pairs["nu"][keep])
        q = np.abs(pairs["q_b"][keep]) + np.abs(pairs["q_t"][keep])
        band = pairs["band"][keep]
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
        for name in ("O", "E", "S", "C", "L", "U"):
            m = band == name
            if not np.any(m):
                continue
            axs[0].scatter(anu[m], e[m], s=13, alpha=0.7, label=name)
            axs[1].scatter(q[m], e[m], s=13, alpha=0.7, label=name)
        for ax, xl in zip(axs, (r"$|\nu|$", r"$|q_b| + |q_t|$")):
            ax.axhline(1.0, color="0.65", lw=0.6, ls=":")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(xl); ax.set_ylabel(r"$|$fast$/$QMC $-\,1|$ [\%]")
            ax.grid(alpha=0.22)
        axs[0].legend(fontsize=8, ncol=3)
        fig.suptitle("XPM pairs of the physical comb (25 GHz pitch, 24.5 GBaud)")
        fig.savefig(out_dir / "campaign_summary_xpm_pairs.png", dpi=170)
        plt.close(fig)


def main() -> None:
    init_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fwm", type=Path, action="append", default=[])
    p.add_argument("--fwm-label", type=str, action="append", default=[])
    p.add_argument("--xpm", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    for i, path in enumerate(args.fwm):
        t = load_table(path, "fwm_rows")
        if not t:
            lg.warning(f"{path}: no fwm_rows")
            continue
        label = args.fwm_label[i] if i < len(args.fwm_label) else path.parent.name
        tables.append((label, derived_fwm(t)))
        lg.info(f"{label}: {t['fast'].size} tuples")
    if tables:
        plot_fwm(tables, args.out_dir)
        lg.info("wrote campaign_summary_fwm{,_map}.png")

    if args.xpm is not None:
        sweep = load_table(args.xpm, "xpm_sweep")
        pairs = load_table(args.xpm, "xpm_pairs")
        if sweep:
            plot_xpm(sweep, pairs, args.out_dir)
            lg.info("wrote campaign_summary_xpm{,_pairs}.png")


if __name__ == "__main__":
    main()
