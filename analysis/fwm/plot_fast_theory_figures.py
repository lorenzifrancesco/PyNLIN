"""Generate the proof figures for docs/source/lorenzi_fast_method.md."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path("src").resolve()))
from pynlin.methods.td.fast_nlin import (  # noqa: E402
    exact_conditional_acceptance,
    kernel_abs2,
    linear_tuple_estimate,
    near_model_masked,
    pointwise_conditional_acceptance,
    qmc_tuple_ground_truth,
    support_acceptance,
    uniform_sum_density,
    xpm_fast_batch,
    _xpm_mass_transform,
    FAR_MARGIN_FACTOR,
    FAR_MARGIN_OFFSET,
    WIDE_HALFWIDTH,
)

OUT = Path("docs/source/_static/lorenzi-fast")
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#6f6e66"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
})


# ------------------------------------------------------------------ fig 1
def fig_kernel_envelope():
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    u = np.linspace(-60, 60, 4000)
    ax.semilogy(u, kernel_abs2(u), color=BLUE, label=None)
    env = np.minimum(1.0, 4.0 / np.maximum(u**2, 1e-12))
    ax.semilogy(u, env, color=ORANGE, ls="--", lw=1.4)
    u0, W = 38.0, 12.0
    ax.axvspan(u0 - W, u0 + W, color=AQUA, alpha=0.18, lw=0)
    gap = u0 - W
    ax.semilogy([gap], [4.0 / gap**2], "o", color=YELLOW, ms=7, zorder=5)
    ax.annotate(r"$\hat K(u)=4\sin^2(u/2)/u^2$", xy=(-57, 3e-1), color=BLUE)
    ax.annotate(r"envelope $\min(1,\,4/u^2)$", xy=(-57, 2e-2), color=ORANGE)
    ax.annotate("reachable set\n$[u_0-W,\\ u_0+W]$", xy=(u0, 2e-5),
                ha="center", color="#1b7a58")
    ax.annotate(r"$\hat K \le 4/g^2$, $g=|u_0|-W$", xy=(gap, 4.0 / gap**2),
                xytext=(8, 2e-1), color="#8a6100",
                arrowprops=dict(arrowstyle="-", color="#8a6100", lw=0.8))
    ax.set_xlabel(r"accumulated mismatch $u=\Delta\beta L$  [rad]")
    ax.set_ylabel(r"efficiency $\hat K(u)$")
    ax.set_ylim(1e-6, 3)
    fig.tight_layout()
    fig.savefig(OUT / "kernel_envelope.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2
def fig_density_regimes():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))

    ax = axes[0]
    u = np.linspace(-4.2, 4.2, 1200)
    cases = [
        (np.array([1.0, 1.0, 1.0]), BLUE, "equal legs $w=(1,1,1)$"),
        (np.array([2.0, 0.7, 0.3]), ORANGE, "generic $w=(2,0.7,0.3)$"),
        (np.array([1.5, 1.45, 0.05]), AQUA, "pathological $w=(1.5,1.45,0.05)$"),
    ]
    for w, color, label in cases:
        ax.plot(u, uniform_sum_density(u, w), color=color, label=label)
    ax.legend(frameon=False)
    ax.set_xlabel(r"offset $u-u_0$")
    ax.set_ylabel(r"density $\rho_{\mathbf{w}}$")
    ax.set_title("(a) 3-uniform mismatch density")

    ax = axes[1]
    W = np.logspace(-1, 5.5, 400)
    far_edge = FAR_MARGIN_FACTOR * W + FAR_MARGIN_OFFSET
    ax.fill_between(W, far_edge, 1e7, color=ORANGE, alpha=0.15, lw=0)
    ax.fill_between(W[W <= WIDE_HALFWIDTH], 1e-2,
                    far_edge[W <= WIDE_HALFWIDTH], color=BLUE, alpha=0.15, lw=0)
    ax.fill_between(W[W > WIDE_HALFWIDTH], 1e-2,
                    far_edge[W > WIDE_HALFWIDTH], color=AQUA, alpha=0.2, lw=0)
    ax.plot(W, far_edge, color=ORANGE, lw=1.2)
    ax.axvline(WIDE_HALFWIDTH, color=AQUA, lw=1.2)
    ax.plot(W, W, color=GRAY, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-1, 3e5)
    ax.set_ylim(1e-2, 1e7)
    ax.annotate("far\n(closed form)", xy=(3, 3e5), color="#a4401a")
    ax.annotate("near\n(quadrature)", xy=(3, 1.5), color="#1a4f8f")
    ax.annotate("wide\n(central+tail)", xy=(1.5e4, 3), color="#157a54")
    ax.annotate(r"$|u_0|=W$", xy=(2e3, 6e2), color=GRAY, rotation=33)
    ax.set_xlabel(r"total width $W=\pi\sum_j|\nu_j|$  [rad]")
    ax.set_ylabel(r"$|u_0|$  [rad]")
    ax.set_title("(b) regime dispatch")
    fig.tight_layout()
    fig.savefig(OUT / "density_regimes.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig_acceptance():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), sharey=True)
    geoms = [
        (np.array([[5.0, 12.0, -30.0]]), 0.4, "(a) generic legs (5, 12, -30)"),
        (np.array([[1.55, -1.52, 0.02]]) * 4.0, 0.4,
         "(b) pathological legs (6.2, -6.1, 0.08)"),
    ]
    numbers = {}
    for ax, (coeffs, d, title) in zip(axes, geoms):
        total = np.pi * np.sum(np.abs(coeffs))
        off = np.linspace(-total, total, 600)[None, :]
        dd = np.array([d])
        exact = exact_conditional_acceptance(off, coeffs, dd)[0]
        approx = pointwise_conditional_acceptance(off, coeffs, dd)[0]
        ax.plot(off[0] / total, exact, color=BLUE, label="exact (zonotope)")
        ax.plot(off[0] / total, approx, color=ORANGE, ls="--",
                label="rescaled 3-uniform model")
        ax.set_xlabel(r"$(u-u_0)/W$")
        ax.set_title(title)
        # masked-F consequence for the caption
        u0 = np.array([0.5 * total])
        w = np.pi * np.abs(coeffs)
        f_ex = near_model_masked(u0, w, coeffs, dd, 2048,
                                 acceptance_fn=exact_conditional_acceptance)[0]
        f_ap = near_model_masked(u0, w, coeffs, dd, 2048)[0]
        # qmc_tuple_ground_truth applies the -nu_c sign itself; undo ours.
        nu_args = (coeffs[0, 0], coeffs[0, 1], -coeffs[0, 2])
        qmc, qerr = qmc_tuple_ground_truth(
            u0=float(u0[0]), nu=nu_args, q=(0, 0, 0, 0), d=d,
            n_points=1 << 16, n_replicates=6, include_quadratic=False)
        numbers[title] = (f_ex, f_ap, qmc, qerr)
    axes[0].set_ylabel(r"$A(u)=P(\mathrm{mask}\mid u)$")
    axes[0].legend(frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(OUT / "acceptance_exact_vs_approx.png", bbox_inches="tight")
    plt.close(fig)
    for k, (f_ex, f_ap, qmc, qerr) in numbers.items():
        print(f"{k}: F_exact={f_ex:.6g} F_approx={f_ap:.6g} "
              f"F_qmc={qmc:.6g}+-{qerr:.2g} "
              f"err_exact={abs(f_ex/qmc-1):.2%} err_approx={abs(f_ap/qmc-1):.2%}")


# ------------------------------------------------------------------ fig 4
def fig_xpm():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    ax = axes[0]
    th = np.linspace(0, 4, 800)
    ax.plot(th, _xpm_mass_transform(th), color=BLUE)
    ax.plot(th[th > 0.4], 1.0 / (np.pi**2 * th[th > 0.4]**2), color=GRAY,
            ls=":", lw=1.2)
    ax.annotate(r"$H(\theta)$", xy=(0.62, 0.42), color=BLUE)
    ax.annotate(r"$1/(\pi\theta)^2$", xy=(1.35, 0.12), color=GRAY)
    ax.annotate(r"$H(0)=2/3$", xy=(1.6, 0.62), color=GRAY)
    ax.set_ylim(0, 0.74)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"masked cosine transform $H$")
    ax.set_title("(a) exact masked XPM transform")

    ax = axes[1]
    nu = np.logspace(-1, 4, 300)
    F = xpm_fast_batch(nu)
    ax.loglog(nu, F, color=BLUE)
    ax.loglog(nu[nu > 5], 1.0 / nu[nu > 5], color=GRAY, ls=":", lw=1.2)
    ax.annotate(r"$F_{\rm XPM}(\nu)$", xy=(0.15, 0.25), color=BLUE)
    ax.annotate(r"$1/|\nu|$ sheet limit", xy=(30, 1.1 / 30), color=GRAY)
    print("sheet-limit check: nu*F at nu=1e3:", 1e3 * xpm_fast_batch(np.array([1e3]))[0],
          " at nu=1e4:", 1e4 * xpm_fast_batch(np.array([1e4]))[0])
    ax.set_xlabel(r"walk-off $|\nu| = |\Delta\beta_1| B L$  [rad]")
    ax.set_ylabel(r"$F_{\rm XPM}$")
    ax.set_title("(b) exact 1-D XPM efficiency")
    fig.tight_layout()
    fig.savefig(OUT / "xpm_reduction.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 5
def fig_fast_vs_qmc():
    # Widths capped at 300 rad: beyond that the Sobol reference itself loses
    # effective samples (the REFINE_MAX_WIDTH phenomenon) and cannot serve as
    # ground truth, so the wide regime is excluded by construction here.
    rng = np.random.default_rng(11)
    n = 60
    total = 10 ** rng.uniform(-0.3, 2.47, n)
    frac = rng.dirichlet(np.ones(3), n)
    widths = total[:, None] * frac
    signs = np.where(rng.random((n, 3)) < 0.5, -1.0, 1.0)
    coeffs = signs * widths / np.pi
    ratio = 10 ** rng.uniform(-2.0, 2.6, n)
    u0 = ratio * (total + 50.0) * np.where(rng.random(n) < 0.5, -1.0, 1.0)
    d = rng.uniform(-1.5, 1.5, n)

    est = linear_tuple_estimate(u0, coeffs, d)
    refined = est.values.copy()
    near = est.regime == 0
    if np.any(near):
        idx = np.where(near)[0]
        nn = int(min(64 + 1.4 * np.max(total[idx]), 9000))
        refined[idx] = near_model_masked(
            u0[idx], np.pi * np.abs(coeffs[idx]), coeffs[idx], d[idx], nn,
            acceptance_fn=exact_conditional_acceptance)
    qmc = np.empty(n)
    qerr = np.empty(n)
    for i in range(n):
        qmc[i], qerr[i] = qmc_tuple_ground_truth(
            u0=float(u0[i]), nu=(coeffs[i, 0], coeffs[i, 1], -coeffs[i, 2]),
            q=(0, 0, 0, 0), d=float(d[i]),
            n_points=1 << 16, n_replicates=6, include_quadratic=False)

    keep = (qmc > 0) & (qerr / np.maximum(qmc, 1e-300) < 0.02)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    sel_n = keep & near
    sel_f = keep & (est.regime == 1)
    ax.errorbar(qmc[sel_n], est.values[sel_n] / qmc[sel_n],
                yerr=qerr[sel_n] / qmc[sel_n], fmt="o", ms=5, lw=0,
                elinewidth=1.0, mfc="none", color=ORANGE,
                label=f"near, bulk model ({int(np.sum(sel_n))})")
    ax.errorbar(qmc[sel_n], refined[sel_n] / qmc[sel_n],
                yerr=qerr[sel_n] / qmc[sel_n], fmt="o", ms=5, lw=0,
                elinewidth=1.0, color=BLUE,
                label=f"near, exact acceptance ({int(np.sum(sel_n))})")
    ax.errorbar(qmc[sel_f], est.values[sel_f] / qmc[sel_f],
                yerr=qerr[sel_f] / qmc[sel_f], fmt="s", ms=5, lw=0,
                elinewidth=1.0, color=AQUA,
                label=f"far, closed form ({int(np.sum(sel_f))})")
    ax.axhline(1.0, color=GRAY, lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"randomized-Sobol ground truth $F$ (linear model, exact mask)")
    ax.set_ylabel("model / ground truth")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fast_vs_qmc.png", bbox_inches="tight")
    plt.close(fig)
    for name, sel, vals in (("bulk", keep & near, est.values),
                            ("refined", keep & near, refined),
                            ("far", keep & (est.regime == 1), est.values)):
        if np.any(sel):
            rel = np.abs(vals[sel] / qmc[sel] - 1.0)
            print(f"fast-vs-QMC [{name}]: n={int(np.sum(sel))}, "
                  f"median={np.median(rel):.3%}, max={np.max(rel):.3%}")


# ------------------------------------------------------------------ fig 6
def fig_support_acceptance():
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    edge = 4.0 * np.pi
    dd = np.linspace(-edge - 1, edge + 1, 900)
    ax.plot(dd, support_acceptance(dd), color=BLUE)
    ax.axvline(edge, color=GRAY, ls=":", lw=1.0)
    ax.axvline(-edge, color=GRAY, ls=":", lw=1.0)
    ax.annotate(r"$A(d)$, $A(0)=2/3$", xy=(0, 0.69), ha="center", color=BLUE)
    ax.annotate("hard support edge\n$|d|=4\\pi$", xy=(7.2, 0.35), color=GRAY)
    ax.set_xlabel(r"support shift $d=\delta\omega/B$  [rad]")
    ax.set_ylabel("acceptance")
    ax.set_ylim(0, 0.78)
    fig.tight_layout()
    fig.savefig(OUT / "support_acceptance.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_kernel_envelope()
    fig_density_regimes()
    fig_acceptance()
    fig_xpm()
    fig_support_acceptance()
    fig_fast_vs_qmc()
    print("figures written to", OUT)
