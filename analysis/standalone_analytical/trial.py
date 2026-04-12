"""Standalone script version of N_vs_Kxci_deltaf_over_B.ipynb.

Plots

    N(x) T^2 L^-2 = 4/9 * (1 + ((1/lambda) * (L/L_eff) * x)^(1/eta))^(-eta)

and

    K_XCI(x) T^2 L^-2 = (L_eff/L) * |ln((x + 1/2)/(x - 1/2))|

as functions of x = Delta f / B.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.grid"] = True


def n_calligraphic(x: np.ndarray, lam: float = 1.0, l_over_leff: float = 1.0, eta: float = 1.0) -> np.ndarray:
    a = (1.0 / lam) * l_over_leff
    return (4.0 / 9.0) * (1.0 + (a * x) ** (1.0 / eta)) ** (-eta)


def k_xci(x: np.ndarray, leff_over_l: float = 1.0) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    mask = x > 0.5
    out[mask] = leff_over_l * np.abs(np.log((x[mask] + 0.5) / (x[mask] - 0.5)))
    return out


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-14)
    out[mask] = num[mask] / den[mask]
    return out


def find_crossover(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float | None:
    d = y1 - y2
    finite = np.isfinite(d)
    idx = np.where(finite[:-1] & finite[1:] & (d[:-1] * d[1:] <= 0))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    x1, x2 = x[i], x[i + 1]
    d1, d2 = d[i], d[i + 1]
    if np.isclose(d2, d1):
        return float(x1)
    return float(x1 - d1 * (x2 - x1) / (d2 - d1))


def finite_min_max(*arrays: np.ndarray | float) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for array in arrays:
        arr = np.asarray(array, dtype=float).ravel()
        if arr.size:
            values.append(arr[np.isfinite(arr)])
    if not values:
        raise ValueError("No finite values available for plotting.")
    merged = np.concatenate(values)
    return float(np.min(merged)), float(np.max(merged))


def add_reference_lines(ax: plt.Axes, x: np.ndarray) -> tuple[float, float]:
    y_ln3 = float(np.log(3.0))
    y_four_ninths = 4.0 / 9.0
    x_annot = float(x[int(0.8 * len(x))])

    ax.axhline(y_ln3, linestyle="--", linewidth=1.2, color="crimson", zorder=10, label=r"$\ln(3)$")
    ax.axhline(y_four_ninths, linestyle="--", linewidth=1.2, color="darkgreen", zorder=10, label=r"$4/9$")
    ax.scatter([x_annot], [y_ln3], color="crimson", s=35, zorder=11)
    ax.scatter([x_annot], [y_four_ninths], color="darkgreen", s=35, zorder=11)
    ax.annotate(r"$\ln(3)$", xy=(x_annot, y_ln3), xytext=(8, 8), textcoords="offset points", color="crimson")
    ax.annotate(r"$4/9$", xy=(x_annot, y_four_ninths), xytext=(8, 8), textcoords="offset points", color="darkgreen")
    return y_ln3, y_four_ninths


def plot_comparison(
    lam: float = 1.0,
    l_over_leff: float = 1.0,
    eta: float = 1.0,
    x_min: float = 0.0,
    x_max: float = 10.0,
    npts: int = 2000,
    show_ratio: bool = True,
    show_crossover: bool = True,
    normalize: bool = False,
    loglog: bool = False,
) -> None:
    if npts < 2:
        raise ValueError("npts must be at least 2.")
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")
    if lam <= 0 or l_over_leff <= 0 or eta <= 0:
        raise ValueError("lambda, L/Leff, and eta must be strictly positive.")
    if loglog and x_min <= 0:
        raise ValueError("log-log plots require x_min > 0.")

    x = np.linspace(x_min, x_max, int(npts))
    leff_over_l = 1.0 / l_over_leff

    y_n = n_calligraphic(x, lam=lam, l_over_leff=l_over_leff, eta=eta)
    y_k = k_xci(x, leff_over_l=leff_over_l)

    if normalize:
        y_n = y_n / np.nanmax(np.abs(y_n))
        if np.any(np.isfinite(y_k)):
            y_k = y_k / np.nanmax(np.abs(y_k))

    fig, ax = plt.subplots()
    ax.plot(x, y_n, label=r"$\mathcal{N}(x)\, T^2 L^{-2}$", linewidth=2)
    ax.plot(x, y_k, label=r"$K_{\mathrm{XCI}}(x)\, T^2 L^{-2}$", linewidth=2)

    y_ln3, y_four_ninths = add_reference_lines(ax, x)

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ymin, ymax = finite_min_max(y_n, y_k, y_ln3, y_four_ninths)
    pad = 0.05 * (ymax - ymin) if ymax > ymin else (0.1 * ymax if ymax != 0 else 0.1)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel(r"$x = \Delta f / B$")
    ax.set_ylabel("normalized value" if normalize else r"$T^2 L^{-2}$")
    ax.set_title(r"Comparison of $\mathcal{N}\, T^2 L^{-2}$ and $K_{\mathrm{XCI}}\, T^2 L^{-2}$")
    ax.legend()

    xc = None
    if show_crossover:
        xc = find_crossover(x, y_n, y_k)
        if xc is not None and np.isfinite(xc):
            ax.axvline(xc, linestyle="--", linewidth=1.5)
            ax.text(xc, ax.get_ylim()[1], f"  crossover ~ {xc:.3f}", va="top")

    if show_ratio:
        ratio = safe_ratio(y_k, y_n)
        fig, ax = plt.subplots()
        ax.plot(x, ratio, linewidth=2)
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(r"$x = \Delta f / B$")
        ax.set_ylabel(r"$(K_{\mathrm{XCI}}\, T^2 L^{-2}) / (\mathcal{N}\, T^2 L^{-2})$")
        ax.set_title(r"Ratio $(K_{\mathrm{XCI}}\, T^2 L^{-2}) / (\mathcal{N}\, T^2 L^{-2})$")
        if xc is not None and np.isfinite(xc):
            ax.axvline(xc, linestyle="--", linewidth=1.5)

    print(f"Parameters: lambda={lam}, L/Leff={l_over_leff}, eta={eta}")
    if xc is None:
        print("No crossover found in the chosen x-range.")
    else:
        print(f"Crossover at x = Delta f / B ≈ {xc:.6g}")
    print("Note: K_XCI is undefined for x <= 1/2 and is masked there.")

    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lam", type=float, default=1.0, help="Lambda parameter.")
    parser.add_argument("--l-over-leff", type=float, default=1.0, help="L / Leff.")
    parser.add_argument("--eta", type=float, default=1.0, help="Eta parameter.")
    parser.add_argument("--x-min", type=float, default=0.0, help="Lower bound for x.")
    parser.add_argument("--x-max", type=float, default=10.0, help="Upper bound for x.")
    parser.add_argument("--npts", type=int, default=2000, help="Number of x samples.")
    parser.add_argument("--show-ratio", action=argparse.BooleanOptionalAction, default=True, help="Show ratio plot.")
    parser.add_argument(
        "--show-crossover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark the first crossover point.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize each curve by its own maximum.",
    )
    parser.add_argument(
        "--loglog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use log-log axes. Requires x_min > 0.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_comparison(
        lam=args.lam,
        l_over_leff=args.l_over_leff,
        eta=args.eta,
        x_min=args.x_min,
        x_max=args.x_max,
        npts=args.npts,
        show_ratio=args.show_ratio,
        show_crossover=args.show_crossover,
        normalize=args.normalize,
        loglog=args.loglog,
    )


if __name__ == "__main__":
    main()
