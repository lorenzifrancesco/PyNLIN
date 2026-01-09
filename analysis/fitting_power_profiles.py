#!/usr/bin/env python3
"""
Compare MMSE finite representations of Raman power profiles using:
  (1) Polynomials with M terms (degree M-1)
  (2) Chebyshev polynomials with M terms (degree M-1)
  (3) Exponential mixtures with M terms chosen by OMP from a fixed dictionary,
      optionally including exp(z-L) terms (implemented as exp(-lambda*(L-z))).

Input file format (your file):
  .npy containing a 0-d object array whose .item() is a dict with:
    - 'z'          : (Nz,)
    - 'signal_sol' : (Nz, Nch)

Outputs:
  - PDF plot of aggregated NMSE vs number of terms for polynomial, Chebyshev,
    and exponential fits.
  - (Optional) PDF with example profile overlays: exact vs poly vs Chebyshev vs exp.

Usage:
  python fit_compare.py \
      --infile /mnt/data/dummy_power_profiles.npy \
      --outpng mse_vs_order.pdf \
      --mmax 25 \
      --nlambda 40 \
      --lam_min 1e-2 \
      --lam_max 1e2 \
      --include_back \
      --agg median \
      --stride 10 \
      --profile_pdf profile_fits.pdf \
      --profile_M 1 5 10 \
      --profile_channels 0 mid last
"""

import argparse
import sys
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def load_profiles_npy(path: str):
    """
    Loads your pickled .npy robustly.
    Some saved objects reference numpy._core.*; this shim maps it to numpy.core.*.
    """
    import numpy.core
    import numpy.core.multiarray as multiarray
    import numpy.core._multiarray_umath as mau

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", multiarray)
    sys.modules.setdefault("numpy._core._multiarray_umath", mau)

    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict-like payload in {path}, got {type(obj)}")
    if "z" not in obj or "signal_sol" not in obj:
        raise KeyError("Expected keys 'z' and 'signal_sol' in the .npy dict.")
    z = np.asarray(obj["z"], dtype=float)
    P = np.asarray(obj["signal_sol"], dtype=float)
    if P.ndim != 2 or z.ndim != 1 or P.shape[0] != z.shape[0]:
        raise ValueError(f"Bad shapes: z={z.shape}, signal_sol={P.shape}")
    return z, P


def nmse(y, yhat):
    """Normalized MSE: ||y-yhat||^2 / ||y||^2."""
    r = y - yhat
    denom = float(y @ y)
    if denom == 0.0:
        return 0.0 if float(r @ r) == 0.0 else np.inf
    return float((r @ r) / denom)


def poly_fit_nmse(y, t, M, V_cache):
    """
    Polynomial MMSE with M terms: yhat = sum_{k=0}^{M-1} a_k t^k.
    Uses least squares (MMSE in L2 sense with uniform samples).
    """
    V = V_cache[M]
    coef, *_ = la.lstsq(V, y, rcond=None)
    yhat = V @ coef
    return nmse(y, yhat)


def poly_fit_yhat(y, V):
    """Return yhat for polynomial LS given design matrix V."""
    coef, *_ = la.lstsq(V, y, rcond=None)
    return V @ coef


def cheb_fit_nmse(y, M, C_cache):
    """
    Chebyshev MMSE with M terms: yhat = sum_{k=0}^{M-1} c_k T_k(x),
    with x mapped to [-1, 1].
    """
    C = C_cache[M]
    coef, *_ = la.lstsq(C, y, rcond=None)
    yhat = C @ coef
    return nmse(y, yhat)


def cheb_fit_yhat(y, C):
    """Return yhat for Chebyshev LS given design matrix C."""
    coef, *_ = la.lstsq(C, y, rcond=None)
    return C @ coef


def build_exp_dictionary(t, lambdas, include_back):
    """
    Builds dictionary A where columns are exponentials.
      forward: exp(-lambda * t)
      backward: exp(-lambda * (1-t))  ~ exp(lambda*(t-1)) = exp((z-L)/L * lambda)
    """
    A_f = np.exp(-np.outer(t, lambdas))
    if not include_back:
        return A_f
    A_b = np.exp(-np.outer(1.0 - t, lambdas))
    return np.hstack([A_f, A_b])


def omp_mmse_nmse_from_gram(y, Aty, yty, G, M):
    """
    OMP using precomputed Gram matrix G = A^T A and A^T y.
    Selects M atoms greedily and solves exact LS on selected set each step.
    Returns NMSE = ||r||^2 / ||y||^2.
    """
    S = []
    aS = np.zeros((0,), dtype=float)
    c = Aty.copy()

    for _ in range(M):
        j = int(np.argmax(np.abs(c)))
        if j in S:
            break
        S.append(j)

        GS = G[np.ix_(S, S)]
        rhs = Aty[S]
        aS = la.solve(GS, rhs)

        c = Aty - G[:, S] @ aS

    res2 = yty - 2.0 * float(aS @ Aty[S]) + float(aS @ (G[np.ix_(S, S)] @ aS))
    if yty == 0.0:
        return 0.0 if res2 == 0.0 else np.inf
    return float(res2 / yty)


def omp_yhat_from_gram(A, Aty, G, M):
    """
    OMP selection + LS reconstruction using precomputed Gram matrix.
    Returns yhat.
    """
    S = []
    aS = np.zeros((0,), dtype=float)
    c = Aty.copy()

    for _ in range(M):
        j = int(np.argmax(np.abs(c)))
        if j in S:
            break
        S.append(j)

        GS = G[np.ix_(S, S)]
        rhs = Aty[S]
        aS = la.solve(GS, rhs)

        c = Aty - G[:, S] @ aS

    if len(S) == 0:
        return np.zeros((A.shape[0],), dtype=float)

    return A[:, S] @ aS


def aggregate(vals, mode):
    vals = np.asarray(vals, dtype=float)
    if mode == "mean":
        return float(np.mean(vals))
    if mode == "median":
        return float(np.median(vals))
    if mode == "p90":
        return float(np.percentile(vals, 90))
    raise ValueError(f"Unknown agg={mode}")


def parse_profile_channels(spec_list, Nch, seed=0):
    """
    spec_list examples: ["0","mid","last","rand3","17"]
    Returns unique channel indices in [0, Nch-1], in the order encountered.
    """
    rng = np.random.default_rng(seed)
    out = []
    for s in spec_list:
        s = str(s).strip().lower()
        if s == "mid":
            idx = Nch // 2
            out.append(idx)
        elif s == "last":
            out.append(Nch - 1)
        elif s.startswith("rand"):
            k = int(s[4:]) if len(s) > 4 else 1
            k = max(1, min(k, Nch))
            picks = rng.choice(Nch, size=k, replace=False).tolist()
            out.extend(picks)
        else:
            idx = int(s)
            if idx < 0 or idx >= Nch:
                raise ValueError(f"Channel index {idx} out of range [0, {Nch-1}]")
            out.append(idx)

    # unique-preserving
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def save_profile_pdf(pdf_path, z, P, V_cache, C_cache, A, G, channels_to_plot, Ms):
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_path) as pdf:
        for ch in channels_to_plot:
            y = P[:, ch].astype(float, copy=False)

            fig, axes = plt.subplots(1, len(Ms), figsize=(5 * len(Ms) * 0.5, 4*0.5), sharey=True)
            if len(Ms) == 1:
                axes = [axes]

            for ax, M in zip(axes, Ms):
                V = V_cache[M]
                y_poly = poly_fit_yhat(y, V)

                C = C_cache[M]
                y_cheb = cheb_fit_yhat(y, C)

                Aty = A.T @ y
                y_exp = omp_yhat_from_gram(A, Aty, G, M)

                ax.plot(z, y, label="exact")
                ax.plot(z, y_poly, label=f"poly", ls="--")
                ax.plot(z, y_cheb, label=f"cheb", ls="-.")
                ax.plot(z, y_exp, label=f"exp (cheating)", ls=":")
                ax.set_xlabel("z [m]")
                ax.set_title(f"Channel {ch}, M={M}")

            axes[0].set_ylabel("Power [W]")
            # One legend per page
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
            fig.tight_layout(rect=[0, 0, 1, 0.90])
            pdf.savefig(fig)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Path to dummy_power_profiles.npy")
    ap.add_argument("--outpng", default="mse_vs_order.pdf", help="Output plot filename (PDF recommended)")
    ap.add_argument("--mmax", type=int, default=25, help="Max number of terms M")
    ap.add_argument("--stride", type=int, default=1, help="Use every stride-th channel (for speed)")
    ap.add_argument("--max_channels", type=int, default=0, help="Limit number of channels (0 = all)")
    ap.add_argument("--agg", choices=["mean", "median", "p90"], default="median",
                    help="Aggregate statistic across channels")
    ap.add_argument("--nlambda", type=int, default=40, help="Number of candidate exponential rates")
    ap.add_argument("--lam_min", type=float, default=1e-2, help="Min lambda in normalized units")
    ap.add_argument("--lam_max", type=float, default=1e2, help="Max lambda in normalized units")
    ap.add_argument("--include_back", action="store_true",
                    help="Include exp(-lambda*(L-z)) terms (aka exp(z-L) family)")

    # NEW: profile overlay PDF options
    ap.add_argument("--profile_pdf", default="",
                    help="If set (non-empty), save profile overlays to this PDF.")
    ap.add_argument("--profile_M", nargs="+", type=int, default=[1, 5, 10],
                    help="M values to plot in the overlay PDF (e.g., 1 5 10).")
    ap.add_argument("--profile_channels", nargs="+", default=["0", "mid", "last"],
                    help="Which channels to plot in the PDF. Supports integers, 'mid', 'last', 'randK' (e.g., rand3).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for randK selection.")

    args = ap.parse_args()

    z, P = load_profiles_npy(args.infile)
    Nz, Nch = P.shape
    z0 = float(z[0])
    L = float(z[-1] - z0)
    if L <= 0:
        raise ValueError("z must be increasing with positive span.")
    t = (z - z0) / L  # normalize to [0,1] for conditioning
    x = 2.0 * t - 1.0  # Chebyshev domain [-1, 1]

    # Choose channels for NMSE curve
    channels = list(range(0, Nch, args.stride))
    if args.max_channels and args.max_channels > 0:
        channels = channels[: args.max_channels]
    if len(channels) == 0:
        raise ValueError("No channels selected; adjust --stride/--max_channels.")
    print(f"Loaded Nz={Nz}, Nch={Nch}. Using {len(channels)} channels for NMSE curve.")

    # Polynomial Vandermonde cache: V_cache[M] is (Nz,M)
    V_cache = {M: np.vander(t, N=M, increasing=True) for M in range(1, args.mmax + 1)}
    # Chebyshev cache: C_cache[M] is (Nz,M)
    C_cache = {M: np.polynomial.chebyshev.chebvander(x, deg=M - 1) for M in range(1, args.mmax + 1)}

    # Exponential dictionary + Gram precompute
    lambdas = np.logspace(np.log10(args.lam_min), np.log10(args.lam_max), args.nlambda)
    A = build_exp_dictionary(t, lambdas, include_back=args.include_back)
    G = A.T @ A  # (K,K)
    print(f"Exponential dictionary size: {A.shape[1]} atoms "
          f"({'with' if args.include_back else 'without'} exp(z-L) family).")

    poly_curve = []
    cheb_curve = []
    exp_curve = []

    for M in range(1, args.mmax + 1):
        poly_vals = []
        cheb_vals = []
        exp_vals = []
        for ch in channels:
            y = P[:, ch].astype(float, copy=False)
            # Polynomial MMSE
            poly_vals.append(poly_fit_nmse(y, t, M, V_cache))
            # Chebyshev MMSE
            cheb_vals.append(cheb_fit_nmse(y, M, C_cache))

            # Exponential OMP-MMSE
            Aty = A.T @ y
            yty = float(y @ y)
            exp_vals.append(omp_mmse_nmse_from_gram(y, Aty, yty, G, M))

        poly_curve.append(aggregate(poly_vals, args.agg))
        cheb_curve.append(aggregate(cheb_vals, args.agg))
        exp_curve.append(aggregate(exp_vals, args.agg))
        print(
            f"M={M:2d}: poly {args.agg} NMSE={poly_curve[-1]:.3e} | "
            f"cheb {args.agg} NMSE={cheb_curve[-1]:.3e} | "
            f"exp {args.agg} NMSE={exp_curve[-1]:.3e}"
        )

    # Plot NMSE vs M
    Ms = np.arange(1, args.mmax + 1)
    plt.figure()
    plt.semilogy(Ms, poly_curve, marker="o", linewidth=1.0,
                 label="Polynomial (M terms, degree M-1)")
    plt.semilogy(Ms, cheb_curve, marker="^", linewidth=1.0,
                 label="Chebyshev (M terms, degree M-1)")
    lbl = "Exponential OMP (M terms, exp(-lambda z))"
    if args.include_back:
        lbl = "Exponential OMP (M terms, exp(-lambda z) + exp(-lambda(L-z)))"
    plt.semilogy(Ms, exp_curve, marker="s", linewidth=1.0, label=lbl)
    plt.xlabel("Number of terms M")
    plt.ylabel(f"{args.agg} NMSE across selected channels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outpng, dpi=200)
    print(f"Saved plot: {args.outpng}")

    # NEW: save profile overlay PDF
    if args.profile_pdf:
        # validate Ms
        prof_Ms = [m for m in args.profile_M if 1 <= m <= args.mmax]
        if len(prof_Ms) == 0:
            raise ValueError("--profile_M contains no values in [1, --mmax].")

        channels_to_plot = parse_profile_channels(args.profile_channels, Nch, seed=args.seed)
        print(f"Saving profile overlays to {args.profile_pdf} for channels {channels_to_plot} and M={prof_Ms}.")

        save_profile_pdf(
            pdf_path=args.profile_pdf,
            z=z,
            P=P,
            V_cache=V_cache,
            C_cache=C_cache,
            A=A,
            G=G,
            channels_to_plot=channels_to_plot,
            Ms=prof_Ms,
        )
        print(f"Saved PDF: {args.profile_pdf}")


if __name__ == "__main__":
    main()
