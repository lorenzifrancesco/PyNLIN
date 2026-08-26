"""Dispatcher seam audit: per-tuple accuracy at the dispatch boundaries.

For one probe target, samples tuples near each regime-dispatch seam and
evaluates them with both adjacent evaluators plus randomized-Sobol ground
truth (linear model, exact output mask), then checks the sheet closed form
against the exact quadrature on the tube survivors and reports where the
survivors sit in the dispatch plane.  The tube report uses the current
mask-aware projected reachable interval; it does not reduce selection to a
boundary in (W, |u0|).  Companion to
docs/source/lorenzi_fast_cost_anatomy.md.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fast_analytic import (
    SHEET_CORE_MARGIN,
    SHEET_MIN_ACCEPTANCE,
    SHEET_MIN_WIDTH,
    masked_linear_phase_outer_interval,
    select_tube,
)
from pynlin.methods.td.fast_nlin import (
    CF_BASE_NODES,
    CF_MAX_NODES,
    NEAR_NODES_PER_RADIAN,
    exact_conditional_acceptance,
    far_model,
    fwm_tuple_variables,
    linear_tuple_estimate,
    near_model_masked,
    qmc_tuple_ground_truth,
    refine_tuples_exact,
    support_acceptance,
    uniform_sum_density,
    wide_model_masked,
)
from pynlin.methods.td.fullband_mc import (
    _beta0_abs_from_fiber,
    decimated_frequency_grid,
)
from pynlin.system import System


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--target", type=int, default=1720)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--n-seam", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    system = System.from_toml(args.config)
    _, freqs = decimated_frequency_grid(system, 1)
    B, L = float(system.pulse.baud_rate), float(system.fiber_length)
    b1g, b2g = system.beta_grids(freqs=freqs)
    b1, b2 = np.asarray(b1g[0], float), np.asarray(b2g[0], float)
    b0 = _beta0_abs_from_fiber(system, freqs, b1)
    v = fwm_tuple_variables(freqs, b0, b1, b2, B, L, args.target)
    coeffs = np.stack([v.nu_a, v.nu_b, -v.nu_c], axis=-1)
    W = np.sum(v.widths, axis=-1)
    absu = np.abs(v.u0)

    def qmc(i: int) -> tuple[float, float]:
        return qmc_tuple_ground_truth(
            u0=float(v.u0[i]),
            nu=(float(v.nu_a[i]), float(v.nu_b[i]), float(v.nu_c[i])),
            q=(0.0, 0.0, 0.0, 0.0),
            d=float(v.d[i]),
            n_points=1 << 16,
            n_replicates=4,
            include_quadratic=False,
        )

    def near_val(i: int) -> float:
        nn = int(min(CF_BASE_NODES + NEAR_NODES_PER_RADIAN * W[i], CF_MAX_NODES))
        return float(
            near_model_masked(v.u0[[i]], v.widths[[i]], coeffs[[i]], v.d[[i]], nn)[0]
        )

    # Seam 1: far/near boundary.
    margin = absu / (3.0 * W + 3000.0)
    cand = np.where((W < 2500) & (margin > 0.95) & (margin < 1.05))[0]
    pick = rng.choice(cand, size=min(args.n_seam, cand.size), replace=False)
    lg.info(f"seam far/near: {cand.size} candidates, auditing {pick.size}")
    for i in pick:
        fa = float(
            far_model(v.u0[[i]], v.widths[[i]])[0] * support_acceptance(v.d[[i]])[0]
        )
        ne = near_val(i)
        g, gs = qmc(i)
        lg.info(
            f"  W={W[i]:7.1f} |u0|={absu[i]:9.1f}  far/qmc={fa / g:6.3f} "
            f"near/qmc={ne / g:6.3f}  (qmc={g:.3e}±{gs:.1e})"
        )

    # Seam 2: wide/near boundary.
    cand = np.where((W > 2700) & (W < 3300) & (margin < 0.8))[0]
    pick = rng.choice(cand, size=min(args.n_seam, cand.size), replace=False)
    lg.info(f"seam wide/near: {cand.size} candidates, auditing {pick.size}")
    for i in pick:
        ne = near_val(i)
        ct, tl = wide_model_masked(v.u0[[i]], v.widths[[i]], coeffs[[i]], v.d[[i]])
        wd = float(ct[0] + tl[0] * support_acceptance(v.d[[i]])[0])
        g, gs = qmc(i)
        lg.info(
            f"  W={W[i]:7.1f} |u0|={absu[i]:9.1f}  wide/qmc={wd / g:6.3f} "
            f"near/qmc={ne / g:6.3f}  (qmc={g:.3e}±{gs:.1e})"
        )

    # Check 3: sheet closed form vs exact quadrature on the tube survivors.
    keep, _ = select_tube(v, args.epsilon)
    Wk = W[keep]
    eligible = (
        (Wk > SHEET_MIN_WIDTH)
        & (absu[keep] < Wk - SHEET_CORE_MARGIN)
        & (absu[keep] <= 3.0 * Wk + 3000.0)
    )
    sel = keep[np.where(eligible)[0]]
    rho0 = uniform_sum_density(-v.u0[sel][:, None], v.widths[sel][:, None, :])[:, 0]
    acc0 = exact_conditional_acceptance(-v.u0[sel][:, None], coeffs[sel], v.d[sel])[
        :, 0
    ]
    good = acc0 >= SHEET_MIN_ACCEPTANCE
    sheet_v = 2.0 * np.pi * rho0[good] * acc0[good]
    sel_g = sel[good]
    est = linear_tuple_estimate(v.u0[sel_g], coeffs[sel_g], v.d[sel_g])
    reg_full = np.zeros(v.u0.size, dtype=int)
    reg_full[sel_g] = est.regime
    exact_v = refine_tuples_exact(v, sel_g, reg_full)
    r = sheet_v / np.where(exact_v != 0, exact_v, np.nan)
    lg.info(
        f"sheet vs exact on {sel_g.size} fired survivors "
        f"({sel.size} eligible, {int(np.sum(~good))} demoted by the "
        f"A_cond guard): ratio median={np.nanmedian(r):.4f} "
        f"p5={np.nanpercentile(r, 5):.4f} p95={np.nanpercentile(r, 95):.4f} "
        f"mass-weighted={np.nansum(sheet_v) / np.nansum(exact_v):.4f}"
    )

    # Check 4: survivor placement across regimes.
    lower, upper = masked_linear_phase_outer_interval(v)
    projected_gap = np.where(
        (lower <= 0.0) & (upper >= 0.0),
        0.0,
        np.minimum(np.abs(lower), np.abs(upper)),
    )
    box_crossing = absu <= W
    lg.info(
        f"tube geometry: {int(np.sum(box_crossing))} unmasked-box crossings; "
        f"{int(np.sum(box_crossing & (projected_gap > 0.0)))} of them are "
        "gapped by the mask-aware outer interval"
    )
    est_k = linear_tuple_estimate(v.u0[keep], coeffs[keep], v.d[keep])
    vals = est_k.values
    for name, m in (
        ("near", est_k.regime == 0),
        ("far", est_k.regime == 1),
        ("wide", est_k.regime == 2),
    ):
        if np.any(m):
            lg.info(
                f"survivors {name}: n={int(np.sum(m))} "
                f"mass={np.sum(vals[m]) / np.sum(vals):.2%} "
                f"W median={np.median(Wk[m]):.0f} "
                f"projected-gap median={np.median(projected_gap[keep][m]):.1f}"
            )


if __name__ == "__main__":
    main()
