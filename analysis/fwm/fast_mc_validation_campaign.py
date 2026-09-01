"""Fast-vs-MC validation campaign for XPM (n-PC) and strict FWM.

Validates the *production* fast path against randomized-QMC / Monte-Carlo
ground truth over an extensive, deliberately stratified set of cases, and
reports the errors against the reduced variables of
``docs/source/lorenzi_fast_method.md`` so that failure regimes are located
in the phase diagram rather than in channel index.

No part of the fast path evaluated here falls back to Monte-Carlo: the
branch-2 ("fallback") tier of :func:`analytic_tuple_values` is the
deterministic Gauss-Legendre exact-acceptance quadrature
(:func:`refine_tuples_exact`), not the deprecated
:func:`refine_tuples_qmc`.  This script asserts that at start-up.

Stages
------
``fwm``
    Per-tuple strict-FWM validation.  Probe targets are placed across
    O/E/S/C/L/U and specifically at the zero-dispersion wavelength; for each
    target the tuple population is stratified over

      * the branch taken by the analytic dispatcher (sheet / far / fallback),
      * loudness ``x_grad = L B ||grad dbeta||``,
      * detuning ``|mu| = |u0| / x_grad``,
      * support offset ``|d|``,
      * zero-line proximity ``zeta = min_j|nu_j| / max_j|nu_j|`` -- the
        distance to the degenerate directions where one leg's walk-off
        vanishes (near-ZDW group-delay matching),

    and a bounded number of tuples is drawn per stratum.  Each drawn tuple is
    evaluated three ways: fast analytic, QMC with the linear phase model, and
    QMC with the full quadratic model.  The split separates *mask/regime
    model* error from *in-channel quadratic* error.

``xpm``
    Per-pair XPM validation.  (a) a controlled sweep in the reduced pair
    variables ``(L/L_W = nu, L/L_D = beta2 B^2 L)`` comparing the fast closed
    form against both the QMC pair ground truth and the direct n-PC sector
    estimator's aggregate ``N1``, with the 2PC/3PCa/3PCb/4PC composition
    reported alongside; (b) the physical pairs of the real comb at the same
    probe targets.

Outputs figures under ``--out-dir`` and raw arrays as ``.npz`` next to them.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
from pynlin.methods.td import fast_analytic
from pynlin.methods.td.fast_analytic import (
    SHEET_CORE_MARGIN,
    SHEET_MIN_WIDTH,
    analytic_tuple_values,
    envelope_bound,
    select_tube,
)
from pynlin.methods.td.fast_nlin import (
    FAR_MARGIN_FACTOR,
    FAR_MARGIN_OFFSET,
    FWMTupleVariables,
    fwm_tuple_variables,
    qmc_tuple_ground_truth,
    qmc_xpm_ground_truth,
    xpm_fast_batch,
    xpm_pair_variables,
)
from pynlin.methods.td.fullband_mc import (
    _beta0_abs_from_fiber,
    decimated_frequency_grid,
)
from pynlin.methods.td.xhkm_mc import estimate_xhkm_sectors_direct_mc
from pynlin.system import System
from pynlin.utils import nu2lambda

# ITU band edges in nm; used only to label probe targets.
BANDS = (
    ("O", 1260.0, 1360.0),
    ("E", 1360.0, 1460.0),
    ("S", 1460.0, 1530.0),
    ("C", 1530.0, 1565.0),
    ("L", 1565.0, 1625.0),
    ("U", 1625.0, 1675.0),
)

BRANCH_NAMES = ("sheet", "far", "fallback")
BRANCH_COLORS = ("#2b7bba", "#d95f02", "#7570b3")


# ---------------------------------------------------------------------------
# Guard: the fast path must never call a Monte-Carlo evaluator
# ---------------------------------------------------------------------------


def assert_no_mc_fallback() -> None:
    """Fail loudly if the production fast path can reach an MC evaluator.

    ``analytic_tuple_values`` dispatches to (0) the sheet closed form,
    (1) the far closed form and (2) ``refine_tuples_exact``.  The last is a
    deterministic bucketed Gauss-Legendre quadrature; the QMC refinement
    (``refine_tuples_qmc``) and the QMC ground truths must not be reachable
    from the module's imported namespace as evaluators.
    """
    forbidden = ("refine_tuples_qmc", "qmc_tuple_ground_truth", "qmc_xpm_ground_truth")
    reachable = [name for name in forbidden if hasattr(fast_analytic, name)]
    if reachable:
        raise RuntimeError(
            "fast path imports Monte-Carlo evaluators: "
            + ", ".join(reachable)
            + " -- stop and report to the user before running the campaign."
        )
    import inspect

    src = inspect.getsource(fast_analytic)
    for token in ("Sobol", "qmc", "default_rng", "np.random"):
        if token in src:
            raise RuntimeError(
                f"fast_analytic references {token!r}: possible MC fallback -- "
                "stop and report to the user before running the campaign."
            )
    lg.info("guard: fast path is MC-free (sheet/far closed forms + exact quadrature)")


# ---------------------------------------------------------------------------
# Probe target selection
# ---------------------------------------------------------------------------


def band_of(freq_hz: float) -> str:
    lam_nm = float(nu2lambda(freq_hz)) * 1e9
    for name, lo, hi in BANDS:
        if lo <= lam_nm < hi:
            return name
    return "?"


@dataclass(frozen=True)
class Probe:
    index: int
    label: str
    freq: float
    beta2: float


def choose_probes(freqs: np.ndarray, beta2: np.ndarray, per_band: int = 2) -> list[Probe]:
    """Probe targets: ``per_band`` per ITU band plus the ZDW and its flanks.

    The ZDW targets are the interesting ones: ``beta2 -> 0`` collapses the
    walk-off of channel pairs straddling it, which is where the zero lines
    ``nu_j = 0`` of the tuple geometry are densely populated.
    """
    probes: dict[int, Probe] = {}

    def add(idx: int, label: str) -> None:
        idx = int(np.clip(idx, 0, freqs.size - 1))
        probes.setdefault(idx, Probe(idx, label, float(freqs[idx]), float(beta2[idx])))

    zdw = int(np.argmin(np.abs(beta2)))
    add(zdw, "ZDW")
    # Flanks at a few |beta2| decades on either side of the ZDW.
    for sign, tag in ((-1, "below"), (+1, "above")):
        for frac in (0.05, 0.2, 0.6):
            side = np.arange(freqs.size)
            # beta2 is monotone in index around the ZDW for a standard fiber;
            # pick by |beta2| quantile on the requested side.
            cand = side[(side - zdw) * sign > 0] if sign > 0 else side[(side - zdw) * sign > 0]
            if cand.size == 0:
                continue
            span = np.abs(beta2[cand])
            tgt = frac * float(np.max(span))
            add(int(cand[np.argmin(np.abs(span - tgt))]), f"ZDW{tag}-{frac:g}")

    by_band: dict[str, list[int]] = {}
    for i, f in enumerate(freqs):
        by_band.setdefault(band_of(float(f)), []).append(i)
    for name, _, _ in BANDS:
        idx = by_band.get(name, [])
        if not idx:
            continue
        for k in np.unique(np.linspace(0, len(idx) - 1, per_band).astype(int)):
            add(idx[int(k)], name)

    return sorted(probes.values(), key=lambda p: p.index)


# ---------------------------------------------------------------------------
# FWM stratified tuple sampling
# ---------------------------------------------------------------------------


def predicted_branch(v: FWMTupleVariables, keep: np.ndarray) -> np.ndarray:
    """Cheap replica of the dispatcher's branch predicate, without evaluating.

    ``analytic_tuple_values`` decides sheet/far/fallback from ``|u0|``, the
    leg widths and two thresholds; only the sheet demotion (conditional
    acceptance below ``SHEET_MIN_ACCEPTANCE``) needs an actual evaluation.
    Reproducing the predicate here lets the stratifier see the whole
    population while the (expensive) quadrature runs on sampled tuples only.
    """
    u0 = v.u0[keep]
    widths = np.pi * np.abs(
        np.stack([v.nu_a[keep], v.nu_b[keep], -v.nu_c[keep]], axis=-1)
    )
    W = np.sum(widths, axis=-1)
    abs_u0 = np.abs(u0)
    far = abs_u0 > FAR_MARGIN_FACTOR * W + FAR_MARGIN_OFFSET
    sheet = (~far) & (W > SHEET_MIN_WIDTH) & (abs_u0 < W - SHEET_CORE_MARGIN)
    out = np.full(keep.size, 2, dtype=int)
    out[far] = 1
    out[sheet] = 0
    return out


def zero_line_ratio(v: FWMTupleVariables) -> np.ndarray:
    """``zeta = min_j |nu_j| / max_j |nu_j|`` per tuple.

    ``zeta -> 0`` is the degenerate direction where one leg is group-delay
    matched to the target (the ``u1,u2,u3`` zero-phase plane normals of
    theory doc section 10.4.1) -- the region where ``linear_tuple_estimate``
    is known to be fragile.
    """
    legs = np.abs(np.stack([v.nu_a, v.nu_b, v.nu_c], axis=-1))
    lo = np.min(legs, axis=-1)
    hi = np.max(legs, axis=-1)
    return np.divide(lo, hi, out=np.zeros_like(lo), where=hi > 0.0)


def stratified_indices(
    v: FWMTupleVariables,
    branch: np.ndarray,
    *,
    per_stratum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw up to ``per_stratum`` tuples from every occupied stratum.

    Strata are the product of branch x loudness-decade x detuning-decade x
    zero-line-decade x support-offset half.  This deliberately over-samples
    the rare corners (that is the point: the mass-weighted picture is
    already covered by S2/S3, what is missing is coverage of the tails).
    """
    x = v.x_grad
    mu = np.abs(v.mu)
    zeta = zero_line_ratio(v)
    d = np.abs(v.d)

    def decade(a: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
        z = np.log10(np.clip(a, lo, hi))
        return np.clip(((z - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) * n).astype(int), 0, n - 1)

    key = (
        branch.astype(np.int64) * 10_000_000
        + decade(x, 1e-1, 1e5, 12) * 100_000
        + decade(np.maximum(mu, 1e-4), 1e-4, 1e3, 10) * 1_000
        + decade(np.maximum(zeta, 1e-6), 1e-6, 1.0, 8) * 10
        + (d > 1.0).astype(np.int64) * 5
        + (v.acceptance > 0.5).astype(np.int64)
    )
    out: list[int] = []
    for k in np.unique(key):
        pool = np.where(key == k)[0]
        if pool.size <= per_stratum:
            out.extend(pool.tolist())
        else:
            out.extend(rng.choice(pool, size=per_stratum, replace=False).tolist())
    return np.sort(np.array(out, dtype=int))


def _qmc_job(payload):
    u0, nu, q, d, n_points, n_reps, seed = payload
    full, full_se = qmc_tuple_ground_truth(
        u0=u0, nu=nu, q=q, d=d, n_points=n_points, n_replicates=n_reps,
        seed=seed, include_quadratic=True,
    )
    lin, lin_se = qmc_tuple_ground_truth(
        u0=u0, nu=nu, q=q, d=d, n_points=n_points, n_replicates=n_reps,
        seed=seed, include_quadratic=False,
    )
    return full, full_se, lin, lin_se


def run_fwm_stage(args, system, freqs, beta0_abs, beta1, beta2, probes) -> dict:
    """Per-tuple FWM validation over the probe set.

    With ``--window W`` each probe is enumerated on a contiguous W-channel
    slice of the grid centred on it, instead of the whole comb.  That is the
    only way to reach the *physical* channel pitch (decimation 1, spacing /
    baud ratio r = 1.02) without enumerating the full O(n^3)-scale tuple
    population: the window truncates the interferer set, which changes the
    channel *sum* but not the per-tuple geometry that is under test here.
    """
    baud = float(system.pulse.baud_rate)
    length = float(system.fiber_length)
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    t_fast_total = 0.0
    probe_budget = int(args.time_full_probes)
    for probe in probes:
        if args.window > 0 and freqs.size > args.window:
            half = args.window // 2
            lo = int(np.clip(probe.index - half, 0, freqs.size - args.window))
            sl = slice(lo, lo + args.window)
            f_w, b0_w, b1_w, b2_w = freqs[sl], beta0_abs[sl], beta1[sl], beta2[sl]
            t_local = probe.index - lo
        else:
            f_w, b0_w, b1_w, b2_w = freqs, beta0_abs, beta1, beta2
            t_local = probe.index
        v = fwm_tuple_variables(
            f_w, b0_w, b1_w, b2_w, baud, length, t_local
        )
        if v.u0.size == 0:
            lg.warning(f"probe {probe.label} (ch {probe.index}): no tuples, skipped")
            continue
        # Production selection, then production evaluation: exactly what
        # target_analytic_sums does, so the branch labels are the real ones.
        keep, _cert = select_tube(v, args.epsilon)
        if keep.size == 0:
            lg.warning(f"probe {probe.label}: tube empty at epsilon={args.epsilon:g}")
            continue
        # Stratify on the *predicted* branch over the whole kept population
        # (cheap), then run the real dispatcher on the drawn tuples only.
        pred = np.full(v.u0.size, -1, dtype=int)
        pred[keep] = predicted_branch(v, keep)
        sel = stratified_indices(v, pred, per_stratum=args.per_stratum, rng=rng)
        sel = sel[pred[sel] >= 0]
        if sel.size > args.max_tuples_per_probe:
            sel = np.sort(rng.choice(sel, size=args.max_tuples_per_probe, replace=False))
        t0 = time.perf_counter()
        values, branch = analytic_tuple_values(v, sel)
        t_fast_total += time.perf_counter() - t0

        branch_full = np.full(v.u0.size, -1, dtype=int)
        branch_full[sel] = branch
        value_full = np.full(v.u0.size, np.nan)
        value_full[sel] = values

        pop = tuple(int(np.sum(pred[keep] == b)) for b in (0, 1, 2))
        lg.info(
            f"probe {probe.label:>14} ch{probe.index:<5} f={probe.freq*1e-12:7.2f} THz "
            f"beta2={probe.beta2:+.3e}: {v.u0.size} tuples, {keep.size} kept "
            f"(sheet/far/fallback = {pop[0]}/{pop[1]}/{pop[2]}), {sel.size} sampled"
        )
        if probe_budget > 0:
            t0 = time.perf_counter()
            all_values, all_branch = analytic_tuple_values(v, keep)
            dt = time.perf_counter() - t0
            probe_budget -= 1
            real_pop = tuple(int(np.sum(all_branch == b)) for b in (0, 1, 2))
            lg.info(
                f"  full-population fast evaluation: {keep.size} tuples in {dt:.2f} s "
                f"({1e6*dt/max(keep.size,1):.1f} us/tuple), realized branches "
                f"sheet/far/fallback = {real_pop[0]}/{real_pop[1]}/{real_pop[2]}, "
                f"sum={all_values.sum():.6e}"
            )

        payloads = [
            (
                float(v.u0[i]),
                (float(v.nu_a[i]), float(v.nu_b[i]), float(v.nu_c[i])),
                (float(v.q_a[i]), float(v.q_b[i]), float(v.q_c[i]), float(v.q_t)),
                float(v.d[i]),
                int(args.qmc_points),
                int(args.qmc_replicates),
                int(args.seed) + int(i),
            )
            for i in sel
        ]
        t0 = time.perf_counter()
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(_qmc_job, payloads, chunksize=8))
        else:
            results = [_qmc_job(p) for p in payloads]
        lg.info(f"  QMC ground truth: {time.perf_counter()-t0:.1f} s for {sel.size} tuples")

        zeta = zero_line_ratio(v)
        widths_sum = np.sum(v.widths, axis=-1)
        bound = envelope_bound(v.u0, widths_sum, v.acceptance)
        probe_scale = float(np.nanmax(values)) if values.size else 1.0
        q_sum = np.abs(v.q_a) + np.abs(v.q_b) + np.abs(v.q_c) + abs(v.q_t)
        for i, (full, full_se, lin, lin_se) in zip(sel, results):
            rows.append(
                dict(
                    probe=probe.label,
                    band=band_of(probe.freq),
                    target=probe.index,
                    target_beta2=probe.beta2,
                    tuple_index=int(i),
                    fast=float(value_full[i]),
                    qmc_full=full,
                    qmc_full_stderr=full_se,
                    qmc_lin=lin,
                    qmc_lin_stderr=lin_se,
                    branch=int(branch_full[i]),
                    x_grad=float(v.x_grad[i]),
                    mu=float(v.mu[i]),
                    u0=float(v.u0[i]),
                    widths_sum=float(widths_sum[i]),
                    d=float(v.d[i]),
                    acceptance=float(v.acceptance[i]),
                    zeta=float(zeta[i]),
                    q_sum=float(q_sum[i]),
                    bound=float(bound[i]),
                    probe_scale=probe_scale,
                    nu_a=float(v.nu_a[i]),
                    nu_b=float(v.nu_b[i]),
                    nu_c=float(v.nu_c[i]),
                    q_a=float(v.q_a[i]),
                    q_b=float(v.q_b[i]),
                    q_c=float(v.q_c[i]),
                    q_t=float(v.q_t),
                    verified=0.0,
                )
            )
    lg.info(f"fast evaluation total: {t_fast_total:.2f} s")
    return {"rows": rows, "fast_seconds": t_fast_total}


def verify_fwm_rows(rows: list[dict], args) -> int:
    """Re-measure the apparent outliers against a much tighter reference.

    A per-tuple discrepancy is only worth reporting if it survives a
    reference refinement: the QMC ground truth carries a *bias*, not just
    scatter, when the Sobol set is too coarse for the tuple's oscillation,
    and replicate stderr does not see it.  Every row whose apparent error
    exceeds ``--verify-threshold`` is recomputed with 8x the points and 2x
    the replicates, and the tighter value replaces the loose one.
    """
    targets = [
        i
        for i, r in enumerate(rows)
        if r["qmc_full"] > 0.0
        and abs(r["fast"] / r["qmc_full"] - 1.0) > args.verify_threshold
    ]
    targets.sort(key=lambda i: -abs(rows[i]["fast"] / rows[i]["qmc_full"] - 1.0))
    targets = targets[: args.verify_max]
    if not targets:
        return 0
    lg.info(
        f"verification tier: recomputing {len(targets)} apparent outliers at "
        f"{8 * args.qmc_points} points x {2 * args.qmc_replicates} replicates"
    )
    payloads = [
        (
            rows[i]["u0"],
            (rows[i]["nu_a"], rows[i]["nu_b"], rows[i]["nu_c"]),
            (rows[i]["q_a"], rows[i]["q_b"], rows[i]["q_c"], rows[i]["q_t"]),
            rows[i]["d"],
            int(8 * args.qmc_points),
            int(2 * args.qmc_replicates),
            int(args.seed) + 977,
        )
        for i in targets
    ]
    t0 = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_qmc_job, payloads, chunksize=2))
    else:
        results = [_qmc_job(p) for p in payloads]
    survived = 0
    for i, (full, full_se, lin, lin_se) in zip(targets, results):
        before = rows[i]["fast"] / rows[i]["qmc_full"] - 1.0
        rows[i].update(
            qmc_full=full, qmc_full_stderr=full_se,
            qmc_lin=lin, qmc_lin_stderr=lin_se, verified=1.0,
        )
        after = rows[i]["fast"] / full - 1.0 if full > 0.0 else np.nan
        if abs(after) > args.verify_threshold:
            survived += 1
        rows[i]["error_before_verification"] = before
    lg.info(
        f"  {time.perf_counter()-t0:.1f} s; {survived}/{len(targets)} outliers "
        "survived the tighter reference (the rest were QMC bias, not model error)"
    )
    return survived


# ---------------------------------------------------------------------------
# XPM stage
# ---------------------------------------------------------------------------


def _xpm_qmc_points(nu: float, base: int, cap: int = 1 << 22) -> int:
    """Sobol budget that actually resolves ``cos(nu y)`` on ``|y| < 2 pi``.

    The pair integrand oscillates ``nu/pi`` times across the support, so a
    fixed budget silently biases the *reference* at large walk-off (at
    ``nu = 1e4`` a 2^14-point estimate overshoots the exact ``1/nu`` sheet
    limit by 33%, and 64 points per unit of ``nu`` is still 24% off at
    ``nu = 2912``).  Scale the budget with ``nu`` and keep it a power of two
    so the Sobol sequence stays balanced.
    """
    need = max(int(base), int(2048.0 * max(abs(nu), 1.0)))
    n = 1 << int(np.ceil(np.log2(max(need, 2))))
    return int(min(n, cap))


def converged_qmc_xpm(
    *, nu: float, q_b: float, q_t: float, base: int, reps: int, seed: int,
    include_quadratic: bool = True, tol: float = 0.005,
) -> tuple[float, float, bool]:
    """QMC pair reference with an explicit budget-doubling convergence test.

    Replicate scatter alone does NOT detect QMC *bias*: a Sobol set too
    coarse for ``cos(nu y)`` is consistently wrong with a small standard
    error.  (Measured: at ``nu = 2912`` a 2^18-point reference sits 24%
    above the 2^22-point one while quoting a 0.1% stderr.)  So evaluate at
    ``n`` and ``4n`` and report convergence only when they agree to ``tol``.
    """
    n_lo = _xpm_qmc_points(nu, base)
    lo, _ = qmc_xpm_ground_truth(
        nu=nu, q_b=q_b, q_t=q_t, n_points=n_lo, n_replicates=reps,
        seed=seed, include_quadratic=include_quadratic,
    )
    hi, hi_se = qmc_xpm_ground_truth(
        nu=nu, q_b=q_b, q_t=q_t, n_points=min(n_lo * 4, 1 << 23),
        n_replicates=reps, seed=seed + 1, include_quadratic=include_quadratic,
    )
    converged = abs(hi) > 0.0 and abs(lo - hi) / abs(hi) <= tol
    return hi, hi_se, converged


def run_xpm_sweep(args) -> dict:
    """Controlled sweep in the reduced pair variables (L/L_W, L/L_D).

    ``nu = L/L_W = 2 pi r beta2_norm L`` is the pair walk-off (Dar collision
    count) and ``L/L_D = beta2_norm L`` is the in-channel quadratic strength;
    the fast closed form is a function of ``nu`` alone, so the ``L/L_D``
    dependence of the ratio *is* the model's residual error.
    """
    nus = np.geomspace(args.nu_min, args.nu_max, args.n_nu)
    llds = np.array(args.lld, dtype=float)
    length = 1.0

    recs = []
    for lld in llds:
        for nu in nus:
            beta2 = lld / length
            r = nu / (2.0 * np.pi * beta2 * length)
            fast = float(xpm_fast_batch(np.array([nu]))[0])
            q_half = 0.5 * beta2 * length
            npts = _xpm_qmc_points(float(nu), args.qmc_points)
            g_full, g_full_se, conv_full = converged_qmc_xpm(
                nu=float(nu), q_b=q_half, q_t=q_half, base=args.qmc_points,
                reps=args.qmc_replicates, seed=args.seed, include_quadratic=True,
            )
            g_lin, g_lin_se, conv_lin = converged_qmc_xpm(
                nu=float(nu), q_b=q_half, q_t=q_half, base=args.qmc_points,
                reps=args.qmc_replicates, seed=args.seed, include_quadratic=False,
            )
            mc = estimate_xhkm_sectors_direct_mc(
                beta2=beta2, alpha=0.0, length=length,
                channel_spacing_over_baud=float(r),
                n_samples=args.sector_samples, seed=args.seed,
            )
            recs.append(
                dict(
                    nu=float(nu), lld=float(lld), r=float(r), fast=fast,
                    qmc_full=g_full, qmc_full_stderr=g_full_se,
                    qmc_lin=g_lin, qmc_lin_stderr=g_lin_se,
                    n1=float(mc.n1), n1_stderr=float(mc.n1_stderr),
                    n_2pc=float(mc.n_2pc), n_3pca=float(mc.n_3pca),
                    n_3pcb=float(mc.n_3pcb), n_4pc=float(mc.n_4pc),
                    n_2pc_stderr=float(mc.n_2pc_stderr),
                    n_4pc_stderr=float(mc.n_4pc_stderr),
                    qmc_points=float(npts),
                    converged=float(conv_full and conv_lin),
                )
            )
            lg.info(
                f"  L/LD={lld:8.3g} nu={nu:10.3g} r={r:9.3g}: fast={fast:.4e} "
                f"qmc={g_full:.4e} ({g_full/fast:.4f}) N1={mc.n1:.4e} ({mc.n1/fast:.4f}) "
                f"| 2PC={mc.n_2pc/mc.n1:.3f} 3PCa={mc.n_3pca/mc.n1:+.3f} "
                f"3PCb={mc.n_3pcb/mc.n1:+.3f} 4PC={mc.n_4pc/mc.n1:+.3f}"
            )
    return {"sweep": recs}


def run_xpm_physical(args, system, freqs, beta1, beta2, probes) -> dict:
    baud = float(system.pulse.baud_rate)
    length = float(system.fiber_length)
    rng = np.random.default_rng(args.seed + 1)
    recs = []
    for probe in probes:
        others, nu_pairs, q_pairs = xpm_pair_variables(
            beta1, beta2, baud, length, probe.index
        )
        fast = xpm_fast_batch(nu_pairs)
        q_t = 0.5 * float(beta2[probe.index]) * baud**2 * length
        # Stratify in log|nu| so the near-zero-walk-off pairs (ZDW-straddling
        # channels) are not swamped by the bulk.
        eligible = np.where(np.abs(nu_pairs) <= args.pair_nu_max)[0]
        if eligible.size == 0:
            lg.warning(f"probe {probe.label}: no pair with |nu| <= {args.pair_nu_max:g}")
            continue
        order = eligible[np.argsort(np.abs(nu_pairs[eligible]))]
        bins = np.linspace(0, order.size - 1, min(args.n_pairs, order.size)).astype(int)
        sel = np.unique(order[bins])
        for j in sel:
            nu = float(nu_pairs[j])
            g_full, g_se, conv = converged_qmc_xpm(
                nu=nu, q_b=float(q_pairs[j]), q_t=q_t, base=args.qmc_points,
                reps=args.qmc_replicates, seed=int(rng.integers(1 << 30)),
            )
            recs.append(
                dict(
                    probe=probe.label, band=band_of(probe.freq),
                    target=probe.index, interferer=int(others[j]),
                    nu=nu, q_b=float(q_pairs[j]), q_t=q_t,
                    fast=float(fast[j]), qmc_full=g_full, qmc_full_stderr=g_se,
                    converged=float(conv),
                )
            )
        lg.info(f"probe {probe.label:>14}: {sel.size} XPM pairs, "
                f"|nu| in [{np.abs(nu_pairs).min():.3g}, {np.abs(nu_pairs).max():.3g}]")
    return {"pairs": recs}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _rel_err(fast: np.ndarray, truth: np.ndarray, floor: float) -> np.ndarray:
    denom = np.where(np.abs(truth) > floor, truth, np.nan)
    return (fast - denom) / denom


def plot_fwm(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    get = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    fast, full, lin = get("fast"), get("qmc_full"), get("qmc_lin")
    branch = get("branch").astype(int)
    se_full = get("qmc_full_stderr")
    se_lin = get("qmc_lin_stderr")
    resolvable = full > 5.0 * se_full
    err_total = np.where(resolvable, (fast - full) / full, np.nan)
    err_model = np.where(lin > 5.0 * se_lin, (fast - lin) / lin, np.nan)
    err_quad = np.where(resolvable & (lin > 5.0 * se_lin), (lin - full) / full, np.nan)

    axes_spec = (
        ("x_grad", r"loudness $x_\nabla = LB\|\nabla\Delta\beta\|$", True),
        ("mu_abs", r"detuning $|\mu| = |u_0|/x_\nabla$", True),
        ("gap", r"gap $(|u_0| - W)/W$", False),
        ("d_abs", r"support offset $|d|$", False),
        ("zeta", r"zero-line proximity $\zeta=\min|\nu_j|/\max|\nu_j|$", True),
        ("q_sum", r"quadratic budget $\sum_j |q_j|$", True),
    )
    derived = {
        "x_grad": get("x_grad"),
        "mu_abs": np.abs(get("mu")),
        "gap": (np.abs(get("u0")) - get("widths_sum")) / np.maximum(get("widths_sum"), 1e-30),
        "d_abs": np.abs(get("d")),
        "zeta": np.maximum(get("zeta"), 1e-7),
        "q_sum": np.maximum(get("q_sum"), 1e-12),
    }

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (key, label, logx) in zip(axs.ravel(), axes_spec):
        for b in (0, 1, 2):
            m = branch == b
            if not np.any(m):
                continue
            ax.scatter(derived[key][m], 100.0 * err_total[m], s=7, alpha=0.45,
                       color=BRANCH_COLORS[b], label=BRANCH_NAMES[b])
        ax.axhline(0.0, color="k", lw=0.7)
        for lvl in (1.0, -1.0):
            ax.axhline(lvl, color="0.6", lw=0.6, ls="--")
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel(label)
        ax.set_ylabel(r"fast / QMC(full) $-$ 1  [\%]")
        ax.grid(alpha=0.25)
    axs[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Fast strict-FWM per-tuple error vs reduced variables "
                 "(dashed: $\\pm1\\%$)")
    fig.savefig(out_dir / "campaign_fwm_reduced.png", dpi=160)
    plt.close(fig)

    # Error decomposition: model error vs quadratic error.
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, (e, title) in zip(
        axs,
        (
            (err_model, "model error: fast vs QMC(linear)"),
            (err_quad, "quadratic error: QMC(linear) vs QMC(full)"),
            (err_total, "total: fast vs QMC(full)"),
        ),
    ):
        for b in (0, 1, 2):
            m = branch == b
            if not np.any(m):
                continue
            ax.scatter(derived["x_grad"][m], 100.0 * e[m], s=7, alpha=0.45,
                       color=BRANCH_COLORS[b], label=BRANCH_NAMES[b])
        ax.axhline(0.0, color="k", lw=0.7)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel(r"$x_\nabla$")
        ax.set_ylabel(r"relative error [\%]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=8)
    fig.savefig(out_dir / "campaign_fwm_error_split.png", dpi=160)
    plt.close(fig)

    # Phase-diagram map: where in (x_grad, |mu|) the error lives.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ok = np.isfinite(err_total)
    sc = axs[0].scatter(
        derived["x_grad"][ok], np.maximum(derived["mu_abs"][ok], 1e-4),
        c=np.clip(100.0 * np.abs(err_total[ok]), 1e-2, 1e2),
        norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1e2),
        s=11, cmap="magma_r",
    )
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$x_\nabla$"); axs[0].set_ylabel(r"$|\mu|$")
    axs[0].set_title(r"$|$fast/QMC $-$ 1$|$ [\%] over the phase diagram", fontsize=10)
    fig.colorbar(sc, ax=axs[0], label=r"[\%]")

    sc = axs[1].scatter(
        derived["zeta"][ok], np.maximum(derived["mu_abs"][ok], 1e-4),
        c=np.clip(100.0 * np.abs(err_total[ok]), 1e-2, 1e2),
        norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1e2),
        s=11, cmap="magma_r",
    )
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel(r"$\zeta$ (zero-line proximity)"); axs[1].set_ylabel(r"$|\mu|$")
    axs[1].set_title("same, against the degenerate directions", fontsize=10)
    fig.colorbar(sc, ax=axs[1], label=r"[\%]")
    fig.savefig(out_dir / "campaign_fwm_phase_map.png", dpi=160)
    plt.close(fig)


def plot_xpm(sweep: list[dict], pairs: list[dict], out_dir: Path) -> None:
    if sweep:
        g = lambda k: np.array([r[k] for r in sweep], dtype=float)  # noqa: E731
        nu, lld = g("nu"), g("lld")
        fast, qfull, qlin, n1 = g("fast"), g("qmc_full"), g("qmc_lin"), g("n1")
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
        for value in np.unique(lld):
            m = lld == value
            o = np.argsort(nu[m])
            axs[0].plot(nu[m][o], 100.0 * (fast[m][o] / qfull[m][o] - 1.0),
                        marker="o", ms=3, lw=1, label=f"$L/L_D$={value:g}")
            axs[1].plot(nu[m][o], 100.0 * (fast[m][o] / qlin[m][o] - 1.0),
                        marker="o", ms=3, lw=1)
            axs[2].plot(nu[m][o], 100.0 * (n1[m][o] / qfull[m][o] - 1.0),
                        marker="o", ms=3, lw=1)
        for ax, title in zip(
            axs,
            ("fast vs QMC(full)", "fast vs QMC(linear)  [model error]",
             "sector-MC $N_1$ vs QMC(full)  [MC cross-check]"),
        ):
            ax.axhline(0.0, color="k", lw=0.7)
            ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.1)
            ax.set_xlabel(r"$\nu = L/L_W$"); ax.set_ylabel(r"relative error [\%]")
            ax.set_title(title, fontsize=10); ax.grid(alpha=0.25)
        axs[0].legend(fontsize=8)
        fig.suptitle("XPM pair efficiency: fast closed form vs Monte-Carlo")
        fig.savefig(out_dir / "campaign_xpm_sweep.png", dpi=160)
        plt.close(fig)

        # n-PC composition.
        fig, axs = plt.subplots(1, len(np.unique(lld)), figsize=(5 * len(np.unique(lld)), 4.2),
                                constrained_layout=True, squeeze=False)
        for ax, value in zip(axs[0], np.unique(lld)):
            m = lld == value
            o = np.argsort(nu[m])
            for key, lab, col in (
                ("n_2pc", "2PC", "#1b9e77"),
                ("n_3pca", "3PCa", "#d95f02"),
                ("n_3pcb", "3PCb", "#7570b3"),
                ("n_4pc", "4PC", "#e7298a"),
            ):
                frac = g(key)[m][o] / n1[m][o]
                ax.plot(nu[m][o], frac, marker="o", ms=3, lw=1, color=col, label=lab)
            ax.axhline(0.0, color="k", lw=0.7)
            ax.axhline(1.0, color="0.6", lw=0.6, ls="--")
            ax.set_xscale("log")
            ax.set_xlabel(r"$\nu = L/L_W$"); ax.set_ylabel(r"sector / $N_1$")
            ax.set_title(f"$L/L_D$ = {value:g}", fontsize=10); ax.grid(alpha=0.25)
        axs[0][0].legend(fontsize=8)
        fig.suptitle("n-PC collision-sector composition of the pair efficiency "
                     "the fast method returns as a single aggregate")
        fig.savefig(out_dir / "campaign_xpm_sectors.png", dpi=160)
        plt.close(fig)

    if pairs:
        pairs = [r for r in pairs if r.get("converged", 1.0) > 0.5]
        if not pairs:
            lg.warning("no XPM pair reference converged; skipping the pair plot")
            return
        g = lambda k: np.array([r[k] for r in pairs], dtype=float)  # noqa: E731
        nu, fast, qfull = np.abs(g("nu")), g("fast"), g("qmc_full")
        bands = [r["band"] for r in pairs]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        for name, _, _ in BANDS:
            m = np.array([b == name for b in bands])
            if not np.any(m):
                continue
            axs[0].scatter(nu[m], 100.0 * (fast[m] / qfull[m] - 1.0), s=10, alpha=0.6, label=name)
            axs[1].scatter(np.abs(g("q_t"))[m] + np.abs(g("q_b"))[m],
                           100.0 * (fast[m] / qfull[m] - 1.0), s=10, alpha=0.6, label=name)
        for ax, xlabel in zip(axs, (r"$|\nu|$ (pair walk-off)", r"$|q_b| + |q_t|$")):
            ax.axhline(0.0, color="k", lw=0.7)
            ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.1)
            ax.set_xlabel(xlabel); ax.set_ylabel(r"fast / QMC $-$ 1 [\%]")
            ax.grid(alpha=0.25)
        axs[0].legend(fontsize=8, ncol=3)
        fig.suptitle("XPM pairs of the physical comb")
        fig.savefig(out_dir / "campaign_xpm_physical.png", dpi=160)
        plt.close(fig)


# ---------------------------------------------------------------------------


def summarize_fwm(rows: list[dict]) -> None:
    """Report the error split, separating real error from unresolvable MC noise.

    A tuple is *MC-resolvable* only if the QMC ground truth is itself several
    standard errors away from zero; deep in the kernel tail the truth is
    smaller than the sampling noise of any affordable MC, and a ratio there
    measures the reference, not the model.  Discrepancies are therefore
    scored both as a relative error and as ``z = (fast - qmc)/sigma_qmc``.
    """
    if not rows:
        return
    g = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    fast, full, se = g("fast"), g("qmc_full"), g("qmc_full_stderr")
    branch = g("branch").astype(int)
    scale = g("probe_scale")
    resolvable = full > np.maximum(5.0 * se, 0.0)
    err = np.where(resolvable, (fast - full) / np.where(full != 0.0, full, np.nan), np.nan)
    z = np.where(se > 0.0, (fast - full) / np.where(se > 0.0, se, np.nan), np.nan)
    rel_scale = (fast - full) / np.maximum(scale, 1e-300)

    lg.info("=" * 78)
    lg.info(
        f"FWM per-tuple: {len(rows)} tuples; "
        f"{int(np.sum(resolvable))} MC-resolvable (qmc > 5 sigma), "
        f"{int(np.sum(~resolvable))} below MC noise floor (not validatable)"
    )
    for b in (0, 1, 2):
        m = resolvable & (branch == b)
        if not np.any(m):
            continue
        a = np.abs(err[m])
        sig = np.abs(z[m]) > 3.0
        lg.info(
            f"  branch {BRANCH_NAMES[b]:>8}: n={int(m.sum()):5d} "
            f"median={100*np.median(a):7.3f}%  p90={100*np.percentile(a,90):8.3f}%  "
            f"p99={100*np.percentile(a,99):9.3f}%  max={100*np.max(a):10.3f}%  "
            f"|z|>3: {100*np.mean(sig):5.1f}%"
        )
    w = np.where(resolvable, full, 0.0)
    if np.sum(w) > 0:
        lg.info(
            f"  mass-weighted signed error (resolvable): "
            f"{100*np.sum(w*np.nan_to_num(err))/np.sum(w):+.4f}%"
        )
    lg.info(
        f"  worst error relative to the probe's largest tuple: "
        f"{100*np.nanmax(np.abs(rel_scale)):.4f}% "
        "(this is what a channel sum actually inherits)"
    )
    # Certified envelope: the analytic bound must dominate the truth.
    bnd = g("bound")
    viol = resolvable & (full > bnd * (1.0 + 1e-6) + 5.0 * se)
    if np.any(viol):
        lg.warning(f"  certified envelope violated on {int(viol.sum())} tuples")
    else:
        lg.info("  certified envelope holds on every resolvable tuple")

    bad = resolvable & (np.abs(err) > 0.05) & (np.abs(z) > 3.0)
    if np.any(bad):
        idx = np.where(bad)[0][np.argsort(-np.abs(err[np.where(bad)[0]]))][:20]
        lg.info(f"  statistically significant offenders (>5% and >3 sigma): {int(bad.sum())}")
        for i in idx:
            r = rows[int(i)]
            lg.info(
                f"    {r['probe']:>14} branch={BRANCH_NAMES[r['branch']]:>8} "
                f"err={100*err[i]:+9.2f}% z={z[i]:+7.1f}  x={r['x_grad']:.3e} "
                f"mu={r['mu']:+.3e} zeta={r['zeta']:.2e} d={r['d']:+.3f} "
                f"W={r['widths_sum']:.3e} q={r['q_sum']:.3e} "
                f"fast={r['fast']:.3e} qmc={r['qmc_full']:.3e}+-{r['qmc_full_stderr']:.1e} "
                f"(rel. to probe max: {100*rel_scale[i]:+.3e}%)"
            )
    else:
        lg.info("  no statistically significant >5% offenders")

    # Per-probe / per-band roll-up.
    lg.info("  per-probe roll-up (resolvable tuples only):")
    for probe in dict.fromkeys(r["probe"] for r in rows):
        m = resolvable & np.array([r["probe"] == probe for r in rows])
        if not np.any(m):
            lg.info(f"    {probe:>14}: no resolvable tuples")
            continue
        a = np.abs(err[m])
        lg.info(
            f"    {probe:>14}: n={int(m.sum()):4d} median={100*np.median(a):7.3f}% "
            f"p95={100*np.percentile(a,95):8.3f}% max={100*np.max(a):9.3f}%"
        )


def main() -> None:
    init_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    p.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    p.add_argument("--stage", choices=("fwm", "xpm", "all"), default="all")
    p.add_argument("--decimation", type=int, default=8)
    p.add_argument("--per-band", type=int, default=2)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--per-stratum", type=int, default=2)
    p.add_argument("--max-tuples-per-probe", type=int, default=300)
    p.add_argument(
        "--window", type=int, default=0,
        help="enumerate each probe on a contiguous W-channel window instead "
             "of the full comb (0 = full comb); lets decimation 1 be used",
    )
    p.add_argument("--qmc-points", type=int, default=1 << 16)
    p.add_argument("--qmc-replicates", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--verify-threshold", type=float, default=0.02,
        help="apparent per-tuple errors above this are recomputed against a "
             "much tighter QMC reference before being reported",
    )
    p.add_argument("--verify-max", type=int, default=400)
    p.add_argument(
        "--time-full-probes", type=int, default=2,
        help="on this many probes, also evaluate the whole kept population "
             "to time the fast path and report its realized branch mix",
    )
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--nu-min", type=float, default=1e-2)
    p.add_argument("--nu-max", type=float, default=1e4)
    p.add_argument("--n-nu", type=int, default=24)
    p.add_argument("--lld", type=float, nargs="+", default=[1e-3, 1e-1, 1.0, 10.0])
    p.add_argument("--sector-samples", type=int, default=1 << 21)
    p.add_argument("--n-pairs", type=int, default=24)
    p.add_argument(
        "--pair-nu-max", type=float, default=2000.0,
        help="restrict physical XPM pairs to |nu| below this; above it the "
             "reference itself needs >2^23 Sobol points, and the sweep "
             "already establishes sub-percent accuracy there",
    )
    p.add_argument(
        "--xpm-decimation", type=int, default=1,
        help="grid decimation for the physical XPM pairs; 1 keeps the true "
             "25 GHz pitch so nearest-neighbour pairs (small nu, large "
             "in-channel q) are represented",
    )
    args = p.parse_args()

    assert_no_mc_fallback()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    system = System.from_toml(args.config)
    _, freqs = decimated_frequency_grid(system, args.decimation)
    b1g, b2g = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(b1g[0], dtype=float)
    beta2 = np.asarray(b2g[0], dtype=float)
    beta0_abs = _beta0_abs_from_fiber(system, freqs, beta1)
    probes = choose_probes(freqs, beta2, per_band=args.per_band)
    lg.info(
        f"grid: {freqs.size} channels ({freqs[0]*1e-12:.1f}-{freqs[-1]*1e-12:.1f} THz), "
        f"{len(probes)} probes; ZDW near {nu2lambda(freqs[int(np.argmin(np.abs(beta2)))])*1e9:.1f} nm"
    )

    payload: dict[str, object] = {}
    if args.stage in ("fwm", "all"):
        res = run_fwm_stage(args, system, freqs, beta0_abs, beta1, beta2, probes)
        for r in res["rows"]:
            r.setdefault("error_before_verification", np.nan)
        verify_fwm_rows(res["rows"], args)
        payload["fwm_rows"] = res["rows"]
        summarize_fwm(res["rows"])
        plot_fwm(res["rows"], args.out_dir)
    if args.stage in ("xpm", "all"):
        sweep = run_xpm_sweep(args)["sweep"]
        if args.xpm_decimation == args.decimation:
            xfreqs, xb1, xb2, xprobes = freqs, beta1, beta2, probes
        else:
            _, xfreqs = decimated_frequency_grid(system, args.xpm_decimation)
            xb1g, xb2g = system.beta_grids(freqs=xfreqs)
            xb1 = np.asarray(xb1g[0], dtype=float)
            xb2 = np.asarray(xb2g[0], dtype=float)
            xprobes = choose_probes(xfreqs, xb2, per_band=args.per_band)
            lg.info(
                f"physical XPM pairs on the decimation-{args.xpm_decimation} grid: "
                f"{xfreqs.size} channels, {len(xprobes)} probes"
            )
        pairs = run_xpm_physical(args, system, xfreqs, xb1, xb2, xprobes)["pairs"]
        payload["xpm_sweep"] = sweep
        payload["xpm_pairs"] = pairs
        plot_xpm(sweep, pairs, args.out_dir)

    np.savez(
        args.out_dir / "campaign_fast_vs_mc.npz",
        **{
            k: np.array([tuple(r.values()) for r in v], dtype=object)
            if isinstance(v, list) and v
            else np.array([])
            for k, v in payload.items()
        },
        **{
            f"{k}_fields": np.array(list(v[0].keys()))
            for k, v in payload.items()
            if isinstance(v, list) and v
        },
    )
    lg.info(f"wrote figures and campaign_fast_vs_mc.npz to {args.out_dir}")


if __name__ == "__main__":
    main()
