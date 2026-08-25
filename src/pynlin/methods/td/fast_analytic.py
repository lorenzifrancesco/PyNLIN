"""Analytical production path for per-tuple FWM efficiency (S3, v1).

Implements the design of docs/source/lorenzi_fast_method.md §15:

* :func:`envelope_bound` -- the certified per-tuple upper bound
  ``A(d) * min(1, 4/g^2)``, ``g = |u0| - W`` (§9 theorem).
* :func:`select_tube` -- the epsilon-tube around the phase-matched
  stationary lines, realized as the certified selection ``bound >= eps``
  with EXACT accumulation of the discarded certificate.
* :func:`analytic_tuple_values` -- branch-dispatched per-tuple efficiency:
    - sheet branch (closed form): F = 2*pi * rho_w(-u0) * A_cond(u=0),
      valid when the mismatch density is flat over the kernel core;
      the CONDITIONAL acceptance is mandatory (the marginal A(d) is off by
      3/2 at d=0 for the equal split, where u and the mask variable are
      perfectly correlated).
    - gapped branch (closed form): the far model of fast_nlin with the
      plain acceptance (mask-kernel correlation second order there).
    - bridge/narrow fallback (v1): the existing regime quadrature with the
      exact conditional acceptance. Correctness-first; a fitted bridge
      C(x, mu_nat) is the documented later optimization.
* :func:`target_analytic_sums` -- per-target FWM+XPM sums through the tube
  and the analytic branches, with the truncation certificate.

The reference implementation (``fast_nlin.target_fast_sums``) remains the
gate for this path; see analysis/fwm/fast_s3_tube.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fast_nlin import (
    FWMTupleVariables,
    exact_conditional_acceptance,
    far_model,
    fwm_tuple_variables,
    linear_tuple_estimate,
    refine_tuples_exact,
    support_acceptance,
    uniform_sum_density,
    xpm_fast_batch,
    xpm_pair_variables,
    FAR_MARGIN_FACTOR,
    FAR_MARGIN_OFFSET,
)

# Sheet-branch validity: density must be flat over the kernel's effective
# width. Requires a wide support and a phase-matched point well inside it.
SHEET_MIN_WIDTH = 2000.0  # rad; W above this puts >97% of kernel mass in a
#                           region where the 3-uniform density varies slowly
SHEET_CORE_MARGIN = 200.0  # rad; distance of u=0 from the support edge
# Below this conditional acceptance at u=0 the sheet value is kernel-tail
# dominated and the local formula's relative error is unbounded; demote.
SHEET_MIN_ACCEPTANCE = 0.05


@dataclass(frozen=True)
class AnalyticTargetResult:
    xpm: float
    fwm: float
    fwm_tuples_total: int
    fwm_tuples_kept: int
    certificate: float  # certified upper bound on the discarded FWM sum
    branch_counts: tuple[int, int, int]  # sheet, far, fallback
    branch_mass: tuple[float, float, float]


def envelope_bound(
    u0: np.ndarray, widths_sum: np.ndarray, acceptance: np.ndarray
) -> np.ndarray:
    """Certified upper bound on F per tuple (§9): A * min(1, 4/g^2)."""
    g = np.abs(u0) - widths_sum
    with np.errstate(divide="ignore"):
        decay = np.where(g > 0.0, 4.0 / np.maximum(g, 1e-300) ** 2, np.inf)
    return acceptance * np.minimum(1.0, decay)


def select_tube(
    variables: FWMTupleVariables, epsilon: float
) -> tuple[np.ndarray, float]:
    """Keep tuples whose certified bound >= epsilon.

    Returns (survivor indices, certificate = exact sum of the discarded
    tuples' bounds -- a rigorous upper bound on the discarded true sum).
    ``epsilon <= 0`` keeps everything with a zero certificate.
    """
    W = np.sum(variables.widths, axis=-1)
    # Quadratic padding P_q (theory doc §9): the confinement argument uses
    # the linear model, so the certificate valid for the full quadratic
    # model must shrink the gap by the maximum quadratic phase shift
    # pi^2 * sum|q_j| (each x_j^2 <= pi^2; x_d^2 <= pi^2 under the mask).
    p_q = np.pi**2 * (
        np.abs(variables.q_a) + np.abs(variables.q_b)
        + np.abs(variables.q_c) + abs(variables.q_t)
    )
    bound = envelope_bound(variables.u0, W + p_q, variables.acceptance)
    if epsilon <= 0.0:
        return np.arange(variables.u0.size), 0.0
    keep = bound >= epsilon
    certificate = float(np.sum(bound[~keep]))
    return np.where(keep)[0], certificate


def analytic_tuple_values(
    variables: FWMTupleVariables, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Branch-dispatched per-tuple efficiency for the selected tuples.

    Returns (values, branch) with branch 0 = sheet closed form, 1 = far
    closed form, 2 = quadrature fallback (bridge/narrow).
    """
    idx = np.asarray(indices, dtype=int)
    u0 = variables.u0[idx]
    d = variables.d[idx]
    coeffs = np.stack(
        [variables.nu_a[idx], variables.nu_b[idx], -variables.nu_c[idx]], axis=-1
    )
    widths = np.pi * np.abs(coeffs)
    W = np.sum(widths, axis=-1)
    abs_u0 = np.abs(u0)

    values = np.zeros(idx.size, dtype=float)
    branch = np.full(idx.size, 2, dtype=int)

    far = abs_u0 > FAR_MARGIN_FACTOR * W + FAR_MARGIN_OFFSET
    sheet = (~far) & (W > SHEET_MIN_WIDTH) & (abs_u0 < W - SHEET_CORE_MARGIN)
    fallback = ~(far | sheet)

    if np.any(far):
        values[far] = far_model(u0[far], widths[far]) * support_acceptance(d[far])
        branch[far] = 1

    if np.any(sheet):
        sel = np.where(sheet)[0]
        # F = 2*pi * rho_w(-u0) * A_cond(u=0); the conditional acceptance is
        # evaluated at offset -u0 (u = u0 + offset = 0).
        rho0 = np.empty(sel.size)
        acc0 = np.empty(sel.size)
        chunk = 65536
        for s0 in range(0, sel.size, chunk):
            ss = sel[s0 : s0 + chunk]
            rho0[s0 : s0 + ss.size] = uniform_sum_density(
                -u0[ss][:, None], widths[ss][:, None, :]
            )[:, 0]
            acc0[s0 : s0 + ss.size] = exact_conditional_acceptance(
                -u0[ss][:, None], coeffs[ss], d[ss]
            )[:, 0]
        # When the mask excludes the phase-matched point (A_cond(0) ~ 0) the
        # local evaluation returns ~0 while the true value is kernel-tail
        # dominated: absolute mass is negligible but relative error is
        # unbounded. Demote those tuples to the quadrature fallback.
        good = acc0 >= SHEET_MIN_ACCEPTANCE
        values[sel[good]] = 2.0 * np.pi * rho0[good] * acc0[good]
        branch[sel[good]] = 0
        fallback = fallback.copy()
        fallback[sel[~good]] = True

    if np.any(fallback):
        # v1 bridge: the exact-acceptance regime quadrature, via the
        # refinement tier's evaluator (bucketed node counts + memory-bounded
        # chunking -- an unchunked (tuples, nodes) batch on a near-ZDW
        # target would be a multi-GB allocation).
        sel = np.where(fallback)[0]
        est = linear_tuple_estimate(u0[sel], coeffs[sel], d[sel])
        sel_global = idx[sel]
        regime_full = np.zeros(variables.u0.size, dtype=int)
        regime_full[sel_global] = est.regime
        values[sel] = refine_tuples_exact(variables, sel_global, regime_full)

    return values, branch


def target_analytic_sums(
    freqs: np.ndarray,
    beta0_abs: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    baud_rate: float,
    length: float,
    target: int,
    *,
    epsilon: float = 1e-9,
) -> AnalyticTargetResult:
    """Per-target prefactor-free sums via tube selection + analytic branches.

    Same conventions as ``fast_nlin.target_fast_sums`` (multiply by L^2 for
    the [m^2] sums). ``certificate`` bounds the FWM sum discarded by the
    epsilon-tube; ``epsilon <= 0`` reproduces the exhaustive tuple set.
    """
    variables = fwm_tuple_variables(
        freqs, beta0_abs, beta1, beta2, baud_rate, length, target
    )
    n_total = int(variables.u0.size)
    if n_total == 0:
        _, nu_pairs, _ = xpm_pair_variables(beta1, beta2, baud_rate, length, target)
        return AnalyticTargetResult(
            xpm=float(np.sum(xpm_fast_batch(nu_pairs))), fwm=0.0,
            fwm_tuples_total=0, fwm_tuples_kept=0, certificate=0.0,
            branch_counts=(0, 0, 0), branch_mass=(0.0, 0.0, 0.0),
        )
    keep, certificate = select_tube(variables, epsilon)
    values, branch = analytic_tuple_values(variables, keep)
    fwm_total = float(np.sum(values))
    counts = tuple(int(np.sum(branch == b)) for b in (0, 1, 2))
    mass = tuple(float(np.sum(values[branch == b])) for b in (0, 1, 2))

    _, nu_pairs, _ = xpm_pair_variables(beta1, beta2, baud_rate, length, target)
    xpm_total = float(np.sum(xpm_fast_batch(nu_pairs)))
    return AnalyticTargetResult(
        xpm=xpm_total,
        fwm=fwm_total,
        fwm_tuples_total=n_total,
        fwm_tuples_kept=int(keep.size),
        certificate=certificate,
        branch_counts=counts,  # type: ignore[arg-type]
        branch_mass=mass,  # type: ignore[arg-type]
    )


__all__ = [
    "AnalyticTargetResult",
    "analytic_tuple_values",
    "envelope_bound",
    "select_tube",
    "target_analytic_sums",
]
