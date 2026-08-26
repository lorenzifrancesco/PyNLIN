"""GN-model analog evaluators on the normalized-variable, lossless kernel.

Implements the two GN reference models of the theory doc
(``docs/source/lorenzi_fast_method.md`` section 5.3) mapped into the
pipeline's normalized per-tuple units:

* **GN-CFM analog** (Gan et al., arXiv:2510.11867): center-family triplets
  only, NLI PSD evaluated at the COI center (the ``x_d = 0`` section, so
  ``x_c = x_a + x_b + d`` and the linear phase has two effective legs
  ``nu_a - nu_c``, ``nu_b - nu_c``), circumscribed-rectangle domain (the
  per-triplet window Pi dropped), linear phase.  Evaluated with the
  closed-form/CF machinery of :mod:`pynlin.methods.td.fast_nlin`.
* **GN-NI analog** (arXiv:2401.18022): all families, exact per-island 2-D
  midpoint integral at ``x_d = 0`` with the full quadratic in-channel phase
  -- the brute-force numerical-integral model's mathematics.

These are **assumption-stack ablations on the lossless single-span kernel**,
not reproductions of the published models (which carry loss, ISRS fits, and
multi-span coherence).  Ratios against ``target_fast_sums`` quantify what
the GN approximation class (locally-white receiver, rectangle domain,
family collapse, linearization) costs in this setting -- they must not be
quoted as the published models' errors.

Conventions match :func:`pynlin.methods.td.fast_nlin.target_fast_sums`:
values are dimensionless per-tuple efficiencies. Multiply an efficiency sum
by ``fiber_length_m**2`` to obtain the coefficient sum in m^2.
"""

from __future__ import annotations

import operator

import numpy as np
from scipy.special import sici

from pynlin.methods.td.fast_nlin import (
    CF_BASE_NODES,
    CF_MAX_NODES,
    FAR_MARGIN_FACTOR,
    FAR_MARGIN_OFFSET,
    NEAR_NODES_PER_RADIAN,
    WIDE_HALFWIDTH,
    FWMTupleVariables,
    cf_gauss_legendre,
    far_model,
    fwm_tuple_variables,
    kernel_abs2,
    uniform_sum_density,
    wide_model,
)


def _positive_int(value: int, name: str) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _rect_mean_near(u0: np.ndarray, widths: np.ndarray, n_nodes: int) -> np.ndarray:
    """Unmasked rectangle mean of the kernel: u-space quadrature of
    Khat(u0 + s) against the uniform-sum density of the effective legs
    (nonnegative integrand, so the node count only resolves the kernel
    oscillation)."""
    total = np.maximum(np.sum(widths, axis=-1), 1e-9)
    nodes, wts = cf_gauss_legendre(n_nodes)
    offset = total[:, None] * (2.0 * nodes[None, :] - 1.0)
    rho = uniform_sum_density(offset, widths[:, None, :])
    integrand = kernel_abs2(u0[:, None] + offset) * rho
    return 2.0 * total * np.sum(wts[None, :] * integrand, axis=-1)


def _gn_cfm_active_values(u0: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Evaluate tuples whose effective rectangle has at least one active leg."""
    total = np.sum(widths, axis=-1)
    values = np.empty(u0.size, dtype=float)

    far = np.abs(u0) > FAR_MARGIN_FACTOR * total + FAR_MARGIN_OFFSET
    wide = (~far) & (total > WIDE_HALFWIDTH)
    near = ~(far | wide)
    if np.any(far):
        values[far] = far_model(u0[far], widths[far])
    if np.any(wide):
        central, tail = wide_model(u0[wide], widths[wide])
        values[wide] = central + tail
    near_idx = np.where(near)[0]
    if near_idx.size:
        n_nodes = np.minimum(
            CF_BASE_NODES + (NEAR_NODES_PER_RADIAN * total[near_idx]).astype(int),
            CF_MAX_NODES,
        )
        buckets = np.ceil(
            np.log2(np.maximum(n_nodes / CF_BASE_NODES, 1.0))
        ).astype(int)
        for bucket in np.unique(buckets):
            bsel = near_idx[buckets == bucket]
            nn = int(CF_BASE_NODES * 2**bucket)
            chunk = max(4_000_000 // max(nn, 1), 16)
            for start in range(0, bsel.size, chunk):
                sub = bsel[start : start + chunk]
                values[sub] = _rect_mean_near(u0[sub], widths[sub], nn)
    return values


def _one_leg_rectangle_mean(u0: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Exact mean of the link kernel over one uniform phase interval."""
    lo = u0 - width
    hi = u0 + width

    def antiderivative(u: np.ndarray) -> np.ndarray:
        si = sici(u)[0]
        ratio = np.empty_like(u)
        small = np.abs(u) < 1e-4
        ratio[small] = -0.5 * u[small] + u[small] ** 3 / 24.0
        ratio[~small] = (np.cos(u[~small]) - 1.0) / u[~small]
        return 2.0 * (ratio + si)

    return (antiderivative(hi) - antiderivative(lo)) / (2.0 * width)


def gn_cfm_tuple_values(
    variables: FWMTupleVariables, indices: np.ndarray | None = None
) -> np.ndarray:
    """Per-tuple GN-CFM analog value (locally-white receiver, rectangle).

    On the ``x_d = 0`` section, ``x_c = x_a + x_b + d`` and the linear phase
    collapses to ``u = (u0 - nu_c d) + (nu_a - nu_c) x_a + (nu_b - nu_c) x_b``.
    Dropping the Pi window makes the value the plain rectangle mean of the
    kernel -- a two-leg unmasked linear model, dispatched over the same
    far/wide/CF tiers the pipeline uses (its "closed form" tier).  No family
    selection is applied here; see :func:`target_gn_cfm_sums`.
    """
    idx = (
        np.arange(variables.u0.size) if indices is None
        else np.asarray(indices, dtype=int)
    )
    u0 = variables.u0[idx] - variables.nu_c[idx] * variables.d[idx]
    widths = np.pi * np.stack(
        [
            np.abs(variables.nu_a[idx] - variables.nu_c[idx]),
            np.abs(variables.nu_b[idx] - variables.nu_c[idx]),
        ],
        axis=-1,
    )
    values = np.empty(idx.size, dtype=float)
    active = widths > 1e-10
    active_count = np.sum(active, axis=-1)
    zero = active_count == 0
    values[zero] = kernel_abs2(u0[zero])
    one = np.where(active_count == 1)[0]
    if one.size:
        active_widths = widths[one][active[one]]
        values[one] = _one_leg_rectangle_mean(u0[one], active_widths)
    two = np.where(active_count == 2)[0]
    if two.size:
        values[two] = _gn_cfm_active_values(u0[two], widths[two])
    return values


def gn_ni_tuple_values(
    variables: FWMTupleVariables,
    indices: np.ndarray | None = None,
    *,
    n_grid: int = 32,
    n_xd: int = 0,
) -> np.ndarray:
    """Per-tuple GN-NI analog: midpoint n x n mean of the masked kernel.

    Per tuple: mean over ``(x_a, x_b)`` in ``(-pi, pi)^2`` of
    ``1(|x_c| < pi) Khat(u)``, with ``x_c = x_a + x_b + d - x_d`` and the
    full quadratic phase.  ``n_xd = 0`` is the GN locally-white receiver
    (the single ``x_d = 0`` section, killing the ``q_t`` term);
    ``n_xd > 0`` averages over that many midpoint receiver sections -- the
    closure test, which undoes the locally-white assumption and converges to
    the masked 3-D local-quadratic integral up to quadrature error.
    Memory-bounded by chunking over tuples.
    """
    n_grid = _positive_int(n_grid, "n_grid")
    n_xd = _nonnegative_int(n_xd, "n_xd")
    idx = (
        np.arange(variables.u0.size) if indices is None
        else np.asarray(indices, dtype=int)
    )
    out = np.empty(idx.size, dtype=float)
    chunk = max(4_000_000 // (n_grid * n_grid), 16)
    for start in range(0, idx.size, chunk):
        out[start : start + chunk] = _gn_ni_batch(
            variables, idx[start : start + chunk], n_grid, n_xd
        )
    return out


def _gn_ni_batch(
    variables: FWMTupleVariables, idx: np.ndarray, n_grid: int, n_xd: int = 0
) -> np.ndarray:
    x = -np.pi + (np.arange(n_grid) + 0.5) * (2.0 * np.pi / n_grid)
    xa = x[None, :, None]
    xb = x[None, None, :]
    u0 = variables.u0[idx][:, None, None]
    nu_a = variables.nu_a[idx][:, None, None]
    nu_b = variables.nu_b[idx][:, None, None]
    nu_c = variables.nu_c[idx][:, None, None]
    q_a = variables.q_a[idx][:, None, None]
    q_b = variables.q_b[idx][:, None, None]
    q_c = variables.q_c[idx][:, None, None]
    q_t = float(variables.q_t)
    d = variables.d[idx][:, None, None]
    if n_xd <= 0:
        xd_values = np.array([0.0])
    else:
        xd_values = -np.pi + (np.arange(n_xd) + 0.5) * (2.0 * np.pi / n_xd)
    acc = np.zeros(idx.size, dtype=float)
    for xd in xd_values:
        xc = xa + xb + d - xd
        inside = np.abs(xc) < np.pi
        u = (
            u0 + nu_a * xa + nu_b * xb - nu_c * xc
            + q_a * xa**2 + q_b * xb**2 - q_c * xc**2 - q_t * xd**2
        )
        acc += np.mean(kernel_abs2(u) * inside, axis=(1, 2))
    return acc / xd_values.size


def center_family_indices(
    variables: FWMTupleVariables, family_step_dimensionless: float
) -> np.ndarray:
    """Indices of the center-family triplets (support shift rounding to 0).

    ``family_step_dimensionless`` is the family spacing in support-shift units,
    ``2 pi * channel_spacing / baud_rate`` on a uniform grid.
    """
    family_step_dimensionless = float(family_step_dimensionless)
    if (
        not np.isfinite(family_step_dimensionless)
        or family_step_dimensionless <= 0.0
    ):
        raise ValueError("family_step_dimensionless must be finite and > 0")
    return np.where(np.rint(variables.d / family_step_dimensionless) == 0)[0]


def target_gn_cfm_sums(
    frequencies_hz: np.ndarray,
    beta_at_carriers_per_m: np.ndarray,
    beta1_s_per_m: np.ndarray,
    beta2_s2_per_m: np.ndarray,
    symbol_rate_baud: float,
    fiber_length_m: float,
    target_idx: int,
    *,
    family_step_dimensionless: float | None = None,
) -> tuple[float, int]:
    """GN-CFM analog efficiency sum for one target: center family.

    Returns ``(efficiency_sum, n_triplets)``. ``family_step_dimensionless``
    defaults to ``2 pi * median_spacing_hz / symbol_rate_baud``.
    """
    variables = fwm_tuple_variables(
        frequencies_hz,
        beta_at_carriers_per_m,
        beta1_s_per_m,
        beta2_s2_per_m,
        symbol_rate_baud,
        fiber_length_m,
        target_idx,
    )
    if family_step_dimensionless is None:
        spacing_hz = float(
            np.median(np.diff(np.sort(np.asarray(frequencies_hz, float))))
        )
        family_step_dimensionless = (
            2.0 * np.pi * spacing_hz / float(symbol_rate_baud)
        )
    sel = center_family_indices(variables, family_step_dimensionless)
    if sel.size == 0:
        return 0.0, 0
    return float(np.sum(gn_cfm_tuple_values(variables, sel))), int(sel.size)


def target_gn_ni_sums(
    frequencies_hz: np.ndarray,
    beta_at_carriers_per_m: np.ndarray,
    beta1_s_per_m: np.ndarray,
    beta2_s2_per_m: np.ndarray,
    symbol_rate_baud: float,
    fiber_length_m: float,
    target_idx: int,
    *,
    n_coarse: int = 8,
    n_fine: int = 32,
    n_refine: int = 65536,
    n_xd: int = 0,
) -> tuple[float, float, int]:
    """GN-NI analog efficiency sum for one target: all families, exact islands.

    Coarse midpoint pass on every tuple, fine pass on the ``n_refine``
    largest coarse values. Returns ``(efficiency_sum, coarse_efficiency_sum,
    refined_count)``.
    """
    variables = fwm_tuple_variables(
        frequencies_hz,
        beta_at_carriers_per_m,
        beta1_s_per_m,
        beta2_s2_per_m,
        symbol_rate_baud,
        fiber_length_m,
        target_idx,
    )
    return gn_ni_from_variables(
        variables, n_coarse=n_coarse, n_fine=n_fine, n_refine=n_refine, n_xd=n_xd
    )


def gn_ni_from_variables(
    variables: FWMTupleVariables,
    *,
    n_coarse: int = 8,
    n_fine: int = 32,
    n_refine: int = 65536,
    n_xd: int = 0,
) -> tuple[float, float, int]:
    """GN-NI analog sum from prebuilt variables (see :func:`target_gn_ni_sums`)."""
    n_coarse = _positive_int(n_coarse, "n_coarse")
    n_fine = _positive_int(n_fine, "n_fine")
    n_refine = _nonnegative_int(n_refine, "n_refine")
    n_xd = _nonnegative_int(n_xd, "n_xd")
    n = variables.u0.size
    if n == 0:
        return 0.0, 0.0, 0
    values = gn_ni_tuple_values(variables, n_grid=n_coarse, n_xd=n_xd)
    coarse_sum = float(np.sum(values))
    k = min(int(n_refine), n)
    if k == 0:
        return coarse_sum, coarse_sum, 0
    top = np.argpartition(values, -k)[-k:] if k < n else np.arange(n)
    values[top] = gn_ni_tuple_values(
        variables, top, n_grid=n_fine, n_xd=n_xd
    )
    return float(np.sum(values)), coarse_sum, int(top.size)


__all__ = [
    "center_family_indices",
    "gn_cfm_tuple_values",
    "gn_ni_from_variables",
    "gn_ni_tuple_values",
    "target_gn_cfm_sums",
    "target_gn_ni_sums",
]
