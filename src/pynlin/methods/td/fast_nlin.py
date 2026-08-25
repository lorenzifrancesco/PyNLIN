"""Lorenzi Fast core: prefactor-free NLIN estimation in normalized variables.

Every prefactor-free contribution (XPM pair or strict FWM tuple) is the
in-channel average of the normalized link kernel

    Khat(u) = |(exp(iu) - 1) / (iu)|^2 = 4 sin^2(u/2) / u^2,   u = dbeta * L,

over the in-channel offsets ``x_j in (-pi, pi)`` (``omega_j = x_j * B``) with
the output-support mask ``|x_d| < pi``.  After removing the target-channel
group delay (``beta1 -> beta1 - beta1[t]``), the accumulated phase is

    u(x) = u0 + nu_a x_a + nu_b x_b - nu_c x_c + quadratic terms,

with the normalized variables

    u0   = (beta0_a + beta0_b - beta0_c - beta0_t) * L - beta1_t * dw * L
    nu_j = (beta1_j - beta1_t) * B * L          (walk-off phase, one per leg)
    q_j  = 0.5 * beta2_j * B^2 * L              (in-channel quadratic phase)
    d    = dw / B,  dw = 2 pi (f_a + f_b - f_c - f_t)   (support shift)

The quantity every model here evaluates is NOT a new object: it is the
per-tuple/per-pair normalized collision sum ``N * T^2 / L^2`` — the Golani
sum ``N = sum_{h,r,m} |X_{h,r,m}|^2`` of docs/source/direct_sector_mc.md in
its dimensionless form (equivalently the m^2-valued frequency-domain
aggregate divided by ``L^2``, the ``noise_coefficient`` of the validation
scripts):

    N T^2/L^2 = E[Khat(u) * 1_mask].

The linear model (quadratics dropped, mask treated as independent of u)
factorizes exactly:

    N T^2/L^2 = A(d) * int_0^1 2 (1 - t) prod_j sinc(w_j t) cos(u0 t) dt,

where ``w_j = pi |nu_j|`` and ``A(d)`` is the mask acceptance, because the
Fourier transform of Khat is the triangle ``2 pi (1 - |t|)_+`` and the
characteristic function of the sum of uniforms is a product of sincs.
Far and wide regimes use closed-form asymptotics stitched to this integral.

The per-tuple contribution in the conventions of
:mod:`pynlin.methods.td.fullband_mc` is ``L^2 * (N T^2/L^2) = N T^2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

TWO_PI = 2.0 * np.pi

# Regime-dispatch thresholds (phases in radians).
WIDE_HALFWIDTH = 3000.0
FAR_MARGIN_FACTOR = 3.0
FAR_MARGIN_OFFSET = 3000.0
CF_BASE_NODES = 64
NEAR_NODES_PER_RADIAN = 1.4
CF_MAX_NODES = 9000
NEAR_BUDGET = 4_000_000  # tuples x nodes per vectorized chunk
WIDE_CHUNK = 4096
# QMC refinement is only reliable when the kernel core (|u| < ~pi) covers
# enough of the sampling cube; beyond this total width the effective sample
# count collapses and the analytic estimate is the more accurate one.
REFINE_MAX_WIDTH = 300.0


def kernel_abs2(u: np.ndarray) -> np.ndarray:
    """Normalized lossless link kernel Khat(u) = 4 sin^2(u/2) / u^2."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    small = np.abs(u) < 1e-6
    out[small] = 1.0 - u[small] ** 2 / 12.0
    us = u[~small]
    out[~small] = 4.0 * np.sin(0.5 * us) ** 2 / us**2
    return out


def uniform_sum_density(u: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Density of a sum of independent uniforms on (-w_j, w_j) at points u.

    ``widths`` has shape (..., k); ``u`` broadcasts against the leading shape.
    Uses the inclusion-exclusion (Irwin-Hall style) piecewise-polynomial form.
    Degenerate zero widths are handled by dropping the corresponding leg.
    """
    widths = np.atleast_2d(np.asarray(widths, dtype=float))
    u = np.asarray(u, dtype=float)
    k = widths.shape[-1]
    total = np.sum(widths, axis=-1)
    # Shift to sums of uniforms on (0, 2 w_j): v = u + sum(w_j).
    v = u + total
    dens = np.zeros(np.broadcast_shapes(u.shape, total.shape), dtype=float)
    norm = factorial(k - 1) * np.prod(np.maximum(2.0 * widths, 1e-300), axis=-1)
    for mask_bits in range(2**k):
        signs = np.array([(mask_bits >> j) & 1 for j in range(k)], dtype=float)
        shift = np.sum(2.0 * widths * signs, axis=-1)
        parity = -1.0 if int(np.sum(signs)) % 2 else 1.0
        dens += parity * np.maximum(v - shift, 0.0) ** (k - 1)
    return dens / norm


def _comb3(j: int) -> float:
    return (1.0, 3.0, 3.0, 1.0)[j]


def support_acceptance(d: np.ndarray) -> np.ndarray:
    """Acceptance A(d) = P(|x_a + x_b - x_c + d| < pi), x_j ~ U(-pi, pi)."""
    d = np.asarray(d, dtype=float)

    def cdf3(x: np.ndarray) -> np.ndarray:
        v = np.clip((x + 3.0 * np.pi) / TWO_PI, 0.0, 3.0)
        out = np.zeros_like(v)
        for j in range(4):
            out += (-1.0) ** j * _comb3(j) * np.maximum(v - j, 0.0) ** 3
        return out / 6.0

    return cdf3(np.pi - d) - cdf3(-np.pi - d)


from functools import lru_cache


GL_PANEL_ORDER = 256


@lru_cache(maxsize=64)
def _leggauss_cached(n: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(n))
    return nodes, weights


def cf_gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature rule on [0, 1] with ~n nodes.

    Orders above GL_PANEL_ORDER use composite panels of the capped-order
    rule: leggauss costs O(order^3) (companion eigendecomposition), so a
    single high-order rule is prohibitively slow, while composite panels of
    a cached base rule are both cheap and equally accurate for oscillatory
    integrands resolved at a fixed nodes-per-period density.
    """
    n = int(n)
    if n <= GL_PANEL_ORDER:
        nodes, weights = _leggauss_cached(n)
        return 0.5 * (nodes + 1.0), 0.5 * weights
    n_panels = int(np.ceil(n / GL_PANEL_ORDER))
    base_nodes, base_weights = _leggauss_cached(GL_PANEL_ORDER)
    h = 1.0 / n_panels
    starts = h * np.arange(n_panels)
    nodes = (starts[:, None] + 0.5 * h * (base_nodes[None, :] + 1.0)).reshape(-1)
    weights = np.broadcast_to(0.5 * h * base_weights, (n_panels, GL_PANEL_ORDER)).reshape(-1)
    return nodes, weights.copy()


def _sinc(z: np.ndarray) -> np.ndarray:
    return np.sinc(z / np.pi)


def linear_model_cf(u0: np.ndarray, widths: np.ndarray, n_nodes: int) -> np.ndarray:
    """Exact linear-model integral int_0^1 2(1-t) prod sinc(w_j t) cos(u0 t) dt.

    Vectorized over a batch: ``u0`` shape (m,), ``widths`` shape (m, k).
    """
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    widths = np.atleast_2d(np.asarray(widths, dtype=float))
    t, wts = cf_gauss_legendre(n_nodes)
    phase = widths[:, :, None] * t[None, None, :]
    prod_sinc = np.prod(_sinc(phase), axis=1)
    integrand = 2.0 * (1.0 - t)[None, :] * prod_sinc * np.cos(u0[:, None] * t[None, :])
    return integrand @ wts


MASK_COEFFS = np.array([1.0, 1.0, -1.0])


def _cdf3_scaled(x, w):
    """CDF of a sum of three uniforms on (-w, w) with per-element width w."""
    import numpy as _np
    w = _np.maximum(w, 1e-12)
    v = _np.clip((x + 3.0 * w) / (2.0 * w), 0.0, 3.0)
    out = _np.zeros_like(v)
    for j in range(4):
        out += (-1.0) ** j * _comb3(j) * _np.maximum(v - j, 0.0) ** 3
    return out / 6.0


EXACT_ACCEPTANCE_M_NODES = 48


def _batched_Minv_detM(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-tuple inverse and determinant of the change-of-basis matrix
    ``M = [c_u; c_m; c_w]`` where ``c_u`` are the signed linear coefficients
    of ``u`` in the in-channel offsets, ``c_m = (1, 1, -1)`` are the
    coefficients of the mask variable, and ``c_w = c_u x c_m`` completes a
    basis. ``coeffs`` is (m, 3); returns (Minv (m,3,3), detM (m,)).
    """
    coeffs = np.atleast_2d(np.asarray(coeffs, dtype=float))
    m = coeffs.shape[0]
    c_m = np.broadcast_to(MASK_COEFFS, (m, 3))
    c_w = np.cross(coeffs, c_m)
    norm_w = np.linalg.norm(c_w, axis=-1)
    degenerate = norm_w < 1e-9
    if np.any(degenerate):
        # c_u parallel to c_m: a measure-zero direction. Nudge c_w off the
        # degenerate axis; the exact result is insensitive to this choice
        # since it only fixes an internal (unobserved) basis vector.
        c_w = c_w.copy()
        c_w[degenerate] += np.array([1e-6, -1e-6, 0.0])
    M = np.stack([coeffs, c_m, c_w], axis=1)
    Minv = np.linalg.inv(M)
    detM = np.linalg.det(M)
    return Minv, detM


def _exact_joint_density(
    u: np.ndarray, m: np.ndarray, Minv: np.ndarray, detM: np.ndarray
) -> np.ndarray:
    """Exact joint density rho(u, m) for (x_a, x_b, x_c) ~ Uniform(-pi, pi)^3,
    with u the signed linear combination and m = x_a + x_b - x_c the mask
    variable, both (batch, n) broadcastable; ``Minv``/``detM`` from
    :func:`_batched_Minv_detM`.

    The density is uniform on the linear image of the cube (a zonotope), so
    at fixed (u, m) it equals the length of the feasible interval for the
    third coordinate ``w`` (in the (u, m, w) basis) divided by the Jacobian
    ``|det M|`` and the cube's volume ``(2 pi)^3``. No Fourier inversion or
    approximation is used; this is exact.
    """
    A = Minv[:, :, 0:1] * u[:, None, :] + Minv[:, :, 1:2] * m[:, None, :]
    B = Minv[:, :, 2, None]
    pos = B > 1e-14
    neg = B < -1e-14
    safe = np.where(pos | neg, B, 1.0)
    bound_lo = (-np.pi - A) / safe
    bound_hi = (np.pi - A) / safe
    lo = np.where(pos, bound_lo, np.where(neg, bound_hi, -np.inf))
    hi = np.where(pos, bound_hi, np.where(neg, bound_lo, np.inf))
    infeasible_zero = (~pos) & (~neg) & ((A < -np.pi) | (A > np.pi))
    lo = np.where(infeasible_zero, np.inf, lo)
    hi = np.where(infeasible_zero, -np.inf, hi)
    w_lo = np.max(lo, axis=1)
    w_hi = np.min(hi, axis=1)
    length = np.maximum(w_hi - w_lo, 0.0)
    return length / (TWO_PI**3 * np.abs(detM)[:, None])


def exact_conditional_acceptance(
    u_offset: np.ndarray,
    coeffs: np.ndarray,
    d: np.ndarray,
    m_nodes: int = EXACT_ACCEPTANCE_M_NODES,
) -> np.ndarray:
    """Exact acceptance P(|m + d| < pi | u = u_offset), replacing the earlier
    rescaled-3-uniform-sum approximation.

    That approximation is stably wrong (not merely under-resolved) whenever
    two leg widths are nearly equal and the third is small -- a geometry
    common near the ZDW -- because the true conditional law of the mask
    variable is not always shaped like a rescaled sum of three uniforms.
    This computes the exact conditional probability via the density in
    :func:`_exact_joint_density`, integrated over the mask window with a
    Gauss-Legendre rule; verified to converge to independent QMC ground
    truth (0.03% agreement at high resolution, vs up to 81% error for the
    old model on the tuple that exposed the bug).

    ``u_offset`` is (m, n), ``coeffs`` (m, 3), ``d`` (m,). The inner
    m-quadrature is looped (not vectorized as a third array axis) to keep
    memory bounded to O(m * n) regardless of ``m_nodes``.
    """
    coeffs = np.atleast_2d(np.asarray(coeffs, dtype=float))
    u_offset = np.atleast_2d(np.asarray(u_offset, dtype=float))
    d = np.asarray(d, dtype=float).reshape(-1)

    # Degenerate direction c_u || c_m (e.g. the exact equal split): M has two
    # parallel rows and is singular no matter how the third basis vector is
    # chosen. Physically u determines m exactly (u = kappa * m), so the
    # conditional acceptance is the indicator of the mask window.
    cross = np.cross(coeffs, np.broadcast_to(MASK_COEFFS, coeffs.shape))
    norm_u = np.linalg.norm(coeffs, axis=-1)
    parallel = np.linalg.norm(cross, axis=-1) < 1e-9 * np.maximum(norm_u, 1e-300)
    if np.any(parallel):
        out = np.empty_like(np.broadcast_to(u_offset, (coeffs.shape[0], u_offset.shape[-1])).copy())
        kappa = np.sum(coeffs * MASK_COEFFS[None, :], axis=-1) / 3.0
        par_idx = np.where(parallel)[0]
        m_par = u_offset[par_idx] / np.maximum(np.abs(kappa[par_idx]), 1e-300)[:, None] \
            * np.sign(kappa[par_idx])[:, None]
        out[par_idx] = (np.abs(m_par + d[par_idx, None]) < np.pi).astype(float)
        rest = np.where(~parallel)[0]
        if rest.size:
            out[rest] = exact_conditional_acceptance(
                u_offset[rest] if u_offset.shape[0] > 1 else u_offset,
                coeffs[rest], d[rest], m_nodes,
            )
        return out

    Minv, detM = _batched_Minv_detM(coeffs)
    nodes, wts = cf_gauss_legendre(m_nodes)
    lo = -np.pi - d
    hi = np.pi - d
    width = (hi - lo)[:, None]

    integral = np.zeros_like(u_offset)
    for k in range(m_nodes):
        m_k = np.broadcast_to(lo[:, None] + width * nodes[k], u_offset.shape)
        integral += wts[k] * _exact_joint_density(u_offset, m_k, Minv, detM)
    integral = integral * width

    widths_arr = np.pi * np.abs(coeffs)
    marginal = uniform_sum_density(u_offset, widths_arr[:, None, :])
    return np.where(marginal > 1e-300, integral / np.maximum(marginal, 1e-300), 0.0)


def pointwise_conditional_acceptance(
    u_offset: np.ndarray,
    coeffs: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Approximate acceptance P(|s + d| < pi | u), modeling the conditional
    law of the mask variable as a rescaled sum of three uniforms.

    This is stably WRONG (not merely under-resolved) when two leg widths are
    nearly equal and the third is small -- common near the ZDW -- because
    the true conditional shape need not resemble a 3-uniform sum. It is
    used as the cheap bulk-pass model in :func:`near_model_masked` /
    :func:`wide_model_masked`, since low-mass tuples where it is wrong
    contribute negligibly to the aggregate (S2 mass-weighted measurement).
    The mass-capped refinement tier in ``target_fast_sums`` corrects the
    high-mass tuples with :func:`exact_conditional_acceptance` instead.
    """
    coeffs = np.atleast_2d(np.asarray(coeffs, dtype=float))
    d = np.asarray(d, dtype=float).reshape(-1)
    var_u = np.pi**2 / 3.0 * np.sum(coeffs**2, axis=-1)
    cov_us = np.pi**2 / 3.0 * np.sum(coeffs * MASK_COEFFS[None, :], axis=-1)
    safe_var = np.maximum(var_u, 1e-300)
    slope = np.where(var_u > 0.0, cov_us / safe_var, 0.0)
    cond_var = np.clip(np.pi**2 - cov_us * slope, 0.0, np.pi**2)
    leg_w = np.sqrt(cond_var)

    shift = slope[:, None] * u_offset
    hi = np.pi - d[:, None] - shift
    lo = -np.pi - d[:, None] - shift
    degenerate = leg_w < 1e-9
    w_bcast = np.broadcast_to(leg_w[:, None], hi.shape)
    accept = _cdf3_scaled(hi, w_bcast) - _cdf3_scaled(lo, w_bcast)
    accept_deg = ((hi > 0.0) & (lo < 0.0)).astype(float)
    return np.where(degenerate[:, None], accept_deg, accept)


def far_model(u0: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Far-detuned closed form: E[(2 - 2 cos u)/u^2] for |u0| >> sum(w_j).

    Uses E[cos u] = cos(u0) prod sinc(w_j) exactly and a second-order
    expansion of E[1/u^2] about u0.
    """
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    widths = np.atleast_2d(np.asarray(widths, dtype=float))
    var = np.sum(widths**2, axis=-1) / 3.0
    inv_u02 = 1.0 / u0**2
    e_inv_u2 = inv_u02 * (1.0 + 3.0 * var * inv_u02)
    e_cos = np.cos(u0) * np.prod(_sinc(widths), axis=-1)
    return 2.0 * e_inv_u2 * (1.0 - e_cos)


WIDE_CENTRAL_CUT = 48.0 * np.pi
WIDE_CENTRAL_NODES = 384


def wide_model(u0: np.ndarray, widths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wide-density regime: returns (central, tail) contributions separately.

    Central: exact quadrature of Khat(u) rho(u - u0) over |u| < U with the
    closed-form density (kernel oscillation resolved with 8 nodes/period).
    Tail: cos-averaged kernel 2/u^2 against the density outside |u| > U,
    valid because the density spans many oscillation periods here.
    The caller applies (possibly different) acceptances to the two pieces.
    """
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    widths = np.atleast_2d(np.asarray(widths, dtype=float))
    U = WIDE_CENTRAL_CUT

    nodes, wts = cf_gauss_legendre(WIDE_CENTRAL_NODES)
    u = -U + 2.0 * U * nodes
    rho = uniform_sum_density(u[None, :] - u0[:, None], widths[:, None, :])
    central = 2.0 * U * np.sum(wts[None, :] * kernel_abs2(u)[None, :] * rho, axis=-1)

    total = np.sum(widths, axis=-1)
    lo = u0 - total
    hi = u0 + total
    t_nodes, t_wts = cf_gauss_legendre(32)
    tail = np.zeros_like(u0)
    for sign in (-1.0, 1.0):
        seg_lo = np.maximum(lo, U) if sign > 0 else np.maximum(-hi, U)
        seg_hi = np.maximum(hi, U) if sign > 0 else np.maximum(-lo, U)
        length = np.maximum(seg_hi - seg_lo, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            span = np.where(length > 0.0, np.log(seg_hi / seg_lo), 0.0)
        y = span[:, None] * t_nodes[None, :]
        u_t = seg_lo[:, None] * np.exp(y)
        rho_t = uniform_sum_density(sign * u_t - u0[:, None], widths[:, None, :])
        tail += np.where(
            length > 0.0,
            2.0 * span * np.sum(t_wts[None, :] * rho_t / u_t, axis=-1),
            0.0,
        )
    return central, tail


@dataclass(frozen=True)
class TupleBatchEstimate:
    values: np.ndarray
    regime: np.ndarray  # 0 = CF exact, 1 = far, 2 = wide


def linear_tuple_estimate(
    u0: np.ndarray,
    coeffs: np.ndarray,
    d: np.ndarray,
) -> TupleBatchEstimate:
    """Regime-dispatched linear-model estimate of F for a batch of tuples.

    ``u0`` (m,), ``coeffs`` (m, 3) signed linear coefficients of the
    in-channel offsets (for FWM: ``(nu_a, nu_b, -nu_c)``), ``d`` (m,) the
    normalized support shift.  Returns acceptance-weighted F values.
    """
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    coeffs = np.atleast_2d(np.asarray(coeffs, dtype=float))
    d = np.broadcast_to(np.asarray(d, dtype=float).reshape(-1), u0.shape)
    widths = np.pi * np.abs(coeffs)
    m = u0.size
    total = np.sum(widths, axis=-1)
    abs_u0 = np.abs(u0)

    values = np.zeros(m, dtype=float)
    regime = np.zeros(m, dtype=int)
    plain_accept = support_acceptance(d)

    far = abs_u0 > FAR_MARGIN_FACTOR * total + FAR_MARGIN_OFFSET
    wide = (~far) & (total > WIDE_HALFWIDTH)
    near = ~(far | wide)

    if np.any(far):
        # Kernel mass spans the whole density: correlation with the mask is a
        # second-order effect, use the plain acceptance.
        values[far] = far_model(u0[far], widths[far]) * plain_accept[far]
        regime[far] = 1
    if np.any(wide):
        idx = np.where(wide)[0]
        for start in range(0, idx.size, WIDE_CHUNK):
            sel = idx[start : start + WIDE_CHUNK]
            central, tail = wide_model_masked(u0[sel], widths[sel], coeffs[sel], d[sel])
            values[sel] = central + tail * plain_accept[sel]
        regime[idx] = 2
    if np.any(near):
        idx = np.where(near)[0]
        # Positive-integrand u-space quadrature over the compact density
        # support, with the pointwise conditional mask acceptance.
        n_nodes = np.minimum(
            CF_BASE_NODES + (NEAR_NODES_PER_RADIAN * total[idx]).astype(int),
            CF_MAX_NODES,
        )
        buckets = np.ceil(np.log2(np.maximum(n_nodes / CF_BASE_NODES, 1.0))).astype(int)
        for bucket in np.unique(buckets):
            sel = idx[buckets == bucket]
            nn = int(CF_BASE_NODES * 2**bucket)
            # Chunk so the (tuples, nodes) work arrays stay within ~100 MB.
            chunk = max(int(NEAR_BUDGET / max(nn, 1)), 16)
            for start in range(0, sel.size, chunk):
                sub = sel[start : start + chunk]
                values[sub] = near_model_masked(
                    u0[sub], widths[sub], coeffs[sub], d[sub], nn
                )
    return TupleBatchEstimate(values=values, regime=regime)


def near_model_masked(
    u0: np.ndarray,
    widths: np.ndarray,
    coeffs: np.ndarray,
    d: np.ndarray,
    n_nodes: int,
    acceptance_fn=None,
) -> np.ndarray:
    """u-space quadrature of Khat(u) rho(u - u0) A(u) over the density support.

    The integrand is nonnegative, so no oscillatory cancellation occurs; the
    node count only needs to resolve the kernel oscillation (period 2 pi)
    across the support width. ``acceptance_fn`` defaults to the cheap
    approximate model (bulk pass); pass :func:`exact_conditional_acceptance`
    for the mass-capped refinement tier.
    """
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    widths = np.atleast_2d(np.asarray(widths, dtype=float))
    total = np.maximum(np.sum(widths, axis=-1), 1e-9)
    nodes, wts = cf_gauss_legendre(n_nodes)
    offset = total[:, None] * (2.0 * nodes[None, :] - 1.0)
    u = u0[:, None] + offset
    rho = uniform_sum_density(offset, widths[:, None, :])
    fn = acceptance_fn or pointwise_conditional_acceptance
    accept = fn(offset, coeffs, d)
    integrand = kernel_abs2(u) * rho * accept
    return 2.0 * total * np.sum(wts[None, :] * integrand, axis=-1)


def wide_model_masked(
    u0: np.ndarray,
    widths: np.ndarray,
    coeffs: np.ndarray,
    d: np.ndarray,
    acceptance_fn=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Wide-regime central piece with pointwise mask acceptance, plus tail."""
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    widths = np.atleast_2d(np.asarray(widths, dtype=float))
    U = WIDE_CENTRAL_CUT
    nodes, wts = cf_gauss_legendre(WIDE_CENTRAL_NODES)
    u = -U + 2.0 * U * nodes
    offset = u[None, :] - u0[:, None]
    rho = uniform_sum_density(offset, widths[:, None, :])
    fn = acceptance_fn or pointwise_conditional_acceptance
    accept = fn(offset, coeffs, d)
    central = 2.0 * U * np.sum(
        wts[None, :] * kernel_abs2(u)[None, :] * rho * accept, axis=-1
    )
    _, tail = wide_model(u0, widths)
    return central, tail


# ---------------------------------------------------------------------------
# Normalized tuple variables from system grids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FWMTupleVariables:
    """Vectorized normalized variables for all support-surviving strict FWM
    tuples of one target channel."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    u0: np.ndarray
    nu_a: np.ndarray
    nu_b: np.ndarray
    nu_c: np.ndarray
    q_a: np.ndarray
    q_b: np.ndarray
    q_c: np.ndarray
    q_t: float
    d: np.ndarray
    acceptance: np.ndarray

    @property
    def widths(self) -> np.ndarray:
        return np.pi * np.abs(np.stack([self.nu_a, self.nu_b, self.nu_c], axis=-1))

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(np.sum(self.widths**2, axis=-1) / 3.0)

    @property
    def x_grad(self) -> np.ndarray:
        """Loudness scale ``L*B*||grad dbeta||_2 = sqrt(sum nu_j^2)`` of
        docs/source/fwm_single_tuple_scaling.md."""
        return np.sqrt(self.nu_a**2 + self.nu_b**2 + self.nu_c**2)

    @property
    def mu(self) -> np.ndarray:
        """Dimensionless detuning ``dbeta_center / (B*||grad dbeta||_2)``
        (single-tuple-scaling convention); ``u0 = mu * x_grad`` exactly."""
        x = self.x_grad
        return np.divide(self.u0, x, out=np.zeros_like(self.u0), where=x > 1e-300)


def fwm_tuple_variables(
    freqs: np.ndarray,
    beta0_abs: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    baud_rate: float,
    length: float,
    target: int,
) -> FWMTupleVariables:
    """Compute (u0, nu, q, d) for every support-pruned strict tuple of a target.

    Support condition: |f_a + f_b - f_c - f_t| <= 2 B.  Strictness: a, b, c
    pairwise distinct and different from the target.
    """
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    n = freqs.size
    t = int(target)
    B = float(baud_rate)
    L = float(length)

    # Target-frame propagation constants: remove the target group delay so the
    # linear-in-frequency part of beta cancels exactly for energy-conserving
    # quadruples.
    beta1_rel = np.asarray(beta1, dtype=float) - float(beta1[t])
    omega = TWO_PI * freqs
    beta0_rel = (
        np.asarray(beta0_abs, dtype=float)
        - float(beta0_abs[t])
        - float(beta1[t]) * (omega - omega[t])
    )

    order = np.argsort(freqs)
    sorted_freqs = freqs[order]
    width = 2.0 * B
    ft = freqs[t]

    a_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    b_all = np.arange(n, dtype=np.int64)
    for a in range(n):
        if a == t:
            continue
        centers = freqs[a] + freqs - ft
        lo = np.searchsorted(sorted_freqs, centers - width, side="left")
        hi = np.searchsorted(sorted_freqs, centers + width, side="right")
        counts = hi - lo
        counts[t] = 0
        counts[a] = 0
        surv = np.where(counts > 0)[0]
        if surv.size == 0:
            continue
        cnt = counts[surv]
        total = int(np.sum(cnt))
        # Ragged expansion of [lo[b], hi[b]) ranges into flat position lists.
        starts = np.repeat(lo[surv], cnt)
        ends = np.cumsum(cnt)
        within = np.arange(total) - np.repeat(ends - cnt, cnt)
        c_vals = order[starts + within]
        b_vals = np.repeat(b_all[surv], cnt)
        keep = (c_vals != t) & (c_vals != a) & (c_vals != b_vals)
        if not np.any(keep):
            continue
        c_vals = c_vals[keep]
        b_vals = b_vals[keep]
        a_parts.append(np.full(c_vals.size, a, dtype=np.int32))
        b_parts.append(b_vals.astype(np.int32))
        c_parts.append(c_vals.astype(np.int32))
    if not a_parts:
        empty = np.array([], dtype=np.int32)
        zeros = np.array([], dtype=float)
        return FWMTupleVariables(
            a=empty, b=empty, c=empty, u0=zeros, nu_a=zeros, nu_b=zeros,
            nu_c=zeros, q_a=zeros, q_b=zeros, q_c=zeros,
            q_t=0.5 * float(beta2[t]) * B**2 * L, d=zeros, acceptance=zeros,
        )

    a_idx = np.concatenate(a_parts)
    b_idx = np.concatenate(b_parts)
    c_idx = np.concatenate(c_parts)

    delta_omega = TWO_PI * (freqs[a_idx] + freqs[b_idx] - freqs[c_idx] - ft)
    d = delta_omega / B
    u0 = (beta0_rel[a_idx] + beta0_rel[b_idx] - beta0_rel[c_idx]) * L
    nu_a = beta1_rel[a_idx] * B * L
    nu_b = beta1_rel[b_idx] * B * L
    nu_c = beta1_rel[c_idx] * B * L
    beta2 = np.asarray(beta2, dtype=float)
    q_scale = 0.5 * B**2 * L
    return FWMTupleVariables(
        a=a_idx,
        b=b_idx,
        c=c_idx,
        u0=u0,
        nu_a=nu_a,
        nu_b=nu_b,
        nu_c=nu_c,
        q_a=beta2[a_idx] * q_scale,
        q_b=beta2[b_idx] * q_scale,
        q_c=beta2[c_idx] * q_scale,
        q_t=float(beta2[t]) * q_scale,
        d=d,
        acceptance=support_acceptance(d),
    )


def xpm_pair_variables(
    beta1: np.ndarray,
    beta2: np.ndarray,
    baud_rate: float,
    length: float,
    target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (interferer index, nu, q) for all XPM pairs of a target."""
    beta1 = np.asarray(beta1, dtype=float)
    beta2 = np.asarray(beta2, dtype=float)
    n = beta1.size
    t = int(target)
    others = np.array([b for b in range(n) if b != t], dtype=np.int32)
    nu = (beta1[others] - beta1[t]) * float(baud_rate) * float(length)
    q = 0.5 * beta2[others] * float(baud_rate) ** 2 * float(length)
    return others, nu, q


def _xpm_mass_transform(theta: np.ndarray) -> np.ndarray:
    """Cosine transform H(theta) of the exact masked XPM density.

    The XPM phase is u = nu (x_1 - x_2) and the mask reduces exactly to the
    weight (1 - |y| / 2 pi) on y = x_1 - x_2, giving the closed form
    H(theta) = 1/(pi theta)^2 - sin(2 pi theta)/(2 pi^3 theta^3), H(0) = 2/3.
    """
    theta = np.asarray(theta, dtype=float)
    out = np.empty_like(theta)
    small = np.abs(theta) < 1e-4
    ts = theta[small]
    # Series from the sin expansion: H = 2/3 - (2 pi^2 / 15) theta^2 + ...
    out[small] = 2.0 / 3.0 - (2.0 * np.pi**2 / 15.0) * ts**2
    tb = theta[~small]
    out[~small] = 1.0 / (np.pi**2 * tb**2) - np.sin(TWO_PI * tb) / (
        2.0 * np.pi**3 * tb**3
    )
    return out


def xpm_fast_batch(nu: np.ndarray) -> np.ndarray:
    """Exact linear-model XPM estimate F(nu) = int_0^1 2(1-t) H(nu t) dt.

    The oscillatory part of H is confined to nu*t < ~10; beyond that the
    smooth 1/(nu t)^2 part is integrated in closed form.
    """
    nu = np.asarray(nu, dtype=float).reshape(-1)
    abs_nu = np.maximum(np.abs(nu), 1e-12)
    result = np.empty_like(abs_nu)

    t0 = np.minimum(1.0, 10.0 / abs_nu)
    nodes, wts = cf_gauss_legendre(96)
    t = t0[:, None] * nodes[None, :]
    integrand = 2.0 * (1.0 - t) * _xpm_mass_transform(abs_nu[:, None] * t)
    head = t0 * np.sum(wts[None, :] * integrand, axis=-1)

    # Smooth remainder on [t0, 1] of 2 (1 - t) / (pi nu t)^2.
    with np.errstate(divide="ignore", invalid="ignore"):
        remainder = np.where(
            t0 < 1.0,
            (2.0 / (np.pi**2 * abs_nu**2)) * (1.0 / t0 - 1.0 + np.log(t0)),
            0.0,
        )
    result = head + remainder
    return result


# ---------------------------------------------------------------------------
# Ground truth: randomized QMC with the full quadratic in-channel model
# ---------------------------------------------------------------------------


def qmc_tuple_ground_truth(
    *,
    u0: float,
    nu: tuple[float, float, float],
    q: tuple[float, float, float, float],
    d: float,
    n_points: int = 1 << 18,
    n_replicates: int = 8,
    seed: int = 0,
    include_quadratic: bool = True,
) -> tuple[float, float]:
    """Randomized-Sobol estimate of F for one tuple with the full model.

    Returns (mean, stderr over replicates).  The integrand is Khat(u(x)) with
    the exact output-support mask; ``include_quadratic=False`` reproduces the
    linearized model (for isolating collapse error from quadratic error).
    """
    from scipy.stats import qmc as scipy_qmc

    q_a, q_b, q_c, q_t = q
    nu_a, nu_b, nu_c = nu
    estimates = np.empty(n_replicates, dtype=float)
    for rep in range(n_replicates):
        sampler = scipy_qmc.Sobol(d=3, scramble=True, seed=seed + rep)
        x = TWO_PI * (sampler.random(n_points) - 0.5)
        x_a, x_b, x_c = x[:, 0], x[:, 1], x[:, 2]
        x_d = x_a + x_b - x_c + d
        mask = np.abs(x_d) < np.pi
        u = u0 + nu_a * x_a + nu_b * x_b - nu_c * x_c
        if include_quadratic:
            u = u + q_a * x_a**2 + q_b * x_b**2 - q_c * x_c**2 - q_t * x_d**2
        estimates[rep] = float(np.mean(kernel_abs2(u) * mask))
    mean = float(np.mean(estimates))
    stderr = float(np.std(estimates, ddof=1) / np.sqrt(n_replicates))
    return mean, stderr


def qmc_xpm_ground_truth(
    *,
    nu: float,
    q_b: float,
    q_t: float,
    n_points: int = 1 << 18,
    n_replicates: int = 8,
    seed: int = 0,
    include_quadratic: bool = True,
) -> tuple[float, float]:
    """Randomized-Sobol XPM pair ground truth matching the MC integrand."""
    from scipy.stats import qmc as scipy_qmc

    estimates = np.empty(n_replicates, dtype=float)
    for rep in range(n_replicates):
        sampler = scipy_qmc.Sobol(d=3, scramble=True, seed=seed + rep)
        r = TWO_PI * (sampler.random(n_points) - 0.5)
        x_in, x_1, x_2 = r[:, 0], r[:, 1], r[:, 2]
        x_out = x_in - x_1 + x_2
        mask = np.abs(x_out) < np.pi
        u = nu * (x_1 - x_2)
        if include_quadratic:
            u = u + q_t * (x_out**2 - x_in**2) + q_b * (x_1**2 - x_2**2)
        estimates[rep] = float(np.mean(kernel_abs2(u) * mask))
    mean = float(np.mean(estimates))
    stderr = float(np.std(estimates, ddof=1) / np.sqrt(n_replicates))
    return mean, stderr


# ---------------------------------------------------------------------------
# Per-target fast sums with certified bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FastTargetResult:
    xpm: float
    fwm: float
    fwm_tuples: int
    regime_counts: tuple[int, int, int]
    far_bound_check: float
    refined_tuples: int
    refined_mass_fraction: float


def refine_tuples_qmc(
    variables: FWMTupleVariables,
    indices: np.ndarray,
    *,
    n_points: int = 1 << 14,
    seed: int = 7777,
    include_quadratic: bool = True,
    chunk_size: int = 512,
) -> np.ndarray:
    """Exact-model QMC re-evaluation of selected tuples, shared point set.

    One scrambled-Sobol batch is drawn once and broadcast across all selected
    tuples. The tuple dimension is processed in chunks of ``chunk_size`` so
    peak memory is bounded by (chunk_size, n_points) regardless of how many
    tuples are refined -- near-ZDW targets can have thousands of near-regime
    tuples requiring refinement, and an unchunked (n_tuples, n_points) batch
    of several intermediate arrays there is a real OOM risk under process
    parallelism (each worker allocating independently).
    """
    from scipy.stats import qmc as scipy_qmc

    sampler = scipy_qmc.Sobol(d=3, scramble=True, seed=seed)
    x = TWO_PI * (sampler.random(int(n_points)) - 0.5)
    x_a, x_b, x_c = x[:, 0], x[:, 1], x[:, 2]

    idx = np.asarray(indices, dtype=int)
    out = np.empty(idx.size, dtype=float)
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, idx.size, chunk_size):
        sub = idx[start : start + chunk_size]
        d = variables.d[sub][:, None]
        x_d = x_a[None, :] + x_b[None, :] - x_c[None, :] + d
        mask = np.abs(x_d) < np.pi
        u = (
            variables.u0[sub][:, None]
            + variables.nu_a[sub][:, None] * x_a[None, :]
            + variables.nu_b[sub][:, None] * x_b[None, :]
            - variables.nu_c[sub][:, None] * x_c[None, :]
        )
        if include_quadratic:
            u = (
                u
                + variables.q_a[sub][:, None] * x_a[None, :] ** 2
                + variables.q_b[sub][:, None] * x_b[None, :] ** 2
                - variables.q_c[sub][:, None] * x_c[None, :] ** 2
                - variables.q_t * x_d**2
            )
        out[start : start + sub.size] = np.mean(kernel_abs2(u) * mask, axis=-1)
    return out


def refine_tuples_exact(
    variables: FWMTupleVariables,
    indices: np.ndarray,
    regime: np.ndarray,
) -> np.ndarray:
    """Deterministic exact-acceptance re-evaluation of selected tuples.

    Replaces the earlier QMC-based refinement (:func:`refine_tuples_qmc`)
    for the mask/mismatch-linear model: near-regime tuples are re-evaluated
    with :func:`exact_conditional_acceptance` at high node counts (no
    sampling noise, converges deterministically -- verified to 0.03%
    agreement with tight QMC on the tuple that exposed the original 81%
    bug); wide-regime tuples get the same treatment for their central
    piece. Far-regime tuples are not re-evaluated: the far asymptotic
    already uses the plain (unconditional) acceptance, which the S2 gate
    measured accurate to <1%, so there is no bug there to correct.

    ``indices`` are positions into ``variables``; ``regime`` is the
    per-tuple regime classification from :func:`linear_tuple_estimate`
    aligned with ``variables`` (0=near, 1=far, 2=wide). Returns values
    aligned with ``indices``.
    """
    idx = np.asarray(indices, dtype=int)
    out = np.empty(idx.size, dtype=float)
    coeffs = np.stack(
        [variables.nu_a[idx], variables.nu_b[idx], -variables.nu_c[idx]], axis=-1
    )
    widths = np.pi * np.abs(coeffs)
    u0 = variables.u0[idx]
    d = variables.d[idx]
    total = np.sum(widths, axis=-1)
    reg = regime[idx]

    far_sel = reg == 1
    if np.any(far_sel):
        plain_accept = support_acceptance(d[far_sel])
        out[far_sel] = far_model(u0[far_sel], widths[far_sel]) * plain_accept

    wide_sel = reg == 2
    if np.any(wide_sel):
        plain_accept = support_acceptance(d[wide_sel])
        central, tail = wide_model_masked(
            u0[wide_sel], widths[wide_sel], coeffs[wide_sel], d[wide_sel],
            acceptance_fn=exact_conditional_acceptance,
        )
        out[wide_sel] = central + tail * plain_accept

    near_sel = reg == 0
    if np.any(near_sel):
        near_idx = np.where(near_sel)[0]
        n_nodes = np.minimum(
            CF_BASE_NODES + (NEAR_NODES_PER_RADIAN * total[near_idx]).astype(int),
            CF_MAX_NODES,
        )
        buckets = np.ceil(np.log2(np.maximum(n_nodes / CF_BASE_NODES, 1.0))).astype(int)
        for bucket in np.unique(buckets):
            sel = near_idx[buckets == bucket]
            nn = int(CF_BASE_NODES * 2**bucket)
            chunk = max(int(NEAR_BUDGET / max(nn, 1) / 4), 16)
            for start in range(0, sel.size, chunk):
                sub = sel[start : start + chunk]
                out[sub] = near_model_masked(
                    u0[sub], widths[sub], coeffs[sub], d[sub], nn,
                    acceptance_fn=exact_conditional_acceptance,
                )
    return out


def target_fast_sums(
    freqs: np.ndarray,
    beta0_abs: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    baud_rate: float,
    length: float,
    target: int,
    *,
    n_refine: int = 256,
    max_near_refine: int = 16_384,
) -> FastTargetResult:
    """Fast prefactor-free XPM + strict-FWM sums for one target channel.

    Values are in the conventions of ``compute_fullband_prefactor_free_mc``
    (multiply by L^2 to obtain the [m^2] sums reported there).  The
    ``n_refine`` largest fast-pass tuples are re-evaluated with the exact
    conditional acceptance (:func:`refine_tuples_exact`), a deterministic
    correction (no QMC sampling noise) that caps the aggregate bias of the
    analytic mask-acceptance approximation to the unrefined remainder. Note
    this omits the (S2-measured, ~0.1-0.3% aggregate) beta2 quadratic-phase
    contribution that the earlier QMC refinement folded in incidentally.
    """
    variables = fwm_tuple_variables(
        freqs, beta0_abs, beta1, beta2, baud_rate, length, target
    )
    refined = 0
    refined_mass = 0.0
    if variables.u0.size:
        coeffs = np.stack(
            [variables.nu_a, variables.nu_b, -variables.nu_c], axis=-1
        )
        est = linear_tuple_estimate(variables.u0, coeffs, variables.d)
        values = est.values.copy()
        widths_sum_all = np.sum(variables.widths, axis=-1)
        # The near regime's analytic mask-acceptance model assumes the
        # conditional law of the mask variable resembles a rescaled 3-uniform
        # sum; it is stably WRONG (not merely under-resolved) when the three
        # leg widths are strongly asymmetric (e.g. two nearly-equal legs plus
        # a near-degenerate third). Near-regime tuples are always refined,
        # rather than gated by REFINE_MAX_WIDTH (which excludes exactly the
        # moderate-to-wide near tuples where this failure mode concentrates).
        # Near the ZDW the near-regime population itself can reach ~1e5-1e6
        # tuples per target; refining literally all of them is both a memory
        # risk (bounded per-call by chunking, but adds up across many tuples)
        # and a multi-minute cost. Mass is heavy-tailed, so capping to the
        # top ``max_near_refine`` by analytic value keeps >99.9% of the near
        # mass refined while bounding worst-case cost independent of how
        # large the near-regime population grows.
        near_idx = np.where(est.regime == 0)[0]
        if near_idx.size > max_near_refine:
            top_near = near_idx[
                np.argpartition(values[near_idx], -max_near_refine)[-max_near_refine:]
            ]
        else:
            top_near = near_idx
        always_refine = top_near
        total_before = float(np.sum(values))
        if n_refine > 0 and values.size:
            narrow = widths_sum_all <= REFINE_MAX_WIDTH
            candidates = np.setdiff1d(
                np.where(narrow)[0], always_refine, assume_unique=False
            )
            k = min(int(n_refine), candidates.size)
            if k > 0:
                top_k = candidates[np.argpartition(values[candidates], -k)[-k:]]
                always_refine = np.union1d(always_refine, top_k)
        refined_mass = (
            float(np.sum(values[always_refine])) / total_before
            if total_before > 0 and always_refine.size
            else 0.0
        )
        if always_refine.size:
            values[always_refine] = refine_tuples_exact(
                variables, always_refine, est.regime
            )
        refined = int(always_refine.size)
        fwm_total = float(np.sum(values))
        counts = (
            int(np.sum(est.regime == 0)),
            int(np.sum(est.regime == 1)),
            int(np.sum(est.regime == 2)),
        )
        far = est.regime == 1
        # Certified envelope bound on the far set: F <= A * 4 / (|u0| - W)^2.
        widths_sum = np.sum(variables.widths, axis=-1)
        gap = np.maximum(np.abs(variables.u0) - widths_sum, 1e-30)
        far_bound = float(np.sum(variables.acceptance[far] * 4.0 / gap[far] ** 2))
    else:
        fwm_total = 0.0
        counts = (0, 0, 0)
        far_bound = 0.0

    _, nu_pairs, _ = xpm_pair_variables(beta1, beta2, baud_rate, length, target)
    xpm_total = float(np.sum(xpm_fast_batch(nu_pairs)))
    return FastTargetResult(
        xpm=xpm_total,
        fwm=fwm_total,
        fwm_tuples=int(variables.u0.size),
        regime_counts=counts,
        far_bound_check=far_bound,
        refined_tuples=refined,
        refined_mass_fraction=refined_mass,
    )


# ---------------------------------------------------------------------------
# Physical prefactor layer
# ---------------------------------------------------------------------------


def physical_nlin_spectrum(
    gamma_t: np.ndarray,
    launch_power_w: float,
    xpm_sum_m2: np.ndarray,
    fwm_sum_m2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert prefactor-free [m^2] sums into physical NLIN variances [W].

    Coefficient counting for the field expansion of |A|^2 A with distinct
    channels, matching the SSFM-validated conventions in
    ``analysis/methods/ssfm_interface.py``:

    - XPM pair (one interferer b, counted once per pair): the cross term
      carries field multiplicity 2, hence variance 4 gamma^2 P_t P_b^2 F.
    - Strict FWM tuple: the term A_a A_b A_c^* carries multiplicity 2, hence
      variance 4 gamma^2 P_a P_b P_c F per unordered {a, b}; the fast path
      enumerates ordered (a, b) pairs, so the coefficient halves to 2.

    Flat launch power and Gaussian symbols are assumed (constellation factor
    unity, as in the SSFM validation pipelines); per-channel powers and
    constellation moments are the documented extension point.
    """
    gamma_t = np.asarray(gamma_t, dtype=float)
    P = float(launch_power_w)
    xpm = 4.0 * gamma_t**2 * P**3 * np.asarray(xpm_sum_m2, dtype=float)
    fwm = 2.0 * gamma_t**2 * P**3 * np.asarray(fwm_sum_m2, dtype=float)
    return xpm, fwm
