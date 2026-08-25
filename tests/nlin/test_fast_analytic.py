import numpy as np
import pytest

from pynlin.methods.td.fast_analytic import (
    analytic_tuple_values,
    envelope_bound,
    select_tube,
    target_analytic_sums,
)
from pynlin.methods.td.fast_nlin import (
    FWMTupleVariables,
    exact_conditional_acceptance,
    linear_tuple_estimate,
    support_acceptance,
    target_fast_sums,
    uniform_sum_density,
    wide_model_masked,
)


def _make_variables(u0, nu, d):
    """Assemble FWMTupleVariables from per-tuple (u0, (nu_a,nu_b,nu_c), d)."""
    u0 = np.asarray(u0, dtype=float)
    nu = np.atleast_2d(np.asarray(nu, dtype=float))
    d = np.asarray(d, dtype=float)
    n = u0.size
    idx = np.zeros(n, dtype=np.int32)
    z = np.zeros(n)
    return FWMTupleVariables(
        a=idx, b=idx, c=idx, u0=u0,
        nu_a=nu[:, 0], nu_b=nu[:, 1], nu_c=-nu[:, 2],
        q_a=z, q_b=z, q_c=z, q_t=0.0, d=d,
        acceptance=support_acceptance(d),
    )


def test_sheet_branch_matches_quadrature_with_conditional_acceptance():
    # Wide equal-split tuple, phase-matched point inside the support: the
    # sheet closed form 2*pi*rho(-u0)*A_cond(0) must match the quadrature.
    W_tot = 6000.0
    coeffs = (W_tot / 3.0 / np.pi) * np.array([[1.0, 1.0, -1.0]])
    # (u0, d, expect_sheet): at |d|=6.41 the mask excludes the phase-matched
    # point (equal split: m = u/kappa, |0 + 6.41| > pi), the value is
    # kernel-tail dominated, and the branch must demote to the fallback.
    cases = ((0.0, 0.0, True), (1500.0, 0.0, True), (1500.0, 3.0, True),
             (0.0, 6.41, False))
    for mu0, d0, expect_sheet in cases:
        v = _make_variables([mu0], coeffs, [d0])
        vals, branch = analytic_tuple_values(v, np.array([0]))
        assert (branch[0] == 0) == expect_sheet
        central, tail = wide_model_masked(
            np.array([mu0]), np.pi * np.abs(coeffs), coeffs, np.array([d0]),
            acceptance_fn=exact_conditional_acceptance,
        )
        ref = central[0] + tail[0] * support_acceptance(np.array([d0]))[0]
        assert vals[0] == pytest.approx(ref, rel=0.03)


def test_sheet_branch_marginal_acceptance_is_wrong():
    # The documented 3/2 lesson: the marginal acceptance A(d) in place of the
    # conditional one is off by ~1.5x for the equal split at d=0 (u and the
    # mask variable are perfectly correlated there).
    W_tot = 6000.0
    coeffs = (W_tot / 3.0 / np.pi) * np.array([[1.0, 1.0, -1.0]])
    v = _make_variables([0.0], coeffs, [0.0])
    vals, _ = analytic_tuple_values(v, np.array([0]))
    rho0 = uniform_sum_density(np.array([0.0]), np.pi * np.abs(coeffs))[0]
    naive = 2.0 * np.pi * rho0 * support_acceptance(np.array([0.0]))[0]
    assert vals[0] / naive == pytest.approx(1.5, rel=0.02)


def test_envelope_bound_dominates_true_values():
    rng = np.random.default_rng(4)
    n = 400
    total = 10 ** rng.uniform(0.0, 3.5, n)
    frac = rng.dirichlet(np.ones(3), n)
    nu = (total[:, None] * frac) / np.pi * np.where(rng.random((n, 3)) < 0.5, -1, 1)
    u0 = 10 ** rng.uniform(-1, 6, n) * np.where(rng.random(n) < 0.5, -1, 1)
    d = rng.uniform(-6.4, 6.4, n)
    v = _make_variables(u0, nu, d)
    coeffs = np.stack([v.nu_a, v.nu_b, -v.nu_c], axis=-1)
    true = linear_tuple_estimate(v.u0, coeffs, v.d).values
    W = np.sum(v.widths, axis=-1)
    bound = envelope_bound(v.u0, W, v.acceptance)
    assert np.all(true <= bound * (1.0 + 1e-9) + 1e-15)


def test_tube_certificate_bounds_discarded_sum_and_eps0_keeps_all():
    rng = np.random.default_rng(9)
    n = 500
    total = 10 ** rng.uniform(0.0, 3.0, n)
    frac = rng.dirichlet(np.ones(3), n)
    nu = (total[:, None] * frac) / np.pi
    u0 = 10 ** rng.uniform(0, 6, n)
    d = rng.uniform(-3, 3, n)
    v = _make_variables(u0, nu, d)
    coeffs = np.stack([v.nu_a, v.nu_b, -v.nu_c], axis=-1)
    true = linear_tuple_estimate(v.u0, coeffs, v.d).values

    keep0, cert0 = select_tube(v, 0.0)
    assert keep0.size == n and cert0 == 0.0

    for eps in (1e-10, 1e-6, 1e-3):
        keep, cert = select_tube(v, eps)
        discarded = np.setdiff1d(np.arange(n), keep)
        assert np.sum(true[discarded]) <= cert * (1.0 + 1e-9) + 1e-15


def test_target_analytic_sums_matches_reference_on_synthetic_grid():
    # Small uniform grid with quadratic dispersion: the eps=0 analytic path
    # must agree with the reference quadrature+refinement pipeline.
    n_ch = 48
    baud = 10e9
    spacing = 25e9
    length = 80e3
    freqs = 190e12 + spacing * np.arange(n_ch)
    beta2 = np.full(n_ch, 2e-27)
    omega = 2.0 * np.pi * freqs
    beta1 = beta2[0] * (omega - omega[0])
    beta0_abs = 0.5 * beta2[0] * (omega - omega[0]) ** 2
    t = n_ch // 2

    ref = target_fast_sums(freqs, beta0_abs, beta1, beta2, baud, length, t)
    ana = target_analytic_sums(
        freqs, beta0_abs, beta1, beta2, baud, length, t, epsilon=0.0
    )
    assert ana.fwm_tuples_kept == ana.fwm_tuples_total == ref.fwm_tuples
    assert ana.xpm == pytest.approx(ref.xpm, rel=1e-12)
    assert ana.fwm == pytest.approx(ref.fwm, rel=0.05)

    # A finite eps keeps a subset, and kept sum + certificate covers eps=0 sum.
    ana_eps = target_analytic_sums(
        freqs, beta0_abs, beta1, beta2, baud, length, t, epsilon=1e-6
    )
    assert ana_eps.fwm_tuples_kept < ana_eps.fwm_tuples_total
    assert ana.fwm <= ana_eps.fwm + ana_eps.certificate * (1.0 + 1e-9)
