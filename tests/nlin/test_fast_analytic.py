import numpy as np
import pytest

from pynlin.methods.td.fast_analytic import (
    analytic_tuple_values,
    envelope_bound,
    masked_linear_phase_outer_interval,
    select_tube,
    target_analytic_sums,
)
from pynlin.methods.td.fast_nlin import (
    FWMTupleVariables,
    exact_conditional_acceptance,
    kernel_abs2,
    kernel_upper_envelope,
    linear_tuple_estimate,
    support_acceptance,
    target_fast_sums,
    uniform_sum_density,
    wide_model_masked,
)


def _make_variables(u0, nu, d, q=None, q_t=0.0):
    """Assemble FWMTupleVariables from per-tuple (u0, (nu_a,nu_b,nu_c), d)."""
    u0 = np.asarray(u0, dtype=float)
    nu = np.atleast_2d(np.asarray(nu, dtype=float))
    d = np.asarray(d, dtype=float)
    n = u0.size
    idx = np.zeros(n, dtype=np.int32)
    q = np.zeros((n, 3)) if q is None else np.atleast_2d(np.asarray(q, dtype=float))
    return FWMTupleVariables(
        a=idx,
        b=idx,
        c=idx,
        u0=u0,
        nu_a=nu[:, 0],
        nu_b=nu[:, 1],
        nu_c=-nu[:, 2],
        q_a=q[:, 0],
        q_b=q[:, 1],
        q_c=q[:, 2],
        q_t=float(q_t),
        d=d,
        acceptance=support_acceptance(d),
    )


def test_upper_envelope_fringe_policy_dominates_resolved_dispatch_values():
    u0 = np.array([3.0, 5000.0, 1000.0])
    coeffs = np.array(
        [
            [0.4, -0.2, 0.1],
            [0.4, -0.2, 0.1],
            [400.0, -350.0, 300.0],
        ]
    )
    d = np.zeros(3)

    resolved = linear_tuple_estimate(u0, coeffs, d)
    envelope = linear_tuple_estimate(u0, coeffs, d, fringe_policy="upper_envelope")

    assert np.array_equal(resolved.regime, np.array([0, 1, 2]))
    assert np.array_equal(envelope.regime, resolved.regime)
    assert np.all(envelope.values >= resolved.values)
    assert np.any(envelope.values > resolved.values)
    assert np.array_equal(
        linear_tuple_estimate(u0, coeffs, d, fringe_policy="resolved").values,
        resolved.values,
    )


def test_upper_envelope_kernel_and_policy_validation():
    u = np.array([0.0, 0.5, 2.0, 2.0 * np.pi, 20.0])
    assert np.all(kernel_upper_envelope(u) >= kernel_abs2(u))
    with pytest.raises(ValueError, match="fringe_policy"):
        linear_tuple_estimate(
            np.array([1.0]),
            np.ones((1, 3)),
            np.zeros(1),
            fringe_policy="invalid",
        )


def test_sheet_branch_matches_quadrature_with_conditional_acceptance():
    # Wide equal-split tuple, phase-matched point inside the support: the
    # sheet closed form 2*pi*rho(-u0)*A_cond(0) must match the quadrature.
    W_tot = 6000.0
    coeffs = (W_tot / 3.0 / np.pi) * np.array([[1.0, 1.0, -1.0]])
    # (u0, d, expect_sheet): at |d|=6.41 the mask excludes the phase-matched
    # point (equal split: m = u/kappa, |0 + 6.41| > pi), the value is
    # kernel-tail dominated, and the branch must demote to the fallback.
    cases = (
        (0.0, 0.0, True),
        (1500.0, 0.0, True),
        (1500.0, 3.0, True),
        (0.0, 6.41, False),
    )
    for mu0, d0, expect_sheet in cases:
        v = _make_variables([mu0], coeffs, [d0])
        vals, branch = analytic_tuple_values(v, np.array([0]))
        assert (branch[0] == 0) == expect_sheet
        central, tail = wide_model_masked(
            np.array([mu0]),
            np.pi * np.abs(coeffs),
            coeffs,
            np.array([d0]),
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


def test_masked_reachable_interval_tracks_shift_for_aligned_coefficients():
    # c_u = 20 c_m makes the projection interval exact.  Reversing d moves
    # the accepted phase interval across zero even though (u0, W, A) match.
    coeffs = np.array([[20.0, 20.0, -20.0], [20.0, 20.0, -20.0]])
    v = _make_variables([40.0, 40.0], coeffs, [2.0, -2.0])
    lower, upper = masked_linear_phase_outer_interval(v)

    assert lower == pytest.approx([-20.0 * np.pi, 80.0 - 20.0 * np.pi])
    assert upper == pytest.approx([20.0 * np.pi, 80.0 + 20.0 * np.pi])
    keep, _ = select_tube(v, epsilon=0.05)
    assert np.array_equal(keep, np.array([0]))


def test_masked_reachable_interval_handles_aligned_anti_aligned_and_degenerate():
    coeffs = np.array(
        [
            [10.0, 10.0, -10.0],  # aligned: mask narrows 3*pi*k to pi*k
            [10.0, -10.0, 0.0],  # anti-aligned: kappa=0, no projection narrowing
            [0.0, 0.0, 0.0],  # degenerate: phase is the single point u0
        ]
    )
    v = _make_variables([50.0, 50.0, 7.0], coeffs, [0.0, 0.0, 1.0])
    lower, upper = masked_linear_phase_outer_interval(v)

    assert lower[0] == pytest.approx(50.0 - 10.0 * np.pi)
    assert upper[0] == pytest.approx(50.0 + 10.0 * np.pi)
    assert lower[1] == pytest.approx(50.0 - 20.0 * np.pi)
    assert upper[1] == pytest.approx(50.0 + 20.0 * np.pi)
    assert lower[2] == upper[2] == pytest.approx(7.0)

    keep, _ = select_tube(v, epsilon=0.06)
    assert np.array_equal(keep, np.array([1]))


def test_quadratic_padding_expands_reachable_interval_for_selection():
    # The linear aligned interval misses zero by 5 rad.  A conservative
    # P_q=6 rad reaches zero, so the padded bound must retain the tuple.
    kappa = 10.0
    u0 = np.pi * kappa + 5.0
    coeffs = np.array([[kappa, kappa, -kappa]])
    linear = _make_variables([u0], coeffs, [0.0])
    quadratic = _make_variables([u0], coeffs, [0.0], q=[[6.0 / np.pi**2, 0.0, 0.0]])

    lower, upper = masked_linear_phase_outer_interval(quadratic)
    assert lower[0] == pytest.approx(5.0)
    assert upper[0] == pytest.approx(2.0 * np.pi * kappa + 5.0)
    assert select_tube(linear, epsilon=0.2)[0].size == 0
    assert np.array_equal(select_tube(quadratic, epsilon=0.2)[0], np.array([0]))


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
    envelope = target_fast_sums(
        freqs,
        beta0_abs,
        beta1,
        beta2,
        baud,
        length,
        t,
        fringe_policy="upper_envelope",
    )
    ana = target_analytic_sums(
        freqs, beta0_abs, beta1, beta2, baud, length, t, epsilon=0.0
    )
    assert ana.fwm_tuples_kept == ana.fwm_tuples_total == ref.fwm_tuples
    assert ana.xpm == pytest.approx(ref.xpm, rel=1e-12)
    assert ana.fwm == pytest.approx(ref.fwm, rel=0.05)
    assert envelope.xpm == ref.xpm
    assert envelope.fwm >= ref.fwm

    # A finite eps keeps a subset, and kept sum + certificate covers eps=0 sum.
    ana_eps = target_analytic_sums(
        freqs, beta0_abs, beta1, beta2, baud, length, t, epsilon=1e-6
    )
    assert ana_eps.fwm_tuples_kept < ana_eps.fwm_tuples_total
    assert ana.fwm <= ana_eps.fwm + ana_eps.certificate * (1.0 + 1e-9)
