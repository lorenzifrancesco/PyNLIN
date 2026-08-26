import numpy as np
import pytest

from pynlin.methods.td.fast_nlin import FWMTupleVariables, kernel_abs2
from pynlin.methods.td.gn_analog import (
    center_family_indices,
    gn_cfm_tuple_values,
    gn_ni_from_variables,
    gn_ni_tuple_values,
)


def _variables(
    u_const,
    linear_coeffs,
    support_shift,
    quad_phase_coeffs=None,
    target_quad_phase=0.0,
):
    u_const = np.asarray(u_const, dtype=float)
    linear_coeffs = np.atleast_2d(np.asarray(linear_coeffs, dtype=float))
    support_shift = np.asarray(support_shift, dtype=float)
    n = u_const.size
    if quad_phase_coeffs is None:
        quad_phase_coeffs = np.zeros((n, 3))
    quad_phase_coeffs = np.atleast_2d(np.asarray(quad_phase_coeffs, dtype=float))
    indices = np.arange(n, dtype=np.int32)
    return FWMTupleVariables(
        a=indices,
        b=indices,
        c=indices,
        u0=u_const,
        nu_a=linear_coeffs[:, 0],
        nu_b=linear_coeffs[:, 1],
        nu_c=linear_coeffs[:, 2],
        q_a=quad_phase_coeffs[:, 0],
        q_b=quad_phase_coeffs[:, 1],
        q_c=quad_phase_coeffs[:, 2],
        q_t=float(target_quad_phase),
        d=support_shift,
        acceptance=np.ones(n),
    )


def test_gn_cfm_zero_width_is_constant_phase_limit():
    variables = _variables(
        [-2.0, 0.0, 3.5],
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [-3.0, -3.0, -3.0]],
        [0.3, -0.2, 0.7],
    )

    expected_phase = variables.u0 - variables.nu_c * variables.d
    assert gn_cfm_tuple_values(variables) == pytest.approx(kernel_abs2(expected_phase))

    near_degenerate = _variables([0.8], [[1.0, 1.0 + 1e-12, 1.0]], [0.1])
    assert gn_cfm_tuple_values(near_degenerate)[0] == pytest.approx(
        kernel_abs2(np.array([0.7]))[0], abs=1e-11
    )


def test_gn_cfm_one_effective_leg_matches_direct_quadrature():
    variables = _variables([0.7], [[1.3, -0.4, 1.3]], [0.2])
    nodes, weights = np.polynomial.legendre.leggauss(512)
    expected_phase = variables.u0[0] - variables.nu_c[0] * variables.d[0]
    expected = np.sum(weights * kernel_abs2(expected_phase - 1.7 * np.pi * nodes)) / 2.0

    assert gn_cfm_tuple_values(variables)[0] == pytest.approx(expected, rel=2e-10)


def test_gn_ni_is_symmetric_under_exchange_of_nonconjugated_legs():
    first = _variables([1.1], [[0.4, -0.9, 0.2]], [0.3], [[0.03, -0.07, 0.02]])
    swapped = _variables([1.1], [[-0.9, 0.4, 0.2]], [0.3], [[-0.07, 0.03, 0.02]])

    assert gn_ni_tuple_values(first, n_grid=20) == pytest.approx(
        gn_ni_tuple_values(swapped, n_grid=20), abs=1e-15
    )


def test_gn_ni_xd_zero_ignores_target_quadratic_phase():
    base = _variables([0.2], [[0.4, -0.1, 0.3]], [0.0], target_quad_phase=0.0)
    changed = _variables([0.2], [[0.4, -0.1, 0.3]], [0.0], target_quad_phase=100.0)

    assert gn_ni_tuple_values(base, n_grid=16) == pytest.approx(
        gn_ni_tuple_values(changed, n_grid=16), abs=0.0
    )
    assert gn_ni_tuple_values(base, n_grid=16, n_xd=8) != pytest.approx(
        gn_ni_tuple_values(changed, n_grid=16, n_xd=8)
    )


def test_gn_ni_refinement_replaces_all_values_with_fine_grid():
    variables = _variables(
        [0.0, 1.0],
        [[0.2, 0.4, -0.1], [0.7, -0.3, 0.5]],
        [0.0, 0.4],
        [[0.02, 0.01, -0.03], [0.04, -0.02, 0.01]],
    )
    coarse_values = gn_ni_tuple_values(variables, n_grid=4)
    fine_values = gn_ni_tuple_values(variables, n_grid=12)

    value, coarse_value, refined_count = gn_ni_from_variables(
        variables, n_coarse=4, n_fine=12, n_refine=10
    )

    assert value == pytest.approx(np.sum(fine_values))
    assert coarse_value == pytest.approx(np.sum(coarse_values))
    assert refined_count == 2


def test_center_family_selection_and_control_validation():
    variables = _variables(
        np.zeros(5), np.zeros((5, 3)), [-0.51, -0.49, 0.0, 0.49, 0.51]
    )
    assert np.array_equal(center_family_indices(variables, 1.0), [1, 2, 3])

    with pytest.raises(ValueError, match="family_step_dimensionless"):
        center_family_indices(variables, 0.0)
    with pytest.raises(ValueError, match="n_grid"):
        gn_ni_tuple_values(variables, n_grid=0)
    with pytest.raises(ValueError, match="n_refine"):
        gn_ni_from_variables(variables, n_refine=-1)
    with pytest.raises(ValueError, match="n_xd"):
        gn_ni_tuple_values(variables, n_xd=-1)
