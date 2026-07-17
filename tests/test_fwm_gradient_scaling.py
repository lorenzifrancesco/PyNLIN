from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.standalone_numerical.plot_fwm_mc_gradient_scaling import (
    compute_dataset,
    convergence_metrics,
    convergence_passes,
    constant_beta2_delta_beta,
    enumerate_fwm_tuples,
    phase_matched_domain_intersects,
    plot_convergence,
    plot_dataset,
    save_convergence,
    save_dataset,
    save_metrics,
    surrogate_geometry,
    surrogate_beta2_from_system,
)


def test_constant_beta2_gradient_matches_finite_difference():
    freqs = np.array([190.00e12, 190.03e12, 190.07e12, 190.01e12])
    beta2 = -20e-27
    geometry = surrogate_geometry(freqs, baud_rate=25e9, length=100e3, beta2_const=beta2)
    omega = 2.0 * np.pi * freqs[[1, 2, 3]]
    step = 1e7
    numerical = np.empty(3)
    for axis in range(3):
        plus = omega.copy()
        minus = omega.copy()
        plus[axis] += step
        minus[axis] -= step
        numerical[axis] = (
            constant_beta2_delta_beta(*plus, beta2)
            - constant_beta2_delta_beta(*minus, beta2)
        ) / (2.0 * step)

    np.testing.assert_allclose(geometry.gradient, numerical, rtol=2e-8, atol=0.0)


def test_surrogate_geometry_shift_and_scale_invariants():
    freqs = np.array([190.00e12, 190.03e12, 190.07e12, 190.01e12])
    base = surrogate_geometry(freqs, baud_rate=25e9, length=100e3, beta2_const=-20e-27)
    shifted = surrogate_geometry(freqs + 40e12, baud_rate=25e9, length=100e3, beta2_const=-20e-27)
    doubled = surrogate_geometry(freqs, baud_rate=25e9, length=200e3, beta2_const=-40e-27)

    np.testing.assert_allclose(shifted.gradient, base.gradient)
    assert shifted.x_grad == base.x_grad
    assert shifted.x_phase == base.x_phase
    assert doubled.x_grad == 4.0 * base.x_grad
    assert doubled.x_phase == 4.0 * base.x_phase
    np.testing.assert_allclose(base.x_combined, base.x_phase + base.x_grad)
    np.testing.assert_allclose(base.x_quadrature, np.hypot(base.x_phase, base.x_grad))


def test_zero_gradient_is_handled():
    geometry = surrogate_geometry(
        np.full(4, 190e12), baud_rate=25e9, length=100e3, beta2_const=-20e-27
    )

    assert geometry.gradient_norm == 0.0
    assert geometry.x_grad == 0.0
    assert np.isinf(geometry.curvature)
    assert geometry.phase_matched_domain


def test_surrogate_beta2_prefers_explicit_config():
    system = SimpleNamespace(
        raw_config={"fiber": {"beta2": 12e-27}},
        fiber=SimpleNamespace(beta2=-20e-27),
    )

    assert surrogate_beta2_from_system(system) == 12e-27


def test_phase_matched_domain_intersection():
    assert phase_matched_domain_intersects(
        np.array([190.00e12, 190.20e12, 190.00e12, 190.20e12]), 25e9
    )
    assert not phase_matched_domain_intersects(
        np.array([190.00e12, 190.20e12, 190.40e12, 190.60e12]), 25e9
    )


def test_tuple_enumeration_excludes_degenerate_and_unsupported_terms():
    freqs = np.arange(7, dtype=float) * 10e9
    count, tuples = enumerate_fwm_tuples(
        freqs, baud_rate=12e9, target=3, cap=None, seed=1234
    )

    assert count == len(tuples) > 0
    for a, b, c in tuples:
        assert 3 not in (a, b, c)
        assert len({a, b, c}) == 3
        assert abs(freqs[a] + freqs[b] - freqs[c] - freqs[3]) <= 24e9


def test_targeted_tuple_selection_prefers_resonance_and_proximity():
    freqs = np.arange(9, dtype=float) * 10e9
    kwargs = dict(freqs=freqs, baud_rate=12e9, target=4, cap=6, seed=1234)
    _, reservoir = enumerate_fwm_tuples(**kwargs, selection_mode="reservoir")
    _, resonant = enumerate_fwm_tuples(**kwargs, selection_mode="near_resonant")
    _, nearby = enumerate_fwm_tuples(**kwargs, selection_mode="nearby")
    omega = 2.0 * np.pi * freqs
    beta2_value = -20e-27
    beta0 = 0.5 * beta2_value * omega**2
    beta1 = beta2_value * omega
    beta2 = np.full(freqs.size, beta2_value)
    _, profile_resonant = enumerate_fwm_tuples(
        **kwargs,
        selection_mode="profile_resonant",
        beta0=beta0,
        beta1=beta1,
        beta2=beta2,
    )

    mismatch = lambda item: abs(
        (freqs[item[0]] - freqs[item[2]]) * (freqs[item[1]] - freqs[item[2]])
    )
    span = lambda item: np.ptp(freqs[[4, *item]])
    assert max(map(mismatch, resonant)) <= max(map(mismatch, reservoir))
    assert max(map(mismatch, profile_resonant)) <= max(map(mismatch, reservoir))
    assert max(map(span, nearby)) <= max(map(span, reservoir))


class _DummyWDM:
    def __init__(self, freqs):
        self._freqs = np.asarray(freqs, dtype=float)
        self._band_slices = {"C": slice(0, len(freqs))}

    def frequency_grid(self):
        return self._freqs


def _dummy_system():
    freqs = 190e12 + np.arange(5) * 10e9
    beta2 = -20e-27
    beta1 = beta2 * 2.0 * np.pi * (freqs - freqs[0])
    return SimpleNamespace(
        wdm=_DummyWDM(freqs),
        pulse=SimpleNamespace(baud_rate=100e9),
        fiber_length=1e3,
        fiber=SimpleNamespace(
            beta2=beta2,
            _beta_profile=None,
            _freq_profile=None,
        ),
        beta_grids=lambda freqs=None: (
            np.asarray([beta1[: len(freqs)]], dtype=float),
            np.full((1, len(freqs)), beta2),
        ),
    )


def test_standalone_dataset_is_deterministic_and_symmetric():
    kwargs = dict(
        decimation=1,
        targets=np.array([2]),
        tuple_cap=None,
        n_samples=128,
        n_seeds=1,
        seed=1234,
        beta2_const=-20e-27,
        workers=1,
    )
    first = compute_dataset(_dummy_system(), **kwargs)
    second = compute_dataset(_dummy_system(), **kwargs)

    np.testing.assert_allclose(first["value"], second["value"])
    lookup = {
        (int(a), int(b), int(c)): float(value)
        for a, b, c, value in zip(first["a"], first["b"], first["c"], first["value"], strict=True)
    }
    for (a, b, c), value in lookup.items():
        np.testing.assert_allclose(lookup[(b, a, c)], value, rtol=1e-14)


def test_standalone_serialization_and_plots(tmp_path):
    data = compute_dataset(
        _dummy_system(),
        decimation=1,
        targets=np.array([2]),
        tuple_cap=8,
        n_samples=64,
        n_seeds=1,
        seed=4321,
        beta2_const=-20e-27,
        workers=1,
    )
    npz_path, csv_path = save_dataset(data, tmp_path / "results")
    paths = plot_dataset(data, tmp_path / "plots")
    metrics_path = save_metrics(
        {"test": {"slope": -1.0, "intercept": 0.0, "rmse": 0.1, "r2": 0.9}},
        tmp_path / "results",
    )

    assert npz_path.exists()
    assert csv_path.exists()
    assert metrics_path.exists()
    assert all(path.exists() for path in paths)
    saved = np.load(npz_path)
    for field in (
        "value",
        "stderr",
        "x_grad",
        "x_phase",
        "x_combined",
        "x_quadrature",
        "x_actual_phase",
        "x_actual_grad",
        "x_actual_combined",
        "phase_matched_domain",
    ):
        assert field in saved


def test_quick_convergence_outputs(tmp_path):
    kwargs = dict(
        system=_dummy_system(),
        decimation=1,
        targets=np.array([2]),
        tuple_cap=8,
        n_seeds=1,
        seed=9876,
        beta2_const=-20e-27,
        workers=1,
    )
    datasets = {
        sample_count: compute_dataset(n_samples=sample_count, **kwargs)
        for sample_count in (32, 64)
    }
    summary = convergence_metrics(datasets)
    csv_path = save_convergence(summary, tmp_path)
    plot_path = plot_convergence(summary, tmp_path)

    assert np.array_equal(summary["n_samples"], np.array([32, 64]))
    assert summary["median_relative_change"][-1] == 0.0
    assert summary["p90_relative_change"][-1] == 0.0
    assert summary["fraction_within_2sigma"][-1] == 1.0
    assert csv_path.exists()
    assert plot_path.exists()


def test_quick_convergence_decision_rule():
    summary = {
        "n_samples": np.array([100, 400, 1600]),
        "median_relative_change": np.array([0.1, 0.03, 0.0]),
        "p90_relative_change": np.array([0.3, 0.12, 0.0]),
        "fraction_within_2sigma": np.array([0.6, 0.9, 1.0]),
        "slope_change": np.array([0.2, 0.08, 0.0]),
    }

    assert convergence_passes(summary)
    summary["p90_relative_change"][-2] = 0.25
    assert not convergence_passes(summary)
