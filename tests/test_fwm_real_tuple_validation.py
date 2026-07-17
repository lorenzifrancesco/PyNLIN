from pathlib import Path
import sys

import numpy as np
import pytest
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.standalone_numerical.validate_fwm_mc_real_tuples import (
    _polynomial_channels,
    estimate_xpm_sector_ensemble,
    load_ssfm_xpm_cache,
    physical_tuple_coordinates,
    profile_sweep_targets,
    tuple_sequences,
)
from analysis.standalone_numerical.validate_ssfm_xpm_spectrum import (
    local_taylor_coefficients,
    select_spectrum_target_indices,
)
from pynlin.methods.td.fwm_kernel import FWMChannels
from pynlin.methods.td.xhkm_mc import XPMTaylorDispersion


def test_tuple_sequences_are_in_band_and_carrier_conserving():
    band_slices = {"A": slice(0, 20), "B": slice(20, 50)}
    sequences = tuple_sequences(band_slices, translation_step=4, max_span=3)

    assert {entry[0] for entry in sequences} == {"translation", "span"}
    for _, band, _, (d, a, b, c) in sequences:
        band_slice = band_slices[band]
        assert all(
            band_slice.start <= index < band_slice.stop for index in (d, a, b, c)
        )
        assert a + b - c - d == 0
        assert d not in (a, b, c)
        assert len({a, b, c}) == 3


def test_physical_coordinates_are_derived_without_modifying_channels():
    baud_rate = 2.0
    length = 5.0
    channels = FWMChannels(
        omega_a=-1.0,
        omega_b=2.0,
        omega_c=1.0,
        omega_d=0.0,
        beta0_a=3.0,
        beta0_b=4.0,
        beta0_c=2.0,
        beta0_d=1.0,
        beta1_a=1.0,
        beta1_b=2.0,
        beta1_c=3.0,
        beta1_d=0.5,
        gvd_a=0.0,
        gvd_b=0.0,
        gvd_c=0.0,
        gvd_d=0.0,
    )

    delta_beta, gradient_norm, x_grad, mu = physical_tuple_coordinates(
        channels, baud_rate=baud_rate, length=length
    )

    expected_norm = np.linalg.norm([0.5, 1.5, -2.5])
    assert delta_beta == 4.0
    np.testing.assert_allclose(gradient_norm, expected_norm)
    np.testing.assert_allclose(x_grad, length * baud_rate * expected_norm)
    np.testing.assert_allclose(mu, delta_beta / (baud_rate * expected_norm))
    assert channels.beta0_a == 3.0


def test_symmetric_degenerate_tuple_is_beta4_sensitive_at_zdw():
    delta = 0.3
    freqs = np.array([-delta, 0.0, 0.0, delta]) / (2.0 * np.pi)
    coefficients = (0.0, 1.0, 0.0, 5.0, 7.0)

    cubic = _polynomial_channels(
        freqs, reference_frequency=0.0, coefficients=coefficients, order=3
    )
    quartic = _polynomial_channels(
        freqs, reference_frequency=0.0, coefficients=coefficients, order=4
    )

    np.testing.assert_allclose(cubic.delta_beta0, 0.0, atol=1e-15)
    np.testing.assert_allclose(quartic.delta_beta0, -7.0 * delta**4 / 12.0)


def test_profile_sweep_keeps_configured_shifts_inside_profile():
    class Fiber:
        _freq_profile = np.array([100.2, 120.8])

    class WDM:
        spacing = 1.0

    class DummySystem:
        fiber = Fiber()
        wdm = WDM()

    targets = profile_sweep_targets(DummySystem(), step=2, xpm_shift=3, fwm_shift=3)

    np.testing.assert_array_equal(targets, np.arange(101.0, 115.0, 2.0))
    assert targets[0] >= np.min(DummySystem.fiber._freq_profile)
    assert targets[-1] + 6.0 <= np.max(DummySystem.fiber._freq_profile)


def test_adaptive_sector_estimate_classifies_resolution(monkeypatch):
    def fake_estimator(**kwargs):
        return SimpleNamespace(
            n_2pc=10.0,
            n_3pca=2.0,
            n_3pcb=0.01,
            n_4pc=-0.1,
            n1=13.0,
            n_2pc_stderr=0.1,
            n_3pca_stderr=0.1,
            n_3pcb_stderr=0.1,
            n_4pc_stderr=0.1,
            n1_stderr=0.1,
            n_samples=500,
            metadata={"stop_reason": "maximum_budget"},
        )

    monkeypatch.setattr(
        "analysis.standalone_numerical.validate_fwm_mc_real_tuples.estimate_xhkm_sums_mc",
        fake_estimator,
    )
    result = estimate_xpm_sector_ensemble(
        dispersion=XPMTaylorDispersion(
            beta0=np.zeros(2),
            beta1=np.zeros(2),
            beta2=np.zeros(2),
            baud_rate=1.0,
        ),
        alpha=0.0,
        length=1.0,
        spacing_over_baud=1.0,
        min_samples=100,
        max_samples=500,
        batch_size=100,
        seed=1,
        sigma_threshold=3.0,
        max_relative_error=0.25,
        max_stderr_over_n1=0.001,
    )

    assert result["n_2pc_resolved"] is True
    assert result["n_3pca_resolved"] is True
    assert result["n_3pcb_resolved"] is False
    assert result["n_4pc_resolved"] is False
    assert result["n_samples"] == 500.0
    assert result["stop_reason"] == "maximum_budget"


def test_ssfm_spectrum_targets_are_real_uniform_channel_pairs():
    system = SimpleNamespace(
        wdm=SimpleNamespace(frequency_grid=lambda: np.arange(100.0, 111.0)),
        fiber=SimpleNamespace(_freq_profile=np.array([102.0, 109.0])),
    )

    targets = select_spectrum_target_indices(system, target_count=3, interferer_shift=2)

    np.testing.assert_array_equal(targets, [2, 4, 7])
    assert np.all(targets + 2 < 11)

    descending = SimpleNamespace(
        wdm=SimpleNamespace(frequency_grid=lambda: np.arange(110.0, 99.0, -1.0)),
        fiber=SimpleNamespace(_freq_profile=np.array([102.0, 109.0])),
    )
    descending_targets = select_spectrum_target_indices(
        descending, target_count=3, interferer_shift=2
    )
    assert np.all(
        descending.wdm.frequency_grid()[descending_targets - 2]
        > descending.wdm.frequency_grid()[descending_targets]
    )


def test_ssfm_xpm_cache_rejects_incompatible_shift(tmp_path):
    path = tmp_path / "ssfm.npz"
    np.savez(
        path,
        target_frequency_hz=np.array([100.0]),
        n1_ssfm_m2=np.array([2.0]),
        n1_ssfm_stderr_m2=np.array([0.1]),
        fiber_length_m=np.array(10.0),
        baud_rate_hz=np.array(2.0),
        xpm_shift_channels=np.array(2),
    )

    loaded = load_ssfm_xpm_cache(
        path,
        expected_length=10.0,
        expected_baud_rate=2.0,
        expected_xpm_shift=2,
    )
    np.testing.assert_array_equal(loaded["n1_ssfm_m2"], [2.0])
    with pytest.raises(ValueError, match="channel shift"):
        load_ssfm_xpm_cache(
            path,
            expected_length=10.0,
            expected_baud_rate=2.0,
            expected_xpm_shift=1,
        )


def test_local_taylor_coefficients_include_higher_orders():
    class Polynomial:
        def derivative(self, order):
            return lambda omega: {2: 2.0, 3: 3.0, 4: 4.0}[order]

    system = SimpleNamespace(
        fiber=SimpleNamespace(beta_spline_omega=lambda **kwargs: Polynomial())
    )
    target = 10.0
    interferer = target + 0.5 / (2.0 * np.pi)

    ssfm_betas, grids = local_taylor_coefficients(system, target, interferer)

    beta0, beta1, beta2, beta3, beta4 = grids
    np.testing.assert_allclose(ssfm_betas, [2e24, 3e36, 4e48])
    np.testing.assert_allclose(beta0, [0.0, 1.0 * 0.5**2 + 0.5 * 0.5**3 + 4.0 * 0.5**4 / 24.0])
    np.testing.assert_allclose(beta1, [0.0, 2.0 * 0.5 + 1.5 * 0.5**2 + 4.0 * 0.5**3 / 6.0])
    np.testing.assert_allclose(beta2, [2.0, 2.0 + 3.0 * 0.5 + 2.0 * 0.5**2])
    np.testing.assert_allclose(beta3, [3.0, 5.0])
    np.testing.assert_allclose(beta4, [4.0, 4.0])
