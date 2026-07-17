from types import SimpleNamespace

import numpy as np
import pytest

from pynlin.methods.td import fullband_mc
from pynlin.methods.td.xhkm_mc import estimate_xhkm_sums_mc


class _DummyWDM:
    def __init__(self, freqs):
        self._freqs = np.asarray(freqs, dtype=float)

    def frequency_grid(self):
        return self._freqs


def test_gamma_grid_scales_with_frequency_and_effective_area():
    fiber = SimpleNamespace(
        gamma=2.0,
        effective_area=80e-12,
        effective_area_at=lambda wavelength: 80e-12 if wavelength > 1.54e-6 else 40e-12,
    )
    system = SimpleNamespace(fiber=fiber, effective_area=80e-12)
    freqs = np.array([190e12, 200e12])

    gamma = fullband_mc.gamma_grid(system, freqs)

    freq_ref = np.mean(freqs)
    assert gamma[0] == pytest.approx(2.0 * (freqs[0] / freq_ref))
    assert gamma[1] == pytest.approx(2.0 * (freqs[1] / freq_ref) * 2.0)


def test_decimated_frequency_grid_strides_across_full_band():
    freqs = np.arange(10, dtype=float)
    system = SimpleNamespace(wdm=_DummyWDM(freqs))

    indices, decimated = fullband_mc.decimated_frequency_grid(system, factor=4)

    assert np.array_equal(indices, np.array([0, 4, 8]))
    assert np.array_equal(decimated, freqs[[0, 4, 8]])


def test_local_taylor_xpm_matches_scalar_dar_for_constant_beta2():
    baud_rate = 10e9
    length = 100e3
    beta2 = 2e-26
    q = 1.5
    delta_omega = 2.0 * np.pi * q * baud_rate
    n_samples = 4096
    seed = 1234

    local, _ = fullband_mc.estimate_xpm_n1_local_taylor_mc(
        beta0_offsets=np.array([0.0, 0.5 * beta2 * delta_omega**2]),
        beta1=np.array([0.0, beta2 * delta_omega]),
        beta2=np.array([beta2, beta2]),
        baud_rate=baud_rate,
        length=length,
        target=0,
        interferer=1,
        n_samples=n_samples,
        seed=seed,
    )
    scalar = estimate_xhkm_sums_mc(
        beta2=beta2 / (1.0 / baud_rate) ** 2,
        alpha=0.0,
        length=length,
        channel_spacing_over_baud=q,
        n_samples=n_samples,
        seed=seed,
    )

    assert local == pytest.approx(scalar.n1, rel=1e-14)


def test_support_pruned_fwm_tuples_exclude_degenerate_terms():
    freqs = np.arange(5, dtype=float) * 10e9

    count, tuples = fullband_mc.collect_support_pruned_fwm_tuples(
        freqs,
        baud_rate=100e9,
        target=1,
        selection_mode="reservoir",
    )

    assert count == 24
    assert len(tuples) == 24
    for a, b, c in tuples:
        assert fullband_mc.is_nondegenerate_fwm_tuple(1, a, b, c)


def test_systematic_fwm_tuple_sample_spans_support():
    freqs = np.arange(8, dtype=float) * 10e9

    count, tuples = fullband_mc.collect_support_pruned_fwm_tuples(
        freqs,
        baud_rate=100e9,
        target=3,
        max_tuples=5,
        seed=1234,
        selection_mode="systematic",
    )

    assert count > len(tuples)
    assert len(tuples) == 5
    assert len({a for a, _, _ in tuples}) > 1
    assert len({b for _, b, _ in tuples}) > 1
    for a, b, c in tuples:
        assert fullband_mc.is_nondegenerate_fwm_tuple(3, a, b, c)


def test_fullband_xpm_uses_calculation_interferer_grid(monkeypatch):
    freqs_full = np.array([190e12, 190.05e12, 190.1e12, 190.15e12])
    beta2_dec = np.array([[2e-26, 6e-26]])
    beta2_full = np.array([[1e-26, 3e-26, 6e-26, 9e-26]])

    propagator_args = []

    def fake_propagator_abs2(db, length, alpha):
        propagator_args.append(db)
        return db * 0.0

    def beta_grids(freqs=None):
        if freqs is None or len(freqs) == 2:
            return (np.zeros_like(beta2_dec), beta2_dec)
        return (np.zeros_like(beta2_full), beta2_full)

    system = SimpleNamespace(
        wdm=_DummyWDM(freqs_full),
        pulse=SimpleNamespace(baud_rate=10e9),
        fiber_length=1.0,
        fiber=SimpleNamespace(gamma=1.0, effective_area=80e-12),
        effective_area=80e-12,
        beta_grids=beta_grids,
    )

    monkeypatch.setattr(fullband_mc, "_propagator_abs2", fake_propagator_abs2)

    diag = fullband_mc.compute_fullband_prefactor_free_mc(
        system,
        include_xpm=True,
        include_fwm=False,
        xpm_samples=1,
        decimation=2,
    )

    assert len(propagator_args) == 2, (
        f"Expected 2 _propagator_abs2 calls (2 targets × 1 calculation-grid interferer), got {len(propagator_args)}"
    )
    assert np.allclose(diag.xpm, [0.0, 0.0])
    assert diag.metadata["xpm_grid"] == "calculation"


def test_fullband_mc_parallel_matches_sequential():
    freqs = np.array([190.0e12, 190.05e12, 190.1e12, 190.15e12])
    beta2 = np.full((1, freqs.size), 2e-26)
    system = SimpleNamespace(
        wdm=_DummyWDM(freqs),
        pulse=SimpleNamespace(baud_rate=10e9),
        fiber_length=1.0,
        fiber=SimpleNamespace(gamma=1.0, effective_area=80e-12),
        effective_area=80e-12,
        beta_grids=lambda freqs=None: (np.zeros((1, len(freqs))), beta2[:, : len(freqs)]),
    )

    kwargs = dict(
        include_xpm=True,
        include_fwm=True,
        xpm_samples=8,
        fwm_samples=4,
        fwm_frequency_samples=3,
        max_fwm_tuples_per_target=4,
        seed=1234,
    )
    sequential = fullband_mc.compute_fullband_prefactor_free_mc(system, n_workers=1, **kwargs)
    parallel = fullband_mc.compute_fullband_prefactor_free_mc(system, n_workers=2, **kwargs)

    assert np.allclose(parallel.xpm, sequential.xpm)
    assert np.allclose(parallel.fwm, sequential.fwm)
    assert np.array_equal(parallel.fwm_support_count, sequential.fwm_support_count)
    assert np.array_equal(parallel.fwm_tuple_count, sequential.fwm_tuple_count)
