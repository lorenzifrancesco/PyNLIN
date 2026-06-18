from types import SimpleNamespace

import numpy as np
import pytest

from pynlin.methods.td.estimation import lo_correction_uwb
from pynlin.methods.td.estimation.raman_integrals_uwb import load_fB, raman_integral
from pynlin.methods.td.estimator import get_kappa2_matrix_uwb, total_nlin_uwb


class _DummyWDM:
    def __init__(self, freqs):
        self._freqs = np.asarray(freqs, dtype=float)

    def frequency_grid(self):
        return self._freqs


def _make_system(freqs=(193.1e12, 193.2e12), length=1.0, baud_rate=1.0):
    fiber = SimpleNamespace(length=length, effective_area=80e-12)
    return SimpleNamespace(
        n_modes=1,
        fiber=fiber,
        effective_area=fiber.effective_area,
        fiber_length=length,
        pulse=SimpleNamespace(baud_rate=baud_rate),
        baud_rate=baud_rate,
        launch_power=0.0,
        raw_config={},
        wdm=_DummyWDM(freqs),
    )


def test_uwb_raman_integral_constant_profile_is_unitary():
    system = _make_system(length=80e3)
    fB = np.ones(17)

    assert raman_integral(system, "LO", fB) == pytest.approx(1.0)
    assert raman_integral(system, "HI", fB) == pytest.approx(1.0)


def test_uwb_load_fB_accepts_npz_signal_sol_channel_z_layout(tmp_path):
    system = _make_system(freqs=(193.1e12, 193.2e12), length=10.0)
    z = np.linspace(0.0, system.fiber_length, 8)
    signal_sol = np.vstack((np.linspace(1.0, 2.0, z.size), np.linspace(2.0, 6.0, z.size)))
    profile_path = tmp_path / "profile.npz"
    np.savez(profile_path, signal_sol=signal_sol, z=z)

    fB, fB_min, fB_max, _, _ = load_fB(system, profile_path=profile_path)

    expected = signal_sol / signal_sol[:, :1]
    assert fB.shape == (z.size, 1, 2)
    assert np.allclose(fB[:, 0, :], expected.T)
    assert np.allclose(fB_min, np.min(expected.T, axis=1))
    assert np.allclose(fB_max, np.max(expected.T, axis=1))


def test_uwb_lo_lookup_returns_min_then_max(monkeypatch):
    system = _make_system(length=100.0, baud_rate=1.0)
    grid_min = np.ones((20, 20), dtype=float)
    grid_max = np.full((20, 20), 2.0, dtype=float)

    def fB_min(z):
        return np.ones_like(np.asarray(z, dtype=float))

    def fB_max(z):
        return np.ones_like(np.asarray(z, dtype=float))

    class _LoadedGrid:
        def __enter__(self):
            return {"s2b_lo_corr_min": grid_min, "s2b_lo_corr_max": grid_max}

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        lo_correction_uwb,
        "load_fB",
        lambda *args, **kwargs: (None, None, None, fB_min, fB_max),
    )
    monkeypatch.setattr(lo_correction_uwb.os.path, "exists", lambda filename: True)
    monkeypatch.setattr(lo_correction_uwb.np, "load", lambda *args, **kwargs: _LoadedGrid())

    lookup_min, lookup_max = lo_correction_uwb.build_lookup_integral_table_with_raman(system)

    assert np.asarray(lookup_min(0.1, 0.1)).item() == pytest.approx(1.0)
    assert np.asarray(lookup_max(0.1, 0.1)).item() == pytest.approx(2.0)


def test_uwb_total_nlin_scales_with_interferer_power_squared():
    system = _make_system(freqs=(193.1e12, 193.2e12), length=1.0, baud_rate=1.0)
    collision_coeffs = np.ones((1, 2, 1, 2), dtype=float)

    launch_a = np.array([1e-3, 1e-3])
    launch_b = np.array([1e-3, 2e-3])

    nlin_a = total_nlin_uwb(
        system,
        collision_coeffs,
        launch_powers_w=launch_a,
        exclude_self_channel=True,
    )
    nlin_b = total_nlin_uwb(
        system,
        collision_coeffs,
        launch_powers_w=launch_b,
        exclude_self_channel=True,
    )

    assert nlin_a[0, 0] > 0.0
    assert nlin_b[0, 0] / nlin_a[0, 0] == pytest.approx(4.0)


def test_uwb_total_nlin_recomputes_unversioned_cache(tmp_path):
    system = _make_system(freqs=(193.1e12, 193.2e12), length=1.0, baud_rate=1.0)
    collision_coeffs = np.ones((1, 2, 1, 2), dtype=float)
    cache_path = tmp_path / "nlin.npy"
    np.save(cache_path, np.zeros((1, 2), dtype=float))

    nlin = total_nlin_uwb(
        system,
        collision_coeffs,
        launch_powers_w=np.array([1e-3, 1e-3]),
        exclude_self_channel=True,
        cache_path=cache_path,
    )

    assert nlin[0, 0] > 0.0
    with np.load(cache_path) as cached:
        assert cached["cache_version"].item() == 2
        assert np.allclose(cached["nlin"], nlin)


def test_uwb_smf_kappa_uses_manakov_fallback_when_csv_is_missing():
    system = _make_system()

    kappa2 = get_kappa2_matrix_uwb(system, use_kappa=True, use_x_mode=False)

    assert kappa2.shape == (1, 1)
    assert kappa2[0, 0] == pytest.approx((8.0 / 9.0) ** 2)
