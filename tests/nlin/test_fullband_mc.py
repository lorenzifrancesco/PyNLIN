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


def test_fullband_xpm_uses_local_taylor_estimator(monkeypatch):
    freqs = np.array([190e12, 190.1e12])
    beta2 = np.array([[2e-26, 6e-26]])
    seen = []

    def fake_estimate_xpm_n1_local_taylor_mc(**kwargs):
        seen.append((kwargs["target"], kwargs["interferer"], tuple(kwargs["beta2"])))
        return 1.0, 0.0

    system = SimpleNamespace(
        wdm=_DummyWDM(freqs),
        pulse=SimpleNamespace(baud_rate=10e9),
        fiber_length=1.0,
        fiber=SimpleNamespace(gamma=1.0, effective_area=80e-12),
        effective_area=80e-12,
        beta_grids=lambda freqs=None: (np.zeros_like(beta2), beta2),
    )
    monkeypatch.setattr(fullband_mc, "estimate_xpm_n1_local_taylor_mc", fake_estimate_xpm_n1_local_taylor_mc)

    diag = fullband_mc.compute_fullband_prefactor_free_mc(
        system,
        include_xpm=True,
        include_fwm=False,
        xpm_samples=1,
    )

    assert seen == [(0, 1, tuple(beta2[0])), (1, 0, tuple(beta2[0]))]
    assert np.allclose(diag.xpm, [1.0, 1.0])
