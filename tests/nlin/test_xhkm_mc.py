from types import SimpleNamespace

import numpy as np
import pytest

import pynlin  # noqa: F401  # initialize loguru fallback before importing darnlin
from darnlin.nlin import _interchannel_components
from pynlin.methods.td.xhkm_mc import assert_flat_signal_power_profile, estimate_xhkm_sums_mc


def test_prefactor_free_mc_matches_dar_interchannel_components_with_fixed_samples():
    n_samples = 512
    rng = np.random.default_rng(1234)
    R = 2.0 * np.pi * (rng.random((4, n_samples)) - 0.5)
    gamma = 1.7
    power = 2.3
    prefactor = 4.0 * gamma**2 * power**3
    beta2 = 0.31
    alpha = 0.17
    length = 1.9
    phase_delay = 0.2
    q = 1.25
    nspan = 2

    mc = estimate_xhkm_sums_mc(
        beta2=beta2,
        alpha=alpha,
        length=length,
        phase_delay=phase_delay,
        channel_spacing_over_baud=q,
        nspan=nspan,
        n_samples=n_samples,
        random_variables=R,
    )
    chi1, chi2 = _interchannel_components(
        gamma,
        beta2,
        alpha,
        nspan,
        length,
        phase_delay,
        power,
        n_samples,
        q,
        R=R,
    )

    assert mc.n1 == pytest.approx(chi1 / prefactor, rel=1e-14)
    assert mc.n2 == pytest.approx(chi2 / prefactor, rel=1e-14)
    assert mc.n1_stderr >= 0.0
    assert mc.n2_stderr >= 0.0


def test_prefactor_free_mc_sector_estimates_are_finite_and_consistent():
    mc = estimate_xhkm_sums_mc(
        beta2=0.31,
        alpha=0.17,
        length=1.9,
        phase_delay=0.2,
        channel_spacing_over_baud=1.25,
        nspan=2,
        n_samples=2048,
        seed=1234,
    )

    values = [
        mc.n_2pc,
        mc.n_3pca,
        mc.n_3pcb,
        mc.n_3pc_total,
        mc.n_4pc,
        mc.n_2pc_stderr,
        mc.n_3pca_stderr,
        mc.n_3pcb_stderr,
        mc.n_3pc_total_stderr,
        mc.n_4pc_stderr,
    ]
    assert np.all(np.isfinite(values))
    assert mc.n_3pc_total == pytest.approx(mc.n_3pca + mc.n_3pcb, rel=1e-14, abs=1e-14)
    assert mc.n1 == pytest.approx(mc.n_2pc + mc.n_3pca + mc.n_3pcb + mc.n_4pc, rel=1e-14, abs=1e-14)
    assert mc.n2 == pytest.approx(mc.n_2pc + mc.n_3pcb, rel=1e-14, abs=1e-14)


def test_prefactor_free_mc_rejects_nonflat_profiles_section():
    system = SimpleNamespace(raw_config={"profiles": {"mode": "recompute"}})

    with pytest.raises(ValueError, match="flat signal power profiles"):
        assert_flat_signal_power_profile(system)


def test_prefactor_free_mc_accepts_flat_profiles_section():
    system = SimpleNamespace(raw_config={"profiles": {"mode": "flat"}})

    assert_flat_signal_power_profile(system)


def test_prefactor_free_mc_rejects_nonflat_legacy_pcfm_mode():
    system = SimpleNamespace(raw_config={"pcfm": {"run": {"power_profiles_mode": "cached"}}})

    with pytest.raises(ValueError, match="flat signal power profiles"):
        assert_flat_signal_power_profile(system)
