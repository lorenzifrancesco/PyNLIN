from types import SimpleNamespace

import numpy as np
import pytest

import pynlin  # noqa: F401  # initialize loguru fallback before importing darnlin
from darnlin.nlin import _interchannel_components
from pynlin.methods.td.fullband_mc import estimate_xpm_n1_local_taylor_mc
from pynlin.methods.td.xhkm_mc import (
    XPMTaylorDispersion,
    _SECTOR_TRANSFORM,
    assert_flat_signal_power_profile,
    estimate_xhkm_sums_mc,
)


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
    assert mc.n1 == pytest.approx(
        mc.n_2pc + mc.n_3pca + mc.n_3pcb + mc.n_4pc, rel=1e-14, abs=1e-14
    )
    assert mc.n2 == pytest.approx(mc.n_2pc + mc.n_3pcb, rel=1e-14, abs=1e-14)
    assert mc.metadata["sector_estimator"] == "aggregate_residuals_covariance_aware"
    assert mc.metadata["sector_estimates_are_direct_modulus_squared_sums"] is False


def test_prefactor_free_mc_rejects_nonflat_profiles_section():
    system = SimpleNamespace(raw_config={"profiles": {"mode": "recompute"}})

    with pytest.raises(ValueError, match="flat signal power profiles"):
        assert_flat_signal_power_profile(system)


def test_prefactor_free_mc_accepts_flat_profiles_section():
    system = SimpleNamespace(raw_config={"profiles": {"mode": "flat"}})

    assert_flat_signal_power_profile(system)


def test_prefactor_free_mc_rejects_nonflat_legacy_pcfm_mode():
    system = SimpleNamespace(
        raw_config={"pcfm": {"run": {"power_profiles_mode": "cached"}}}
    )

    with pytest.raises(ValueError, match="flat signal power profiles"):
        assert_flat_signal_power_profile(system)


def test_local_taylor_dispersion_matches_scalar_beta2_for_every_sector():
    baud_rate = 10e9
    beta2 = 2e-26
    q = 1.5
    carrier = 2.0 * np.pi * q * baud_rate
    dispersion = XPMTaylorDispersion(
        beta0=np.array([0.0, 0.5 * beta2 * carrier**2]),
        beta1=np.array([0.0, beta2 * carrier]),
        beta2=np.array([beta2, beta2]),
        baud_rate=baud_rate,
    )
    kwargs = dict(
        alpha=0.0,
        length=100e3,
        channel_spacing_over_baud=q,
        n_samples=4096,
        seed=1234,
    )

    scalar = estimate_xhkm_sums_mc(beta2=beta2 * baud_rate**2, **kwargs)
    local = estimate_xhkm_sums_mc(beta2=0.0, dispersion=dispersion, **kwargs)

    for name in ("n1", "n2", "n_2pc", "n_3pca", "n_3pcb", "n_4pc"):
        assert getattr(local, name) == pytest.approx(getattr(scalar, name), rel=2e-14)


def test_local_taylor_dispersion_includes_beta4():
    base = XPMTaylorDispersion(
        beta0=np.zeros(2),
        beta1=np.zeros(2),
        beta2=np.zeros(2),
        beta4=np.zeros(2),
        baud_rate=1.0,
    )
    quartic = XPMTaylorDispersion(
        beta0=np.zeros(2),
        beta1=np.zeros(2),
        beta2=np.zeros(2),
        beta4=np.array([24.0, 0.0]),
        baud_rate=1.0,
    )
    kwargs = dict(
        beta2=0.0,
        alpha=0.0,
        length=0.7,
        channel_spacing_over_baud=1.5,
        n_samples=2048,
        seed=42,
    )

    assert estimate_xhkm_sums_mc(dispersion=quartic, **kwargs).n1 != pytest.approx(
        estimate_xhkm_sums_mc(dispersion=base, **kwargs).n1
    )


def test_local_taylor_n1_matches_direct_xpm_with_common_samples():
    rng = np.random.default_rng(19)
    samples = 2.0 * np.pi * (rng.random((4, 2048)) - 0.5)
    dispersion = XPMTaylorDispersion(
        beta0=np.array([0.0, 0.3]),
        beta1=np.array([0.0, 0.2]),
        beta2=np.array([0.04, -0.03]),
        beta3=np.array([0.002, -0.001]),
        beta4=np.array([0.0002, 0.0001]),
        baud_rate=1.0,
    )
    direct, _ = estimate_xpm_n1_local_taylor_mc(
        beta0_offsets=dispersion.beta0,
        beta1=dispersion.beta1,
        beta2=dispersion.beta2,
        beta3=dispersion.beta3,
        beta4=dispersion.beta4,
        baud_rate=dispersion.baud_rate,
        length=0.8,
        target=0,
        interferer=1,
        n_samples=samples.shape[1],
        seed=None,
        random_variables=samples[:3],
    )
    sectors = estimate_xhkm_sums_mc(
        beta2=0.0,
        dispersion=dispersion,
        alpha=0.0,
        length=0.8,
        channel_spacing_over_baud=1.5,
        n_samples=samples.shape[1],
        random_variables=samples,
        seed=19,
    )

    assert sectors.n1 == pytest.approx(direct, rel=2e-14)


def test_zero_mismatch_has_finite_coherent_limit():
    mc = estimate_xhkm_sums_mc(
        beta2=0.0,
        alpha=0.0,
        length=2.0,
        channel_spacing_over_baud=1.5,
        n_samples=1024,
        seed=1,
    )

    assert np.isfinite(mc.n1)
    assert mc.n1 > 0.0


def test_sector_covariance_linear_transform_matches_direct_transform():
    rng = np.random.default_rng(9)
    aggregates = rng.normal(size=(20, 4))
    transformed = aggregates @ _SECTOR_TRANSFORM.T

    expected = np.cov(transformed, rowvar=False, ddof=1)
    propagated = (
        _SECTOR_TRANSFORM
        @ np.cov(aggregates, rowvar=False, ddof=1)
        @ _SECTOR_TRANSFORM.T
    )

    np.testing.assert_allclose(propagated, expected, rtol=1e-14, atol=1e-14)


def test_adaptive_batches_are_deterministic_and_bounded():
    kwargs = dict(
        beta2=0.31,
        alpha=0.17,
        length=1.9,
        channel_spacing_over_baud=1.25,
        n_samples=2048,
        min_samples=512,
        batch_size=256,
        target_relative_stderr=1e-12,
        target_stderr_over_n1=1e-12,
        seed=1234,
    )

    first = estimate_xhkm_sums_mc(**kwargs)
    second = estimate_xhkm_sums_mc(**kwargs)

    assert first == second
    assert first.n_samples == 2048
    assert first.metadata["stop_reason"] == "maximum_budget"
    assert first.metadata["covariance_method"] == "batch_means_linear_transform"
