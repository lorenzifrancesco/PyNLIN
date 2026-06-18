import numpy as np

from pynlin.fiber import SMFiber
from scipy.constants import c

from pynlin.methods.pcfm import (
    N2_SIO2,
    PcfmConfig,
    _aeff_array,
    _beta2_array,
    _beta2_eff,
    _to_per_channel_power,
    _to_per_polarization_power,
    compute_pcfm_nlin,
    compute_sci_numeric,
    fit_spp_polynomials,
    load_signal_profiles,
    normalize_spp,
)
from pynlin.pulses import GaussianPulse
from pynlin.system import System
from pynlin.wdm import Amplification, RegularWDM


def _make_constant_profile(tmp_path, n_channels: int, length: float, n_z: int = 20):
    z = np.linspace(0.0, length, n_z)
    signal_sol = np.ones((n_channels, z.size), dtype=float)
    profile_path = tmp_path / "profile.npz"
    np.savez(profile_path, signal_sol=signal_sol, z=z)
    return profile_path


def _make_minimal_system(n_channels: int, spacing_hz: float, center_frequency: float, length: float):
    fiber = SMFiber(beta2=20e-27, effective_area=80e-12, length=length)
    wdm = RegularWDM(spacing=spacing_hz, num_channels=n_channels, center_frequency=center_frequency)
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=64, samples_per_symbol=8)
    amp = Amplification(n_pumps=0, raman_gain=0.0, pumps=None)
    return System(fiber=fiber, wdm=wdm, pulse=pulse, amplification=amp)


def test_beta2_eff_matches_eq28():
    beta2 = 20e-27
    beta3 = 1.2e-40
    beta4 = 3.4e-55
    fc = 193.1e12
    fm = 193.2e12
    fk = 193.25e12
    coeffs = (beta2, beta3, beta4, fc)

    val = _beta2_eff(fm, fk, coeffs)
    val_swap = _beta2_eff(fk, fm, coeffs)
    val_center = _beta2_eff(fc, fc, coeffs)

    assert np.isclose(val, val_swap, rtol=1e-12, atol=0.0)
    assert np.isclose(val_center, beta2, rtol=1e-12, atol=0.0)

    dfm = fm - fc
    dfk = fk - fc
    ref = beta2 + np.pi * beta3 * (dfm + dfk) + (2.0 / 3.0) * (np.pi ** 2) * beta4 * (
        dfm * dfm + dfm * dfk + dfk * dfk
    )
    assert np.isclose(val, ref, rtol=1e-12, atol=0.0)


def test_xci_scales_with_interferer_power_squared(tmp_path):
    length = 50e3
    system = _make_minimal_system(
        n_channels=2,
        spacing_hz=50e9,
        center_frequency=193.1e12,
        length=length,
    )
    profile_path = _make_constant_profile(tmp_path, n_channels=2, length=length, n_z=20)

    cfg = PcfmConfig(
        degree=3,
        use_numeric_xci=False,
        use_beta2_eff=False,
        n_f=4,
        n_z=6,
    )

    launch_a = np.array([1e-3, 1e-3])
    _, _, xci_a = compute_pcfm_nlin(
        system,
        profile_path,
        launch_powers_w=launch_a,
        config=cfg,
        return_components=True,
    )
    xci_a_cut = float(xci_a[0, 0])
    assert xci_a_cut > 0.0

    launch_b = np.array([1e-3, 2e-3])
    _, _, xci_b = compute_pcfm_nlin(
        system,
        profile_path,
        launch_powers_w=launch_b,
        config=cfg,
        return_components=True,
    )
    xci_b_cut = float(xci_b[0, 0])

    ratio = xci_b_cut / xci_a_cut
    assert np.isclose(ratio, 4.0, rtol=1e-2, atol=0.0)


def test_pcfm_uses_dual_pol_input_and_per_pol_output_scaling(tmp_path):
    length = 50e3
    system = _make_minimal_system(
        n_channels=1,
        spacing_hz=50e9,
        center_frequency=193.1e12,
        length=length,
    )
    profile_path = _make_constant_profile(tmp_path, n_channels=1, length=length, n_z=20)
    launch = np.array([1e-3])
    cfg = PcfmConfig(
        degree=3,
        use_numeric_xci=False,
        use_beta2_eff=False,
        n_f=4,
        n_z=6,
    )

    total, sci, xci = compute_pcfm_nlin(
        system,
        profile_path,
        launch_powers_w=launch,
        config=cfg,
        return_components=True,
    )

    freqs = system.wdm.frequency_grid()
    b_ch = float(system.pulse.baud_rate)
    signal_power_ch_z, z = load_signal_profiles(profile_path, system)
    spp = normalize_spp(signal_power_ch_z, z)
    coeffs = fit_spp_polynomials(z, spp, cfg.degree)
    beta2 = _beta2_array(system, freqs)[0]
    aeff = _aeff_array(system, freqs)[0]
    gamma = 2.0 * np.pi * freqs[0] / c * (N2_SIO2 / aeff)
    k_sci = compute_sci_numeric(coeffs[0], length, beta2, b_ch, cfg.n_f, cfg.n_z, cfg.phase_coeff)
    g_ch = _to_per_channel_power(launch[0]) / b_ch
    expected_sci = _to_per_polarization_power(
        (16.0 / 27.0) * (g_ch ** 3) * (gamma ** 2) * k_sci * b_ch
    )

    assert np.allclose(xci, 0.0)
    assert np.allclose(sci, expected_sci)
    assert np.allclose(total, expected_sci)
