from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c
from scipy.integrate import solve_ivp, trapezoid

from analysis.runners.methods import _fullband_mc_estimate_nlin
from pynlin.constellation_stats import gaussian_stats, qam_stats
from pynlin.fiber import SMFiber
from pynlin.methods.mc import compute_chi1_chi2, nlin_from_chi
from pynlin.methods.pcfm import N2_SIO2, PcfmConfig, compute_pcfm_nlin
from pynlin.methods.td.estimator import total_nlin_uwb
from pynlin.methods.td.fwm_kernel import FWMChannels, compute_fwm_kernel_direct
from pynlin.methods.td.xhkm_sums import compute_xhkm_sums
from pynlin.methods.td.xpm_kernel import classify_collision
from pynlin.pulses import GaussianPulse, NyquistPulse
from pynlin.raman.solvers import RamanAmplifier
from pynlin.raman.solvers_jiang import JiangIterativeConfig, SMFWidebandAmplifier
from pynlin.raman.undepleted import (
    pump_power_coprop,
    pump_power_counterprop,
    signal_power_undepleted_coprop,
    signal_power_undepleted_counterprop,
)
from pynlin.system import System
from pynlin.utils import (
    alpha_to_linear,
    beta2_to_dispersion,
    dBm_to_watt,
    dispersion_to_beta2,
    frequency_to_wavelength,
    watt_to_dBm,
    wavelength_to_frequency,
)
from pynlin.wdm import Amplification, RegularWDM


class _DummyWDM:
    def __init__(self, freqs):
        self._freqs = np.asarray(freqs, dtype=float)

    def frequency_grid(self):
        return self._freqs


def _minimal_td_system(freqs=(193.1e12, 193.2e12)):
    fiber = SimpleNamespace(length=1.0, effective_area=80e-12)
    return SimpleNamespace(
        n_modes=1,
        fiber=fiber,
        effective_area=fiber.effective_area,
        fiber_length=1.0,
        pulse=SimpleNamespace(baud_rate=1.0),
        baud_rate=1.0,
        launch_power=0.0,
        raw_config={},
        wdm=_DummyWDM(freqs),
    )


def _constant_profile(tmp_path, n_channels, length):
    z = np.linspace(0.0, length, 12)
    path = tmp_path / "constant-profile.npz"
    np.savez(path, signal_sol=np.ones((n_channels, z.size)), z=z)
    return path


def test_unit_conversions_are_involutions_and_use_power_attenuation():
    wavelength = np.array([1260e-9, 1550e-9, 1675e-9])
    np.testing.assert_allclose(
        frequency_to_wavelength(wavelength_to_frequency(wavelength)),
        wavelength,
        rtol=2e-16,
    )

    powers_dbm = np.array([-30.0, 0.0, 10.0])
    np.testing.assert_allclose(watt_to_dBm(dBm_to_watt(powers_dbm)), powers_dbm)

    beta2 = -21.7e-27
    np.testing.assert_allclose(
        dispersion_to_beta2(beta2_to_dispersion(beta2, 1550e-9), 1550e-9),
        beta2,
        rtol=2e-16,
    )
    assert alpha_to_linear(1e-3) == pytest.approx(np.log(10.0) / 10.0 * 1e-3)


@pytest.mark.parametrize("pulse_cls", [GaussianPulse, NyquistPulse])
def test_analytical_pulses_have_unit_energy(pulse_cls):
    pulse = pulse_cls(baud_rate=10e9, num_symbols=64, samples_per_symbol=32)
    g, t = pulse.data()
    assert trapezoid(np.abs(g) ** 2, t) == pytest.approx(1.0, rel=2e-14)


def test_constellation_moments_match_closed_forms():
    assert qam_stats(4).mu0 == pytest.approx(1.0)
    assert qam_stats(16).mu0 == pytest.approx(33.0 / 25.0)
    assert gaussian_stats().mu0 == pytest.approx(2.0)
    assert gaussian_stats().e6 == pytest.approx(6.0)


def test_pcfm_xci_matches_single_polarization_paper_normalization(tmp_path):
    length = 50e3
    bandwidth = 10e9
    beta2 = 20e-27
    area = 80e-12
    center = 193.1e12
    spacing = 50e9
    fiber = SMFiber(beta2=beta2, effective_area=area, length=length)
    system = System(
        fiber=fiber,
        wdm=RegularWDM(spacing=spacing, num_channels=2, center_frequency=center),
        pulse=GaussianPulse(baud_rate=bandwidth, num_symbols=32, samples_per_symbol=8),
        amplification=Amplification(n_pumps=0, raman_gain=0.0, pumps=None),
    )
    profile = _constant_profile(tmp_path, 2, length)
    launch_single_pol = np.array([1.1e-3, 0.7e-3])

    _, _, xci = compute_pcfm_nlin(
        system,
        profile,
        launch_powers_w=launch_single_pol,
        config=PcfmConfig(
            degree=3,
            use_numeric_xci=False,
            use_beta2_eff=False,
            n_f=4,
            n_z=6,
        ),
        return_components=True,
    )

    freqs = system.wdm.frequency_grid()
    delta_f = abs(freqs[1] - freqs[0])
    log_term = np.log((delta_f + bandwidth / 2.0) / (delta_f - bandwidth / 2.0))
    kernel = length / (2.0 * np.pi * abs(beta2)) * log_term
    gamma = 2.0 * np.pi * freqs[0] / c * (N2_SIO2 / area)
    expected = (
        (128.0 / 27.0)
        * launch_single_pol[0]
        * launch_single_pol[1] ** 2
        * gamma**2
        * kernel
        / bandwidth**2
    )
    assert xci[0, 0] == pytest.approx(expected, rel=2e-12)


@pytest.mark.xfail(
    strict=True,
    reason="TD modulation reconstruction omits interferer-specific P_j^2 weights",
)
def test_td_modulation_reconstruction_matches_total_for_unequal_powers():
    system = _minimal_td_system()
    collision_coeffs = np.ones((1, 2, 1, 2), dtype=float)
    launch = np.array([1e-3, 2e-3])

    total = total_nlin_uwb(
        system,
        collision_coeffs,
        launch_powers_w=launch,
        exclude_self_channel=True,
    )
    chi1, chi2, prefactor = compute_chi1_chi2(
        system,
        collision_coeffs,
        launch,
        exclude_self_channel=True,
    )
    reconstructed = nlin_from_chi(chi1, chi2, prefactor, qam_stats(64).mu0)
    np.testing.assert_allclose(reconstructed, total, rtol=2e-14, atol=0.0)


@pytest.mark.xfail(
    strict=True,
    reason="SMFiber.loss_profile is redefined and bypasses its attenuation profile",
)
def test_raman_linear_losses_use_smf_attenuation_profile():
    wavelengths = np.array([1.4e-6, 1.6e-6])
    losses_db_per_m = np.array([1e-3, 3e-3])
    fiber = SMFiber(
        attenuation_profile=(wavelengths, losses_db_per_m),
        effective_area=80e-12,
    )
    actual = RamanAmplifier(fiber).get_linear_losses(np.array([1.5e-6]))
    expected = alpha_to_linear(np.array([2e-3]))
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0.0)


@pytest.mark.xfail(
    strict=True,
    reason="Fullband conversion supplies two powers although gamma^2 S requires P^3",
)
def test_fullband_nlin_scales_cubically_with_uniform_launch_power():
    freqs = np.array([190e12, 191e12])
    system = SimpleNamespace(
        wdm=_DummyWDM(freqs),
        fiber=SimpleNamespace(gamma=1.0, effective_area=80e-12),
        effective_area=80e-12,
    )
    diagnostic = SimpleNamespace(total=np.array([2.0]), target_indices=np.array([0]))
    low = _fullband_mc_estimate_nlin(system, diagnostic, np.array([1e-3, 1e-3]))
    high = _fullband_mc_estimate_nlin(system, diagnostic, np.array([2e-3, 2e-3]))
    assert high[0] / low[0] == pytest.approx(8.0)


@pytest.mark.xfail(
    strict=True,
    reason="The public collision classifier disagrees with the Xhkm contraction sectors",
)
def test_collision_classifier_matches_xhkm_sector_partition():
    h_values = np.array([0, 1])
    r_values = np.array([0, 1])
    m_values = np.array([0])
    X = np.zeros((2, 2, 1), dtype=complex)
    X[1, 1, 0] = 1.0  # h=k=1, but h!=0 and k!=m

    sums = compute_xhkm_sums(X, h_values, r_values, m_values)
    assert sums.n_4pc == pytest.approx(1.0)
    assert classify_collision(h=1, r=1, m=0) == "4pc"


@pytest.mark.xfail(
    strict=True,
    reason="The FWM checker divides temporal walk-off by the longitudinal z span",
)
def test_fwm_checker_rejects_walkoff_larger_than_pulse_time_window():
    pulse = GaussianPulse(baud_rate=1e9, num_symbols=16, samples_per_symbol=8)
    channels = FWMChannels(
        omega_a=0.0,
        omega_b=0.0,
        omega_c=0.0,
        omega_d=0.0,
        beta1_a=2e-8,
    )
    with pytest.raises(ValueError, match="time-window wrap"):
        compute_fwm_kernel_direct(
            pulse,
            np.linspace(0.0, 1.0, 101),
            np.array([0]),
            np.array([0]),
            np.array([0]),
            channels=channels,
            discretization_action="assert",
        )


class _SignedUnitGainAmplifier(RamanAmplifier):
    def _interpolate_gain(self, delta_freqs, resolution_hz):
        del resolution_hz
        return np.sign(np.asarray(delta_freqs, dtype=float))


@pytest.mark.xfail(
    strict=True,
    reason="SM Raman gain scales both matrix halves by nu_i/nu_j and violates Manley-Rowe",
)
def test_raman_gain_matrix_conserves_two_wave_photon_rate():
    fiber = SMFiber(raman_coefficient=1.0, effective_area=1.0)
    amplifier = _SignedUnitGainAmplifier(fiber)
    frequencies = np.array([190e12, 205e12])  # Stokes signal, pump
    gain = amplifier.compute_gain_matrix(frequencies)
    derivative = amplifier.raman_ode(
        np.ones(2),
        0.0,
        np.zeros(2),
        gain,
        np.ones(2),
    )

    assert derivative[0] > 0.0
    assert derivative[1] < 0.0
    assert derivative[0] / frequencies[0] + derivative[1] / frequencies[1] == pytest.approx(
        0.0, abs=1e-28
    )


@pytest.mark.xfail(
    strict=True,
    reason="Automatic Jiang iteration performs only one update when no pumps are present",
)
def test_jiang_no_pump_isrs_matches_coupled_ode():
    z = np.linspace(0.0, 10.0, 101)
    signal_in = np.array([1.0, 0.8])
    gain = np.array([[0.0, 0.03], [-0.03, 0.0]])
    losses = np.zeros(2)
    signal_indices = np.array([0, 1])

    fixed_point = SMFWidebandAmplifier._solve_jiang_unidirectional_inner(
        SimpleNamespace(),
        sigP=signal_in,
        pumP=np.empty(0),
        losses=losses,
        G=gain,
        z=z,
        dz=float(z[1] - z[0]),
        direction=np.ones(2),
        pump_idx=np.empty(0, dtype=int),
        sig_idx=signal_indices,
        cfg=JiangIterativeConfig(inner_iters=None, early_stop_rtol=None),
    )
    reference = solve_ivp(
        lambda position, power: power * (gain @ power),
        (z[0], z[-1]),
        signal_in,
        t_eval=z,
        rtol=1e-11,
        atol=1e-13,
    ).y
    np.testing.assert_allclose(fixed_point, reference, rtol=1e-3, atol=1e-8)


@pytest.mark.parametrize("counterprop", [False, True])
@pytest.mark.parametrize("alpha_p", [0.0, 2e-4])
def test_undepleted_raman_signal_matches_independent_ode(counterprop, alpha_p):
    length = 1000.0
    z = np.linspace(0.0, length, 31)
    signal_in = 1e-3
    pump_in = 0.2
    alpha_s = 5e-5
    gain = 8e-4

    if counterprop:
        pump = lambda position: pump_in * np.exp(-alpha_p * (length - position))
        expected = signal_power_undepleted_counterprop(
            z, signal_in, alpha_s, gain, pump_in, alpha_p, length
        )
        np.testing.assert_allclose(
            pump_power_counterprop(z, pump_in, alpha_p, length),
            pump(z),
            rtol=2e-15,
        )
    else:
        pump = lambda position: pump_in * np.exp(-alpha_p * position)
        expected = signal_power_undepleted_coprop(
            z, signal_in, alpha_s, gain, pump_in, alpha_p
        )
        np.testing.assert_allclose(
            pump_power_coprop(z, pump_in, alpha_p),
            pump(z),
            rtol=2e-15,
        )

    numerical = solve_ivp(
        lambda position, power: (-alpha_s + gain * pump(position)) * power,
        (0.0, length),
        np.array([signal_in]),
        t_eval=z,
        rtol=1e-11,
        atol=1e-14,
    ).y[0]
    np.testing.assert_allclose(expected, numerical, rtol=1e-9, atol=1e-14)
