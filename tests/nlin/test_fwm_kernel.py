import numpy as np

from pynlin.methods.td.fwm_kernel import (
    FWMChannels,
    compute_fwm_coefficient_direct,
    compute_fwm_kernel_direct,
)
from pynlin.methods.td.xpm_kernel import compute_xpm_coefficient_direct
from pynlin.pulses import GaussianPulse


def test_fwm_direct_reduces_to_xpm_direct_on_k_eq_m_sector():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=64, samples_per_symbol=32)
    z = np.linspace(0.0, 1000.0, 7)
    dgd = 2.0e-13
    gvda = 1.0e-27
    gvdb = -2.0e-27
    channels = FWMChannels(
        omega_a=0.0,
        omega_b=0.0,
        omega_c=0.0,
        omega_d=0.0,
        beta1_a=0.0,
        beta1_b=dgd,
        beta1_c=dgd,
        beta1_d=0.0,
        gvd_a=gvda,
        gvd_b=gvdb,
        gvd_c=gvdb,
        gvd_d=gvda,
    )

    fwm = compute_fwm_coefficient_direct(
        pulse,
        z,
        h=1,
        k=0,
        m=0,
        channels=channels,
    )
    xpm = compute_xpm_coefficient_direct(
        pulse,
        z,
        h=1,
        k=0,
        m=0,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )

    np.testing.assert_allclose(
        fwm,
        xpm,
        rtol=1e-12,
        atol=1e-12 * max(abs(xpm), 1.0),
    )


def test_fwm_frequency_mismatch_phase_suppresses_large_detuning():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=96, samples_per_symbol=32)
    z = np.linspace(0.0, 1.0, 3)
    matched = FWMChannels(omega_a=0.0, omega_b=0.0, omega_c=0.0, omega_d=0.0)
    detuned = FWMChannels(
        omega_a=2.0 * np.pi * 80e9,
        omega_b=0.0,
        omega_c=0.0,
        omega_d=0.0,
    )

    x_matched = compute_fwm_coefficient_direct(
        pulse, z, h=0, k=0, m=0, channels=matched
    )
    x_detuned = compute_fwm_coefficient_direct(
        pulse, z, h=0, k=0, m=0, channels=detuned
    )

    assert abs(x_detuned) < 0.2 * abs(x_matched)


def test_fwm_frequency_mismatch_can_be_disabled():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=64, samples_per_symbol=32)
    z = np.linspace(0.0, 1.0, 3)
    matched = FWMChannels(omega_a=0.0, omega_b=0.0, omega_c=0.0, omega_d=0.0)
    detuned = FWMChannels(
        omega_a=2.0 * np.pi * 60e9,
        omega_b=0.0,
        omega_c=0.0,
        omega_d=0.0,
    )

    x_matched = compute_fwm_coefficient_direct(
        pulse, z, h=0, k=0, m=0, channels=matched
    )
    x_disabled = compute_fwm_coefficient_direct(
        pulse,
        z,
        h=0,
        k=0,
        m=0,
        channels=detuned,
        include_frequency_mismatch=False,
    )

    np.testing.assert_allclose(
        x_disabled,
        x_matched,
        rtol=1e-12,
        atol=1e-12 * max(abs(x_matched), 1.0),
    )


def test_fwm_phase_mismatch_can_be_disabled():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=64, samples_per_symbol=32)
    z = np.linspace(0.0, 1000.0, 401)
    matched = FWMChannels(omega_a=0.0, omega_b=0.0, omega_c=0.0, omega_d=0.0)
    mismatched = FWMChannels(
        omega_a=0.0,
        omega_b=0.0,
        omega_c=0.0,
        omega_d=0.0,
        beta0_a=2.0 * np.pi / 20.0,
    )

    x_matched = compute_fwm_coefficient_direct(
        pulse, z, h=0, k=0, m=0, channels=matched
    )
    x_mismatched = compute_fwm_coefficient_direct(
        pulse, z, h=0, k=0, m=0, channels=mismatched
    )
    x_disabled = compute_fwm_coefficient_direct(
        pulse,
        z,
        h=0,
        k=0,
        m=0,
        channels=mismatched,
        include_phase_mismatch=False,
    )

    assert abs(x_mismatched) < 0.2 * abs(x_matched)
    np.testing.assert_allclose(
        x_disabled,
        x_matched,
        rtol=1e-12,
        atol=1e-12 * max(abs(x_matched), 1.0),
    )


def test_fwm_kernel_direct_shape_and_metadata():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=32, samples_per_symbol=16)
    z = np.linspace(0.0, 10.0, 3)
    channels = FWMChannels(
        omega_a=1.0,
        omega_b=2.0,
        omega_c=0.5,
        omega_d=1.25,
        beta0_a=0.1,
        beta0_b=0.2,
        beta0_c=0.05,
        beta0_d=0.1,
    )
    h_values = np.array([-1, 0])
    k_values = np.array([0, 1])
    m_values = np.array([-1, 1])

    result = compute_fwm_kernel_direct(
        pulse, z, h_values, k_values, m_values, channels=channels
    )

    assert result.X.shape == (2, 2, 2)
    np.testing.assert_array_equal(result.h_values, h_values)
    np.testing.assert_array_equal(result.k_values, k_values)
    np.testing.assert_array_equal(result.m_values, m_values)
    assert result.metadata["delta_omega"] == 1.25
    np.testing.assert_allclose(result.metadata["delta_beta0"], 0.15)
    assert result.metadata["frequency_matched"] is False
    assert "g_d*" in result.metadata["convention"]


def test_fwm_amplification_function_validation():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=16, samples_per_symbol=16)
    z = np.linspace(0.0, 1.0, 3)
    channels = FWMChannels(omega_a=0.0, omega_b=0.0, omega_c=0.0, omega_d=0.0)

    try:
        compute_fwm_coefficient_direct(
            pulse,
            z,
            h=0,
            k=0,
            m=0,
            channels=channels,
            amplification_function=lambda z_values: np.ones(z_values.size + 1),
        )
    except ValueError as exc:
        assert "amplification_function" in str(exc)
    else:
        raise AssertionError("wrong-length amplification function was accepted")


def test_fwm_zero_mismatch_result_is_nearly_real_for_real_pulse():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=64, samples_per_symbol=32)
    z = np.linspace(0.0, 10.0, 3)
    channels = FWMChannels(omega_a=0.0, omega_b=0.0, omega_c=0.0, omega_d=0.0)

    x = compute_fwm_coefficient_direct(pulse, z, h=0, k=0, m=0, channels=channels)

    assert abs(x.imag) < 1e-12 * abs(x.real)
    assert x.real > 0.0
