import numpy as np

from pynlin.methods.td.fwm_kernel import FWMChannels, compute_fwm_kernel_direct
from pynlin.methods.td.fwm_mc import estimate_fwm_term_sum_dar_mc
from pynlin.pulses import NyquistPulse


def _two_channel_fwm_channels(target: str, term: str, q: float, baud_rate: float, beta2: float) -> FWMChannels:
    delta_omega = 2.0 * np.pi * q * baud_rate
    omegas = {"a": 0.0, "b": delta_omega}

    def beta0(label: str) -> float:
        omega = omegas[label]
        return 0.5 * beta2 * omega * omega

    def beta1(label: str) -> float:
        return beta2 * omegas[label]

    a, b, c = term
    return FWMChannels(
        omega_a=omegas[a],
        omega_b=omegas[b],
        omega_c=omegas[c],
        omega_d=omegas[target],
        beta0_a=beta0(a),
        beta0_b=beta0(b),
        beta0_c=beta0(c),
        beta0_d=beta0(target),
        beta1_a=beta1(a),
        beta1_b=beta1(b),
        beta1_c=beta1(c),
        beta1_d=beta1(target),
        gvd_a=beta2,
        gvd_b=beta2,
        gvd_c=beta2,
        gvd_d=beta2,
    )


def _direct_total_for_radius(pulse, z, channels, radius: int) -> float:
    values = np.arange(-radius, radius + 1)
    result = compute_fwm_kernel_direct(
        pulse,
        z,
        values,
        values,
        values,
        channels=channels,
        auto_refine=True,
        min_pts_per_period=3.0,
        max_z_points=500,
        discretization_action="silent",
    )
    return float(np.sum(np.abs(result.X) ** 2))


def test_dar_fwm_mc_rejects_impossible_output_support():
    baud_rate = 24.5e9
    length = 100e3
    beta2 = -20e-27
    q = 1.0
    channels = _two_channel_fwm_channels("a", "bba", q, baud_rate, beta2)

    estimate = estimate_fwm_term_sum_dar_mc(
        channels=channels,
        baud_rate=baud_rate,
        length=length,
        n_samples=4096,
        seed=1234,
    )

    assert estimate.total == 0.0
    assert estimate.total_stderr == 0.0
    assert estimate.metadata["support_fraction"] == 0.0


def test_direct_fwm_sum_converges_toward_dar_frequency_mc():
    baud_rate = 24.5e9
    length = 100e3
    beta2 = -20e-27
    q = 1.0
    channels = _two_channel_fwm_channels("a", "abb", q, baud_rate, beta2)
    pulse = NyquistPulse(baud_rate=baud_rate, num_symbols=64, samples_per_symbol=24, rolloff=0.0)
    z = np.linspace(0.0, length, 7)
    T2 = (1.0 / baud_rate) ** 2

    mc = estimate_fwm_term_sum_dar_mc(
        channels=channels,
        baud_rate=baud_rate,
        length=length,
        n_samples=20000,
        seed=1234,
    )
    direct_r3 = _direct_total_for_radius(pulse, z, channels, radius=3) * T2
    direct_r5 = _direct_total_for_radius(pulse, z, channels, radius=5) * T2

    ratio_r3 = direct_r3 / mc.total
    ratio_r5 = direct_r5 / mc.total

    assert 0.45 < ratio_r3 < 0.75
    assert 0.65 < ratio_r5 < 0.95
    assert ratio_r5 > ratio_r3


def test_dar_fwm_mc_includes_local_beta4():
    random_variables = np.array([[0.2], [-0.1], [0.05]])
    channels = FWMChannels(
        omega_a=0.0,
        omega_b=0.0,
        omega_c=0.0,
        omega_d=0.0,
        beta4_a=24.0,
    )

    estimate = estimate_fwm_term_sum_dar_mc(
        channels=channels,
        baud_rate=1.0,
        length=0.7,
        n_samples=1,
        random_variables=random_variables,
    )

    delta_beta = random_variables[0, 0] ** 4
    expected = abs((np.exp(1j * delta_beta * 0.7) - 1.0) / (1j * delta_beta)) ** 2
    np.testing.assert_allclose(estimate.total, expected, rtol=1e-13)
    assert estimate.metadata["beta4_a"] == 24.0
