import numpy as np
from scipy.constants import c

import pynlin.nlin.nlin_estimator_uwb as nlin_uwb
from pynlin.constellation_stats import qam_mu0
from pynlin.system import System
from pynlin.utils import dBm2watt


def _qam_mu0(order: int) -> float:
    """Return mu0 = <|b|^4>/<|b|^2>^2 for uniform QAM."""
    return qam_mu0(order)


def _td_prefactor_coeffs(mode_a: int, mode_b: int, n_modes: int) -> tuple[float, float]:
    """Return (a, b) so prefactor = a * mu0 + b."""
    if n_modes == 1:
        mode_a = mode_b = 0
    if mode_a == mode_b:
        a_coeff = 2.0 * nlin_uwb.SPATIAL_MODES[mode_a] + 3.0
        b_coeff = -4.0
    else:
        a_coeff = 2.0 * nlin_uwb.SPATIAL_MODES[mode_b]
        b_coeff = -2.0 * nlin_uwb.SPATIAL_MODES[mode_b]
    return a_coeff, b_coeff


def _td_modulation_components(
    system: System,
    collision_coeffs: np.ndarray,
    launch_powers_w: np.ndarray | None,
    use_kappa: bool = True,
    use_x_mode: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (constant_prefactor, sum_a, sum_b) for TD modulation scaling."""
    length = float(system.fiber_length)
    baud_rate = float(system.pulse.baud_rate)
    y_norm = 1.0 / (length * baud_rate) ** 2
    collision_coeffs_si = collision_coeffs / y_norm

    n_modes, n_freqs, _, _ = collision_coeffs_si.shape
    if launch_powers_w is None:
        power_dbm = system.launch_power if system.launch_power is not None else -5.0
        power_in = np.full((n_modes, n_freqs), dBm2watt(power_dbm))
    else:
        power_raw = np.asarray(launch_powers_w, dtype=float)
        if power_raw.ndim == 1:
            if power_raw.size != n_freqs:
                raise ValueError(
                    f"launch_powers_w length {power_raw.size} != n_freqs {n_freqs}"
                )
            power_in = np.broadcast_to(power_raw[None, :], (n_modes, n_freqs))
        elif power_raw.shape == (n_modes, n_freqs):
            power_in = power_raw
        else:
            raise ValueError(
                f"launch_powers_w shape {power_raw.shape} incompatible with "
                f"(n_modes,n_freqs)=({n_modes},{n_freqs})"
            )

    freqs = system.wdm.frequency_grid()
    n2 = 2.6e-20
    aeff = nlin_uwb._effective_area_array(system, freqs)
    gamma = n2 * (2.0 * np.pi * freqs) / (aeff * c)
    gamma = gamma[None, :]
    constant_prefactor = (power_in**3) * (gamma**2) / (baud_rate**2)

    if use_kappa:
        kappa2 = nlin_uwb.get_kappa2_matrix_uwb(system, use_kappa=True, use_x_mode=use_x_mode)
    else:
        kappa2 = np.ones((n_modes, n_modes), dtype=float)
        if not use_x_mode:
            kappa2 = np.multiply(kappa2, np.eye(n_modes))

    sum_a = np.zeros((n_modes, n_freqs), dtype=float)
    sum_b = np.zeros_like(sum_a)
    for mode_a in range(n_modes):
        for nu_a in range(n_freqs):
            for mode_b in range(n_modes):
                a_coeff, b_coeff = _td_prefactor_coeffs(mode_a, mode_b, n_modes)
                weight = kappa2[mode_a, mode_b]
                coeff_sum = float(np.sum(collision_coeffs_si[mode_a, nu_a, mode_b, :]))
                sum_a[mode_a, nu_a] += weight * coeff_sum * a_coeff
                sum_b[mode_a, nu_a] += weight * coeff_sum * b_coeff
    return constant_prefactor, sum_a, sum_b
