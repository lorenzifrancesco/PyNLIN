from __future__ import annotations

import numpy as np

from pynlin.constellation_stats import qam_mu0
from pynlin.methods.td.fwm_kernel import (
    FWMChannels,
    FWMKernelResult,
    compute_fwm_coefficient_direct,
    compute_fwm_kernel_direct,
)
from pynlin.methods.td.fwm_mc import (
    FWMDarMCSum,
    FWMTermMCSum,
    estimate_fwm_term_sum_dar_mc,
    estimate_fwm_term_sum_mc,
)
from pynlin.system import System
from pynlin.utils import dBm2watt


def _nlin_uwb():
    import pynlin.methods.td.estimator as nlin_uwb

    return nlin_uwb


def _qam_mu0(order: int) -> float:
    """Return mu0 = <|b|^4>/<|b|^2>^2 for uniform QAM."""
    return qam_mu0(order)


def _td_prefactor_coeffs(mode_a: int, mode_b: int, n_modes: int) -> tuple[float, float]:
    """Return (a, b) so prefactor = a * mu0 + b."""
    nlin_uwb = _nlin_uwb()
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
    exclude_self_channel: bool = False,
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
    nlin_uwb = _nlin_uwb()
    gamma2 = nlin_uwb._gamma_matrix_uwb(system, freqs) ** 2
    constant_prefactor = (power_in**3) / (baud_rate**2)

    if use_kappa:
        kappa2 = nlin_uwb.get_kappa2_matrix_uwb(
            system, use_kappa=True, use_x_mode=use_x_mode
        )
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
                coeffs_b = collision_coeffs_si[mode_a, nu_a, mode_b, :]
                coeffs_b = coeffs_b * gamma2[nu_a, :]
                if exclude_self_channel:
                    coeff_sum = float(np.sum(coeffs_b) - coeffs_b[nu_a])
                else:
                    coeff_sum = float(np.sum(coeffs_b))
                sum_a[mode_a, nu_a] += weight * coeff_sum * a_coeff
                sum_b[mode_a, nu_a] += weight * coeff_sum * b_coeff
    return constant_prefactor, sum_a, sum_b


def chi1_chi2(
    system: System,
    collision_coeffs: np.ndarray,
    launch_powers_w: np.ndarray | None = None,
    use_kappa: bool = True,
    use_x_mode: bool = True,
    exclude_self_channel: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (chi1, chi2, prefactor) = (sum_b, sum_a, const_pref)."""
    const_pref, sum_a, sum_b = _td_modulation_components(
        system,
        collision_coeffs,
        launch_powers_w,
        use_kappa=use_kappa,
        use_x_mode=use_x_mode,
        exclude_self_channel=exclude_self_channel,
    )
    return sum_b, sum_a, const_pref
