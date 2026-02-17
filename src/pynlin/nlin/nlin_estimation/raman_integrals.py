"""Raman profile and integral helpers for legacy NLIN estimator workflows."""

from typing import Tuple

import numpy as np

from pynlin.constellation_stats import qam_mu0
from pynlin.log_init import init_logging
from pynlin.system import System

init_logging()

SPATIAL_MODES = np.array([1, 2, 2, 1])
LLW_MIN = 0.01  # target L/LW
LLW_MAX = 100.0
# 64-QAM <|b_0|^4>/<|b_0|^2>^2 from analytical constellation stats.
MU0 = qam_mu0(64)

def load_fB(cf: System) -> Tuple[np.ndarray, np.ndarray, np.ndarray, callable, callable]:
    """Load normalized Raman gain profiles fB(z) and polynomial approximations from a cached solution."""
    # assert (cf.launch_power == -5.0 and cf.raman_gain == 0.0)
    # all the information about the numerosity and stuff is here.
    # Beware, -5dBm is right: it is obtained using the -2dBm solutions so to have equalizaiton without recomputing all
    sol_path = "results/ct_solution-5_gain_0.0.npy"
    solutions = np.load(sol_path, allow_pickle=True).item()

    signal_powers = solutions['signal_sol']
    # indices: (z, mode, channel)
    signal_powers = np.swapaxes(signal_powers, 1, 2)
    fB = signal_powers / signal_powers[0, :, :]  # normalize to input power
    assert np.all(fB[0, :, :] == 1.0)
    fB_max = np.max(fB, axis=(1, 2))
    fB_min = np.min(fB, axis=(1, 2))
    z_axis = np.linspace(0, cf.fiber_length, len(fB_max))
    assert ((fB_min <= fB_max).all())
    assert (fB.shape[0] == len(z_axis))

    coeffs_max = np.polyfit(z_axis, fB_max, 6)
    coeffs_min = np.polyfit(z_axis, fB_min, 6)

    def fB_max_function(z):
        return np.polyval(coeffs_max, z)

    def fB_min_function(z):
        return np.polyval(coeffs_min, z)

    return fB, fB_min, fB_max, fB_min_function, fB_max_function

def raman_integral(cf,
                   regime: str,
                   fB: np.ndarray):
    """Compute LO or HI Raman integrals for a given longitudinal gain profile."""
    z_axis = np.linspace(0, cf.fiber_length, len(fB))
    dz = z_axis[1] - z_axis[0]
    if regime == "LO":
        return (np.sum(fB) * dz / cf.fiber_length)**2
    else:
        return np.sum(fB**2) * dz / cf.fiber_length


def load_raman_integral_extremes(cf,
                                 ) -> Tuple[float, float, float, float]:
    """Return LO/HI Raman integrals for minimum and maximum gain envelopes."""
    _, fB_min, fB_max, _, _ = load_fB(cf)
    r_lo_min = raman_integral(cf, "LO", fB_min)
    r_lo_max = raman_integral(cf, "LO", fB_max)
    r_hi_min = raman_integral(cf, "HI", fB_min)
    r_hi_max = raman_integral(cf, "HI", fB_max)

    return r_lo_min, r_lo_max, r_hi_min, r_hi_max
