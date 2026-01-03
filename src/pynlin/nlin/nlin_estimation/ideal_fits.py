from typing import Tuple

import numpy as np
from loguru import logger as lg
from scipy.optimize import curve_fit

import pynlin.io_utils as cfg
from pynlin.log_init import init_logging

init_logging()

SPATIAL_MODES = np.array([1, 2, 2, 1])
LLW_MIN = 0.01  # target L/LW
LLW_MAX = 100.0
# 64-QAM <|b_0|^4>/<|b_0|^2>^2 this is compatible with the (mu_0 - 1)=0.32*1.19 previously used.
MU0 = 1.3809


def softplus(x, a, b, c):
    """Smoothly increasing three-parameter curve used to fit NLIN vs walk-off."""
    return a * (1 + (x / b)**(1 / c))**(-c)

"""
ideal := no Raman, no GVD. It is flexible to also compute the GVD, but it is not recommended.
"""
def ideal_fit_coefficients(gvda: float = 0.0,
                           gvdb: float = 0.0,
                           ipulse: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Fit NLIN curves for the Raman/GVD-free case and return softplus parameters."""
    cf = cfg.load_toml_to_struct("./input/mmf.toml")
    nc = cfg.load_nc_toml_to_struct("./input/numerical_config.toml")

    # Override DGD ranges using global targets
    # lg.warning(f"Overriding DGD ranges to [{LLW_MIN}, {LLW_MAX}] ps/sqrt(km)")
    nc.dgd1 = LLW_MIN / (cf.fiber_length * cf.baud_rate)
    # build the fit from the minimum to the maximum
    nc.dgd2_n = LLW_MAX / (cf.fiber_length * cf.baud_rate)
    dgd2 = nc.dgd2_n

    cf_path = "./input/mmf.toml"
    cf = cfg.load_toml_to_struct(cf_path)

    dgds_numeric = np.logspace(
        np.log10(nc.dgd1), np.log10(dgd2), nc.n_samples_numeric_n)

    T = 1 / cf.baud_rate
    L = cf.fiber_length
    x_norm = L / T
    y_norm = x_norm**(-2)
    p0 = [0.4, 4.5, 0.5]

    if ipulse == 0:
        pulse_shape = "gaussian"
    else:
        pulse_shape = "nyquist"

    nlin_numeric = None
    if gvda != 0.0 or gvdb != 0.0:
        raise ("You are trying to cheat! Instead of fitting from a case computed with dispersion, you should use the dispersion correction given by correct_fit_coefficients")
        nlin_numeric = np.load(
            f"results/partial_nlin_{pulse_shape}_perfect_{gvda}_{gvdb}.npy")
    else:
        nlin_numeric = np.load(
            f"results/partial_nlin_{pulse_shape}_perfect_0.0_0.0.npy")

    assert len(dgds_numeric) == len(
        nlin_numeric), f"Nyquist: {len(dgds_numeric)} vs {len(nlin_numeric)}"
    popt, _ = curve_fit(softplus,
                        dgds_numeric * x_norm,
                        nlin_numeric * y_norm,
                        p0=p0)
    lg.debug(f"Fitting ideal case with gvda={gvda}, gvdb={gvdb}, pulse_shape={pulse_shape}")
    # lg.info(
    #     f"Ideal fit coefficients (a, b, c): {popt[0]:.3e}, {popt[1]:.3e}, {popt[2]:.3e}")
    # lg.info(f"Ideal fit coefficients (a, b, c): {popt[0]*L**2/T**2 * 1e-30:.3e} km^2/ps^2, {popt[1]*T/L * 1e12:.3e} ps/m, {popt[2]}")
    return popt

if __name__ == "__main__":
    for ipulse in [0, 1]:
        gf = ideal_fit_coefficients(ipulse=ipulse)
