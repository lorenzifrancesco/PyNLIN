"""UWB-oriented ideal softplus fit utilities for NLIN lookup curves."""

from functools import lru_cache

import numpy as np
from loguru import logger as lg
from scipy.optimize import curve_fit

from pynlin.constellation_stats import qam_mu0
from pynlin.log_init import init_logging

init_logging()

SPATIAL_MODES = np.array([1, 2, 2, 1])
LLW_MIN = 0.01  # target L/LW
LLW_MAX = 100.0
MU0 = qam_mu0(64)


def softplus(x, a, b, c):
    """Smoothly increasing three-parameter curve used to fit NLIN vs walk-off."""
    return a * (1 + (x / b) ** (1 / c)) ** (-c)


@lru_cache(maxsize=64)
def _ideal_fit_coefficients_cached(
    ipulse: int,
    x_norm: float,
    n_samples_numeric_n: int,
) -> np.ndarray:
    """Fit NLIN curves for the Raman/GVD-free case and return softplus parameters.

    This variant is TOML-independent for system parameters: it only needs
    numerical sampling settings and partial_nlin lookup tables.
    """
    llw_numeric = np.logspace(np.log10(LLW_MIN), np.log10(LLW_MAX), int(n_samples_numeric_n))
    y_norm = x_norm ** (-2)
    p0 = [0.4, 4.5, 0.5]

    pulse_shape = "gaussian" if ipulse == 0 else "nyquist"

    nlin_numeric = np.load(f"results/partial_nlin_{pulse_shape}_perfect_0.0_0.0.npy")

    assert len(llw_numeric) == len(nlin_numeric), f"{pulse_shape}: {len(llw_numeric)} vs {len(nlin_numeric)}"
    popt, _ = curve_fit(softplus,
                        llw_numeric,
                        nlin_numeric * y_norm,
                        p0=p0)
    lg.debug(
        "[ideal_fits_uwb] Fitted (a,b,c)=({:.3e},{:.3e},{:.3e}) with x_norm={:.3e}".format(
            popt[0], popt[1], popt[2], x_norm
        )
    )
    return popt


def ideal_fit_coefficients(gvda: float = 0.0,
                           gvdb: float = 0.0,
                           ipulse: int = 1,
                           fiber_length: float | None = None,
                           baud_rate: float | None = None,
                           n_samples_numeric_n: int | None = None) -> np.ndarray:
    """Fit NLIN curves for the Raman/GVD-free case and return softplus parameters.

    Cached by pulse type to avoid repeated disk I/O and curve fitting.
    """
    if gvda != 0.0 or gvdb != 0.0:
        raise RuntimeError("Use dispersion correction for nonzero GVD; ideal fit expects gvda=gvdb=0.")
    if fiber_length is None or baud_rate is None:
        raise ValueError(
            "ideal_fit_coefficients requires fiber_length and baud_rate from the active run config."
        )
    if n_samples_numeric_n is None:
        raise ValueError(
            "ideal_fit_coefficients requires n_samples_numeric_n from the active numerical config."
        )
    x_norm = float(fiber_length) * float(baud_rate)
    if x_norm <= 0:
        raise ValueError(f"Invalid x_norm derived from fiber_length/baud_rate: {x_norm}")
    if int(n_samples_numeric_n) <= 0:
        raise ValueError(f"Invalid n_samples_numeric_n: {n_samples_numeric_n}")
    return _ideal_fit_coefficients_cached(ipulse, x_norm, int(n_samples_numeric_n)).copy()


__all__ = ["ideal_fit_coefficients", "softplus", "LLW_MIN", "LLW_MAX", "MU0", "SPATIAL_MODES"]
