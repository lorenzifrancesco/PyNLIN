"""UWB-oriented ideal softplus fit utilities for NLIN lookup curves."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger as lg
from scipy.optimize import curve_fit

from pynlin.constellation_stats import qam_mu0
from pynlin.log_init import init_logging
from pynlin.methods.td.reference_curves import ensure_s1_ref_nlin_curve, load_s1_ref_dataset

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
    ref_path: str,
    ref_mtime_ns: int,
    time_integral_backend: str,
) -> np.ndarray:
    """Fit NLIN curves for the Raman/GVD-free case and return softplus parameters.

    This variant is TOML-independent for system parameters: it only needs
    numerical sampling settings and partial_nlin lookup tables.
    """
    p0 = [0.4, 4.5, 0.5]
    _ = ref_mtime_ns
    dataset = load_s1_ref_dataset(
        path=ref_path,
        mode="perfect",
        gvda=0.0,
        gvdb=0.0,
        time_integral_backend=time_integral_backend,
    )
    pulse_shape = str(dataset["pulse_shape"])
    llw_numeric = np.asarray(dataset["llw_grid"], dtype=float)
    nlin_numeric = np.asarray(dataset["ref_nlin_curve"], dtype=float)
    popt, _ = curve_fit(softplus,
                        llw_numeric,
                        nlin_numeric,
                        p0=p0)
    lg.debug(
        "[ideal_fits_uwb] Fitted (a,b,c)=({:.3e},{:.3e},{:.3e}) from normalized S1 reference {}".format(
            popt[0], popt[1], popt[2], Path(ref_path).name
        )
    )
    return popt


def ideal_fit_coefficients(gvda: float = 0.0,
                           gvdb: float = 0.0,
                           ipulse: int = 1,
                           fiber_length: float | None = None,
                           baud_rate: float | None = None,
                           n_samples_numeric_n: int | None = None,
                           time_integral_backend: str = "direct") -> np.ndarray:
    """Fit NLIN curves for the Raman/GVD-free case and return softplus parameters.

    Cached by pulse type to avoid repeated disk I/O and curve fitting.
    """
    if gvda != 0.0 or gvdb != 0.0:
        raise RuntimeError("Use dispersion correction for nonzero GVD; ideal fit expects gvda=gvdb=0.")
    pulse_shape = "gaussian" if ipulse == 0 else "nyquist"
    ref_path = ensure_s1_ref_nlin_curve(
        pulse_shape=pulse_shape,
        mode="perfect",
        gvda=0.0,
        gvdb=0.0,
        time_integral_backend=time_integral_backend,
    )
    return _ideal_fit_coefficients_cached(
        ipulse,
        str(ref_path),
        ref_path.stat().st_mtime_ns,
        time_integral_backend,
    ).copy()


__all__ = ["ideal_fit_coefficients", "softplus", "LLW_MIN", "LLW_MAX", "MU0", "SPATIAL_MODES"]
