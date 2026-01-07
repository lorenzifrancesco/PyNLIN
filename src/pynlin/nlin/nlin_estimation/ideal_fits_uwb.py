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
MU0 = 1.3809


def softplus(x, a, b, c):
    """Smoothly increasing three-parameter curve used to fit NLIN vs walk-off."""
    return a * (1 + (x / b) ** (1 / c)) ** (-c)


def ideal_fit_coefficients(gvda: float = 0.0,
                           gvdb: float = 0.0,
                           ipulse: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Fit NLIN curves for the Raman/GVD-free case and return softplus parameters.

    This variant tries multiple config files (mmf, uwb, dummy) to avoid failures
    when amplification fields are missing in a single file.
    """
    candidate_structs = ["./input/mmf.toml", "./input/uwb_struct.toml", "./input/dummy_struct.toml"]
    cf = None
    last_exc = None
    for path in candidate_structs:
        try:
            cf = cfg.load_toml_to_struct(path)
            lg.debug(f"[ideal_fits_uwb] Loaded config from {path}")
            break
        except Exception as exc:
            last_exc = exc
            lg.debug(f"[ideal_fits_uwb] Skipping {path}: {exc}")
            continue
    if cf is None:
        raise RuntimeError(f"ideal_fit_coefficients could not load any config; last error: {last_exc}")

    nc = cfg.load_nc_toml_to_struct("./input/numerical_config.toml")

    nc.dgd1 = LLW_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = LLW_MAX / (cf.fiber_length * cf.baud_rate)
    dgd2 = nc.dgd2_n

    dgds_numeric = np.logspace(np.log10(nc.dgd1), np.log10(dgd2), nc.n_samples_numeric_n)

    T = 1 / cf.baud_rate
    L = cf.fiber_length
    x_norm = L / T
    y_norm = x_norm ** (-2)
    p0 = [0.4, 4.5, 0.5]

    pulse_shape = "gaussian" if ipulse == 0 else "nyquist"

    if gvda != 0.0 or gvdb != 0.0:
        raise RuntimeError("Use dispersion correction for nonzero GVD; ideal fit expects gvda=gvdb=0.")
    nlin_numeric = np.load(f"results/partial_nlin_{pulse_shape}_perfect_0.0_0.0.npy")

    assert len(dgds_numeric) == len(nlin_numeric), f"{pulse_shape}: {len(dgds_numeric)} vs {len(nlin_numeric)}"
    popt, _ = curve_fit(softplus,
                        dgds_numeric * x_norm,
                        nlin_numeric * y_norm,
                        p0=p0)
    cfg_label = getattr(cf, "source", None) or "inline-config"
    lg.debug(f"[ideal_fits_uwb] Fitted (a,b,c)=({popt[0]:.3e},{popt[1]:.3e},{popt[2]:.3e}) using {cfg_label}")
    return popt


__all__ = ["ideal_fit_coefficients", "softplus", "LLW_MIN", "LLW_MAX", "MU0", "SPATIAL_MODES"]
