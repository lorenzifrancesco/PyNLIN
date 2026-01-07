import os
from pathlib import Path
from typing import Tuple

import numpy as np
from loguru import logger as lg
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

from pynlin.log_init import init_logging
from pynlin.nlin.collision import MAX_LLD, build_I_low_interpolator
from pynlin.nlin.nlin_estimation.raman_integrals_uwb import load_fB

init_logging()


def build_lookup_integral_table_with_raman(cf,
                                           m_lo_truncation: int = 2,
                                           ipulse: int = 1,
                                           recompute=False,
                                           profile_path: Path | str | None = None) -> Tuple[callable, callable]:
    """Generate interpolants for Raman-inclusive LO corrections at fB_min and fB_max."""
    _, _, _, fB_min, fB_max = load_fB(cf, profile_path=profile_path)
    n_samples = 20
    fiber_length = cf.fiber_length
    lld = np.linspace(1e-30, MAX_LLD, n_samples)
    ld = fiber_length / lld
    lg.debug(
        f"Useful range of L/LD from LO time integral data: {lld[0]:.2e} to {lld[-1]:.2e}")
    raman_correction_grid_max = np.zeros((n_samples, n_samples))
    raman_correction_grid_min = np.zeros((n_samples, n_samples))

    filename = f"results/raman_correction_grid_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.npy"

    if os.path.exists(filename) and not recompute:
        lg.info(f"Loading precomputed Raman correction grid from {filename}")
        data = np.load(filename, allow_pickle=True).item()
        raman_correction_grid_max = data['raman_correction_grid_max']
        raman_correction_grid_min = data['raman_correction_grid_min']
    else:
        lg.info(f"Computing Raman correction grid and saving to {filename}")
        for m_lo in range(m_lo_truncation + 1):
            add_min = np.zeros((n_samples, n_samples))
            add_max = np.zeros((n_samples, n_samples))
            lg.info(f"Calculating m_lo={m_lo}")
            I_low_dataset = np.load(
                f"results/I_low_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo}.npz")
            interp = build_I_low_interpolator(I_low_dataset, ipulse=ipulse)
            for ida, lda in enumerate(ld):
                for idb, ldb in enumerate(ld):
                    if idb < ida:
                        add_max[ida, idb] = add_max[idb, ida]
                        add_min[ida, idb] = add_min[idb, ida]
                    else:
                        def I_specific(x):
                            return interp(x/lda, x/ldb)
                        lg.debug(f"Maximum of fB_max: {fB_max(fiber_length):.2e}")
                        lg.debug(f"Maximum of fB_min: {fB_min(fiber_length):.2e}")
                        add_max[ida, idb] += (
                            quad(lambda x: I_specific(x) * fB_max(x), 0, fiber_length)[0] / fiber_length) ** 2
                        add_min[ida, idb] += (
                            quad(lambda x: I_specific(x) * fB_min(x), 0, fiber_length)[0] / fiber_length) ** 2
            if m_lo != 0:
                add_max *= 2
                add_min *= 2
            raman_correction_grid_max += add_max
            raman_correction_grid_min += add_min

        np.save(filename, {
            'raman_correction_grid_max': raman_correction_grid_max,
            'raman_correction_grid_min': raman_correction_grid_min,
        })

    inter_max = RegularGridInterpolator(
        (lld, lld),
        raman_correction_grid_max,
        bounds_error=False,
        fill_value=None)
    inter_min = RegularGridInterpolator(
        (lld, lld),
        raman_correction_grid_min,
        bounds_error=False,
        fill_value=None)

    def func_wrapper(func):
        def func_wrapper(x, y):
            if x < 0 or y < 0:
                raise ValueError("Input has negative values; check that your LD is positive")
            max_allowed = 1.01 * lld[-1]
            if x > max_allowed or y > max_allowed:
                x = min(x, max_allowed)
                y = min(y, max_allowed)
                lg.warning(
                    f"Clamping Raman correction lookup to L/LD={max_allowed:.2f}; inputs were ({x:.2f}, {y:.2f}). "
                    f"Consider regenerating the grid with a larger range."
                )
            return func((x, y))
        return func_wrapper

    return func_wrapper(inter_max), func_wrapper(inter_min)


__all__ = ["build_lookup_integral_table_with_raman"]
