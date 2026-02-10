import os
from pathlib import Path
from typing import Tuple

import numpy as np
from loguru import logger as lg
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import RegularGridInterpolator

from pynlin.log_init import init_logging
from pynlin.nlin.collision import MAX_LLD, build_I_low_interpolator, ensure_i_low_dataset
from pynlin.nlin.nlin_estimation.raman_integrals_uwb import load_fB

init_logging()


def build_lookup_integral_table_with_raman(cf,
                                           m_lo_truncation: int = 2,
                                           ipulse: int = 1,
                                           recompute=False,
                                           profile_path: Path | str | None = None,
                                           max_lld: float | None = None) -> Tuple[callable, callable]:
    """Generate interpolants for Raman-inclusive LO corrections at fB_min and fB_max."""
    _, _, _, fB_min, fB_max = load_fB(cf, profile_path=profile_path)
    n_samples = 20
    fiber_length = cf.fiber_length
    lld_max = MAX_LLD if max_lld is None else max(float(max_lld), MAX_LLD)
    lld = np.linspace(1e-30, lld_max, n_samples)
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
            ensure_i_low_dataset(
                m_lo=m_lo,
                ipulse=ipulse,
                baud_rate=float(cf.baud_rate),
                fiber_length=float(fiber_length),
                max_lld=float(lld[-1]),
                recompute=recompute,
            )
            add_min = np.zeros((n_samples, n_samples))
            add_max = np.zeros((n_samples, n_samples))
            lg.info(f"Calculating m_lo={m_lo}")
            I_low_dataset = np.load(
                f"results/I_low_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo}.npz")
            interp = build_I_low_interpolator(I_low_dataset, ipulse=ipulse)
            def _safe_integrate(func, a: float, b: float, label: str) -> float:
                import warnings

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", IntegrationWarning)
                    val, _err = quad(func, a, b, limit=200)
                if any(issubclass(w.category, IntegrationWarning) for w in caught):
                    lg.warning(
                        f"IntegrationWarning in {label}; falling back to trapezoid on a coarse grid."
                    )
                    x = np.linspace(a, b, 2001)
                    y = np.vectorize(func)(x)
                    val = float(np.trapezoid(y, x))
                return float(val)

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
                        int_max = _safe_integrate(
                            lambda x: I_specific(x) * fB_max(x), 0, fiber_length, "fB_max"
                        )
                        int_min = _safe_integrate(
                            lambda x: I_specific(x) * fB_min(x), 0, fiber_length, "fB_min"
                        )
                        add_max[ida, idb] += (int_max / fiber_length) ** 2
                        add_min[ida, idb] += (int_min / fiber_length) ** 2
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
        warned = False

        def func_wrapper(x, y):
            nonlocal warned
            if x < 0 or y < 0:
                raise ValueError("Input has negative values; check that your LD is positive")
            max_allowed = 1.01 * lld[-1]
            if x > max_allowed or y > max_allowed:
                x0, y0 = x, y
                x = min(x, max_allowed)
                y = min(y, max_allowed)
                if not warned:
                    warned = True
                    lg.warning(
                        f"Clamping Raman correction lookup to L/LD={max_allowed:.2f}; "
                        f"inputs were ({x0:.2f}, {y0:.2f}). "
                        f"Consider regenerating the grid with a larger range."
                    )
            return func((x, y))
        return func_wrapper

    return func_wrapper(inter_max), func_wrapper(inter_min)


__all__ = ["build_lookup_integral_table_with_raman"]
