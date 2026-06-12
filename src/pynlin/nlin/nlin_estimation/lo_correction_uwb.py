"""Low-order Raman correction lookup-table builders for UWB/SMF NLIN paths."""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
from loguru import logger as lg
from scipy.interpolate import RegularGridInterpolator

from pynlin.log_init import init_logging
from pynlin.nlin.nlin_estimation.config import flat_profiles_enabled
from pynlin.nlin.collision import MAX_LLD, ensure_i_low_dataset
from pynlin.nlin.cache_names import (
    s2a_lo_timeint_path,
    s2b_lo_extrema_path,
)
from pynlin.nlin.nlin_estimation.raman_integrals_uwb import load_fB


init_logging()


def build_lookup_integral_table_with_raman(cf,
                                           m_lo_truncation: int = 40,
                                           ipulse: int = 1,
                                           recompute=False,
                                           profile_path: Path | str | None = None,
                                           profile_channel_idx: int | None = None,
                                           max_lld: float | None = None) -> Tuple[callable, callable]:
    """Generate interpolants for Raman-inclusive LO corrections at fB_min and fB_max."""
    _, _, _, fB_min, fB_max = load_fB(
        cf,
        profile_path=profile_path,
        profile_channel_idx=profile_channel_idx,
    )
    use_trapezoid_only = flat_profiles_enabled(cf)
    z_samples = int(os.getenv("PYNLIN_RAMAN_LO_Z_SAMPLES", "2001"))
    z_samples = max(z_samples, 33)
    if use_trapezoid_only:
        lg.info("flat_profiles enabled; using trapezoid integration for Raman LO corrections.")
    else:
        lg.info("Using sampled trapezoid integration for Raman LO corrections.")
    n_samples = 20
    fiber_length = cf.fiber_length
    lld_max = MAX_LLD if max_lld is None else max(float(max_lld), MAX_LLD)
    lld = np.linspace(1e-30, lld_max, n_samples)
    ld = fiber_length / lld
    lg.debug(
        f"Useful range of L/LD from LO time integral data: {lld[0]:.2e} to {lld[-1]:.2e}")
    raman_correction_grid_max = np.zeros((n_samples, n_samples))
    raman_correction_grid_min = np.zeros((n_samples, n_samples))

    filename = s2b_lo_extrema_path(
        ipulse=ipulse,
        m_lo_truncation=m_lo_truncation,
        fiber_length=fiber_length,
        lld_max=lld[-1],
        custom_fB_index=profile_channel_idx,
    )
    if os.path.exists(filename) and not recompute:
        lg.info(
            "Loading precomputed Raman correction grid from {} "
            "(keys: s2b_lo_corr_max, s2b_lo_corr_min; pulse={}; m_lo_truncation={}; "
            "L={:.1f} km; L/LD_max={:.2f})".format(
                filename,
                "gaussian" if ipulse == 0 else "nyquist",
                m_lo_truncation,
                fiber_length / 1e3,
                lld[-1],
            )
        )  # FIXME: extra cache-provenance logging; remove or downgrade for high-throughput runs.
        with np.load(filename, allow_pickle=False) as data:
            raman_correction_grid_max = data["s2b_lo_corr_max"]
            raman_correction_grid_min = data["s2b_lo_corr_min"]
    else:
        lg.info(
            "Computing Raman correction grid and saving to {} "
            "(pulse={}; m_lo_truncation={}; L={:.1f} km; L/LD_max={:.2f})".format(
                filename,
                "gaussian" if ipulse == 0 else "nyquist",
                m_lo_truncation,
                fiber_length / 1e3,
                lld[-1],
            )
        )  # FIXME: extra cache-provenance logging; remove or downgrade for high-throughput runs.
        z_axis = np.linspace(0.0, fiber_length, z_samples, dtype=float)
        # x/LD = x * (L/LD)/L so each column corresponds to one LLD sample.
        x_over_ld = z_axis[:, None] / ld[None, :]
        fB_min_axis = np.asarray(fB_min(z_axis), dtype=float).reshape(-1)
        fB_max_axis = np.asarray(fB_max(z_axis), dtype=float).reshape(-1)
        if not (np.all(np.isfinite(fB_min_axis)) and np.all(np.isfinite(fB_max_axis))):
            raise ValueError("Non-finite fB_min/fB_max samples in Raman correction builder.")
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
            s2a_path = s2a_lo_timeint_path(ipulse=ipulse, m_lo=m_lo)
            lg.info(
                f"Calculating m_lo={m_lo} using S2A cache {s2a_path}"
            )  # FIXME: extra per-order cache logging; remove or downgrade for high-throughput runs.
            I_low_dataset = np.load(s2a_path)
            interp = RegularGridInterpolator(
                (I_low_dataset["lld_range"], I_low_dataset["lld_range"]),
                I_low_dataset["I_low_values"],
                bounds_error=False,
                fill_value=None,
            )

            for ida, lda in enumerate(ld):
                for idb, ldb in enumerate(ld):
                    if idb < ida:
                        add_max[ida, idb] = add_max[idb, ida]
                        add_min[ida, idb] = add_min[idb, ida]
                    else:
                        points = np.column_stack((x_over_ld[:, ida], x_over_ld[:, idb]))
                        I_axis = np.asarray(interp(points), dtype=float).reshape(-1)
                        int_max = float(np.trapezoid(I_axis * fB_max_axis, z_axis))
                        int_min = float(np.trapezoid(I_axis * fB_min_axis, z_axis))
                        add_max[ida, idb] += (int_max / fiber_length) ** 2
                        add_min[ida, idb] += (int_min / fiber_length) ** 2
            if m_lo != 0:
                add_max *= 2
                add_min *= 2
            raman_correction_grid_max += add_max
            raman_correction_grid_min += add_min

        np.savez(
            filename,
            s2b_lo_corr_max=raman_correction_grid_max,
            s2b_lo_corr_min=raman_correction_grid_min,
        )

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

    return func_wrapper(inter_min), func_wrapper(inter_max)


__all__ = ["build_lookup_integral_table_with_raman"]
