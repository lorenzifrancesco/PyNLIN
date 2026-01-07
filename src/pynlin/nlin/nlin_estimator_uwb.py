"""
UWB-friendly NLIN helpers that mirror nlin_estimator but add SMF support and
per-channel launch power overrides, while keeping the original code unchanged.
"""

import itertools as it
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from loguru import logger as lg
from scipy.constants import c
from scipy.integrate import quad

import pynlin
import pynlin.io_utils as cfg
from pynlin.system import System
from pynlin.fiber import MMFiber
from pynlin.fiber_data.load_fiber_values import load_group_delay, load_rms_gvd
from pynlin.log_init import init_logging
from pynlin.nlin.collision import build_I_low_interpolator
from pynlin.nlin.nlin_estimation.ideal_fits_uwb import ideal_fit_coefficients, softplus, LLW_MIN, LLW_MAX
from pynlin.nlin.nlin_estimation.lo_correction_uwb import build_lookup_integral_table_with_raman
from pynlin.nlin.nlin_estimation.raman_integrals_uwb import (
    load_fB,
    load_raman_integral_extremes,
    raman_integral,
)
from pynlin.utils import dBm2watt, watt2dBm

init_logging()

# Local constants mirroring the legacy estimator
# Constants imported from ideal_fits_uwb; kept for clarity
SPATIAL_MODES = np.array([1, 2, 2, 1])
MU0 = 1.3809

# --------------------
# Kappa handling
# --------------------

def get_kappa2_matrix_uwb(system: System,
                          use_kappa: bool = False,
                          use_x_mode: bool = False) -> np.ndarray:
    """Build squared coupling matrix with SMF fallback (n_modes=1 -> [[1]])."""
    n_modes = getattr(system, "n_modes", 1)
    kappa2 = np.ones((n_modes, n_modes))
    if use_kappa and n_modes > 1:
        kappa = np.loadtxt('input/fiber_data/kappa.csv', delimiter=',')
        kappa2 = kappa ** 2
    if not use_x_mode:
        kappa2 = np.multiply(kappa2, np.eye(n_modes))
    return kappa2


# --------------------
# NLIN prefactor and total NLIN
# --------------------

def _get_effective_area(system: System) -> float:
    area = getattr(system, "effective_area", None)
    if area is None and hasattr(system, "fiber"):
        area = getattr(system.fiber, "effective_area", None)
    if area is None:
        raise ValueError("effective_area not found on System or fiber")
    return area


def _get_launch_power_w(system: System) -> float:
    lp_dbm = getattr(system, "launch_power", None)
    if lp_dbm is None:
        lp_dbm = getattr(getattr(system, "wdm", None), "launch_power_dbm", None)
    if lp_dbm is None:
        raise ValueError("launch_power not defined on System/wdm")
    return dBm2watt(lp_dbm)


def _get_baud_rate(system: System) -> float:
    br = getattr(system, "baud_rate", None)
    if br is None and hasattr(system, "pulse"):
        br = getattr(system.pulse, "baud_rate", None)
    if br is None:
        raise ValueError("baud_rate not found on System or pulse")
    return br


def _get_fiber_length(system: System) -> float:
    fl = getattr(system, "fiber_length", None)
    if fl is None and hasattr(system, "fiber"):
        fl = getattr(system.fiber, "length", None)
    if fl is None:
        raise ValueError("fiber length not found on System or fiber")
    return fl


def nlin_prefactor(system: System, mode_a, mode_b):
    """Multiplicity/constellation prefactor for NLIN; SMF-friendly."""
    n_modes = getattr(system, "n_modes", 1)
    if n_modes == 1:
        mode_a = mode_b = 0
    prefactor = 1
    if mode_a == mode_b:
        prefactor *= MU0 * (2 * SPATIAL_MODES[mode_a] + 3) - 4
    else:
        prefactor *= 2 * SPATIAL_MODES[mode_b] * (MU0 - 1)
    return prefactor


def gvd_correction(system: System,
                   gvda: float,
                   gvdb: float,
                   fiber_length: Optional[float] = None,
                   m_lo_truncation: int = 3,
                   ipulse: int = 1) -> float:
    """Integrate low-order collision terms to correct the LO plateau with GVD."""
    L = _get_fiber_length(system) if fiber_length is None else fiber_length
    br = _get_baud_rate(system)
    lda = 1/(br**2 * gvda) if gvda != 0 else 1e30
    ldb = 1/(br**2 * gvdb) if gvdb != 0 else 1e30
    lo_value = 0.0
    for m_lo in range(m_lo_truncation+1):
        I_low_dataset = np.load(
            f"results/I_low_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo}.npz")
        interp = build_I_low_interpolator(I_low_dataset, ipulse=ipulse)

        def I_specific(x):
            return interp(x/lda, x/ldb)

        added_noise = (quad(I_specific, 0, L)[0] / L)**2
        if m_lo != 0:
            added_noise *= 2
        lo_value += added_noise
    return lo_value


def apply_plateau_correction(ps: Tuple[float, float, float],
                             lo_value: float,
                             ) -> Tuple[float, float, float]:
    """Rescale softplus parameters when the LO plateau value changes."""
    old_lo_value = ps[0]
    ps = list(ps)
    ps[0] = lo_value
    ps[1] = ps[1] * old_lo_value / lo_value
    return tuple(ps)


def apply_turning_point_correction(ps: Tuple[float, float, float],
                                   hi_factor: float) -> Tuple[float, float, float]:
    """Shift softplus turning point based on Raman HI correction."""
    ps = list(ps)
    ps[1] = ps[1] * hi_factor
    return tuple(ps)


def fit_nlin(system: System,
             gvda: float,
             gvdb: float,
             fB: np.ndarray,
             raman_gvd_correction_min: callable,
             raman_gvd_correction_max: callable,
             ipulse: int,
             m_lo_truncation: int = 3) -> callable:
    """Return a fitted NLIN curve for a channel pair with given GVDs and Raman profile."""

    br = _get_baud_rate(system)
    lda = 1/(br**2 * gvda) if gvda != 0 else 1e30
    ldb = 1/(br**2 * gvdb) if gvdb != 0 else 1e30
    try:
        ps_ideal = ideal_fit_coefficients(0.0, 0.0, ipulse=ipulse)
    except Exception as exc:
        lg.warning(f"ideal_fit_coefficients failed ({exc}); using fallback softplus params.")
        ps_ideal = np.array([0.4, 4.5, 0.5])

    if np.all(fB == 1.0):
        lo_value_perfect = gvd_correction(
            system, gvda, gvdb, _get_fiber_length(system), ipulse=ipulse, m_lo_truncation=m_lo_truncation)
        ps_ramanless = apply_plateau_correction(ps_ideal.copy(), lo_value_perfect)
        lg.info("You are using a flat fB, no Raman correction will be applied")
        ps = apply_plateau_correction(ps_ideal.copy(), lo_value_perfect)

        def nlin_basic(dgd):
            return softplus(dgd * _get_fiber_length(system) * br, *ps)
        nlin_basic.ps_params = ps
        return nlin_basic

    L = _get_fiber_length(system)
    lo_value_max = raman_gvd_correction_max(L/lda, L/ldb)
    lo_value_min = raman_gvd_correction_min(L/lda, L/ldb)
    extremes = _G.get("raman_extremes")
    if extremes is None:
        extremes = load_raman_integral_extremes(system)
    r_lo_min, r_lo_max, _, _ = extremes

    raman_integral_fB_lo = raman_integral(system, "LO", fB)
    raman_integral_fB_hi = raman_integral(system, "HI", fB)
    lo_value_fB = (raman_integral_fB_lo - r_lo_min) / (r_lo_max - r_lo_min) * \
        (lo_value_max - lo_value_min) + lo_value_min
    lg.trace(f"LO value: {lo_value_fB:.2e}")
    ps_ramanful = apply_plateau_correction(ps_ideal.copy(), lo_value_fB)
    ps_ramanful = apply_turning_point_correction(ps_ramanful, raman_integral_fB_hi)

    def nlin_megafit(d):
        d = d * _get_fiber_length(system) * br
        return softplus(d, *ps_ramanful)
    nlin_megafit.ps_params = ps_ramanful
    return nlin_megafit


def total_nlin_uwb(system: System,
                   collision_coeffs: np.ndarray,
                   use_kappa: bool = False,
                   use_x_mode: bool = False,
                   launch_powers_w: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert collision coefficients to NLIN PSD per channel with per-channel launch override."""
    L = _get_fiber_length(system)
    br = _get_baud_rate(system)
    x_norm = L * br
    y_norm = 1/(L * br)**2
    lg.info(f"Normalization units: x_norm = {x_norm:.2e} s, y_norm = {y_norm:.2e} s^2/m^2")
    collision_coeffs_si = collision_coeffs / y_norm

    n_modes, n_freqs, _, _ = collision_coeffs_si.shape
    if launch_powers_w is None:
        P_in_arr = np.full((n_modes, n_freqs), _get_launch_power_w(system))
    else:
        P_raw = np.asarray(launch_powers_w, dtype=float)
        if P_raw.ndim == 1:
            if P_raw.size != n_freqs:
                raise ValueError(f"launch_powers_w length {P_raw.size} != n_freqs {n_freqs}")
            P_in_arr = np.broadcast_to(P_raw[None, :], (n_modes, n_freqs))
        elif P_raw.shape == (n_modes, n_freqs):
            P_in_arr = P_raw
        else:
            raise ValueError(f"launch_powers_w shape {P_raw.shape} incompatible with (n_modes,n_freqs)=({n_modes},{n_freqs})")
        lg.info("Using per-channel launch power override for NLIN.")

    omega_0 = 2 * np.pi * system.center_frequency
    n2 = 2.6e-20  # constant of SiO2
    gamma = n2 * omega_0 / (_get_effective_area(system) * c)
    constant_prefactor = (P_in_arr**3) * (gamma**2) / (br**2)
    lg.trace(
        f"gamma={gamma:.2e} 1/(W m), P_in mean={P_in_arr.mean():.2e} W -> prefactor range [{constant_prefactor.min():.2e}, {constant_prefactor.max():.2e}]")

    kappa2 = get_kappa2_matrix_uwb(system, use_kappa, use_x_mode)

    total_nlin = np.zeros((n_modes, n_freqs))
    for mA in range(n_modes):
        for nuA in range(n_freqs):
            for mB in range(n_modes):
                for nuB in range(n_freqs):
                    prefactor = nlin_prefactor(system, mA, mB)
                    total_nlin[mA, nuA] += collision_coeffs_si[mA, nuA, mB, nuB] * kappa2[mA, mB] * prefactor
    total_nlin *= constant_prefactor
    return total_nlin


# --------------------
# Collision coefficients (SMF/MMF)
# --------------------

def _load_overlap_integrals(system: System) -> np.ndarray:
    """Load overlap integrals with SMF fallback (identity)."""
    try:
        return np.load('results/oi_fit.npy')
    except Exception:
        lg.warning("oi_fit.npy not found; using identity overlap integrals.")
        return np.ones((1, 1, 1))


def _build_fiber(system: System) -> MMFiber:
    """Return a fiber instance usable for beta1/beta2 evaluations."""
    oi = _load_overlap_integrals(system)
    gd = getattr(getattr(system, "fiber", None), "group_delay", None) or load_group_delay()
    return MMFiber(
        effective_area=_get_effective_area(system),
        overlap_integrals=oi,
        group_delay=gd,
        length=_get_fiber_length(system),
    )


def _beta_grids_from_system(system: System, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute beta1/beta2 grids from the System helper."""
    return system.beta_grids(freqs=freqs)


def collision_coeffs_system_uwb(system: System,
                                ipulse: int = 1,
                                recompute: bool = False,
                                reserve_cpus: int = 3,
                                profile_path: Path | str | None = None) -> np.ndarray:
    """Compute or load channel-pair collision coefficients (SMF or MMF)."""
    fiber_type = "smf" if system.n_modes == 1 else "mmf"
    profile_tag = Path(profile_path).stem if profile_path is not None else None
    filename = f"results/collision_coefficients_ipulse{ipulse}_{fiber_type}"
    if profile_tag:
        filename = f"{filename}_{profile_tag}"
    filename = f"{filename}.npy"
    if os.path.exists(filename) and not recompute:
        lg.info(f"Loading precomputed collision coefficients from {filename} of shape {np.load(filename).shape}")
        return np.load(filename)

    lg.info("Computing collision coefficients from scratch")
    nc = cfg.load_nc_toml_to_struct("input/numerical_config.toml")
    wdm = system.wdm
    freqs = wdm.frequency_grid()
    beta1, beta2 = _beta_grids_from_system(system, freqs)

    d_min = nc.dgd1
    d_max = nc.dgd2_g
    raman_gvd_correction_max, raman_gvd_correction_min = build_lookup_integral_table_with_raman(
        system, ipulse=ipulse, profile_path=profile_path)
    fB, fB_min, fB_max, fB_min_function, fB_max_function = load_fB(system, profile_path=profile_path)

    beta1_path = "/tmp/beta1_grid.npy"
    beta2_path = "/tmp/beta2_grid.npy"
    fB_path = "/tmp/fB.npy"
    np.save(beta1_path, beta1)
    np.save(beta2_path, beta2)
    np.save(fB_path, fB)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    A_tasks = list(it.product(range(system.n_modes), range(len(freqs))))
    n_workers = max((os.cpu_count() or 1) - reserve_cpus, 1)

    def _init_worker(beta1_path, beta2_path, fB_path, n_modes, n_freqs,
                     system, ipulse, raman_min, raman_max, n_workers, raman_extremes):
        import os
        import numpy as np
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        _G["beta1"] = np.load(beta1_path, mmap_mode="r")
        _G["beta2"] = np.load(beta2_path, mmap_mode="r")
        _G["fB"] = np.load(fB_path, mmap_mode="r")
        _G["n_modes"] = n_modes
        _G["n_freqs"] = n_freqs
        _G["n_pairs"] = (n_modes * n_freqs) ** 2
        _G["cf"] = system
        _G["ipulse"] = ipulse
        _G["raman_min"] = raman_min
        _G["raman_max"] = raman_max
        _G["n_workers"] = n_workers
        _G["raman_extremes"] = raman_extremes

    raman_extremes = load_raman_integral_extremes(system, profile_path=profile_path)
    collision_coeffs = np.zeros((system.n_modes, len(freqs), system.n_modes, len(freqs)))
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(beta1_path, beta2_path, fB_path,
                  system.n_modes, len(freqs),
                  system, ipulse, raman_gvd_correction_min, raman_gvd_correction_max,
                  n_workers, raman_extremes),
    ) as ex:
        futures = [ex.submit(_work_A, a) for a in A_tasks]
        for fut in as_completed(futures):
            mA, nuA, block, elapsed = fut.result()
            collision_coeffs[mA, nuA, :, :] = block
            lg.info(f"Finished NLIN for A(m={mA},nu={nuA:>5}) in {elapsed:>6.2f} s")

    np.save(filename, collision_coeffs)
    return collision_coeffs


# Worker global store for the UWB wrapper (kept separate to avoid clobbering originals)
_G = {
    "beta1": None, "beta2": None, "fB": None,
    "n_modes": None, "n_freqs": None, "n_pairs": None,
    "cf": None, "ipulse": None,
    "raman_min": None, "raman_max": None,
    "n_workers": None,
    "raman_extremes": None,
}


def _work_A(task_A):
    """Wrapper around nlin_estimator.work_A with the UWB globals."""
    import os
    import time
    mA, nuA = task_A
    pid = os.getpid()

    beta1 = _G["beta1"]
    beta2 = _G["beta2"]
    fBmm = _G["fB"]
    n_modes = _G["n_modes"]
    n_freqs = _G["n_freqs"]
    n_pairs = _G["n_pairs"]
    system = _G["cf"]
    ipulse = _G["ipulse"]
    rmin = _G["raman_min"]
    rmax = _G["raman_max"]
    nw = _G["n_workers"]

    beta1_A = beta1[mA, nuA]
    gvda = beta2[mA, nuA]
    block = np.empty((n_modes, n_freqs), dtype=float)

    start = time.time()
    for mB in range(n_modes):
        for nuB in range(n_freqs):
            idx = (mA * n_freqs + nuA) * n_modes * n_freqs + (mB * n_freqs + nuB) + 1
            try:
                import logging
                lg_local = logging.getLogger(__name__)
                lg_local.info(
                    f"[worker {pid:>6}/{nw:>2}] "
                    f"Computing NLIN A(m={mA},nu={nuA:>5}) vs B(m={mB},nu={nuB:>5}) "
                    f"({idx:>7}/{n_pairs:>7})"
                )
            except Exception:
                pass

            gvdb = beta2[mB, nuB]
            dgd = abs(beta1_A - beta1[mB, nuB])
            val = fit_nlin(
                system,
                abs(gvda),
                abs(gvdb),
                fB=fBmm[:, mB, nuB],
                raman_gvd_correction_min=rmin,
                raman_gvd_correction_max=rmax,
                ipulse=ipulse,
            )(dgd)
            block[mB, nuB] = val

    return mA, nuA, block, time.time() - start


# Re-export softplus and ideal_fit_coefficients for convenience
__all__ = [
    "collision_coeffs_system_uwb",
    "collision_coeffs_from_system",
    "total_nlin_uwb",
    "total_nlin_from_system",
    "get_kappa2_matrix_uwb",
    "nlin_prefactor",
    "gvd_correction",
    "apply_plateau_correction",
    "apply_turning_point_correction",
    "fit_nlin",
    "softplus",
    "ideal_fit_coefficients",
]


# ---------------------------------
# System-centric convenience APIs
# ---------------------------------

def collision_coeffs_from_system(system,
                                 ipulse: int = 1,
                                 recompute: bool = False,
                                 reserve_cpus: int = 15,
                                 profile_path: Path | str | None = None) -> np.ndarray:
    """Modern helper that accepts a System and delegates to collision_coeffs_system_uwb."""
    return collision_coeffs_system_uwb(
        system,
        ipulse=ipulse,
        recompute=recompute,
        reserve_cpus=reserve_cpus,
        profile_path=profile_path,
    )


def total_nlin_from_system(system,
                           collision_coeffs: Optional[np.ndarray] = None,
                           use_kappa: bool = False,
                           use_x_mode: bool = False,
                           launch_powers_w: Optional[np.ndarray] = None,
                           reserve_cpus: int = 3,
                           profile_path: Path | str | None = None) -> np.ndarray:
    """Modern helper that accepts a System and computes NLIN end-to-end."""
    ccfs = collision_coeffs if collision_coeffs is not None else collision_coeffs_system_uwb(
        system, ipulse=1, recompute=False, reserve_cpus=reserve_cpus, profile_path=profile_path)
    return total_nlin_uwb(
        system,
        ccfs,
        use_kappa=use_kappa,
        use_x_mode=use_x_mode,
        launch_powers_w=launch_powers_w,
    )
