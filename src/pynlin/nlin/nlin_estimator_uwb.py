"""
UWB-friendly NLIN helpers that mirror nlin_estimator but add SMF support and
per-channel launch power overrides, while keeping the original code unchanged.
"""

import itertools as it
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from loguru import logger as lg
from scipy.constants import c
from scipy.integrate import quad

import pynlin
from pynlin.constellation_stats import qam_mu0
from pynlin.system import System
from pynlin.fiber import MMFiber
from pynlin.fiber_data.load_fiber_values import load_group_delay, load_rms_gvd
from pynlin.log_init import init_logging
from pynlin.nlin.cache_names import (
    s2_beta1_grid_path,
    s2_beta2_grid_path,
    s2_fB_grid_path,
    s2a_lo_timeint_path,
    s3_pair_nlin_kernel_path,
)
from pynlin.nlin.collision import build_I_low_interpolator, ensure_i_low_dataset
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
MU0 = qam_mu0(64)


def _flat_profiles_enabled(system) -> bool:
    """Return whether flat-profile mode is enabled in run configuration."""
    raw = getattr(system, "raw_config", None)
    if not isinstance(raw, dict):
        return False
    pog = raw.get("pcfm")
    if isinstance(pog, dict):
        run = pog.get("run")
        if isinstance(run, dict):
            mode = run.get("power_profiles_mode")
            if isinstance(mode, str):
                mode = mode.strip().lower()
                if mode == "flat":
                    return True
                if mode in {
                    "cached",
                    "recompute",
                    "cached_no_profile_launch",
                    "recompute_no_profile_launch",
                }:
                    return False
    nlin_section = raw.get("nlin")
    if not isinstance(nlin_section, dict):
        return False
    return bool(nlin_section.get("flat_profiles") or nlin_section.get("flat_profile"))


def _resolve_n_workers(reserve_cpus: int | None) -> int:
    """Resolve worker count from reserve policy and optional environment override."""
    total_cpus = os.cpu_count() or 1
    workers_env = os.getenv("PYNLIN_NLIN_WORKERS")
    if workers_env:
        try:
            workers = int(workers_env)
        except ValueError:
            lg.warning(f"Invalid PYNLIN_NLIN_WORKERS={workers_env!r}; ignoring override.")
        else:
            if workers > 0:
                return min(workers, total_cpus)
            lg.warning(f"PYNLIN_NLIN_WORKERS={workers} is not positive; ignoring override.")

    if reserve_cpus is None:
        reserve_env = os.getenv("PYNLIN_RESERVE_CPUS")
        if reserve_env:
            try:
                reserve = int(reserve_env)
            except ValueError:
                lg.warning(f"Invalid PYNLIN_RESERVE_CPUS={reserve_env!r}; using default reserve=1.")
                reserve = 1
        else:
            reserve = 1
    else:
        reserve = int(reserve_cpus)

    reserve = max(reserve, 0)
    if reserve >= total_cpus:
        lg.warning(
            f"reserve_cpus={reserve} leaves no free worker slots on this host ({total_cpus} CPUs); "
            "using a single worker."
        )
        return 1
    return max(total_cpus - reserve, 1)

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
        lg.warning("Applying kappa.csv coupling weights. Check the Manakov averaging.")
        kappa = np.loadtxt('input/fiber_data/kappa_uwb.csv', delimiter=',') # 8/9 inserted by hand
        kappa2 = kappa ** 2
    if not use_x_mode:
        kappa2 = np.multiply(kappa2, np.eye(n_modes))
    return kappa2


# --------------------
# NLIN prefactor and total NLIN
# --------------------

def _get_effective_area(system: System) -> float:
    """Get scalar effective area from ``system`` or ``system.fiber``."""
    area = getattr(system, "effective_area", None)
    if area is None and hasattr(system, "fiber"):
        area = getattr(system.fiber, "effective_area", None)
    if area is None:
        raise ValueError("effective_area not found on System or fiber")
    return area


def _effective_area_array(system: System, freqs: np.ndarray) -> np.ndarray:
    """Return effective area per channel, falling back to scalar if needed."""
    fiber = getattr(system, "fiber", None)
    if fiber is not None and hasattr(fiber, "effective_area_at"):
        wl = c / freqs
        vals = np.array([fiber.effective_area_at(float(w)) for w in wl], dtype=float)
        if np.all(np.isfinite(vals)) and np.all(vals > 0):
            return vals
    return np.full_like(freqs, _get_effective_area(system), dtype=float)


def _get_launch_power_w(system: System) -> float:
    """Resolve scalar launch power from system settings and convert to Watt."""
    lp_dbm = getattr(system, "launch_power", None)
    if lp_dbm is None:
        lp_dbm = getattr(getattr(system, "wdm", None), "launch_power_dbm", None)
    if lp_dbm is None:
        raise ValueError("launch_power not defined on System/wdm")
    return dBm2watt(lp_dbm)


def _get_baud_rate(system: System) -> float:
    """Resolve baud rate from ``system`` or ``system.pulse``."""
    br = getattr(system, "baud_rate", None)
    if br is None and hasattr(system, "pulse"):
        br = getattr(system.pulse, "baud_rate", None)
    if br is None:
        raise ValueError("baud_rate not found on System or pulse")
    return br


def _get_fiber_length(system: System) -> float:
    """Resolve fiber length from ``system`` or ``system.fiber``."""
    fl = getattr(system, "fiber_length", None)
    if fl is None and hasattr(system, "fiber"):
        fl = getattr(system.fiber, "length", None)
    if fl is None:
        raise ValueError("fiber length not found on System or fiber")
    return fl


def _get_n_samples_numeric_n(system: System) -> int:
    """Read and validate ``system.numerics.n_samples_numeric_n``."""
    numerics = getattr(system, "numerics", None)
    n_samples = getattr(numerics, "n_samples_numeric_n", None)
    if n_samples is None:
        raise ValueError(
            "system.numerics.n_samples_numeric_n is required for ideal_fit_coefficients. "
            "Provide a numerical_config.toml for this run."
        )
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError(f"Invalid system.numerics.n_samples_numeric_n: {n_samples}")
    return n_samples


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
    lld_max = max(abs(br**2 * gvda * L), abs(br**2 * gvdb * L), 1e-30)
    lo_value = 0.0
    use_trapezoid_only = _flat_profiles_enabled(system)
    for m_lo in range(m_lo_truncation+1):
        ensure_i_low_dataset(
            m_lo=m_lo,
            ipulse=ipulse,
            baud_rate=br,
            fiber_length=L,
            max_lld=lld_max,
            recompute=False,
        )
        I_low_dataset = np.load(s2a_lo_timeint_path(ipulse=ipulse, m_lo=m_lo))
        interp = build_I_low_interpolator(I_low_dataset, ipulse=ipulse)

        def I_specific(x):
            return interp(x/lda, x/ldb)

        if use_trapezoid_only:
            x = np.linspace(0.0, L, 2001)
            y = np.vectorize(I_specific)(x)
            added_noise = (float(np.trapezoid(y, x)) / L) ** 2
        else:
            added_noise = (quad(I_specific, 0, L)[0] / L) ** 2
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
    L = _get_fiber_length(system)
    n_samples_numeric_n = _get_n_samples_numeric_n(system)
    lda = 1/(br**2 * gvda) if gvda != 0 else 1e30
    ldb = 1/(br**2 * gvdb) if gvdb != 0 else 1e30
    ps_ideal = ideal_fit_coefficients(
        0.0,
        0.0,
        ipulse=ipulse,
        fiber_length=L,
        baud_rate=br,
        n_samples_numeric_n=n_samples_numeric_n,
    )

    if np.all(fB == 1.0):
        lg.trace("You are using a flat fB, no Raman correction will be applied")
        lo_value_max = raman_gvd_correction_max(L/lda, L/ldb)
        lo_value_min = raman_gvd_correction_min(L/lda, L/ldb)
        extremes = _G.get("raman_extremes")
        if extremes is None:
            extremes = load_raman_integral_extremes(system)
        r_lo_min, r_lo_max, _, _ = extremes
        if np.isclose(r_lo_max, r_lo_min):
            lo_value_fB = lo_value_min
        else:
            raman_integral_fB_lo = raman_integral(system, "LO", fB)
            lo_value_fB = (raman_integral_fB_lo - r_lo_min) / (r_lo_max - r_lo_min) * \
                (lo_value_max - lo_value_min) + lo_value_min
        ps = apply_plateau_correction(ps_ideal.copy(), lo_value_fB)

        def nlin_basic(dgd):
            return softplus(dgd * L * br, *ps)
        nlin_basic.ps_params = ps
        return nlin_basic

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
                   launch_powers_w: Optional[np.ndarray] = None,
                   exclude_self_channel: bool = True,
                   cache_path: Path | str | None = None,
                   recompute: bool = False) -> np.ndarray:
    """Convert collision coefficients to NLIN power per channel with optional caching."""
    cache_target = Path(cache_path) if cache_path is not None else None
    if cache_target is not None and cache_target.exists() and not recompute:
        lg.info(f"Loading cached total NLIN from {cache_target}")
        cached = np.load(cache_target, allow_pickle=True)
        if isinstance(cached, np.lib.npyio.NpzFile):
            data = cached.get("nlin")
            if data is not None:
                cached = data
            else:
                cached = None
            cached.close()
        if cached is not None:
            expected_shape = (collision_coeffs.shape[0], collision_coeffs.shape[1])
            if getattr(cached, "shape", None) != expected_shape:
                lg.warning(
                    f"Cached NLIN shape {getattr(cached, 'shape', None)} != expected {expected_shape}; recomputing."
                )
            else:
                return cached
    L = _get_fiber_length(system)
    br = _get_baud_rate(system)
    x_norm = L * br
    y_norm = 1/(L * br)**2
    lg.info(f"Normalization units: x_norm = {x_norm:.2e} s, y_norm = {y_norm:.2e} s^2/m^2")
    collision_coeffs_si = collision_coeffs / y_norm
    if exclude_self_channel:
        lg.warning(
            "\n"
            "====================================================================\n"
            " WARNING: UWB TD SELF-CHANNEL EXCLUSION ENABLED                   \n"
            " - total_nlin_uwb() is running with exclude_self_channel=True      \n"
            " - Terms with nuB == nuA are removed from the TD summation         \n"
            " - Output is NOT full TD NLIN; it is an XCI-like diagnostic metric \n"
            "===================================================================="
        )

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

    freqs = system.wdm.frequency_grid()
    if freqs.size != n_freqs:
        raise ValueError(f"freq grid size {freqs.size} != n_freqs {n_freqs}")
    n2 = 2.6e-20  # constant of SiO2
    aeff = _effective_area_array(system, freqs)
    gamma = n2 * (2 * np.pi * freqs) / (aeff * c)
    gamma = gamma[None, :]
    constant_prefactor = (P_in_arr**3) * (gamma**2) / (br**2)
    lg.trace(
        "gamma[min/mean/max]=({:.2e}, {:.2e}, {:.2e}) 1/(W m), "
        "P_in mean={:.2e} W -> prefactor range [{:.2e}, {:.2e}]".format(
            float(np.min(gamma)),
            float(np.mean(gamma)),
            float(np.max(gamma)),
            float(P_in_arr.mean()),
            float(constant_prefactor.min()),
            float(constant_prefactor.max()),
        )
    )

    kappa2 = get_kappa2_matrix_uwb(system, use_kappa, use_x_mode)
    assert(np.isclose(kappa2[0, 0], (8.0/9.0)**2, atol=0.01), f"Expected kappa2[0,0] to be (8/9)^2 for SMF fallback, got {kappa2[0,0]}")
    total_nlin = np.zeros((n_modes, n_freqs))
    for mA in range(n_modes):
        for nuA in range(n_freqs):
            for mB in range(n_modes):
                for nuB in range(n_freqs):
                    if exclude_self_channel and nuB == nuA:
                        continue
                    prefactor = nlin_prefactor(system, mA, mB)
                    total_nlin[mA, nuA] += collision_coeffs_si[mA, nuA, mB, nuB] * kappa2[mA, mB] * prefactor
    total_nlin *= constant_prefactor
    if cache_target is not None:
        cache_target.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_target, total_nlin)
        lg.info(f"Saved total NLIN cache to {cache_target}")
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


def _max_lld_from_beta2(system: System, beta2: np.ndarray) -> float | None:
    """Return max L/LD from beta2 grid."""
    try:
        L = float(system.fiber_length)
        br = float(system.baud_rate)
    except (TypeError, ValueError):
        return None
    max_b2 = float(np.nanmax(np.abs(beta2)))
    if not np.isfinite(max_b2) or max_b2 <= 0.0:
        return None
    return L * br * br * max_b2


def collision_coeffs_system_uwb(system: System,
                                ipulse: int = 1,
                                recompute: bool = False,
                                reserve_cpus: int | None = None,
                                profile_path: Path | str | None = None) -> np.ndarray:
    """Compute or load channel-pair collision coefficients (SMF or MMF)."""
    fiber_type = "smf" if system.n_modes == 1 else "mmf"
    profile_tag = Path(profile_path).stem if profile_path is not None else None
    br_hz = _get_baud_rate(system)
    spacing_hz = getattr(system.wdm, "spacing", None)
    freqs = system.wdm.frequency_grid()
    n_ch = len(freqs)
    beta1, beta2 = _beta_grids_from_system(system, freqs)

    disp_signature = np.ascontiguousarray(
        np.stack([beta1, beta2], axis=0),
        dtype=np.float64,
    ).view(np.uint8)
    disp_tag = hashlib.sha1(disp_signature).hexdigest()[:12]

    filename = s3_pair_nlin_kernel_path(
        ipulse=ipulse,
        fiber_type=fiber_type,
        br_hz=br_hz,
        n_ch=n_ch,
        spacing_hz=spacing_hz,
        profile_tag=profile_tag,
        disp_tag=disp_tag,
    )
    if os.path.exists(filename) and not recompute:
        lg.info(f"Loading precomputed collision coefficients from {filename} of shape {np.load(filename).shape}")
        return np.load(filename)

    lg.info("Computing collision coefficients from scratch")
    n_samples_numeric_n = _get_n_samples_numeric_n(system)
    max_lld = _max_lld_from_beta2(system, beta2)
    # Prebuild shared lookup tables once in the parent process to avoid
    # concurrent worker writes to the same cached npz/npy files.
    build_lookup_integral_table_with_raman(
        system,
        ipulse=ipulse,
        recompute=recompute,
        profile_path=profile_path,
        max_lld=max_lld,
    )
    ideal_fit_coefficients(
        0.0,
        0.0,
        ipulse=ipulse,
        fiber_length=_get_fiber_length(system),
        baud_rate=_get_baud_rate(system),
        n_samples_numeric_n=n_samples_numeric_n,
    )
    fB, fB_min, fB_max, fB_min_function, fB_max_function = load_fB(system, profile_path=profile_path)

    beta1_path = str(s2_beta1_grid_path())
    beta2_path = str(s2_beta2_grid_path())
    fB_path = str(s2_fB_grid_path())
    np.save(beta1_path, beta1)
    np.save(beta2_path, beta2)
    np.save(fB_path, fB)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    A_tasks = list(it.product(range(system.n_modes), range(len(freqs))))
    n_workers = _resolve_n_workers(reserve_cpus)
    lg.info(
        f"UWB NLIN workers: {n_workers} (cpu_count={os.cpu_count() or 1}, reserve_cpus={reserve_cpus})"
    )

    # FIXME taking only the first 100 frequs for benchmarking
    # freqs = freqs[:100]
    raman_extremes = load_raman_integral_extremes(system, profile_path=profile_path)
    collision_coeffs = np.zeros((system.n_modes, len(freqs), system.n_modes, len(freqs)))
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
    executor_kwargs = {
        "max_workers": n_workers,
        "initializer": _init_collision_worker,
        "initargs": (
            beta1_path,
            beta2_path,
            fB_path,
            system.n_modes,
            len(freqs),
            system,
            ipulse,
            profile_path,
            max_lld,
            False,
            n_workers,
            raman_extremes,
        ),
    }
    if os.name == "posix":
        executor_kwargs["mp_context"] = mp.get_context("fork")

    with ProcessPoolExecutor(**executor_kwargs) as ex:
        futures = [ex.submit(_work_A, a) for a in A_tasks]
        progress = tqdm(total=len(futures), desc="TD channels", unit="ch") if tqdm else None
        for fut in as_completed(futures):
            mA, nuA, block, elapsed = fut.result()
            collision_coeffs[mA, nuA, :, :] = block
            lg.trace(f"Finished NLIN for A(m={mA},nu={nuA:>5}) in {elapsed:>6.2f} s")
            if progress is not None:
                progress.update(1)
        if progress is not None:
            progress.close()

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


def _init_collision_worker(
    beta1_path,
    beta2_path,
    fB_path,
    n_modes,
    n_freqs,
    system,
    ipulse,
    profile_path,
    max_lld,
    recompute_lookup_tables,
    n_workers,
    raman_extremes,
):
    """Populate worker globals for ProcessPool collision-coefficient jobs."""
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
    raman_max, raman_min = build_lookup_integral_table_with_raman(
        system,
        ipulse=ipulse,
        recompute=bool(recompute_lookup_tables),
        profile_path=profile_path,
        max_lld=max_lld,
    )
    _G["raman_min"] = raman_min
    _G["raman_max"] = raman_max
    _G["n_workers"] = n_workers
    _G["raman_extremes"] = raman_extremes


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
                                 reserve_cpus: int | None = None,
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
                           reserve_cpus: int | None = None,
                           profile_path: Path | str | None = None,
                           cache_path: Path | str | None = None,
                           recompute: bool = False) -> np.ndarray:
    """Modern helper that accepts a System and computes NLIN end-to-end."""
    ccfs = collision_coeffs if collision_coeffs is not None else collision_coeffs_system_uwb(
        system, ipulse=1, recompute=False, reserve_cpus=reserve_cpus, profile_path=profile_path)
    return total_nlin_uwb(
        system,
        ccfs,
        use_kappa=use_kappa,
        use_x_mode=use_x_mode,
        launch_powers_w=launch_powers_w,
        cache_path=cache_path,
        recompute=recompute,
    )
