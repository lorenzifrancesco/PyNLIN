"""Legacy NLIN estimator with Raman/GVD softplus fitting and collision summation."""

import itertools as it
import os
import time
from typing import Tuple

import numpy as np
from loguru import logger as lg
from scipy.constants import c
from scipy.integrate import quad

import pynlin
from pynlin.constellation_stats import qam_mu0
from pynlin.fiber_data.load_fiber_values import load_group_delay, load_rms_gvd
from pynlin.log_init import init_logging
from pynlin.nlin.cache_names import (
    s2_beta1_grid_path,
    s2_beta2_grid_path,
    s2_fB_grid_path,
    s2a_lo_timeint_path,
    s3_pair_nlin_kernel_path,
)
from pynlin.nlin.collision import MAX_LLD, build_I_low_interpolator
from pynlin.nlin.nlin_estimation.raman_integrals import load_fB
from pynlin.fiber import MMFiber
from pynlin.utils import dBm2watt, watt2dBm

init_logging()

from pynlin.nlin.nlin_estimation.ideal_fits import ideal_fit_coefficients, softplus
from pynlin.nlin.nlin_estimation.lo_correction import (
    build_lookup_integral_table_with_raman,
)
from pynlin.nlin.nlin_estimation.raman_integrals import (
    load_raman_integral_extremes,
    raman_integral,
)

SPATIAL_MODES = np.array([1, 2, 2, 1])
LLW_MIN = 0.01  # target L/LW
LLW_MAX = 100.0
# 64-QAM <|b_0|^4>/<|b_0|^2>^2 from analytical constellation stats.
MU0 = qam_mu0(64)

# --- module-level globals that workers will read ---
_G = {
    "beta1": None, "beta2": None, "fB": None,
    "raman_lo": None, "raman_hi": None,
    "n_modes": None, "n_freqs": None, "n_pairs": None,
    "cf": None, "ipulse": None,
    "raman_min": None, "raman_max": None,
    "raman_extremes": None,
    "ps_ideal": None,
    "dgd_scale": None,
    "n_workers": None,
}


def _get_n_samples_numeric_n(cf) -> int:
    """Read and validate ``n_samples_numeric_n`` from the active run config."""
    numerics = getattr(cf, "numerics", None)
    n_samples = getattr(numerics, "n_samples_numeric_n", None)
    if n_samples is None:
        n_samples = getattr(cf, "n_samples_numeric_n", None)
    if n_samples is None:
        raise ValueError(
            "n_samples_numeric_n is required on the active run config "
            "(cf.n_samples_numeric_n or cf.numerics.n_samples_numeric_n)."
        )
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError(f"Invalid n_samples_numeric_n: {n_samples}")
    return n_samples


def _max_lld_from_beta2(cf, beta2: np.ndarray) -> float | None:
    """Return max L/LD from beta2 grid."""
    try:
        L = float(cf.fiber_length)
        br = float(cf.baud_rate)
    except (TypeError, ValueError):
        return None
    max_b2 = float(np.nanmax(np.abs(beta2)))
    if not np.isfinite(max_b2) or max_b2 <= 0.0:
        return None
    return L * br * br * max_b2


def _init_worker(beta1_path, beta2_path, fB_path, n_modes, n_freqs,
                 cf, ipulse, raman_min, raman_max, n_workers,
                 raman_extremes, ps_ideal):
    """Initializer for multiprocessing workers to mmap shared grids and configs."""
    import os

    import numpy as np
    # keep BLAS threads sane per process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    _G["beta1"] = np.load(beta1_path, mmap_mode="r")
    _G["beta2"] = np.load(beta2_path, mmap_mode="r")
    _G["fB"] = np.load(fB_path,    mmap_mode="r")
    fB = _G["fB"]
    n_z = int(fB.shape[0])
    if n_z > 1:
        dz = float(cf.fiber_length) / float(n_z - 1)
    else:
        dz = 0.0
    fB_sum = np.sum(fB, axis=0)
    # Cache Raman LO/HI integrals per (mode, channel) once per worker process.
    # This avoids recomputing raman_integral(...) in the innermost channel loops.
    _G["raman_lo"] = (fB_sum * dz / float(cf.fiber_length))**2
    _G["raman_hi"] = np.sum(fB * fB, axis=0) * dz / float(cf.fiber_length)
    _G["n_modes"] = n_modes
    _G["n_freqs"] = n_freqs
    _G["n_pairs"] = (n_modes * n_freqs) ** 2
    _G["cf"] = cf            # only if cf is picklable; else pass plain params instead
    _G["ipulse"] = ipulse
    _G["raman_min"] = raman_min
    _G["raman_max"] = raman_max
    _G["raman_extremes"] = raman_extremes
    _G["ps_ideal"] = ps_ideal
    _G["dgd_scale"] = float(cf.fiber_length) * float(cf.baud_rate)
    _G["n_workers"] = n_workers


def work_A(task_A):
    """Compute the full NLIN block for a given (mA, nuA) against all (mB, nuB).
    
    this strangely takes about 10s. Maybe it is the manual iteration on all the channels
    """
    mA, nuA = task_A

    beta1 = _G["beta1"]
    beta2 = _G["beta2"]
    fBmm = _G["fB"]
    raman_lo = _G["raman_lo"]
    raman_hi = _G["raman_hi"]
    n_modes = _G["n_modes"]
    n_freqs = _G["n_freqs"]
    cf = _G["cf"]
    ipulse = _G["ipulse"]
    rmin = _G["raman_min"]
    rmax = _G["raman_max"]
    raman_extremes = _G["raman_extremes"]
    ps_ideal = _G["ps_ideal"]
    dgd_scale = _G["dgd_scale"]

    beta1_A = beta1[mA, nuA]
    gvda = beta2[mA, nuA]

    # or nlin.dtype if known here
    block = np.empty((n_modes, n_freqs), dtype=float)

    start = time.time()
    for mB in range(n_modes):
        for nuB in range(n_freqs):
            # pretty progress
            # try:
            #     import logging
            #     lg = logging.getLogger(__name__)
            #     lg.info(
            #         f"[worker {pid:>6}/{nw:>2}] "
            #         f"Computing NLIN A(m={mA},nu={nuA:>5}) vs B(m={mB},nu={nuB:>5}) "
            #         f"({idx:>7}/{n_pairs:>7})"
            #     )
            # except Exception:
            #     pass

            gvdb = beta2[mB, nuB]
            dgd = abs(beta1_A - beta1[mB, nuB])

            # _fit_nlin_params accepts cached invariants to avoid repeated
            # ideal-fit and Raman-extreme lookups on every channel pair.
            ps = _fit_nlin_params(
                cf,
                abs(gvda),
                abs(gvdb),
                fB=fBmm[:, mB, nuB],
                raman_gvd_correction_min=rmin,
                raman_gvd_correction_max=rmax,
                ipulse=ipulse,
                ps_ideal=ps_ideal,
                raman_extremes=raman_extremes,
                raman_integral_fB_lo=float(raman_lo[mB, nuB]),
                raman_integral_fB_hi=float(raman_hi[mB, nuB]),
            )
            block[mB, nuB] = softplus(dgd * dgd_scale, *ps)

    # block_sum = np.sum(block)
    return mA, nuA, block, time.time() - start



def gvd_correction(cf,
                   gvda,
                   gvdb,
                   fiber_length: float,
                   m_lo_truncation: int = 3,
                   ipulse: int = 1) -> float:
    """Integrate low-order collision terms to correct the LO plateau with GVD."""
    lda = 1/(cf.baud_rate**2 * gvda) if gvda != 0 else 1e30
    ldb = 1/(cf.baud_rate**2 * gvdb) if gvdb != 0 else 1e30
    lo_value = 0.0
    for m_lo in range(m_lo_truncation+1):
        I_low_dataset = np.load(s2a_lo_timeint_path(ipulse=ipulse, m_lo=m_lo))
        interp = build_I_low_interpolator(I_low_dataset, ipulse=ipulse) # FIXME this may be hot if taken without caching
        # this is also in normalized units
        def I_specific(x): return interp(x/lda, x/ldb)
        lg.trace(
            f"Correcting for L/LD=[~0, {fiber_length/lda:.1e}] for m_lo={m_lo}")
        # compute the integral
        # INCRIMINATED LINEEEEEE USELESS
        added_noise = (quad(I_specific, 0, fiber_length)[0] / fiber_length)**2
        if m_lo != 0:
            added_noise *= 2
        lo_value += added_noise
    return lo_value


def apply_plateau_correction(ps: Tuple[float, float, float],
                             lo_value: float,
                             ) -> Tuple[float, float, float]:
    """Rescale softplus parameters when the LO plateau value changes."""
    old_lo_value = ps[0]
    lg.trace(
        f"Correcting N^circ (LO val): {old_lo_value:.2e} --> {lo_value:.2e}")
    lg.trace(f"Correcting delta beta 1 of a factor {old_lo_value/(lo_value)}")
    ps[0] = lo_value
    ps[1] = ps[1] * old_lo_value / lo_value
    # ps[2] = ps[2] * np.sqrt(lo_value / old_lo_value)
    return ps


def apply_turning_point_correction(ps: Tuple[float, float, float],
                                   hi_factor: float):
    """Shift softplus turning point based on Raman HI correction."""
    lg.trace(f"Correcting delta beta 1 of a factor {hi_factor}")
    # beware of the sign of the correction (in log units)
    ps[1] = ps[1] * hi_factor
    # ps[2] = ps[2] / np.sqrt(hi_factor)
    return ps


def _fit_nlin_params(cf,
                     gvda: float,
                     gvdb: float,
                     fB: np.ndarray,
                     raman_gvd_correction_min: callable,
                     raman_gvd_correction_max: callable,
                     ipulse: int,
                     m_lo_truncation: int = 3,
                     ps_ideal: np.ndarray | None = None,
                     raman_extremes: Tuple[float, float, float, float] | None = None,
                     raman_integral_fB_lo: float | None = None,
                     raman_integral_fB_hi: float | None = None) -> np.ndarray:
    """Compute fitted softplus parameters for one NLIN channel pair."""
    lda = 1/(cf.baud_rate**2 * gvda) if gvda != 0 else 1e30
    ldb = 1/(cf.baud_rate**2 * gvdb) if gvdb != 0 else 1e30
    if ps_ideal is None:
        ps_ideal = ideal_fit_coefficients(
            0.0,
            0.0,
            ipulse=ipulse,
            fiber_length=float(cf.fiber_length),
            baud_rate=float(cf.baud_rate),
            n_samples_numeric_n=_get_n_samples_numeric_n(cf),
        )

    if np.all(fB == 1.0):
        lo_value_perfect = gvd_correction(
            cf,
            gvda,
            gvdb,
            cf.fiber_length,
            ipulse=ipulse,
            m_lo_truncation=m_lo_truncation,
        )
        lg.info("You are using a flat fB, no Raman correction will be applied")
        return apply_plateau_correction(ps_ideal.copy(), lo_value_perfect)

    lo_value_max = raman_gvd_correction_max(
        cf.fiber_length/lda, cf.fiber_length/ldb)
    lo_value_min = raman_gvd_correction_min(
        cf.fiber_length/lda, cf.fiber_length/ldb)
    if raman_extremes is None:
        raman_extremes = load_raman_integral_extremes(cf)
    r_lo_min, r_lo_max, _, _ = raman_extremes

    # Optional precomputed integrals are used by worker fast-path.
    if raman_integral_fB_lo is None:
        raman_integral_fB_lo = raman_integral(cf, "LO", fB)
    if raman_integral_fB_hi is None:
        raman_integral_fB_hi = raman_integral(cf, "HI", fB)
    lo_value_fB = (raman_integral_fB_lo - r_lo_min) / (r_lo_max - r_lo_min) * \
        (lo_value_max - lo_value_min) + lo_value_min
    lg.trace(f"LO value: {lo_value_fB:.2e}")
    ps_ramanful = apply_plateau_correction(ps_ideal.copy(), lo_value_fB)
    ps_ramanful = apply_turning_point_correction(
        ps_ramanful, raman_integral_fB_hi)
    return ps_ramanful


# remark: this functions takes DGD in SI units, and outputs channel pair NLIN in normalized units
def fit_nlin(cf,
             gvda: float,
             gvdb: float,
             fB: np.ndarray,
             raman_gvd_correction_min: callable,
             raman_gvd_correction_max: callable,
             ipulse: int,
             m_lo_truncation: int = 3) -> callable:
    """Return a fitted NLIN curve for a channel pair with given GVDs and Raman profile."""
    ps = _fit_nlin_params(
        cf,
        gvda,
        gvdb,
        fB=fB,
        raman_gvd_correction_min=raman_gvd_correction_min,
        raman_gvd_correction_max=raman_gvd_correction_max,
        ipulse=ipulse,
        m_lo_truncation=m_lo_truncation,
    )
    dgd_scale = float(cf.fiber_length) * float(cf.baud_rate)

    def nlin_megafit(d):
        d = d * dgd_scale
        return softplus(d, *ps)

    nlin_megafit.ps_params = ps
    return nlin_megafit

"""
corrections due to mode multiplicity
"""
def nlin_prefactor_general(cf, mode_a: int, mode_b: int):
    """Multiplicity/constellation prefactor for MMF NLIN between two modes."""
    prefactor = 1
    if mode_a == mode_b:
        prefactor *= MU0 * (2*SPATIAL_MODES[mode_a] + 3) - 4
    else:
        prefactor *= 2 * SPATIAL_MODES[mode_b] * (MU0-1)
    assert prefactor >= 0
    return prefactor


"""
wrapper for the nlin_prefactor_mmf to handle single-mode case
"""


def nlin_prefactor(cf, mode_a, mode_b):
    """Wrapper handling SMF vs MMF to compute NLIN prefactors."""
    if cf.n_modes == 1:
        return nlin_prefactor_general(cf, 0, 0)
    else:
        return nlin_prefactor_general(cf, mode_a, mode_b)


# this only compute the collision coefficients \sum_m X_{0mm}^2
def collision_coeffs_system(cf,
                            ipulse: int = 1,
                            recompute: bool = False,):
    """Compute or load channel-pair collision coefficients for the given config."""
    # assert cf.launch_power == -5
    # assert cf.raman_gain == 0.0
    # assert cf.baud_rate == 33e9
    # assert cf.channel_spacing == 50e9
    # assert cf.n_channels == 200
    # assert cf.center_frequency == 195.94e12
    # assert cf.fiber_length == 70e3
    if cf.n_modes == 1:
        fiber_type = "smf"
    else:
        fiber_type = "mmf"

    br_hz = float(cf.baud_rate)
    spacing_hz = getattr(cf, "channel_spacing", None)
    n_ch = int(cf.n_channels)

    filename = s3_pair_nlin_kernel_path(
        ipulse=ipulse,
        fiber_type=fiber_type,
        br_hz=br_hz,
        n_ch=n_ch,
        spacing_hz=spacing_hz,
    )
    if os.path.exists(filename) and not recompute:
        lg.info(f"Loading precomputed collision coefficients from {filename} of shape {np.load(filename).shape}")
        return np.load(filename)
    else: # this is the very intensive part.
        lg.info(f"Computing collision coefficients from scratch")
        n_samples_numeric_n = _get_n_samples_numeric_n(cf)
        T = 1 / cf.baud_rate
        L = cf.fiber_length
        x_norm = L / T
        y_norm = x_norm**(-2)

        oi_fit = np.load('results/oi_fit.npy')

        beta1_params = load_group_delay()
        wdm = pynlin.wdm.WDM(
            spacing=cf.channel_spacing,
            num_channels=cf.n_channels,
            center_frequency=cf.center_frequency
        )
        freqs = wdm.frequency_grid()
        modes = range(cf.n_modes)
        fiber = MMFiber(
            effective_area=cf.effective_area,
            overlap_integrals=oi_fit,
            group_delay=beta1_params,
            length=cf.fiber_length,
        )

        beta1 = np.zeros((cf.n_modes, len(freqs)))

        for i in modes:
            beta1[i, :] = fiber.group_delay.evaluate_beta1(i, freqs)
        beta2 = np.zeros((cf.n_modes, len(freqs)))
        for i in modes:
            beta2[i, :] = fiber.group_delay.evaluate_beta2(i, freqs)
        beta1 = np.array(beta1)
        beta2 = np.array(beta2)
        # for each channel, we compute the total number of collisions that
        # needs to be computed for evaluating the total noise on that channel.
        T = 1 / cf.baud_rate
        L = cf.fiber_length
        collision_coeffs = np.zeros(
            (cf.n_modes, len(freqs), cf.n_modes, len(freqs)))

        max_lld = _max_lld_from_beta2(cf, beta2)
        raman_gvd_correction_max, raman_gvd_correction_min = build_lookup_integral_table_with_raman(
            cf, ipulse=ipulse, max_lld=max_lld) # this could be intensive, but it is actually ok.
        fB, _, _, _, _ = load_fB(cf)
        # Reuse across all pair fits: these are global for this run.
        raman_extremes = load_raman_integral_extremes(cf)
        ps_ideal = ideal_fit_coefficients(
            0.0,
            0.0,
            ipulse=ipulse,
            fiber_length=float(cf.fiber_length),
            baud_rate=float(cf.baud_rate),
            n_samples_numeric_n=n_samples_numeric_n,
        )

        # precompute the Raman corrections from the numerical results of the integrals
        # precompute the GVD-dependent fitting parameters (perfect amplification)

        from concurrent.futures import ProcessPoolExecutor, as_completed

        beta1_path = str(s2_beta1_grid_path())
        beta2_path = str(s2_beta2_grid_path())
        fB_path = str(s2_fB_grid_path())
        np.save(beta1_path, beta1)
        np.save(beta2_path, beta2)
        np.save(fB_path,    fB)  # shape (K, n_modes, n_freqs)

        A_tasks = list(it.product(range(cf.n_modes), range(len(freqs))))
        n_workers = os.cpu_count() or 1

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(beta1_path, beta2_path, fB_path,
                      cf.n_modes, len(freqs),
                      cf, ipulse, raman_gvd_correction_min, raman_gvd_correction_max,
                      n_workers, raman_extremes, ps_ideal),
        ) as ex:
            futures = [ex.submit(work_A, a) for a in A_tasks]
            for fut in as_completed(futures):
                mA, nuA, block, elapsed = fut.result()
                collision_coeffs[mA, nuA, :, :] = block
                lg.trace(
                    f"Finished NLIN for A(m={mA},nu={nuA:>5}) in {elapsed:>6.2f} s")

        # --- saving
        np.save(filename, collision_coeffs)

        return collision_coeffs


def get_kappa2_matrix(cf,
                      use_kappa: bool = False,
                      use_x_mode: bool = False,
                      ) -> np.ndarray:
    """Build squared coupling matrix and optionally disable cross-mode terms."""
    kappa2 = np.zeros((cf.n_modes, cf.n_modes))
    if use_kappa:
        # FIXME: Check the Manakov averaging.
        lg.warning("Applying kappa.csv coupling weights. Check the Manakov averaging.")
        kappa = np.loadtxt('input/fiber_data/kappa.csv', delimiter=',')
        kappa2 = kappa**2
    else:
        kappa2 = np.ones((cf.n_modes, cf.n_modes))  # kind of innatural, but ok
    switchoff_matrix = np.eye(cf.n_modes)

    if not use_x_mode:
        kappa2 = np.multiply(kappa2, switchoff_matrix)
    return kappa2


# all in SI units
def total_nlin(cf,
               collision_coeffs: np.ndarray,
               use_kappa: bool = False,
               use_x_mode: bool = False,
               exclude_self_channel: bool = True,
               ) -> np.ndarray:
    """Convert collision coefficients to NLIN power per channel.

    Parameters
    ----------
    exclude_self_channel:
        If ``True``, removes terms with ``nuB == nuA`` from the channel summation
        before aggregation. This is useful to inspect an XCI-like contribution
        without the self-channel (SCI-like) term.
    """

    x_norm = cf.fiber_length * cf.baud_rate
    y_norm = 1/(cf.fiber_length * cf.baud_rate)**2
    lg.info(
        f"Normalization units: x_norm = {x_norm:.2e} s, y_norm = {y_norm:.2e} s^2/m^2")
    collision_coeffs_si = collision_coeffs / y_norm  # bring back to SI units

    if exclude_self_channel:
        lg.warning(
            "\n"
            "====================================================================\n"
            " WARNING: LEGACY TD SELF-CHANNEL EXCLUSION ENABLED                \n"
            " - total_nlin() is running with exclude_self_channel=True          \n"
            " - Terms with nuB == nuA are removed from the TD summation         \n"
            " - Output is NOT full TD NLIN; it is an XCI-like diagnostic metric \n"
            "===================================================================="
        )
        n_modes, n_freqs, _, _ = collision_coeffs_si.shape
        diag_idx = np.arange(n_freqs)
        collision_coeffs_si = np.array(collision_coeffs_si, copy=True)
        collision_coeffs_si[:, diag_idx, :, diag_idx] = 0.0

    P_in = dBm2watt(cf.launch_power)
    omega_0 = 2 * np.pi * cf.center_frequency
    n2 = 2.6e-20  # constant of SiO2
    omega_0 = 2 * np.pi * cf.center_frequency
    gamma = n2 * omega_0 / (cf.effective_area * c)  # Delta f / f << 1
    lg.trace(
        f"Computed gamma: {gamma:.2e} 1/(W m), P_in: {P_in:.2e} W / {watt2dBm(P_in):.2f} dBm")
    constant_prefactor = P_in**3 * gamma**2 / (cf.baud_rate**2)

    lg.trace(
        f"Computed constant prefactor: {constant_prefactor:.2e} W * s^2 / m^2")
    kappa2 = get_kappa2_matrix(cf, use_kappa, use_x_mode)

    n_modes, n_freqs, _, _ = collision_coeffs_si.shape
    prefactor_matrix = np.zeros((n_modes, n_modes), dtype=float)
    for mA in range(n_modes):
        for mB in range(n_modes):
            prefactor_matrix[mA, mB] = nlin_prefactor(cf, mA, mB)
    coupling = kappa2 * prefactor_matrix
    # Contract over (mB, nuB): total_nlin[a, i] = sum_{b,j} coeff[a,i,b,j]*coupling[a,b]
    total_nlin = np.einsum("aibj,ab->ai", collision_coeffs_si, coupling, optimize=True)
    return total_nlin * constant_prefactor


if __name__ == "__main__":
    import sys
    import time
    from pynlin.system import System

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m pynlin.nlin.nlin_estimator <system.toml> [numerical_config.toml]"
        )
    system_path = sys.argv[1]
    numerics_path = sys.argv[2] if len(sys.argv) > 2 else None
    cf = System.from_toml(system_path, numerical_path=numerics_path)

    start = time.perf_counter()
    ccfs = collision_coeffs_system(cf,
                                   ipulse=1,
                                   recompute=False)
    lg.debug(
        f"A few collisions (should be of order 1e-1, 1e-2): {ccfs[0, 0, :, :5]}")
    ttnl = total_nlin(cf,
                      ccfs,
                      use_kappa=True,
                      use_x_mode=False,
                      )
    lg.debug(f"Total NLIN shape: {ttnl.shape}")
    lg.debug(
        f"A few total NLIN: \n {ttnl[0, 0:5]} W, \n {watt2dBm(ttnl[0, 0:5])} dBm")
