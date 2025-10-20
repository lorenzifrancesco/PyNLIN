import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import threading
import itertools as it
from concurrent.futures import ThreadPoolExecutor, as_completed
from pynlin.utils import watt2dBm
import matplotlib.pyplot as plt
import os
from scripts.modules.collision import build_I_low_interpolator, MAX_LLD
import scripts.modules.cfg as cfg
from scripts.modules.load_fiber_values import load_group_delay, load_rms_gvd
from scipy.constants import c
from pynlin.utils import dBm2watt
from pynlin.fiber import MMFiber
import pynlin
import numpy as np
from typing import Tuple
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from loguru import logger as lg
from scripts.modules.log_init import init_logging
init_logging()

SPATIAL_MODES = np.array([1, 2, 2, 1])
LLW_MIN = 0.01  # target L/LW
LLW_MAX = 100.0
MU0 = 1.3809  # 64-QAM <|b_0|^4>/<|b_0|^2>^2


# --- module-level globals that workers will read ---
_G = {
    "beta1": None, "beta2": None, "fB": None,
    "n_modes": None, "n_freqs": None, "n_pairs": None,
    "cf": None, "ipulse": None,
    "raman_min": None, "raman_max": None,
    "n_workers": None,
}


def _init_worker(beta1_path, beta2_path, fB_path, n_modes, n_freqs,
                 cf, ipulse, raman_min, raman_max, n_workers):
    import os
    import numpy as np
    # keep BLAS threads sane per process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    _G["beta1"] = np.load(beta1_path, mmap_mode="r")
    _G["beta2"] = np.load(beta2_path, mmap_mode="r")
    _G["fB"] = np.load(fB_path,    mmap_mode="r")
    _G["n_modes"] = n_modes
    _G["n_freqs"] = n_freqs
    _G["n_pairs"] = (n_modes * n_freqs) ** 2
    _G["cf"] = cf            # only if cf is picklable; else pass plain params instead
    _G["ipulse"] = ipulse
    _G["raman_min"] = raman_min
    _G["raman_max"] = raman_max
    _G["n_workers"] = n_workers


def work_A(task_A):
    """
    Top-level, picklable worker. Computes the whole (mB,nuB) block for one (mA,nuA).
    """
    import os
    import time
    import numpy as np
    mA, nuA = task_A
    pid = os.getpid()

    beta1 = _G["beta1"]
    beta2 = _G["beta2"]
    fBmm = _G["fB"]
    n_modes = _G["n_modes"]
    n_freqs = _G["n_freqs"]
    n_pairs = _G["n_pairs"]
    cf = _G["cf"]
    ipulse = _G["ipulse"]
    rmin = _G["raman_min"]
    rmax = _G["raman_max"]
    nw = _G["n_workers"]

    beta1_A = beta1[mA, nuA]
    gvda = beta2[mA, nuA]

    # or nlin.dtype if known here
    block = np.empty((n_modes, n_freqs), dtype=float)

    start = time.time()
    for mB in range(n_modes):
        for nuB in range(n_freqs):
            idx = (mA*n_freqs + nuA) * n_modes * \
                n_freqs + (mB*n_freqs + nuB) + 1

            # pretty progress
            try:
                import logging
                lg = logging.getLogger(__name__)
                lg.info(
                    f"[worker {pid:>6}/{nw:>2}] "
                    f"Computing NLIN A(m={mA},nu={nuA:>5}) vs B(m={mB},nu={nuB:>5}) "
                    f"({idx:>7}/{n_pairs:>7})"
                )
            except Exception:
                pass

            gvdb = beta2[mB, nuB]
            dgd = abs(beta1_A - beta1[mB, nuB])

            val = fit_nlin(          # ensure this is importable at module scope too
                cf,
                abs(gvda),
                abs(gvdb),
                fB=fBmm[:, mB, nuB],
                raman_gvd_correction_min=rmin,
                raman_gvd_correction_max=rmax,
                ipulse=ipulse,
            )(dgd)
            block[mB, nuB] = val

    # block_sum = np.sum(block)
    return mA, nuA, block, time.time() - start


def load_fB(cf: cfg.Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, callable, callable]:
    assert (cf.launch_power == -5.0 and cf.raman_gain == 0.0)
    # all the information about the numerosity and stuff is here.
    sol_path = "results/ct_solution-5_gain_0.0.npy"
    solutions = np.load(sol_path, allow_pickle=True).item()

    signal_powers = solutions['signal_sol']
    # indices: (z, mode, channel)
    signal_powers = np.swapaxes(signal_powers, 1, 2)
    fB = signal_powers / signal_powers[0, :, :]  # normalize to input power
    assert np.all(fB[0, :, :] == 1.0)
    fB_max = np.max(fB, axis=(1, 2))
    fB_min = np.min(fB, axis=(1, 2))
    z_axis = np.linspace(0, cf.fiber_length, len(fB_max))
    assert ((fB_min <= fB_max).all())
    assert (fB.shape[0] == len(z_axis))

    coeffs_max = np.polyfit(z_axis, fB_max, 6)
    coeffs_min = np.polyfit(z_axis, fB_min, 6)

    def fB_max_function(z):
        return np.polyval(coeffs_max, z)

    def fB_min_function(z):
        return np.polyval(coeffs_min, z)

    return fB, fB_min, fB_max, fB_min_function, fB_max_function


def softplus2(x, a, b, c):
    return a * (1 + (x / b)**(1 / c))**(-c)


def raman_integral(cf,
                   regime: str,
                   fB: np.ndarray):
    z_axis = np.linspace(0, cf.fiber_length, len(fB))
    dz = z_axis[1] - z_axis[0]
    if regime == "LO":
        return (np.sum(fB) * dz / cf.fiber_length)**2
    else:
        return np.sum(fB**2) * dz / cf.fiber_length


def load_raman_integral_extremes(cf,
                                 ) -> Tuple[float, float, float, float]:
    _, fB_min, fB_max, _, _ = load_fB(cf)
    r_lo_min = raman_integral(cf, "LO", fB_min)
    r_lo_max = raman_integral(cf, "LO", fB_max)
    r_hi_min = raman_integral(cf, "HI", fB_min)
    r_hi_max = raman_integral(cf, "HI", fB_max)

    return r_lo_min, r_lo_max, r_hi_min, r_hi_max


def build_lookup_integral_table_with_raman(cf,
                                           m_lo_truncation: int = 2,
                                           ipulse: int = 1,
                                           recompute=False) -> Tuple[callable, callable]:
    # sampling the gvda, gvdb space, build the callable function
    # giving the correction integrals for fB_max and fB_min: integral(L/gvda, L/gvdb).
    _, _, _, fB_min, fB_max = load_fB(cf)
    n_samples = 20
    fiber_length = cf.fiber_length
    lld = np.linspace(1e-30, MAX_LLD, n_samples)
    ld = fiber_length / lld
    lg.debug(
        f"Useful range of L/LD from LO time integral data: {lld[0]:.2e} to {lld[-1]:.2e}")
    raman_correction_grid_max = np.zeros((n_samples, n_samples))
    raman_correction_grid_min = np.zeros((n_samples, n_samples))

    # plot max and min fB
    # exit()
    # save to file with exhaustive namefile information
    filename = f"results/raman_correction_grid_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.npy"

    if os.path.exists(filename) and not recompute:
        lg.info(f"Loading precomputed Raman correction grid from {filename}")
        data = np.load(filename, allow_pickle=True).item()
        raman_correction_grid_max = data['raman_correction_grid_max']
        raman_correction_grid_min = data['raman_correction_grid_min']
    else:
        lg.info(f"Computing Raman correction grid and saving to {filename}")
        for m_lo in range(m_lo_truncation+1):
            add_min = np.zeros((n_samples, n_samples))
            add_max = np.zeros((n_samples, n_samples))
            lg.info(f"Calculating m_lo={m_lo}")
            I_low_dataset = np.load(
                f"results/I_low_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo}.npz")
            interp = build_I_low_interpolator(I_low_dataset, ipulse=ipulse)
            for ida, lda in enumerate(ld):
                for idb, ldb in enumerate(ld):
                    # apply symmetry:
                    if idb < ida:
                        add_max[ida, idb] = add_max[idb, ida]
                        add_min[ida, idb] = add_min[idb, ida]
                    else:
                        lg.debug(
                            f"Point {ida*n_samples+idb+1}/{n_samples*n_samples}, spanning LLDA={cf.fiber_length/lda:.2e}, LLDB={cf.fiber_length/ldb:.2e}")
                        # this is also in normalized units
                        def I_specific(x): return interp(x/lda, x/ldb)
                        # compute the integral
                        add_max[ida, idb] += (
                            quad(lambda x: I_specific(x) * fB_max(x), 0, fiber_length)[0] / fiber_length)**2
                        add_min[ida, idb] += (
                            quad(lambda x: I_specific(x) * fB_min(x), 0, fiber_length)[0] / fiber_length)**2
                        lg.trace(
                            f"Contribution of the m_lo={m_lo} integral: max       {add_max[ida, idb]:.2e}, min {add_min[ida, idb]:.2e}")
            if m_lo != 0:
                add_max *= 2
                add_min *= 2
            raman_correction_grid_max += add_max
            raman_correction_grid_min += add_min

            plt.figure(figsize=(4, 4))
            plt.imshow(raman_correction_grid_max, extent=(
                lld[0], lld[-1], lld[0], lld[-1]), origin='lower')
            plt.colorbar()
            plt.title(
                f"Raman correction grid max, m_lo={m_lo_truncation}, {cf.fiber_length/1e3:.1f} km")
            plt.xlabel("L/LDa")
            plt.ylabel("L/LD b")
            plt.savefig(
                f"results/raman_correction_grid_max_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.png", dpi=300)
            plt.close()
            plt.figure(figsize=(4, 4))
            plt.imshow(raman_correction_grid_min, extent=(
                lld[0], lld[-1], lld[0], lld[-1]), origin='lower')
            plt.colorbar()
            plt.title(
                f"Raman correction grid max, m_lo={m_lo_truncation}, {cf.fiber_length/1e3:.1f} km")
            plt.xlabel("L/LDa")
            plt.ylabel("L/LD b")
            plt.savefig(
                f"media/debug/raman_correction_grid_min_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.png", dpi=300)
            plt.close()
            # raman_correction_grid_max = np.zeros((n_samples, n_samples))
            # raman_correction_grid_min = np.zeros((n_samples, n_samples))
        np.save(filename, {
            'raman_correction_grid_max': raman_correction_grid_max,
            'raman_correction_grid_min': raman_correction_grid_min,
        })

    # build the interpolator and return it
    # checking
    plt.figure(figsize=(4, 4))
    plt.imshow(raman_correction_grid_max, extent=(
        lld[0], lld[-1], lld[0], lld[-1]), origin='lower')
    plt.colorbar()
    plt.title(
        f"Raman correction grid max, m_lo={m_lo_truncation}, {cf.fiber_length/1e3:.1f} km")
    plt.xlabel("L/LDa")
    plt.ylabel("L/LD b")
    plt.savefig(
        f"results/raman_correction_grid_max_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.png", dpi=300)
    plt.close()
    plt.figure(figsize=(4, 4))
    plt.imshow(raman_correction_grid_min, extent=(
        lld[0], lld[-1], lld[0], lld[-1]), origin='lower')
    plt.colorbar()
    plt.title(
        f"Raman correction grid max, m_lo={m_lo_truncation}, {cf.fiber_length/1e3:.1f} km")
    plt.xlabel("L/LDa")
    plt.ylabel("L/LD b")
    plt.savefig(
        f"media/debug/raman_correction_grid_min_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.png", dpi=300)
    plt.close()

    interp_func_max = RegularGridInterpolator(
        (lld, lld),
        raman_correction_grid_max,
        bounds_error=False,
        fill_value=None)

    interp_func_min = RegularGridInterpolator(
        (lld, lld),
        raman_correction_grid_min,
        bounds_error=False,
        fill_value=None)

    def func_wrapper(func):
        def func_wrapper(x, y):
            assert (x <= 1.01 * lld[-1] and y <= 1.01 * lld[-1]
                    ), f"Input {x} exceeds the 110% of the interpolation range [{lld[0]}, {lld[-1]}]"
            assert (x >= 0 and y >=
                    0), f"Input has negative values check that your LD is positive"
            return func((x, y))
        return func_wrapper

    return func_wrapper(interp_func_min), func_wrapper(interp_func_max)


"""
ideal := no Raman, no GVD. It is flexible to also compute the GVD, but it is not recommended.
"""


def ideal_fit_coefficients(gvda: float = 0.0,
                           gvdb: float = 0.0,
                           ipulse: int = 1) -> Tuple[np.ndarray, np.ndarray]:
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
    popt, _ = curve_fit(softplus2,
                        dgds_numeric * x_norm,
                        nlin_numeric * y_norm,
                        p0=p0)
    return popt


def gvd_correction(cf,
                   gvda,
                   gvdb,
                   fiber_length: float,
                   m_lo_truncation: int = 3,
                   ipulse: int = 1) -> float:
    lda = 1/(cf.baud_rate**2 * gvda) if gvda != 0 else 1e30
    ldb = 1/(cf.baud_rate**2 * gvdb) if gvdb != 0 else 1e30
    lo_value = 0.0
    for m_lo in range(m_lo_truncation+1):
        I_low_dataset = np.load(
            f"results/I_low_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo}.npz")
        interp = build_I_low_interpolator(I_low_dataset, ipulse=ipulse)
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
    lg.trace(f"Correcting delta beta 1 of a factor {hi_factor}")
    # beware of the sign of the correction (in log units)
    ps[1] = ps[1] * hi_factor
    # ps[2] = ps[2] / np.sqrt(hi_factor)
    return ps


# remark: this functions takes DGD in SI units, and outputs channel pair NLIN in normalized units
def fit_nlin(cf,
             gvda: float,
             gvdb: float,
             fB: np.ndarray,
             raman_gvd_correction_min: callable,
             raman_gvd_correction_max: callable,
             ipulse: int,
             m_lo_truncation: int = 3) -> callable:

    lda = 1/(cf.baud_rate**2 * gvda) if gvda != 0 else 1e30
    ldb = 1/(cf.baud_rate**2 * gvdb) if gvdb != 0 else 1e30
    ps_ideal = ideal_fit_coefficients(0.0, 0.0)

    if np.all(fB == 1.0):
        lo_value_perfect = gvd_correction(cf,
        gvda, gvdb, cf.fiber_length, ipulse=ipulse, m_lo_truncation=m_lo_truncation) # FIXME this is the sucker
        ps_ramanless = apply_plateau_correction(ps_ideal.copy(), lo_value_perfect)
        lg.info("You are using a flat fB, no Raman correction will be applied")
        ps = apply_plateau_correction(ps_ideal.copy(), lo_value_perfect)
        return lambda dgd: softplus2(dgd * cf.fiber_length * cf.baud_rate, *ps)

    # correct in the LO regime (Raman + GVD)
    lo_value_max = raman_gvd_correction_max(
        cf.fiber_length/lda, cf.fiber_length/ldb)
    lo_value_min = raman_gvd_correction_min(
        cf.fiber_length/lda, cf.fiber_length/ldb)
    r_lo_min, r_lo_max, _, _ = load_raman_integral_extremes(cf)

    raman_integral_fB_lo = raman_integral(cf, "LO", fB)
    raman_integral_fB_hi = raman_integral(cf, "HI", fB)
    # special case of zero GVD
    # if gvda == 0.0 and gvdb == 0.0:
    #     # we compute the fB effect using
    #     lo_value_fB = raman_integral_fB_lo
    # else:
    lo_value_fB = (raman_integral_fB_lo - r_lo_min) / (r_lo_max - r_lo_min) * \
        (lo_value_max - lo_value_min) + lo_value_min
    lg.trace(f"LO value: {lo_value_fB:.2e}")
    ps_ramanful = apply_plateau_correction(ps_ideal.copy(), lo_value_fB)
    ps_ramanful = apply_turning_point_correction(
        ps_ramanful, raman_integral_fB_hi)

    # applying the Raman value is not the same as multiplying ever
    # build the linear composition to match the HI correction (in form of a simple Raman integral)
    def nlin_megafit(d):
        d = d * cf.fiber_length * cf.baud_rate  # FIXME apply normalization..
        DGD_MAX = LLW_MAX
        DGD_MIN = LLW_MIN
        xi = (d-DGD_MIN)/(DGD_MAX-DGD_MIN)
        return softplus2(d, *ps_ramanful)
        return ((softplus2(d, *ps_ramanless) * raman_integral_fB_hi))
        return softplus2(d, *ps_ramanful) * (1-xi) + softplus2(d, *ps_ramanless) * xi * raman_integral_fB_hi
        return softplus2(d, *ps_ramanful)
        return softplus2(d, *ps_ramanless) * raman_integral_fB_lo
    return nlin_megafit


"""
corrections due to mode multiplicity
"""


def nlin_prefactor_general(cf: cfg.Config, mode_a: int, mode_b: int):
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


def nlin_prefactor(cf: cfg.Config, mode_a, mode_b):
    if cf.n_modes == 1:
        return nlin_prefactor_general(cf, 0, 0)
    else:
        return nlin_prefactor_general(cf, mode_a, mode_b)

# this only compute the collision coefficients \sum_m X_{0mm}^2


def collision_coeffs_system(cf,
                            ipulse: int = 1,
                            recompute: bool = False,):
    assert (cf.n_modes == 4)
    assert (cf.launch_power == -5)
    filename = f"results/collision_coefficients_ipulse{ipulse}.npy"
    if os.path.exists(filename) and not recompute:
        lg.info(f"Loading precomputed collision coefficients from {filename}")
        return np.load(filename)
    else:
        lg.info(f"Computing collision coefficients from scratch")
        # load numerical config
        nc = cfg.load_nc_toml_to_struct("input/numerical_config.toml")
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

        d_min = nc.dgd1
        d_max = nc.dgd2_g
        d_span = d_max - d_min
        raman_gvd_correction_max, raman_gvd_correction_min = build_lookup_integral_table_with_raman(
            cf, ipulse=ipulse)
        fB, fB_min, fB_max, fB_min_function, fB_max_function = load_fB(cf)

        # precompute the Raman corrections from the numerical results of the integrals
        # precompute the GVD-dependent fitting parameters (perfect amplification)

        from concurrent.futures import ProcessPoolExecutor, as_completed
        # import os, numpy as np, itertools as it

        # Precompute & save grids so workers don’t need your `fiber` object:
        beta1 = np.empty((cf.n_modes, len(freqs)))
        beta2 = np.empty_like(beta1)
        for m in range(cf.n_modes):
            for j, f in enumerate(freqs):
                beta1[m, j] = fiber.group_delay.evaluate_beta1(m, f)
                beta2[m, j] = fiber.group_delay.evaluate_beta2(m, f)

        beta1_path = "/tmp/beta1_grid.npy"
        beta2_path = "/tmp/beta2_grid.npy"
        fB_path = "/tmp/fB.npy"
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
                      n_workers),
        ) as ex:
            futures = [ex.submit(work_A, a) for a in A_tasks]
            for fut in as_completed(futures):
                mA, nuA, block, elapsed = fut.result()
                collision_coeffs[mA, nuA, :, :] = block
                lg.info(
                    f"Finished NLIN for A(m={mA},nu={nuA:>5}) in {elapsed:>6.2f} s")

        # --- saving
        np.save(filename, collision_coeffs)

        return collision_coeffs


def get_kappa2_matrix(cf,
                      use_kappa: bool = False,
                      use_x_mode: bool = False,
                      ) -> np.ndarray:
    kappa2 = np.zeros((cf.n_modes, cf.n_modes))
    if use_kappa:
        kappa = np.loadtxt('input/kappa.csv', delimiter=',')
        kappa2 = kappa**2
    else:
        kappa2 = np.ones((cf.n_modes, cf.n_modes)) # kind of innatural, but ok
    switchoff_matrix = np.eye(cf.n_modes)
    
    if not use_x_mode:
        kappa2 = np.multiply(kappa2, switchoff_matrix)
    return kappa2

## all in SI units
def total_nlin(cf,
               collision_coeffs: np.ndarray, # warn: in normalized units
               use_kappa: bool = False,
               use_x_mode: bool = False,
               ) -> np.ndarray:
    
    x_norm = cf.fiber_length * cf.baud_rate
    y_norm = 1/(cf.fiber_length * cf.baud_rate)**2
    lg.info(f"Normalization units: x_norm = {x_norm:.2e} s, y_norm = {y_norm:.2e} s^2/m^2")
    collision_coeffs_si = collision_coeffs / y_norm # bring back to SI units
    
    P_in = dBm2watt(cf.launch_power)
    omega_0 = 2 * np.pi * cf.center_frequency
    n2 = 2.6e-20  # constant of SiO2
    omega_0 = 2 * np.pi * cf.center_frequency
    gamma = n2 * omega_0 / (cf.effective_area * c)  # Delta f / f << 1
    lg.trace(f"Computed gamma: {gamma:.2e} 1/(W m), P_in: {P_in:.2e} W / {watt2dBm(P_in):.2f} dBm")
    constant_prefactor = P_in**3 * gamma**2 / (cf.baud_rate**2)
    
    lg.trace(f"Computed constant prefactor: {constant_prefactor:.2e} W * s^2 / m^2")
    kappa2 = get_kappa2_matrix(cf, use_kappa, use_x_mode)

    n_modes, n_freqs, _, _ = collision_coeffs_si.shape
    total_nlin = np.zeros((n_modes, n_freqs))
    for mA in range(n_modes):
        for nuA in range(n_freqs):
            for mB in range(n_modes):
                for nuB in range(n_freqs):
                    prefactor = nlin_prefactor(cf, mA, mB)
                    total_nlin[mA, nuA] += collision_coeffs_si[mA,
                                                            nuA, mB, nuB] * kappa2[mA, mB] * prefactor
    total_nlin *= constant_prefactor
    return total_nlin


if __name__ == "__main__":
    import time
    start = time.perf_counter()
    ccfs = collision_coeffs_system(cfg.load_toml_to_struct("./input/mmf.toml"),
                                   ipulse=1,
                                   recompute=False)
    lg.debug(f"A few collisions (should be of order 1e-1, 1e-2): {ccfs[0,0, :, :5]}")
    ttnl = total_nlin(cfg.load_toml_to_struct("./input/mmf.toml"),
               ccfs,
               use_kappa=True,
               use_x_mode=False,
               )
    lg.debug(f"Total NLIN shape: {ttnl.shape}")
    lg.debug(f"A few total NLIN: \n {ttnl[0,0:5]} W, \n {watt2dBm(ttnl[0,0:5])} dBm")
    exit() 
    import scripts.modules.cfg as cfg
    cf = cfg.load_toml_to_struct("./input/mmf.toml")

    # build the interpolator and pass it to the corrector
    I_low_n = np.load(f"results/I_low_nyquist.npz")
    interp = build_I_low_interpolator(I_low_dataset=I_low_n, ipulse=0)
    lg.debug("Useful range of L/LD from LO time integral data: ",
             I_low_n['lld_range'])
