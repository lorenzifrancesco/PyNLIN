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
import os

SPATIAL_MODES = np.array([1, 2, 2, 1])
LLW_MIN = 0.01  # target L/LW
LLW_MAX = 100.0


def load_fB(cf: cfg.Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, callable, callable]:
    assert (cf.launch_power == -5.0 and cf.raman_gain == 0.0)
    # all the information about the numerosity and stuff is here.
    sol_path = "results/ct_solution-6_gain_0.0.npy"
    solutions = np.load(sol_path, allow_pickle=True).item()

    signal_powers = solutions['signal_sol']
    # indices: (z, mode, channel)
    signal_powers = np.swapaxes(signal_powers, 1, 2)
    fB = signal_powers / signal_powers[0, :, :]  # normalize to input power
    fB_max = np.max(signal_powers, axis=(1, 2))
    fB_min = np.min(signal_powers, axis=(1, 2))
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
                                           ipulse: int = 1):
    # sampling the gvda, gvdb space, build the callable function
    # giving the correction integrals for fB_max and fB_min: integral(L/gvda, L/gvdb).
    _, _, _, fB_min, fB_max = load_fB(cf)
    n_samples = 20
    fiber_length = cf.fiber_length
    lld = np.linspace(1e-30, MAX_LLD, n_samples)
    ld = fiber_length / lld
    raman_correction_grid_max = np.zeros((n_samples, n_samples))
    raman_correction_grid_min = np.zeros((n_samples, n_samples))
    
    # save to file with exhaustive namefile information
    filename = f"results/raman_correction_grid_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.npy"
    if os.path.exists(filename):
        lg.info(f"Loading precomputed Raman correction grid from {filename}")
        data = np.load(filename, allow_pickle=True).item()
        return data['raman_correction_grid_max'], data['raman_correction_grid_min']
    else:
        lg.info(f"Computing Raman correction grid and saving to {filename}")
        for m_lo in range(m_lo_truncation+1):
            lg.info(f"Calculating m_lo={m_lo}")
            I_low_dataset = np.load(
                f"results/I_low_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo}.npz")
            interp = build_I_low_interpolator(I_low_dataset, ipulse=ipulse)
            for ida, lda in enumerate(ld):
                for idb, ldb in enumerate(ld):
                    # apply symmetry: 
                    if idb < ida:
                        raman_correction_grid_max[ida, idb] = raman_correction_grid_max[idb, ida]
                        raman_correction_grid_min[ida, idb] = raman_correction_grid_min[idb, ida]
                        continue
                    lg.debug(f"Point {ida*n_samples+idb+1}/{n_samples*n_samples}")
                    # this is also in normalized units
                    def I_specific(x): return interp(x/lda, x/ldb)
                    # compute the integral
                    raman_correction_grid_max[ida, idb] += (
                        quad(lambda x: I_specific(x) * fB_max(x), 0, fiber_length)[0] / fiber_length)**2
                    raman_correction_grid_min[ida, idb] += (
                        quad(lambda x: I_specific(x) * fB_min(x), 0, fiber_length)[0] / fiber_length)**2
                    if m_lo != 0:
                        raman_correction_grid_max[ida, idb] *= 2
                        raman_correction_grid_min[ida, idb] *= 2
        np.save(filename, {
            'raman_correction_grid_max': raman_correction_grid_max,
            'raman_correction_grid_min': raman_correction_grid_min,
        })
        
        
    # build the interpolator and return it
    interp_func = RegularGridInterpolator(
        (lld, lld),
        raman_correction_grid_max,
        bounds_error=False,
        fill_value=None)

    def interp_func_wrapped_max(x, y):
        assert (x <= 1.01 * lld[-1] and y <= 1.01 * lld[-1]
                ), f"Input {x} exceeds the 110% of the interpolation range [{lld[0]}, {lld[-1]}]"
        assert (x >= 0 and y >=
                0), f"Input has negative values check that your LD is positive"
        return interp_func([x, y])
    interp_func = RegularGridInterpolator(
        (lld, lld),
        raman_correction_grid_min,
        bounds_error=False,
        fill_value=None)

    def interp_func_wrapped_min(x, y):
        assert (x <= 1.01 * lld[-1] and y <= 1.01 * lld[-1]
                ), f"Input {x} exceeds the 110% of the interpolation range [{lld[0]}, {lld[-1]}]"
        assert (x >= 0 and y >=
                0), f"Input has negative values check that your LD is positive"
        return interp_func([x, y])
    return interp_func_wrapped_max, interp_func_wrapped_min



"""
ideal := no Raman, no GVD. It is flexible to also compute the GVD, but it is not recommended.
"""
def ideal_fit_coefficients(gvda: float = 0.0,
                           gvdb: float = 0.0,
                           ipulse: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    cf = cfg.load_toml_to_struct("./input/mmf.toml")
    nc = cfg.load_nc_toml_to_struct("./input/numerical_config.toml")

    # Override DGD ranges using global targets
    lg.warning(f"Overriding DGD ranges to [{LLW_MIN}, {LLW_MAX}] ps/sqrt(km)")
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
    lg.warning("nlin_numeric")
    lg.warning(nlin_numeric)
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
        lg.info(
            f"We are calculating in the range L/LD=[1e-90, {fiber_length/lda:.1e}] for m_lo={m_lo}")
        # compute the integral
        added_noise = (quad(I_specific, 0, fiber_length)[0] / fiber_length)**2
        if m_lo != 0:
            added_noise *= 2
        lo_value += added_noise
    return lo_value

def apply_fit_correction(ps: Tuple[float, float, float],
                         lo_value: float,
                        ) -> Tuple[float, float, float]:
    old_lo_value = ps[0]
    lg.info(f"Correcting N^circ (LO val): {old_lo_value} --> {lo_value}")
    lg.info(f"Correcting delta beta1a of a factor {old_lo_value/(lo_value)}")
    ps[0] = lo_value
    ps[1] = ps[1] * old_lo_value / lo_value
    return ps

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
    lo_value_perfect = gvd_correction(cf, 
        gvda, gvdb, cf.fiber_length, ipulse=ipulse, m_lo_truncation=m_lo_truncation)

    if np.all(fB == 1.0):
        lg.info("You are using a flat fB, no Raman correction will be applied")
        ps = apply_fit_correction(ps_ideal.copy(), lo_value_perfect)
        return lambda dgd: softplus2(dgd * cf.fiber_length * cf.baud_rate, *ps) 
    
    # correct in the LO regime (Raman + GVD)
    lo_value_max = raman_gvd_correction_max(
        cf.fiber_length/lda, cf.fiber_length/ldb)
    lo_value_min = raman_gvd_correction_min(
        cf.fiber_length/lda, cf.fiber_length/ldb)
    r_lo_min, r_lo_max, r_hi_min, r_hi_max = load_raman_integral_extremes(cf)

    raman_integral_fB_lo = raman_integral(cf, "LO", fB)
    raman_integral_fB_hi = raman_integral(cf, "HI", fB)
    lo_value_fB = (raman_integral_fB_lo - r_lo_min) / (r_lo_max - r_lo_min) * \
        (lo_value_max - lo_value_min) + lo_value_min    
    ps_ramanless = apply_fit_correction(ps_ideal.copy(), lo_value_perfect)
    lg.info(f"Ramanless fit parameters (no Raman, with GVD): {ps_ramanless}")
    ps_ramanful = apply_fit_correction(ps_ideal.copy(), lo_value_fB)
    # build the linear composition to match the HI correction (in form of a simple Raman integral)
    def nlin_megafit(d):
        raise NotImplementedError("Raman not implemented yet")
        xi = (d-LLW_MIN)/(LLW_MAX-LLW_MIN)
        return softplus2(d * cf.fiber_length * cf.baud_rate, *ps_ramanful) * (1-xi) + softplus2(d * cf.fiber_length * cf.baud_rate, *ps_ramanless) * xi * raman_integral_fB_hi
    return nlin_megafit

"""
corrections due to mode multiplicity and constellation shaping
"""
def nlin_prefactor_mmf(cf, mode_a, mode_b):
    n2 = 2.6e-20
    omega_0 = 2 * np.pi * cf.center_frequency
    gamma = n2 * omega_0 / (cf.effective_area * c)
    P_in = dBm2watt(cf.launch_power)
    constellation_factor = 0.32 * 1.19  # 64-QAM
    di = np.diag_indices(len(mode_a))
    mode_b_prefactor = 2 * np.outer(np.ones((cf.n_modes)), SPATIAL_MODES)
    var = constellation_factor * mode_b_prefactor
    var[di] = (constellation_factor + 1) * (2*SPATIAL_MODES[mode_a[0]] + 3) - 4
    assert ((var > 0).all())
    nlin_prefactor = (P_in)**3 * gamma**2 * var / (cf.baud_rate**2)
    # print("denom", (2*ns[mode_a])**3)
    return nlin_prefactor

"""
wrapper for the nlin_prefactor_mmf to handle single-mode case
"""
def nlin_prefactor(cf, mode_a, mode_b):
    if len(mode_a) == 1:
        return nlin_prefactor_mmf(cf, [0], [0])[0, 0]
    else:
        return nlin_prefactor_mmf(cf, mode_a, mode_b)


def get_nlin_system(cf,
                    use_kappa=False,
                    use_fB=False,
                    use_x_mode_interactions=True,
                    use_dBm_scale=False, 
                    ipulse: int = 1,):
    assert (cf.n_modes == 4)
    assert (cf.launch_power == -5)
    if use_fB:
        raise NotImplementedError("Raman not implemented yet")
    nc = cfg.load_nc_toml_to_struct("input/numerical_config.toml")
    T = 1 / cf.baud_rate
    L = cf.fiber_length
    x_norm = L / T
    y_norm = x_norm**(-2)
    
    oi_fit = np.load('results/oi_fit.npy')
    
    # FIXME send this to the prefactor methods
    if use_kappa:
        kappa = np.loadtxt('input/kappa.csv', delimiter=',')
        kappa = kappa**2
    else:
        kappa = np.ones((cf.n_modes, cf.n_modes))
    switchoff_matrix = np.eye(cf.n_modes)
    modal_prefactor = kappa  # FIXME check this modal prefactor thing

    beta1_params = load_group_delay()
    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )
    freqs = wdm.frequency_grid()
    modes = range(cf.n_modes)
    mmfiber = MMFiber(
        effective_area=cf.effective_area,
        overlap_integrals=oi_fit,
        group_delay=beta1_params,
        length=cf.fiber_length,
    )

    beta1 = np.zeros((cf.n_modes, len(freqs)))

    for i in modes:
        beta1[i, :] = mmfiber.group_delay.evaluate_beta1(i, freqs)
    beta2 = np.zeros((cf.n_modes, len(freqs)))
    for i in modes:
        beta2[i, :] = mmfiber.group_delay.evaluate_beta2(i, freqs)
    beta1 = np.array(beta1)
    beta2 = np.array(beta2)
    # for each channel, we compute the total number of collisions that
    # needs to be computed for evaluating the total noise on that channel.
    T = 1 / cf.baud_rate
    L = cf.fiber_length
    nlin = np.zeros((cf.n_modes, len(freqs)))

    d_min = nc.dgd1
    d_max = nc.dgd2_g
    d_span = d_max - d_min
    raman_gvd_correction_max, raman_gvd_correction_min =                    build_lookup_integral_table_with_raman(cf, ipulse=ipulse)
    if use_fB:
        pass
    else:
        pass

    if use_dBm_scale:
        modal_prefactor = np.multiply(
            modal_prefactor,
            nlin_prefactor(cf, np.array(modes), np.array(modes))
        )
    # precompute the Raman corrections from the numerical results of the integrals
    # precompute the GVD-dependent fitting parameters (perfect amplification)
    rms_gvd = load_rms_gvd()
    if cf.pulse_shape == 'Gaussian':
        param_select = 0
    else:
        param_select = 1

    for mA in modes:
        for fA in range(len(freqs)):
            for mB in modes:
                for fB in range(len(freqs)):
                    gvd = rms_gvd[mA, mB]

    # rescaling for cross-mode interactions
    if not use_x_mode_interactions:
        modal_prefactor = np.multiply(modal_prefactor, switchoff_matrix)
        # modal_prefactor *= 0.9
    modal_prefactor = modal_prefactor[:cf.n_modes, :cf.n_modes]
    nlin = modal_prefactor @ nlin
    return nlin


if __name__ == "__main__":
    ideal_fit_coefficients(0.0, 0.0, 1)
    exit()

    import scripts.modules.cfg as cfg
    cf = cfg.load_toml_to_struct("./input/mmf.toml")

    # build the interpolator and pass it to the corrector
    I_low_n = np.load(f"results/I_low_nyquist.npz")
    interp = build_I_low_interpolator(I_low_dataset=I_low_n, ipulse=0)
    lg.debug("Useful range of L/LD from LO time integral data: ",
             I_low_n['lld_range'])