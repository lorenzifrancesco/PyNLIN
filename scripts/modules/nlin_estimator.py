from scipy.constants import c
from pynlin.utils import dBm2watt
from pynlin.fiber import MMFiber
import pynlin
import numpy as np
from typing import Tuple
from scipy.optimize import curve_fit
from scipy.integrate import quad
from loguru import logger as lg
from scripts.modules.log_init import init_logging
init_logging()

from scripts.modules.load_fiber_values import load_group_delay, load_rms_gvd
import scripts.modules.cfg as cfg
from scripts.modules.collision import build_I_low_interpolator

SPATIAL_MODES = np.array([1, 2, 2, 1])
LLW_MIN = 0.01  # target L/LW
LLW_MAX = 100.0


def fB_preprocessing(cf):
    fBs = load_fB(cf)
    # for the maximum and the minimum Raman profiles, we need to compute the m = 0 integral for the GVD grid.
    # Mapping to a angle - lenght space.
    

def load_fB(cf: cfg.Config) -> Tuple[np.ndarray, callable, callable]:
    assert (cf.launch_power == -5.0 and cf.raman_gain == 0.0)
    sol_path = "results/ct_solution-6_gain_0.0.npy" # all the information about the numerosity and stuff is here.
    solutions = np.load(sol_path, allow_pickle=True).item()

    signal_powers = solutions['signal_sol']
    signal_powers = np.swapaxes(signal_powers, 1, 2) # indices: (z, mode, channel)
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


# a hot function !! gets evaluated (Nch)^2 times
def get_raman_corrections(cf, # FIXME get_raman_extremes 
                          gvda, 
                          gvdb, 
                          fB, 
                          fB_min, 
                          fB_max, 
                          asymptotic,
                          ) -> Tuple[float, float, float, float]:
    """
    Given the optimized signal profiles, calculate max and min asymptotic values
    for the correction coefficients f_B.
    Requires: input/mmf.toml or smf.toml,
              results/ct_solution-6_gain_0.0[_SMF].npy
    """
    # toml_path = "./input/smf.toml" if smf else "./input/mmf.toml"
    # select_string = "_SMF" if smf else ""
    # cf = cfg.load_toml_to_struct(toml_path)

    fB, fB_min, fB_max, fB_min_func, fB_max_func = load_fB(cf)
    z_axis = np.linspace(0, cf.fiber_length, len(fB_max))
    dz = z_axis[1] - z_axis[0]
    if asymptotic == Regime.HI:
        r_lo_min = (np.sum(fB_min) * dz / cf.fiber_length)**2 # insert the Raman simple here.
        r_lo_max = (np.sum(fB_max) * dz / cf.fiber_length)**2
        r_hi_min = np.sum(fB_min**2) * dz / cf.fiber_length
        r_hi_max = np.sum(fB_max**2) * dz / cf.fiber_length

        rcal_hi = np.sum(fB**2, axis=0) * dz / cf.fiber_length
        rcal_lo = (np.sum(fB, axis=0) * dz / cf.fiber_length)**2
        assert ((rcal_hi >= r_hi_min - 1e-15).all())
        assert ((rcal_hi <= r_hi_max + 1e-15).all())
        assert ((rcal_lo >= r_lo_min - 1e-15).all())
        assert ((rcal_lo <= r_lo_max + 1e-15).all())
    else: # Raman complicated (need the fitting -->
        # - We can work in a faster way. Estimate only t 
        # # REUSE THE CONCEPT ALREADY WRITTEN IN THE ARTICLE.
        noise_min = np.load(f"results/partial_nlin_gaussian_min{gvd}B2.npy")
        noise_max = np.load(f"results/partial_nlin_gaussian_max{gvd}B2.npy")
        noise_perfect = np.load(f"results/partial_nlin_gaussian_perfect{gvd}B2.npy") # load integral here: 
        
        r_lo_min = noise_min[0] / noise_perfect[0]
        r_hi_min = noise_min[-1] / noise_perfect[-1]
        r_lo_max = noise_max[0] / noise_perfect[0]
        r_hi_max = noise_max[-1] / noise_perfect[-1]

    return r_lo_min, r_lo_max, r_hi_min, r_hi_max


def correct_fit_coefficients(ps: Tuple[float, float, float], 
                             lda: float, 
                             ldb: float,
                             fiber_length: float,
                             interp: callable):
    """
    This only corrects for the non negligible gvd
    """
    I_specific = lambda x: interp(x/lda, x/ldb) # this is also in normalized units
    lg.info(f"a sample of I_specific at L/LD=1: {I_specific(1.0)}")
    lo_value = quad(I_specific, 0, fiber_length)[0]
    # correction of fit params: check that this is ok. Keep eta the same
    old_lo_value = ps[0]
    lg.debug(f"Correcting N^circ (LO val): {old_lo_value} --> {lo_value}")
    ps[0] = lo_value
    ps[1] = ps[1] * lo_value / old_lo_value
    return ps


def get_fit_coefficients(gvda: float = 0.0,
                         gvdb: float = 0.0,
                         ipulse: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    cf = cfg.load_toml_to_struct("./input/mmf.toml")
    nc = cfg.load_nc_toml_to_struct("./input/numerical_config.toml")

    # Override DGD ranges using global targets
    lg.warning(f"Overriding DGD ranges to [{LLW_MIN}, {LLW_MAX}] ps/sqrt(km)")
    nc.dgd1 = LLW_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = LLW_MAX / (cf.fiber_length * cf.baud_rate) # build the fit from the minimum to the maximum 
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
        raise("You are trying to cheat! Instead of fitting from a case computed with dispersion, you should use the dispersion correction given by correct_fit_coefficients")
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


def get_nlin_prefactor_mmf(cf, mode_a, mode_b):
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


def get_nlin_prefactor(cf, mode_a, mode_b):
    if len(mode_a) == 1:
        return get_nlin_prefactor_mmf(cf, [0], [0])[0, 0]
    else:
        return get_nlin_prefactor_mmf(cf, mode_a, mode_b)


def get_nlin_system(cf,
             use_kappa=False,
             use_fB=False,
             use_x_mode_interactions=True,
             use_dBm_scale=False):
    assert (cf.n_modes == 4)
    assert (cf.launch_power == -5)
    if use_fB:
        raise NotImplementedError("Raman not implemented yet")
    nc = cfg.load_nc_toml_to_struct("input/numerical_config.toml") # FIXME
    T = 1 / cf.baud_rate
    L = cf.fiber_length
    x_norm = L / T
    y_norm = x_norm**(-2)
    oi_fit = np.load('results/oi_fit.npy')
    if use_kappa:
        kappa = np.loadtxt('input/kappa.csv', delimiter=',')
        kappa = kappa**2
    else:
        kappa = np.ones((cf.n_modes, cf.n_modes))
    switchoff_matrix = np.eye(cf.n_modes)
    
    
    # # FIXME
    # if cf.n_modes == 1:
    #     solutions = np.load("results/ct_solution"+str(int(round(cf.launch_power)))+"_gain_0.0_SMF.npy",
    #                         allow_pickle=True).item()
    # else:
    #     solutions = np.load("results/ct_solution"+str(int(round(cf.launch_power)))+"_gain_0.0.npy",
    #                         allow_pickle=True).item()

    # signal_powers = solutions['signal_sol']
    # signal_powers_swp = np.swapaxes(signal_powers, 1, 2)
    # assert (cf.n_modes == signal_powers_swp.shape[1])
    # assert (cf.n_channels == signal_powers_swp.shape[2])
    # initial_powers = signal_powers_swp[0, :, :]
    # fB = np.divide(signal_powers_swp, initial_powers)
    # assert ((fB <= 1.5).all())
    # assert ((fB > 0).all())
    z_axis = np.linspace(0, cf.fiber_length, len(fB))
    dz = z_axis[1] - z_axis[0]
    # # coeffs = np.polyfit(z_axis, fB, 6)
    # if use_fB:
    #     rcal_minus = (np.sum(fB, axis=0) * dz / cf.fiber_length)**2
    #     rcal_plus = np.sum(fB**2, axis=0) * dz / cf.fiber_length
    # else:
    #     rcal_plus = 1.0
    #     rcal_minus = 1.0

    beta1_params = load_group_delay()
    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )
    freqs = wdm.frequency_grid()
    modes = range(cf.n_modes)

    fiber = MMFiber(
        effective_area=80e-12,
        overlap_integrals=oi_fit,
        group_delay=beta1_params,
        length=100e3
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
    nlin = np.zeros((cf.n_modes, len(freqs)))

    d_min = nc.dgd1
    d_max = nc.dgd2_g
    d_span = d_max - d_min
    if use_fB:
        r_lo_min, r_lo_max, r_hi_min, r_hi_max = get_raman_corrections(
            smf=(cf.n_modes == 1))
        r_bar_lo = (rcal_minus - r_lo_min) / (r_lo_max - r_lo_min)
        r_bar_hi = (rcal_plus - r_hi_min) / (r_hi_max - r_hi_min)
        # assert((rcal_plus<=r_hi_max).all())
        # assert((rcal_plus>=r_hi_min).all())
        # assert((rcal_minus<=f_minus_max).all())
        # assert((rcal_minus>=f_minus_min).all())
        assert (nc.dgd2_n == nc.dgd2_g)
        # returns a (4, 250) matrix for all the b channels
        # assert((r_bar_lo>-1e-15).all())
        # assert((r_bar_hi>-1e-15).all())
    else:
        f_lo_plus = 1.0
        f_lo_minus = 1.0
        f_hi_plus = 1.0
        f_hi_minus = 1.0
        def zeta(d): return np.ones((len(modes), len(freqs)))

    modal_prefactor = kappa  # FIXME check this modal prefactor thing
    if use_dBm_scale:
        modal_prefactor = np.multiply(
            modal_prefactor,
            get_nlin_prefactor(cf, np.array(modes), np.array(modes))
        )
    # precompute the Raman corrections from the numerical results of the integrals
    # precompute the GVD-dependent fitting parameters (perfect amplification)
    rms_gvd = load_rms_gvd()
    ps_matrix = np.zeros((cf.n_modes, cf.n_modes, 3))
    r_bar_hi_min = np.zeros((cf.n_modes, cf.n_modes))
    r_bar_lo_min = np.zeros((cf.n_modes, cf.n_modes))
    r_bar_hi_max = np.zeros((cf.n_modes, cf.n_modes))
    r_bar_lo_max = np.zeros((cf.n_modes, cf.n_modes))
    if cf.pulse_shape == 'Gaussian':
        param_select = 0
    else:
        param_select = 1

    for mA in modes:
        for mB in modes:
            gvd = rms_gvd[mA, mB]
            # FIXME inside of this function, implement the shift of the variables with
            ps_matrix[mA, mB, :] = get_fit_coefficients(gvd=gvd)[param_select]
            r_bar_lo_min[mA, mB], r_bar_lo_max[mA, mB], r_bar_hi_min[mA, mB], r_bar_hi_max[mA,
                                                                                           mB] = get_raman_corrections(smf=(cf.n_modes == 1), gvd=gvd)

    # nlin_megafit = lambda d: softplus2(d * x_norm, *ps_perf[0, :]) * raman_correction(d) / y_norm
    rms_gvd = load_rms_gvd()
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
    get_fit_coefficients(0.0, 0.0, 1)
    exit()
    
    import scripts.modules.cfg as cfg
    cf = cfg.load_toml_to_struct("./input/mmf.toml")
    correct_fit_coefficients([1, 2, 3], 0.5, 0.5, cf.fiber_length, lambda x: x) # this is broken
    
    # build the interpolator and pass it to the corrector
    I_low_n = np.load(f"results/I_low_nyquist.npz")
    interp = build_I_low_interpolator(I_low_dataset = I_low_n, ipulse = 0)
    lg.debug("Useful range of L/LD from LO time integral data: ", I_low_n['lld_range'])
    
    correct_fit_coefficients([1, 2, 3], 0.4, 0.4, interp)