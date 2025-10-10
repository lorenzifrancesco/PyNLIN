from scipy.constants import c
from pynlin.utils import dBm2watt
from scripts.modules.load_fiber_values import load_group_delay, load_rms_gvd
from pynlin.fiber import MMFiber
import pynlin
import numpy as np
import scripts.modules.cfg as cfg
from typing import Tuple
from scipy.optimize import curve_fit

from loguru import logger as lg
from scripts.modules.log_init import init_logging
init_logging()

SPATIAL_MODES = np.array([1, 2, 2, 1])
LLW_MIN = 0.01 # target L/LW
LLW_MAX = 100.0


def softplus2(x, a, b, c):
    return a * (1 + (x / b)**(1 / c))**(-c)





# perfect -> nonzero gvd / nonzero Raman
# Additive effect





































def get_raman_corrections(smf: bool = False, gvd=0.0) -> Tuple[float, float, float, float]:
    """
    Given the optimized signal profiles, calculate max and min asymptotic values
    for the correction coefficients f_B.
    Requires: results/oi_fit.npy, input/mmf.toml or smf.toml,
              results/ct_solution-6_gain_0.0[_SMF].npy
    """
    toml_path = "./input/smf.toml" if smf else "./input/mmf.toml"
    select_string = "_SMF" if smf else ""
    cf = cfg.load_toml_to_struct(toml_path)

    assert (cf.launch_power == -5.0 and cf.raman_gain == 0.0)

    sol_path = f"results/ct_solution-6_gain_0.0{select_string}.npy"
    solutions = np.load(sol_path, allow_pickle=True).item()

    signal_powers = solutions['signal_sol']
    signal_powers_swp = np.swapaxes(signal_powers, 1, 2)
    initial_powers = signal_powers_swp[0, :, :]
    fB = np.divide(signal_powers_swp, initial_powers)
    assert ((fB >= 0).all())
    fB_max = np.max(fB, axis=(1, 2))
    fB_min = np.min(fB, axis=(1, 2))
    assert ((fB_min <= fB_max).all())

    z_axis = np.linspace(0, cf.fiber_length, len(fB_max))
    dz = z_axis[1] - z_axis[0]
    assert (fB.shape[0] == len(z_axis))

    if gvd == 0:
        r_lo_min = (np.sum(fB_min) * dz / cf.fiber_length)**2
        r_lo_max = (np.sum(fB_max) * dz / cf.fiber_length)**2
        r_hi_min = np.sum(fB_min**2) * dz / cf.fiber_length
        r_hi_max = np.sum(fB_max**2) * dz / cf.fiber_length

        # Sanity checks
        rcal_hi = np.sum(fB**2, axis=0) * dz / cf.fiber_length
        rcal_lo = (np.sum(fB, axis=0) * dz / cf.fiber_length)**2
        assert ((rcal_hi >= r_hi_min - 1e-15).all())
        assert ((rcal_hi <= r_hi_max + 1e-15).all())
        assert ((rcal_lo >= r_lo_min - 1e-15).all())
        assert ((rcal_lo <= r_lo_max + 1e-15).all())
    else:
        noise_min = np.load(f"results/partial_nlin_gaussian_min{gvd}B2.npy")
        noise_max = np.load(f"results/partial_nlin_gaussian_max{gvd}B2.npy")
        noise_perfect = np.load(
            f"results/partial_nlin_gaussian_perfect{gvd}B2.npy")
        r_lo_min = noise_min[0] / noise_perfect[0]
        r_hi_min = noise_min[-1] / noise_perfect[-1]
        r_lo_max = noise_max[0] / noise_perfect[0]
        r_hi_max = noise_max[-1] / noise_perfect[-1]

    return r_lo_min, r_lo_max, r_hi_min, r_hi_max

def get_dispersion_corrected_raman_coefficient(gvd_rms: float,
                                               dless_r: float,
                                               dless_r_extremes: Tuple[float, float, float], d_r_extremes: Tuple[float, float]) -> float:

    raise NotImplementedError(
        "This function is not currently used, please implement it if needed")
    return d_r_extremes[1]


def get_fit_coefficients(fB_simple_interpolation: bool = False, gvd=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the numerical noise results for a representative GVD, find the best
    fit coefficients in the Nyquist and Gaussian cases. Only consider max and min profiles.
    Requires: results/partial_nlin_gaussian_*.npy, results/partial_nlin_nyquist_*.npy,
              input/mmf.toml, input/numerical_config.toml, results/oi_fit.npy
    """
    cf = cfg.load_toml_to_struct("./input/mmf.toml")
    nc = cfg.load_nc_toml_to_struct("./input/numerical_config.toml")

    # Override DGD ranges using global targets
    lg.warning(f"Overriding DGD ranges to [{LLW_MIN}, {LLW_MAX}] ps/sqrt(km)")
    old = {"dgd1": getattr(nc, "dgd1", None), "dgd2_n": getattr(
        nc, "dgd2_n", None), "dgd2_g": getattr(nc, "dgd2_g", None)}
    nc.dgd1 = LLW_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = LLW_MAX / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_g = nc.dgd2_n
    if gvd is None:
        gvd = nc.gvd

    oi_fit = np.load('results/oi_fit.npy')
    beta1_params = load_group_delay()
    # fiber = MMFiber(
    #     effective_area=cf.effective_area,
    #     overlap_integrals=oi_fit,
    #     group_delay=beta1_params,
    #     length=cf.fiber_length,
    #     n_modes=cf.n_modes
    # )

    modes = ["perfect"] if fB_simple_interpolation else ["min", "max"]
    dgds_numeric_g = np.logspace(
        np.log10(nc.dgd1), np.log10(nc.dgd2_g), nc.n_samples_numeric_g)
    dgds_numeric_n = np.logspace(
        np.log10(nc.dgd1), np.log10(nc.dgd2_n), nc.n_samples_numeric_n)
    T = 1 / cf.baud_rate
    L = fiber.length # HOW TO FIXME
    x_norm = L / T
    y_norm = x_norm**(-2)
    p0 = [0.2, 4.5, 0.5]

    ps_g = np.zeros((len(modes), 3))
    ps_n = np.zeros((len(modes), 3))

    for mode in modes:
        partial_B2g = np.load(
            f"results/partial_nlin_gaussian_{mode}{gvd}B2.npy")
        partial_B2n = np.load(
            f"results/partial_nlin_nyquist_{mode}{gvd}B2.npy")

        assert len(dgds_numeric_g) == len(
            partial_B2g), f"Gaussian: {len(dgds_numeric_g)} vs {len(partial_B2g)}"
        assert len(dgds_numeric_n) == len(
            partial_B2n), f"Nyquist: {len(dgds_numeric_n)} vs {len(partial_B2n)}"

        popt_n, _ = curve_fit(softplus2, dgds_numeric_n *
                              x_norm, partial_B2n * y_norm, p0=p0)
        popt_g, _ = curve_fit(softplus2, dgds_numeric_g *
                              x_norm, partial_B2g * y_norm, p0=p0)
        ps_g[modes.index(mode), :] = popt_g
        ps_n[modes.index(mode), :] = popt_n

    return ps_g, ps_n



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

# system-wide nlin
def get_nlin(cf,
             use_kappa=False,
             use_fB=False,
             use_x_mode_interactions=True,
             use_dBm_scale=False):
    assert (cf.n_modes == 4)
    assert (cf.launch_power == -5)
    nc = cfg.load_nc_toml_to_struct("input/numerical_config.toml")
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
    if cf.n_modes == 1:
        solutions = np.load("results/ct_solution"+str(int(round(cf.launch_power)))+"_gain_0.0_SMF.npy",
                            allow_pickle=True).item()
    else:
        solutions = np.load("results/ct_solution"+str(int(round(cf.launch_power)))+"_gain_0.0.npy",
                            allow_pickle=True).item()

    signal_powers = solutions['signal_sol']
    signal_powers_swp = np.swapaxes(signal_powers, 1, 2)
    assert (cf.n_modes == signal_powers_swp.shape[1])
    assert (cf.n_channels == signal_powers_swp.shape[2])
    initial_powers = signal_powers_swp[0, :, :]
    fB = np.divide(signal_powers_swp, initial_powers)
    assert ((fB <= 1.5).all())
    assert ((fB > 0).all())
    z_axis = np.linspace(0, cf.fiber_length, len(fB))
    dz = z_axis[1] - z_axis[0]
    # coeffs = np.polyfit(z_axis, fB, 6)
    if use_fB:
        rcal_minus = (np.sum(fB, axis=0) * dz / cf.fiber_length)**2
        rcal_plus = np.sum(fB**2, axis=0) * dz / cf.fiber_length
    else:
        rcal_plus = 1.0
        rcal_minus = 1.0

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

        def zeta(d):
            return (d-d_max) / d_span

        def raman_correction(d):
            return (zeta(d) * r_bar_lo + (1-zeta(d)) * r_bar_hi)
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