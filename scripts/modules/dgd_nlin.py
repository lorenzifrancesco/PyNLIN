import logging
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# import plotly.graph_objects as go
# import seaborn as sns
import pynlin.wdm
from pynlin.utils import nu2lambda
from scripts.modules.load_fiber_values import load_group_delay, load_dummy_group_delay
from scripts.modules.threshold import get_raman_corrections, get_fit_coefficients, softplus2
from numpy import polyval
from pynlin.fiber import MMFiber
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
import scripts.modules.cfg as cfg
from pynlin.utils import watt2dBm, dBm2watt
from scipy.optimize import curve_fit
from scipy.constants import c

ns = np.array([1, 2, 2, 1])

# # SMF
# def get_nlin_prefactor_smf(cf):
#     n2 = 2.6e-20 # constant of SiO2 # FIXME offload
#     omega_0 = 2 * np.pi * cf.center_frequency # Hz
#     gamma = n2 * omega_0/ (cf.effective_area * c) # WDM band center approximation
#     print("Using gamma = ", gamma)
#     P_in = dBm2watt(cf.launch_power)
#     constellation_factor = 0.32 * 1.19 # 64-QAM (<|b|^4>/<|b|^2>^2 - 1)
#     nlin_prefactor = (P_in)**3 * gamma**2 * constellation_factor / (cf.baud_rate**2)
#     print("nlin prefactor", nlin_prefactor)
#     return nlin_prefactor

SPATIAL_MODES = [1, 2, 2, 1]

def get_nlin_prefactor_mmf(cf, mode_a, mode_b):
    n2 = 2.6e-20
    omega_0 = 2 * np.pi * cf.center_frequency
    gamma = n2 * omega_0 / (cf.effective_area * c)
    P_in = dBm2watt(cf.launch_power)
    constellation_factor = 0.32 * 1.19 # 64-QAM
    di = np.diag_indices(len(mode_a))
    mode_b_prefactor = 2 * np.outer(np.ones((cf.n_modes)), SPATIAL_MODES)
    var = constellation_factor * mode_b_prefactor 
    var[di] =  (constellation_factor + 1) * (2*ns[mode_a[0]] + 3) - 4
    assert((var > 0).all())
    nlin_prefactor = (P_in)**3 * gamma**2 * var / (cf.baud_rate**2)
    # print("denom", (2*ns[mode_a])**3)
    return nlin_prefactor

def get_nlin_prefactor(cf, mode_a, mode_b):
    if len(mode_a) == 1:
        return get_nlin_prefactor_mmf(cf, [0], [0])[0, 0]
    else: 
        return get_nlin_prefactor_mmf(cf, mode_a, mode_b)

def get_nlin(cf,
             use_kappa=False,
             use_fB=False,
             use_x_mode_interactions=True, 
             use_dBm_scale=False):
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
    assert(cf.launch_power==-6)
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
    assert((fB<=1.5).all())
    assert((fB>0).all())
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

    # collisions = np.zeros((len(modes), len(freqs)))
    # for i in range(len(modes)):
    #     for j in range(len(freqs)):
    #         collisions[i, j] = np.floor(np.abs(np.sum(beta1 - beta1[i, j])) * L / T)

    # collisions_single = np.zeros((1, len(freqs)))
    # for j in range(len(freqs)):
    #     collisions_single[0, j] = np.floor(
    #         np.abs(np.sum(beta1[0:] - beta1[0, j])) * L / T)

    nlin = np.zeros((cf.n_modes, len(freqs)))
    
    d_min = nc.dgd1
    d_max = nc.dgd2_g
    d_span = d_max - d_min
    if use_fB:
        f_minus_min, f_minus_max, f_plus_min, f_plus_max = get_raman_corrections(smf=(cf.n_modes==1))
        lincomb_lo = (rcal_minus - f_minus_min) / (f_minus_max - f_minus_min)
        lincomb_hi = (rcal_plus  - f_plus_min) / (f_plus_max - f_plus_min)
        # print(lincomb_lo)
        # print(lincomb_hi)
        assert((rcal_plus<=f_plus_max).all())
        assert((rcal_plus>=f_plus_min).all())
        # assert((rcal_minus<=f_minus_max).all())
        # assert((rcal_minus>=f_minus_min).all())
        assert (nc.dgd2_n == nc.dgd2_g)
        # returns a (4, 250) matrix for all the b channels
        assert((lincomb_lo>-1e-15).all())
        assert((lincomb_hi>-1e-15).all())
        def lc(d):
            return 1/d_span * ((d_max-d) * lincomb_lo + (d-d_min) * lincomb_hi)
    else:
        f_lo_plus = 1.0
        f_lo_minus = 1.0
        f_hi_plus = 1.0
        f_hi_minus = 1.0
        lc = lambda d: np.ones((len(modes), len(freqs)))

    ps_g, ps_n = get_fit_coefficients()
    ps = ps_g if cf.pulse_shape == 'Gaussian' else ps_n

    # beware, we have a mixed unit system here, so we need to be careful
    # print("Optimal parameters MAX: ", ps[0,  :])
    # print("Optimal parameters MIN: ", ps[1, :])
    lc_softplus = lambda d: (lc(d) * softplus2(d * x_norm, *ps[0, :]) + (1-lc(d)) * softplus2(d*x_norm, *ps[1, :])) / y_norm

    def pair_noise(dgd):
        return lc_softplus(dgd)
    #
    modal_prefactor = kappa
        
    if use_dBm_scale:
        modal_prefactor = np.multiply(
            modal_prefactor,
            get_nlin_prefactor(cf, np.array(modes), np.array(modes))
            )

    if not use_x_mode_interactions:
        modal_prefactor = np.multiply(modal_prefactor, switchoff_matrix)
        # modal_prefactor *= 0.9
    modal_prefactor = modal_prefactor[:cf.n_modes, :cf.n_modes]
    for i in modes:
        for j in range(len(freqs)):
            # print("WARN: we are neglecting the fact that the kappa matrix is not symmetric.")
            nlin_unweighted = np.sum(pair_noise(np.abs(beta1 - beta1[i, j])), axis = 1)
            assert((nlin_unweighted > 0).all())
            nlin[i, j] = (modal_prefactor @ nlin_unweighted)[i]
    return nlin

def noise_plot(use_kappa=False,
               use_smf=False,
               use_fB=False,
               use_dBm_scale=False,
               use_plot_without_x_mode=True, 
               name = "xxx"):
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits([0, 0])
    # rc('text', usetex=True)
    dpi = 300
    grid = False

    smf_file = "./input/smf.toml"
    mmf_file = "./input/mmf.toml"
    cf_smf = cfg.load_toml_to_struct(smf_file)
    cf_mmf = cfg.load_toml_to_struct(mmf_file)
    print(
        f"Loading a ITU-T standardized WDM grid \n [spacing: {cf_smf.channel_spacing * 1e-9:.3e}GHz, center: {cf_smf.center_frequency * 1e-12:.3e}THz] \n")
    assert (cf_smf.fiber_length == cf_mmf.fiber_length)
    assert (cf_smf.baud_rate == cf_mmf.baud_rate)
    if (cf_smf.n_channels == cf_mmf.n_channels * cf_mmf.n_modes):
        print("We are comparing the same total rate")
    else:
        print(
            f"Warning: the number of channels is different: {cf_smf.n_channels} vs {cf_mmf.n_channels * cf_mmf.n_modes}")
    assert (cf_smf.center_frequency == cf_mmf.center_frequency)
    wdm = pynlin.wdm.WDM(
        spacing=cf_smf.channel_spacing,
        num_channels=cf_smf.n_channels,
        center_frequency=cf_smf.center_frequency
    )
    freqs_smf = wdm.frequency_grid()
    wdm = pynlin.wdm.WDM(
        spacing=cf_mmf.channel_spacing,
        num_channels=cf_mmf.n_channels,
        center_frequency=cf_mmf.center_frequency
    )
    freqs_mmf = wdm.frequency_grid()
    # print("MMF", get_nlin_prefactor_mmf(cf_mmf, 1, 1))
    # print("SMF", get_nlin_prefactor_smf(cf_smf))
    # print("aeff1/2", (cf_smf.effective_area/cf_mmf.effective_area)**2)
    # exit()
    nlin_mmf                = get_nlin(cf_mmf,
                                       use_kappa=use_kappa,
                                       use_fB=use_fB,
                                       use_x_mode_interactions=True,
                                       use_dBm_scale=use_dBm_scale,
                                       )
    nlin_mmf_noninteracting =  get_nlin(cf_mmf,
                                       use_kappa=use_kappa,
                                       use_fB=use_fB,
                                       use_x_mode_interactions=False,
                                       use_dBm_scale=use_dBm_scale,)
    nlin_smf                = get_nlin(cf_smf,
                                       use_kappa=use_kappa,
                                       use_fB=use_fB,
                                       use_x_mode_interactions=True,
                                       use_dBm_scale=use_dBm_scale,)
    # for each channel, we compute the total number of collisions that
    # needs to be computed for evaluating the total noise on that channel.
    T = 1 / cf_mmf.baud_rate
    L = cf_mmf.fiber_length

    colors = ["blue", "orange", "green", "red", "gray"]
    linestyles = ["-", "-", "-", "-", "--"]
    labels = ["LP01", "LP1", "LP11", "LP11", "SMF(LP01)"]

    if use_dBm_scale:
        ylabel = r'$P_\mathrm{NLIN} \; [\mathrm{dBm}]$'
        plot_function = plt.plot
        def y_function_mmf(x): return watt2dBm(x)
        def y_function_smf(x): return watt2dBm(x)
    else:
        ylabel = r'$\sum\limits_{B\neq A}\mathcal{N}_{AB} \quad [\mathrm{km}^2/\mathrm{ps}^{2}]$'
        plot_function = plt.semilogy
        def y_function_mmf(x): return x * 1e-30
        def y_function_smf(x): return x * 1e-30

    plt.clf()
    lw = 1.4
    plt.figure(figsize=(3.6, 3.2))
    for i in range(cf_mmf.n_modes):
        plot_function(freqs_mmf * 1e-12,
                      y_function_mmf(nlin_mmf[i, :]),
                      lw=lw,
                      color=colors[i],
                      ls=linestyles[i],
                      label=labels[i])
        if use_plot_without_x_mode:
            plot_function(freqs_mmf * 1e-12,
                          y_function_mmf(nlin_mmf_noninteracting[i, :]),
                          lw=lw,
                          color=colors[i],
                          ls='-.')
    if use_smf:
        plot_function(freqs_smf * 1e-12,
                      y_function_smf(nlin_smf[0, :]),
                      lw=lw,
                      color=colors[-1],
                      ls=linestyles[-1],
                      label=labels[-1])
    #
    plt.xlabel(r'$f \; [\mathrm{THz}]$')
    plt.ylabel(ylabel)
    # plt.legend(labelspacing=0.1)
    plt.grid(grid)
    plt.tight_layout()
    # plt.ylim([2e-2, 1e0])
    plt.savefig(f"media/nlin"+name+".pdf", dpi=dpi)
    print("The figure is saved as media/nlin"+name+".pdf")

    functions = [np.mean, np.median, np.max, np.min]
    function_names = ["mean  ", "median", "max   ", "min   "]
    for foo, name in zip(functions, function_names):
        avg_nlin_mmf = foo(nlin_mmf)
        avg_nlin_smf = foo(nlin_smf)
        print(
            name + f" NLIN coeff per channel: MMF -> {avg_nlin_mmf:4.3e} | SMF -> {avg_nlin_smf:4.3e}")
        # apply QAM 16 and -10 dBm input power
        print(
            name + f" NLIN power per channel: MMF -> {watt2dBm(avg_nlin_mmf):4.1f} dBm | SMF -> {watt2dBm(avg_nlin_smf):4.1f} dBm")
        print("-" * 20)


def noise_histogram(use_kappa=False,
                    use_smf=False,
                    use_fB=False,
                    use_dBm_scale=False):
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits([0, 0])
    rc('text', usetex=True)
    dpi = 300
    grid = False

    smf_file = "./input/smf.toml"
    mmf_file = "./input/mmf.toml"
    cf_smf = cfg.load_toml_to_struct(smf_file)
    cf_mmf = cfg.load_toml_to_struct(mmf_file)
    print(
        f"Loading a ITU-T standardized WDM grid \n [spacing: {cf_smf.channel_spacing * 1e-9:.3e}GHz, center: {cf_smf.center_frequency * 1e-12:.3e}THz] \n")
    assert (cf_smf.fiber_length == cf_mmf.fiber_length)
    assert (cf_smf.baud_rate == cf_mmf.baud_rate)
    # assert (cf_smf.n_channels == cf_mmf.n_channels * cf_mmf.n_modes)
    assert (cf_smf.center_frequency == cf_mmf.center_frequency)
    wdm = pynlin.wdm.WDM(
        spacing=cf_smf.channel_spacing,
        num_channels=cf_smf.n_channels,
        center_frequency=cf_smf.center_frequency
    )
    freqs_smf = wdm.frequency_grid()
    wdm = pynlin.wdm.WDM(
        spacing=cf_mmf.channel_spacing,
        num_channels=cf_mmf.n_channels,
        center_frequency=cf_mmf.center_frequency
    )
    freqs_mmf = wdm.frequency_grid()

    nlin_mmf = get_nlin(cf_mmf,
                        use_kappa=use_kappa,
                        use_fB=use_fB,
                        use_dBm_scale=use_dBm_scale,)
    nlin_smf = get_nlin(cf_smf,
                        use_kappa=False,
                        use_fB=True,
                        use_dBm_scale=use_dBm_scale,)
    # for each channel, we compute the total number of collisions that
    # needs to be computed for evaluating the total noise on that channel.
    T = 1 / cf_mmf.baud_rate
    L = cf_mmf.fiber_length

    colors = ["blue", "orange", "green", "red", "gray"]
    linestyles = ["-", "-", "-", "-", "--"]
    labels = ["LP01", "LP1", "LP11", "LP11", "SMF(LP01)"]

    if use_dBm_scale:
        ylabel = r'$\mathrm{n.\;  of\; channels}$'
        plot_function = plt.hist
        def y_function_mmf(x): return watt2dBm(x)
        def y_function_smf(x): return watt2dBm(x)
    else:
        ylabel = r'$\mathrm{NLIN} \; [\mathrm{km}^2/\mathrm{ps}^{2}]$'
        plot_function = plt.hist
        def y_function_mmf(x): return x * 1e-30
        def y_function_smf(x): return x * 1e-30

    plt.clf()
    plt.figure(figsize=(3.6, 2.8))
    y_data = y_function_mmf(nlin_mmf)

    y_extremes = [np.min(y_data), -40]
    n_bins = 25
    bin_width = (y_extremes[1] - y_extremes[0]) / n_bins
    bins = np.arange(y_extremes[0], y_extremes[1] + bin_width, bin_width)
    alpha   = 0.1
    alpha2  = 0.99
    lw      = 5
    for i in range(cf_mmf.n_modes):
        plot_function(y_data[i, :],
                      bins=bins,
                      histtype='stepfilled',
                      alpha=alpha,
                      lw=lw,
                      color=colors[i],)
        plot_function(y_data[i, :],
                      bins=bins,
                      histtype='step',
                      lw=lw,
                      alpha=alpha2,
                      color=colors[i],)
    if use_smf:
        plot_function(y_function_smf(nlin_smf[0, :]),
                      bins=bins,
                      histtype='stepfilled',
                      alpha=alpha,
                      lw=lw,
                      color=colors[-1],)
        plot_function(y_function_smf(nlin_smf[0, :]),
                      bins=bins,
                      histtype='step',
                      lw=lw,
                      alpha=alpha2,
                      linestyle='--',
                      color=colors[-1],)
    #
    plt.xlabel(r'$P_\mathrm{NLIN} \; [\mathrm{dBm}]$')
    plt.ylabel(ylabel)
    plt.yscale('log')
    # plt.legend(labelspacing=0.1)
    plt.grid(grid)
    plt.tight_layout()
    # plt.ylim([2e-2, 1e0])
    plt.savefig(f"media/6-noise.pdf", dpi=dpi)
    print("The figure is saved as media/6-noise.pdf")

if __name__ == "__main__":
    for realistic in [True, False]:
        if realistic: 
            name = "realistic"
        else:
            name = "idealized"
        noise_plot(use_kappa=realistic,
               use_smf=realistic,
               use_fB=realistic,
               use_dBm_scale=realistic,
               use_plot_x_mode=not realistic,
               name = name)
        
    noise_histogram(use_kappa=True, 
                    use_smf=True,
                    use_fB=True,
                    use_dBm_scale=True)