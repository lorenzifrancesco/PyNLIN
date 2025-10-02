import logging
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pynlin.wdm
from pynlin.utils import nu2lambda
from scripts.modules.load_fiber_values import load_group_delay, load_dummy_group_delay
from numpy import polyval
from pynlin.fiber import *
from pynlin.pulses import *
from pynlin.nlin import compute_all_collisions_time_integrals, get_dgd, X0mm_space_integral, get_gvd
from matplotlib.gridspec import GridSpec
import scripts.modules.cfg as cfg
from scipy.interpolate import interp1d
from pynlin.collisions import get_m_values, get_collision_location
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from typing import Tuple

DGD_MIN = 0.01  # target L/LW
DGD_MAX = 100.0


def softplus2(x, a, b, c):
    return a * (1 + (x / b)**(1 / c))**(-c)


def get_raman_corrections(smf=False) -> Tuple[float, float, float, float]:
    '''
    Given the optimized signal profiles, calculate max and min asymptotic values 
    for the correction coefficients f_B
    requires:  
            results/oi_fit.npy
            input/mmf.toml or smf.toml
            results/ct_solution-2_gain_0.0.npy

    Choice is fixed to input power -6.0 dBm, and gain 0.0 dB.
    '''
    if smf:
        cf = cfg.load_toml_to_struct("./input/smf.toml")
        select_string = "_SMF"
    else:
        cf = cfg.load_toml_to_struct("./input/mmf.toml")
        select_string = ""
    assert (cf.launch_power == -6.0 and cf.raman_gain == 0.0)
    solutions = np.load("results/ct_solution-6_gain_0.0" +
                        select_string + ".npy", allow_pickle=True).item()
    signal_powers = solutions['signal_sol']
    signal_powers_swp = np.swapaxes(signal_powers, 1, 2)
    initial_powers = signal_powers_swp[0, :, :]
    fB = np.divide(signal_powers_swp, initial_powers)
    assert ((fB >= 0).all())
    fB_max = np.max(fB, axis=(1, 2))
    fB_min = np.min(fB, axis=(1, 2))
    assert ((fB_min <= fB_max).all())
    # assert(len(fB_max) == fB.shape.0)
    z_axis = np.linspace(0, cf.fiber_length, len(fB_max))
    dz = z_axis[1] - z_axis[0]
    assert (fB.shape[0] == len(z_axis))
    rcal_lo_min = (np.sum(fB_min) * dz / cf.fiber_length)**2
    rcal_lo_max = (np.sum(fB_max) * dz / cf.fiber_length)**2
    rcal_hi_min = np.sum(fB_min**2) * dz / cf.fiber_length
    rcal_hi_max = np.sum(fB_max**2) * dz / cf.fiber_length
    rcal_hi = np.sum(fB**2, axis=0) * dz / cf.fiber_length
    rcal_lo = (np.sum(fB, axis=0) * dz / cf.fiber_length)**2
    assert ((rcal_hi >= rcal_hi_min - 1e-15).all())
    assert ((rcal_hi <= rcal_hi_max + 1e-15).all())
    assert ((rcal_lo >= rcal_lo_min - 1e-15).all())
    assert ((rcal_lo <= rcal_lo_max + 1e-15).all())
    # in this notation,
    # "minus/plus" is the DGD limiting case
    # "max/min"    is the signal power profile extrema
    return rcal_lo_min, rcal_lo_max, rcal_hi_min, rcal_hi_max


def get_fit_coefficients(fB_simple_interpolation=False) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Given the numerical noise results for a representative GVD, find the best
    fit coefficients in the Nyquist and Gaussian cases.
    Only consider max and min signal power profiles.
    requires:  results/partial_nlin_gaussian.npy
            results/partial_nlin_nyquist.npy
    '''
    cf = cfg.load_toml_to_struct("./input/mmf.toml")
    nc = cfg.load_nc_toml_to_struct("./input/numerical_config.toml")
    print("\033[91m warn: \033[0m overriding the dgd values of numerical_config.toml")
    # LW = T/dgd = 1/(dgd*baud)
    # L/LW = L*dgd*baud = target
    # dgd = target / (L*baud)
    nc.dgd1 = DGD_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = DGD_MAX / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_g = nc.dgd2_n

    oi_fit = np.load('results/oi_fit.npy')
    beta1_params = load_group_delay()
    fiber = MMFiber(
        effective_area=cf.effective_area,
        overlap_integrals=oi_fit,
        group_delay=beta1_params,
        length=cf.fiber_length,
        n_modes=cf.n_modes
    )
    if fB_simple_interpolation:
        # we will apply the interpolation for the fB in the calling function in this case
        # we just need the perfect amplification case
        modes = ["perfect"]
    else:
        modes = ["min", "max"]
    dgds_numeric_g = np.logspace(
        np.log10(nc.dgd1), np.log10(nc.dgd2_g), nc.n_samples_numeric_g)
    dgds_numeric_n = np.logspace(
        np.log10(nc.dgd1), np.log10(nc.dgd2_n), nc.n_samples_numeric_n)
    T = 1 / cf.baud_rate
    L = fiber.length
    x_norm = L / T
    y_norm = x_norm**(-2)
    p0 = [0.2, 4.5, 0.5]  # initial guess for curve fitting
    ps_g = np.zeros((len(modes), 3))
    ps_n = np.zeros((len(modes), 3))
    # FIXME, this methos only works if the dgds are the same..
    for mode in modes:
        partial_B2g = (np.load("results/partial_nlin_gaussian_" +
                       mode + str(nc.gvd) + "B2.npy"))
        partial_B2n = (np.load("results/partial_nlin_nyquist_" +
                       mode + str(nc.gvd) + "B2.npy"))
        popt_n, _ = curve_fit(softplus2, dgds_numeric_n * x_norm,
                              partial_B2n * y_norm, p0=p0)
        popt_g, _ = curve_fit(softplus2, dgds_numeric_g * x_norm,
                              partial_B2g * y_norm, p0=p0)
        ps_g[modes.index(mode), :] = popt_g
        ps_n[modes.index(mode), :] = popt_n
    return ps_g, ps_n


def adjust_luminosity(color, factor):
    rgb = np.array(mcolors.to_rgb(color))  # Convert to RGB
    return np.clip(rgb * factor, 0, 1)  # Scale and clip values


def get_space_integrals(m, z, I):
    X0mm = np.zeros_like(m)
    X0mm = X0mm_space_integral(z, I, amplification_function=None)
    return X0mm


def get_nlin_threshold(
        recompute=False,
        use_fB=False,
        fB_simple_interpolation=False):
    rc('text', usetex=True)
    cf = cfg.load_toml_to_struct("./input/mmf.toml")
    nc = cfg.load_nc_toml_to_struct("./input/numerical_config.toml")
    print("\033[91m warn: \033[0m overriding the dgd values of numerical_config.toml")
    # LW = T/dgd = 1/(dgd*baud)
    # L/LW = L*dgd*baud = target
    # dgd = target / (L*baud)
    nc.dgd1 = DGD_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = DGD_MAX / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_g = nc.dgd2_n

    # remark on the dispersion
    print(f"Within L/LD = {cf.fiber_length * np.abs(nc.gvd) * cf.baud_rate**2:.1f}")

    oi_fit = np.load('results/oi_fit.npy')
    beta1_params = load_group_delay()
    fiber = MMFiber(
        effective_area=cf.effective_area,
        overlap_integrals=oi_fit,
        group_delay=beta1_params,
        length=cf.fiber_length,
        n_modes=cf.n_modes
    )
    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )
    # assert(cf.fiber_length == 70e3)
    # assert(cf.baud_rate == 35e9)
    # assert(cf.n_channels == 250)
    # assert(cf.channel_spacing == 40e9)
    # assert(cf.center_frequency == 195.94e12)
    freqs = wdm.frequency_grid()
    mode_idx = [0, 1, 2, 3]
    mode_names = ['LP01', 'LP11', 'LP21', 'LP02']
    #
    beta1 = np.zeros((len(mode_idx), len(freqs)))
    for i in mode_idx:
        beta1[i, :] = fiber.group_delay.evaluate_beta1(i, freqs)
    beta2 = np.zeros((len(mode_idx), len(freqs)))
    for i in mode_idx:
        beta2[i, :] = fiber.group_delay.evaluate_beta2(i, freqs)
    beta1 = np.array(beta1)
    beta2 = np.array(beta2)
    #
    # dgd1 = nc.dgd1
    # dgd2_g = nc.dgd2_g
    # dgd2_n = nc.dgd2_n
    # n_samples_numeric_g = nc.n_samples_numeric_g
    # n_samples_numeric_n = nc.n_samples_numeric_n
    px = 0
    gvds = [nc.gvd]
    #

    print(
        f"Computing the channel-pair NLIN coefficient insides [{nc.dgd1 * 1e12:.1e}, {nc.dgd2_g * 1e12:.1e}] ps/m ")
    #
    if use_fB:
        rcal_lo_min, rcal_lo_max, rcal_hi_min, rcal_hi_max = get_raman_corrections()
        modes = ["min", "max"]
        assert (cf.launch_power == -6.0 and cf.raman_gain == 0.0)
        solutions = np.load("results/ct_solution-6_gain_0.0.npy",
                            allow_pickle=True).item()
        signal_powers = solutions['signal_sol']
        signal_powers = np.swapaxes(signal_powers, 1, 2)

        fB_max = np.max(signal_powers, axis=(1, 2))
        fB_min = np.min(signal_powers, axis=(1, 2))
        fB_max /= fB_max[0]
        fB_min /= fB_min[0]

        z_axis = np.linspace(0, fiber.length, len(fB_max))
        coeffs_max = np.polyfit(z_axis, fB_max, 6)
        coeffs_min = np.polyfit(z_axis, fB_min, 6)

        def fB_max_function(z):
            return np.polyval(coeffs_max, z)

        def fB_min_function(z):
            return np.polyval(coeffs_min, z)
    else:
        modes = ["perfect"]

        def fB_max_function(z):
            return 1.0

        def fB_min_function(z):
            return 1.0

    def get_space_integrals_max(m, z, I):
        X0mm = X0mm_space_integral(z, I, amplification_function=fB_max_function)
        return X0mm

    def get_space_integrals_min(m, z, I):
        X0mm = X0mm_space_integral(z, I, amplification_function=fB_min_function)
        return X0mm

    def antonio_rescale_max(dgd):
        acc = 0.0
        for m in get_m_values(fiber, wdm, a_chan, b_chan, T, 0, dgd):
            acc += fB_max_function(get_collision_location(m,
                                   fiber, wdm, a_chan, b_chan, pulse, dgd))**2
        return acc

    def antonio_rescale_min(dgd):
        acc = 0.0
        for m in get_m_values(fiber, wdm, a_chan, b_chan, T, 0, dgd):
            acc += fB_min_function(get_collision_location(m,
                                   fiber, wdm, a_chan, b_chan, pulse, dgd))**2
        return acc
    # ----------------------------------
    # Computation of the NLIN coefficient 
    # ----------------------------------
    for gvd in gvds:
        for px in [0, 1]:
            if px == 0:
                pulse = GaussianPulse(
                    baud_rate=cf.baud_rate,
                    num_symbols=1e2,
                    samples_per_symbol=2**5,
                )
            else:
                pulse = NyquistPulse(
                    baud_rate=cf.baud_rate,
                    num_symbols=1e3,
                    samples_per_symbol=2**5,
                    rolloff=0.0,
                )

            n_samples_analytic = 500
            if px == 0:
                dgd2 = nc.dgd2_g
                n_samples_numeric = nc.n_samples_numeric_g
            else:
                dgd2 = nc.dgd2_n
                n_samples_numeric = nc.n_samples_numeric_n
            # 6e-9 for our fiber
            dgds_numeric = np.logspace(
                np.log10(nc.dgd1), np.log10(dgd2), n_samples_numeric)
            dgds_analytic = np.linspace(nc.dgd1, dgd2, n_samples_analytic)
            if px == 1:
                # print(dgds_numeric)
                pass
                # exit()
            partial_nlin = np.zeros(n_samples_numeric)
            partial_nlin_min = np.zeros(n_samples_numeric)
            partial_nlin_max = np.zeros(n_samples_numeric)
            a_chan = (1, 1)
            b_chan = (1, 2)
            if recompute:
                for id, dgd in enumerate(dgds_numeric):
                    z, I, m = compute_all_collisions_time_integrals(
                        a_chan, b_chan, fiber, wdm, pulse, dgd, gvd)
                    # space integrals
                    X0mm_min = get_space_integrals_min(m, z, I)
                    X0mm_max = get_space_integrals_max(m, z, I)
                    X0mm = get_space_integrals(m, z, I)

                    for xx in [X0mm, X0mm_max, X0mm_min]:
                        assert (np.all(np.imag(xx[np.real(xx) != 0])
                                < 1e-6 * np.real(xx[np.real(xx) != 0])))

                    partial_nlin[id] = np.sum(np.real(X0mm)**2)
                    partial_nlin_min[id] = np.sum(np.real(X0mm_min)**2)
                    partial_nlin_max[id] = np.sum(np.real(X0mm_max)**2)

                if px == 0:
                    np.save("results/partial_nlin_gaussian_perfect" +
                            str(gvd) + "B2.npy", partial_nlin)
                    np.save("results/partial_nlin_gaussian_min" +
                            str(gvd) + "B2.npy", partial_nlin_min)
                    np.save("results/partial_nlin_gaussian_max" +
                            str(gvd) + "B2.npy", partial_nlin_max)
                else:
                    np.save("results/partial_nlin_nyquist_perfect" +
                            str(gvd) + "B2.npy", partial_nlin)
                    np.save("results/partial_nlin_nyquist_min" +
                            str(gvd) + "B2.npy", partial_nlin_min)
                    np.save("results/partial_nlin_nyquist_max" +
                            str(gvd) + "B2.npy", partial_nlin_max)

    T = 1 / cf.baud_rate
    L = fiber.length
    assert (T == pulse.T0)
    LD_eff = pulse.T0**2 / np.abs(gvd)
    dgds_analytic = np.linspace(nc.dgd1, nc.dgd2_g, n_samples_analytic)
    analytic_nlin = L / (T * dgds_analytic)
    nlin_analytic_max = np.zeros_like(dgds_analytic)
    nlin_analytic_min = np.zeros_like(dgds_analytic)
    for ix, i in enumerate(dgds_analytic):
        nlin_analytic_max[ix] = antonio_rescale_max(i) / (i**2)
        nlin_analytic_min[ix] = antonio_rescale_min(i) / (i**2)

    analytic_nlin[analytic_nlin > 1e100] = np.nan
    dgds_numeric_g = np.logspace(
        np.log10(nc.dgd1), np.log10(nc.dgd2_g), nc.n_samples_numeric_g)
    dgds_numeric_n = np.logspace(
        np.log10(nc.dgd1), np.log10(nc.dgd2_n), nc.n_samples_numeric_n)

    x_norm = L / T
    y_norm = x_norm**(-2)

    dpi = 300
    grid = False
    
    # ----------------------------------
    # plotting the threshold
    # ----------------------------------
    plt.figure(figsize=(3.6, 3))
    color_modes = [adjust_luminosity('magenta', 0.8),
                   adjust_luminosity('cyan', 0.8), 'green']
    ps_g, ps_n = get_fit_coefficients(fB_simple_interpolation=fB_simple_interpolation)
    for im, mode in enumerate(modes):
        na_nlin = L / (T * dgds_numeric_g)
        if mode == "max":
            na_nlin = na_nlin * rcal_hi_max
        elif mode == "min":
            na_nlin = na_nlin * rcal_hi_min

        assert(nc.gvd == 0)
        gauss = np.ones_like(dgds_analytic) * np.sqrt(np.pi) * (LD_eff / (T * np.sqrt(2 * np.pi))
                                                                * np.arcsinh(L / LD_eff))**2
        gauss = np.ones_like(dgds_analytic) * (L / T)**2 / (2*np.sqrt(np.pi))
        nyquist = np.ones_like(dgds_analytic) * 4 / 9 / \
            y_norm  # 0.444=4/9 instead of 0.406
        if use_fB:
            rcal_hi = rcal_hi_max if mode == "max" else rcal_hi_min
            rcal_lo = rcal_lo_max if mode == "max" else rcal_lo_min
            gauss *= rcal_lo
            nyquist *= rcal_lo

        plt.plot(dgds_numeric_g * x_norm,
                 na_nlin * y_norm,
                 lw=1,
                 color=adjust_luminosity('orange', 0.9))
        if use_fB:# TODO redundant
            plt.plot(dgds_analytic * x_norm,
                    gauss * y_norm,
                    color=color_modes[im],
                    lw=1,
                    ls=":",
                    label=r'$N^>$')
            plt.plot(dgds_analytic * x_norm,
                    nyquist * y_norm,
                    color=color_modes[im],
                    ls="--",
                    lw=1,
                    label='Marco')
        else:
            plt.plot(dgds_analytic * x_norm,
                    gauss * y_norm,
                    color="blue",
                    lw=1,
                    ls=":",
                    label=r'$N^>$')
            plt.plot(dgds_analytic * x_norm,
                    nyquist * y_norm,
                    color="green",
                    ls="--",
                    lw=1,
                    label='Marco')
            
        lowest_dgd = 0.0
        lw = 1
        ss = 20

        for ix, gvd in enumerate(gvds):
            partial_B2g = (np.load("results/partial_nlin_gaussian_" +
                           mode + str(gvd) + "B2.npy"))
            partial_B2n = (np.load("results/partial_nlin_nyquist_" +
                           mode + str(gvd) + "B2.npy"))
            # p0 = [0.2, 4.5, 0.5]
            # popt_n, pcov = curve_fit(softplus2, dgds_numeric_n * x_norm,
            #                         partial_B2n * y_norm, p0=p0)
            # # p0 = [partial_B2g[0], 0.4, 0.5]
            # popt_g, pcov = curve_fit(softplus2, dgds_numeric_g * x_norm,
            #                         partial_B2g * y_norm, p0=p0)
            if fB_simple_interpolation and use_fB:
                d_lo = dgds_analytic[0]
                d_hi = dgds_analytic[-1]
                dgd_span = d_hi - d_lo
                fitted_data_g = softplus2(dgds_analytic * x_norm, * ps_g[0, :]) * \
                    ((dgds_analytic - d_lo) * rcal_hi -
                     (dgds_analytic - d_hi) * rcal_lo) / dgd_span
                fitted_data_n = softplus2(dgds_analytic * x_norm, *ps_n[0, :]) * \
                    ((dgds_analytic - d_lo) * rcal_hi -
                     (dgds_analytic - d_hi) * rcal_lo) / dgd_span
                fitted_data_g_flat = softplus2(dgds_analytic * x_norm, *ps_g[0, :])
                fitted_data_n_flat = softplus2(dgds_analytic * x_norm, *ps_n[0, :])

            else:
                fitted_data_g = softplus2(dgds_analytic * x_norm, *ps_g[im, :])
                fitted_data_n = softplus2(dgds_analytic * x_norm, *ps_n[im, :])
            if ix == 0:
                lowest_dgd = partial_B2g[0]
            plt.plot(dgds_analytic * x_norm,
                     fitted_data_g,
                     color="gray",
                     lw=0.6,
                     #  ls="-."
                     )
            plt.plot(dgds_analytic * x_norm,
                     fitted_data_n,
                     color="gray",
                     lw=0.6,
                     ls="-."
                     )
            # if fB_simple_interpolation and use_fB:
            #     plt.plot(dgds_analytic * x_norm,
            #              fitted_data_g_flat,
            #              color=adjust_luminosity('blue', 0.5),
            #              lw=0.6,
            #              ls=":"
            #              )
            #     plt.plot(dgds_analytic * x_norm,
            #              fitted_data_n_flat,
            #              color=adjust_luminosity('green', 0.5),
            #              lw=0.6,
            #              ls=":"
            #              )
            if use_fB: # TODO redundant lines....
                plt.scatter(dgds_numeric_g * x_norm,
                            partial_B2g * y_norm,
                            label='Gauss.' + str(gvd),
                            color=color_modes[im],
                            marker="x",
                            s=ss,
                            lw=lw
                            )
                plt.scatter(dgds_numeric_n * x_norm,
                            partial_B2n * y_norm,
                            label='Nyq.' + str(gvd),
                            color=color_modes[im],
                            marker="*",
                            s=ss,
                            lw=lw
                            )
            else:
                plt.scatter(dgds_numeric_g * x_norm,
                            partial_B2g * y_norm,
                            label='Gauss.' + str(gvd),
                            color="blue",
                            marker="x",
                            s=ss,
                            lw=lw
                            )
                plt.scatter(dgds_numeric_n * x_norm,
                            partial_B2n * y_norm,
                            label='Nyq.' + str(gvd),
                            color="green",
                            marker="*",
                            s=ss,
                            lw=lw
                            )


        # print(            # f"DGD low. num = {lowest_dgd:.3e}, ra < = {(L / (T * np.sqrt(2 * np.pi)))**2:.3e}")
        # plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, 1.0)
        if use_fB:
            plt.ylim([0.5e-3, 0.5])
            plt.ylim([0.5e-3, 0.11])
        else:
            plt.ylim([0.7e-2, 1])
        plt.xlabel(r'$L/L_W$')
        plt.ylabel(r'$\mathcal{N} \, T^2 / L^2$')
        plt.tight_layout()
        if use_fB:
            plt.savefig(f"media/2-threshold_raman.pdf", dpi=dpi)
            print(
                f"Saving the figure in media/2-threshold_raman.pdf, dpi={dpi}")
        else:
            plt.savefig(f"media/2-threshold.pdf", dpi=dpi)
            print(
                f"Saving the figure in media/2-threshold.pdf, dpi={dpi}")

        print("\nPlotting the ", mode, " case")
        print("GAUSSIAN:")
        print("fitted_data_g      :", fitted_data_g[0])
        if fB_simple_interpolation and use_fB:
            print("fitted_data_g_flat :", fitted_data_g_flat[0], " -> Raman correction ",
                  rcal_lo, " applied -> ", fitted_data_g_flat[0] * rcal_lo / 2)
        print("partial_B2g*y_norm :", (partial_B2g * y_norm)[0])

        print("\nNYQUIST:")
        print("fitted_data_n      :", fitted_data_n[0])
        if fB_simple_interpolation and use_fB:
            print("fitted_data_n_flat :", fitted_data_n_flat[0], " -> Raman correction ",
                  rcal_lo, " applied -> ", fitted_data_n_flat[0] * rcal_lo / 2)
        print("partial_B2n*y_norm :", (partial_B2n * y_norm)[0])
        
        # ----------------------------------
        # plotting the error
        # ----------------------------------
        if not use_fB:
            plt.clf()
            plt.figure(figsize=(3.6, 2))
            lowest_dgd = 0.0
            lw = 1
            ss = 5
            skip_g = n_samples_analytic // nc.n_samples_numeric_g
            skip_n = n_samples_analytic // nc.n_samples_numeric_n

            interp_g = interp1d(dgds_analytic, gauss, kind='cubic',
                                bounds_error=False, fill_value=(gauss[0], gauss[-1]))
            interp_n = interp1d(dgds_analytic, nyquist, kind='cubic',
                                bounds_error=False, fill_value=(nyquist[0], nyquist[-1]))
            interp_a = interp1d(dgds_analytic, analytic_nlin, kind='linear',
                                bounds_error=False, fill_value=(analytic_nlin[0], analytic_nlin[-1]))

            gauss_sampled = interp_g(dgds_numeric_g)
            nyquist_sampled = interp_n(dgds_numeric_n)
            analytic_nlin_sampled = interp_a(dgds_numeric_g)
            
            for ix, gvd in enumerate(gvds):
                partial_B2g = (
                    np.load("results/partial_nlin_gaussian_perfect" + str(gvd) + "B2.npy"))
                if ix == 0:
                    lowest_dgd = partial_B2g[0]
                plt.plot(dgds_numeric_g * x_norm,
                         np.abs(partial_B2g - gauss_sampled) / gauss_sampled,
                         label='Gauss.' + str(gvd),
                         color="blue",
                         marker="x",
                         markersize=ss,
                         lw=lw
                         )
                plt.plot(dgds_numeric_g * x_norm,
                         np.abs(partial_B2g - analytic_nlin_sampled) /
                         analytic_nlin_sampled,
                         label='Gauss.' + str(gvd),
                         color="blue",
                         marker="x",
                         markersize=ss,
                         lw=lw,
                         ls="-."
                         )
                # print(partial_B2g[:10] * y_norm)
                # print(softplus2(dgds_numeric_g * x_norm, *ps_g[0, :])[:10])
                plt.plot(dgds_numeric_n * x_norm,
                         np.abs(partial_B2g * y_norm - softplus2(dgds_numeric_g * x_norm, *ps_g[0, :])) /
                         softplus2(dgds_numeric_g * x_norm, *ps_g[0, :]),
                         color="gray",
                         ls=":",
                         lw=lw,
                         marker="x", markerfacecolor='none', markersize=ss
                         )
                partial_B2n = (
                    np.load("results/partial_nlin_nyquist_perfect" + str(gvd) + "B2.npy"))
                plt.plot(dgds_numeric_n * x_norm,
                         np.abs(partial_B2n - nyquist_sampled) / nyquist_sampled,
                         label='Nyq.' + str(gvd),
                         color="green",
                         marker="*",
                         markersize=ss,
                         lw=lw
                         )
                plt.plot(dgds_numeric_n * x_norm,
                         np.abs(partial_B2n - analytic_nlin_sampled) /
                         analytic_nlin_sampled,
                         label='Nyq.' + str(gvd),
                         color="green",
                         marker="*",
                         markersize=ss,
                         lw=lw,
                         ls="-.",
                         )
                plt.plot(dgds_numeric_n * x_norm,
                         np.abs(partial_B2n * y_norm - softplus2(dgds_numeric_n * x_norm, *ps_n[0, :])) /
                         softplus2(dgds_numeric_n * x_norm, *ps_n[0, :]),
                         color="gray",
                         lw=lw,
                         ls=":",
                         marker="*", markerfacecolor='none', markersize=ss)
            # print(                f"DGD low. num = {lowest_dgd:.3e}, ra < = {(L / (T * np.sqrt(2 * np.pi)))**2:.3e}")
            plt.xscale('log')
            plt.ylim([-0.05, 0.3])
            plt.xlabel(r'$L/L_W$')
            plt.ylabel(r'$\varepsilon$')
            plt.tight_layout()
            plt.savefig(f"media/2-error.pdf", dpi=dpi)
            print(f"Saving the figure in media/2-error.pdf, dpi={dpi}")


if __name__ == "__main__":
    # plot the case-study figure
    get_nlin_threshold(recompute=True,
                       use_fB=True,
                       fB_simple_interpolation=True)
    exit()
    # plot the theoretical figure
    get_nlin_threshold(recompute=False,
                       use_fB=False)

    print(get_raman_corrections())
    get_fit_coefficients()
