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


def adjust_luminosity(color, factor):
    rgb = np.array(mcolors.to_rgb(color))  # Convert to RGB
    return np.clip(rgb * factor, 0, 1)  # Scale and clip values


def get_space_integrals(m, z, I):
    '''
      Read the time integral file and compute the space integrals 
    '''
    X0mm = np.zeros_like(m)
    X0mm = X0mm_space_integral(z, I, amplification_function=None)
    return X0mm


def get_nlin_threshold(
        recompute=False,
        use_fB=False):
    rc('text', usetex=True)
    cf = cfg.load_toml_to_struct("./input/config_collision.toml")
    oi_fit = np.load('results/oi_fit.npy')

    beta1_params = load_group_delay()
    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )
    freqs = wdm.frequency_grid()
    mode_idx = [0, 1, 2, 3]
    mode_names = ['LP01', 'LP11', 'LP21', 'LP02']
    #
    dummy_fiber = MMFiber(
        effective_area=80e-12,
        overlap_integrals=oi_fit,
        group_delay=beta1_params,
        length=100e3,
        n_modes=4
    )
    #
    beta1 = np.zeros((len(mode_idx), len(freqs)))
    for i in mode_idx:
        beta1[i, :] = dummy_fiber.group_delay.evaluate_beta1(i, freqs)
    beta2 = np.zeros((len(mode_idx), len(freqs)))
    for i in mode_idx:
        beta2[i, :] = dummy_fiber.group_delay.evaluate_beta2(i, freqs)
    beta1 = np.array(beta1)
    beta2 = np.array(beta2)
    #
    dgd1 = 1e-16
    dgd2g = 1e-13
    dgd2n = 1e-13
    n_samples_numeric_g = 10
    n_samples_numeric_n = 10
    px = 0
    gvds = [-35e-27]
    #
    print(
        f"Computing the channel-pair NLIN coefficient insides [{dgd1 * 1e12:.1e}, {dgd2g * 1e12:.1e}] ps/m ")
    #
    if use_fB:
        modes = ["min", "max"]
        solutions = np.load("results/ct_solution-2_gain_0.0.npy",
                            allow_pickle=True).item()
        signal_powers = solutions['signal_sol']
        signal_powers = np.swapaxes(signal_powers, 1, 2)
        print(np.shape(signal_powers))
        fB_max = np.max(signal_powers, axis=(1, 2))
        fB_min = np.min(signal_powers, axis=(1, 2))
        fB_max /= fB_max[0]
        fB_min /= fB_min[0]

        z_axis = np.linspace(0, dummy_fiber.length, len(fB_max))
        dz = z_axis[1] - z_axis[0]
        coeffs_max = np.polyfit(z_axis, fB_max, 6)
        coeffs_min = np.polyfit(z_axis, fB_min, 6)

        fB_integral_max = np.sum(fB_max) * dz / dummy_fiber.length
        fB_integral_min = np.sum(fB_min) * dz / dummy_fiber.length

        fB2_antonio_max = np.sum(fB_max**2) * dz / dummy_fiber.length
        fB2_antonio_min = np.sum(fB_min**2) * dz / dummy_fiber.length

        np.save("results/fits/fB_integral_max.npy", fB_integral_max)
        np.save("results/fits/fB_integral_min.npy", fB_integral_min)
        np.save("results/fits/fB2_antonio_max.npy", fB2_antonio_max)
        np.save("results/fits/fB2_antonio_min.npy", fB2_antonio_min)

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
        for m in get_m_values(dummy_fiber, wdm, a_chan, b_chan, T, 0, dgd):
            acc += fB_max_function(get_collision_location(m,
                                   dummy_fiber, wdm, a_chan, b_chan, pulse, dgd))**2
        return acc

    def antonio_rescale_min(dgd):
        acc = 0.0
        for m in get_m_values(dummy_fiber, wdm, a_chan, b_chan, T, 0, dgd):
            acc += fB_min_function(get_collision_location(m,
                                   dummy_fiber, wdm, a_chan, b_chan, pulse, dgd))**2
        return acc
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
            dgd1 = 1e-17
            if px == 0:
                dgd2 = dgd2g
                n_samples_numeric = n_samples_numeric_g
            else:
                dgd2 = dgd2n
                n_samples_numeric = n_samples_numeric_n
            # 6e-9 for our fiber
            dgds_numeric = np.logspace(
                np.log10(dgd1), np.log10(dgd2), n_samples_numeric)
            dgds_analytic = np.linspace(dgd1, dgd2, n_samples_analytic)

            partial_nlin = np.zeros(n_samples_numeric)
            partial_nlin_min = np.zeros(n_samples_numeric)
            partial_nlin_max = np.zeros(n_samples_numeric)
            a_chan = (1, 1)
            b_chan = (1, 2)
            if recompute:
                for id, dgd in enumerate(dgds_numeric):
                    z, I, m = compute_all_collisions_time_integrals(
                        a_chan, b_chan, dummy_fiber, wdm, pulse, dgd, gvd)
                    # space integrals
                    X0mm_min = get_space_integrals_min(m, z, I)
                    X0mm_max = get_space_integrals_max(m, z, I)
                    X0mm = get_space_integrals(m, z, I)
                    partial_nlin[id] = np.sum(X0mm**2)
                    partial_nlin_min[id] = np.sum(X0mm_min**2)
                    partial_nlin_max[id] = np.sum(X0mm_max**2)

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
    L = dummy_fiber.length
    LD_eff = pulse.T0**2 / np.abs(gvd)
    dgd2 = dgd2g
    dgds_analytic = np.linspace(dgd1, dgd2, n_samples_analytic)
    analytic_nlin = L / (T * dgds_analytic)
    nlin_analytic_max = np.zeros_like(dgds_analytic)
    nlin_analytic_min = np.zeros_like(dgds_analytic)
    for ix, i in enumerate(dgds_analytic):
        nlin_analytic_max[ix] = antonio_rescale_max(i) / (i**2)
        nlin_analytic_min[ix] = antonio_rescale_min(i) / (i**2)

    analytic_nlin[analytic_nlin > 1e30] = np.nan
    dgds_numeric_g = np.logspace(np.log10(dgd1), np.log10(dgd2g), n_samples_numeric_g)
    dgds_numeric_n = np.logspace(np.log10(dgd1), np.log10(dgd2n), n_samples_numeric_n)

    x_norm = L / T
    y_norm = x_norm**(-2)

    dpi = 300
    grid = False

    ### plotting the threshold
    plt.figure(figsize=(3.6, 3))
    color_modes = [adjust_luminosity('magenta', 0.8),
                   adjust_luminosity('cyan', 0.8), 'green']
    for im, mode in enumerate(modes):
        # antonio_function = antonio_rescale_max if mode == "max" else antonio_rescale_min
        # analytic_nlin = np.zeros_like(dgds_numeric_g)
        # for ix, i in enumerate(dgds_numeric_g):
        #   analytic_nlin[ix] = antonio_function(i)/(i**2)

        na_nlin = L / (T * dgds_numeric_g)
        if mode == "max":
            na_nlin = na_nlin * fB2_antonio_max
        elif mode == "min":
            na_nlin = na_nlin * fB2_antonio_min

        gauss = np.ones_like(dgds_analytic) * np.sqrt(np.pi) * (LD_eff / (T * np.sqrt(2 * np.pi))
                                                                * np.arcsinh(L / LD_eff))**2
        nyquist = np.ones_like(dgds_analytic) * 4/9 / \
            y_norm  # 0.444=4/9 instead of 0.406
        if mode == "max":
            gauss *= fB_integral_max**2
            nyquist *= fB_integral_max**2
        elif mode == "min":
            gauss *= fB_integral_min**2
            nyquist *= fB_integral_min**2

        plt.plot(dgds_numeric_g * x_norm,
                 na_nlin * y_norm,
                 lw=1,
                 color=adjust_luminosity('orange', 0.9))
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
        lowest_dgd = 0.0
        lw = 1
        ss = 20

        def softplus2(x, a, b, c):
            return a * (1 + (x/b)**(1 / c))**(-c)


        for ix, gvd in enumerate(gvds):
            partial_B2g = (np.load("results/partial_nlin_gaussian_" +
                           mode + str(gvd) + "B2.npy"))
            partial_B2n = (np.load("results/partial_nlin_nyquist_" +
                           mode + str(gvd) + "B2.npy"))
            p0 = [0.2, 4.5, 0.5]
            poptN, pcov = curve_fit(softplus2, dgds_numeric_n * x_norm,
                               partial_B2n * y_norm, p0=p0)
            # p0 = [partial_B2g[0], 0.4, 0.5]
            poptG, pcov = curve_fit(softplus2, dgds_numeric_g * x_norm,
                               partial_B2g * y_norm, p0=p0)
            
            np.save("results/popt_nyquist_" + mode + ".npy", poptN)
            np.save("results/popt_gaussian_" + mode + ".npy", poptG)
            print("OPT-G", poptG)
            print("OPT-N", poptN)
            if ix == 0:
                lowest_dgd = partial_B2g[0]
            plt.plot(dgds_analytic * x_norm,
                     softplus2(dgds_analytic * x_norm, *poptG),
                     color="gray",
                     lw=0.6,
                    #  ls="-."
                     )
            plt.plot(dgds_analytic * x_norm,
                     softplus2(dgds_analytic * x_norm, *poptN),
                     color="gray",
                     lw=0.6,
                    #  ls="-."
                     )
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

        print(
            f"DGD low. num = {lowest_dgd:.3e}, ra < = {(L / (T * np.sqrt(2 * np.pi)))**2:.3e}")
        # plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, 1.0)
        if use_fB:
            plt.ylim([0.5e-3, 0.1])
        else:
            plt.ylim([0.7e-2, 1])
        plt.xlabel(r'$L_W/L$')
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
            
        ### plotting the error
        if not use_fB:
            plt.clf()
            plt.figure(figsize=(3.6, 2))
            lowest_dgd = 0.0
            lw = 1
            ss = 5
            skip_g = n_samples_analytic // n_samples_numeric_g
            skip_n = n_samples_analytic // n_samples_numeric_n

            interp_g = interp1d(dgds_analytic, gauss, kind='cubic')
            interp_n = interp1d(dgds_analytic, nyquist, kind='cubic')
            interp_a = interp1d(dgds_analytic, analytic_nlin, kind='linear')

            gauss_sampled = interp_g(dgds_numeric_g)
            nyquist_sampled = interp_n(dgds_numeric_n)
            analytic_nlin_sampled = interp_a(dgds_numeric_g)
            for ix, gvd in enumerate(gvds):
                partial_B2g = (
                    np.load("results/partial_nlin_gaussian" + str(gvd) + "B2.npy"))
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
                print(partial_B2g[:10]*y_norm)
                print(softplus2(dgds_numeric_g * x_norm, *poptG)[:10])
                plt.plot(dgds_numeric_n * x_norm,
                         np.abs(partial_B2g*y_norm - softplus2(dgds_numeric_g * x_norm, *poptG)) /
                           softplus2(dgds_numeric_g * x_norm, *poptG),
                           color="gray",
                           ls=":",
                           lw=lw,
                           marker="x", markerfacecolor='none', markersize=ss
                           )
                partial_B2n = (
                    np.load("results/partial_nlin_nyquist" + str(gvd) + "B2.npy"))
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
                         np.abs(partial_B2n*y_norm - softplus2(dgds_numeric_n * x_norm, *poptN)) /
                           softplus2(dgds_numeric_n * x_norm, *poptN),
                           color="gray", 
                           lw=lw,
                           ls=":",
                           marker="*", markerfacecolor='none', markersize=ss)
            # print(y_norm * partial_B2g)
            # print(y_norm * gauss_sampled)
            # print("*" * 30)
            # print(y_norm * partial_B2n)
            # print(y_norm * nyquist_sampled)
            # print("*" * 30)
            # print(y_norm * analytic_nlin_sampled)
            print(
                f"DGD low. num = {lowest_dgd:.3e}, ra < = {(L / (T * np.sqrt(2 * np.pi)))**2:.3e}")
            # plt.legend()
            # plt.yscale('log')
            plt.xscale('log')
            plt.ylim([-0.05, 0.3])
            plt.xlabel(r'$L_W/L$')
            plt.ylabel(r'$\varepsilon$')
            plt.tight_layout()
            plt.savefig(f"media/2-error.pdf", dpi=dpi)
            print(f"Saving the figure in media/2-error.pdf, dpi={dpi}")

def get_fig2(recompute=False):
    return get_nlin_threshold(recompute=recompute, use_fB=False)


def get_fig2_raman(recompute=False):
    return get_nlin_threshold(recompute=recompute, use_fB=True)

if __name__ == "__main__":
    # get_fig2(recompute=False)
    get_fig2_raman(recompute=False)