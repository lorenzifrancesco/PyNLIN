import os
import sys
import time
import warnings
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pynlin.wdm
from scripts.modules.load_fiber_values import load_group_delay, load_rms_gvd
from pynlin.fiber import *
from pynlin.pulses import *
from pynlin.nlin import compute_all_collisions_time_integrals, X0mm_space_integral
from matplotlib.gridspec import GridSpec
import scripts.modules.cfg as cfg
from scipy.interpolate import interp1d
from pynlin.collisions import get_m_values, get_collision_location
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Any

from scripts.modules.nlin_estimator import LLW_MAX, LLW_MIN

from loguru import logger as lg
from scripts.modules.log_init import init_logging
init_logging()

from scripts.modules.nlin_estimator import get_raman_corrections, get_fit_coefficients, softplus2

# ==========================
# Core functions
# ==========================
def adjust_luminosity(color, factor):
    rgb = np.array(mcolors.to_rgb(color))  # Convert to RGB
    return np.clip(rgb * factor, 0, 1)  # Scale and clip values



def get_nlin_threshold(
    recompute: bool = False,
    use_fB: bool = False,
    fB_simple_interpolation: bool = False,
):
    if not use_fB:
        fB_simple_interpolation = True
        lg.trace("Disabling fB_simple_interpolation since use_fB is False")

    rc('text', usetex=True)

    cf_path = "./input/mmf.toml"
    nc_path = "./input/numerical_config.toml"
    cf = cfg.load_toml_to_struct(cf_path)
    nc = cfg.load_nc_toml_to_struct(nc_path)

    nc.dgd1 = LLW_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = LLW_MAX / (cf.fiber_length * cf.baud_rate)
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

    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )

    freqs = wdm.frequency_grid()
    mode_idx = [0, 1, 2, 3]
    mode_names = ['LP01', 'LP11', 'LP21', 'LP02']

    beta1 = np.zeros((len(mode_idx), len(freqs)))
    for i in mode_idx:
        beta1[i, :] = fiber.group_delay.evaluate_beta1(i, freqs)
    beta2 = np.zeros((len(mode_idx), len(freqs)))
    for i in mode_idx:
        beta2[i, :] = fiber.group_delay.evaluate_beta2(i, freqs)
    px = 0
    # here we iterate over all the possible gvds
    # gvds = [nc.gvd]
    gvds = np.unique(load_rms_gvd().flatten())
    gvds = [0.0, 30e-27]

    if use_fB:
        rcal_lo_min, rcal_lo_max, rcal_hi_min, rcal_hi_max = get_raman_corrections()
        modes = ["min", "max"]
        assert (cf.launch_power == -5.0 and cf.raman_gain == 0.0)
        sol_path = "results/ct_solution-6_gain_0.0.npy"
        solutions = np.load(sol_path, allow_pickle=True).item()

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
        return X0mm_space_integral(z, I, amplification_function=fB_max_function)

    def get_space_integrals_min(m, z, I):
        return X0mm_space_integral(z, I, amplification_function=fB_min_function)
    
    def get_space_integrals(m, z, I):
        return X0mm_space_integral(z, I, amplification_function=None)

    def antonio_rescale_max(dgd):
        acc = 0.0
        for m in get_m_values(fiber, pulse, 0, dgd):
            acc += fB_max_function(get_collision_location(m, pulse, dgd))**2
        return acc

    def antonio_rescale_min(dgd):
        acc = 0.0
        for m in get_m_values(fiber, pulse, 0, dgd):
            acc += fB_min_function(get_collision_location(m, pulse, dgd))**2
        return acc

    # ----------------------------------
    # Computation of the NLIN coefficient
    # ----------------------------------
    for gvd in gvds:
        for px in [0, 1]:
            if px == 0:
                pulse = GaussianPulse(
                    baud_rate=cf.baud_rate, num_symbols=1e2, samples_per_symbol=2**5)
            else:
                pulse = NyquistPulse(
                    baud_rate=cf.baud_rate, num_symbols=1e3, samples_per_symbol=2**5, rolloff=0.0)

            n_samples_analytic = 500
            if px == 0:
                dgd2 = nc.dgd2_g
                n_samples_numeric = nc.n_samples_numeric_g
            else:
                dgd2 = nc.dgd2_n
                n_samples_numeric = nc.n_samples_numeric_n

            dgds_numeric = np.logspace(
                np.log10(nc.dgd1), np.log10(dgd2), n_samples_numeric)
            dgds_analytic = np.linspace(nc.dgd1, dgd2, n_samples_analytic)

            partial_nlin = np.zeros(n_samples_numeric)
            partial_nlin_min = np.zeros(n_samples_numeric)
            partial_nlin_max = np.zeros(n_samples_numeric)

            gvda = gvd
            gvdb = gvd

            if recompute:
                for idx, dgd in enumerate(dgds_numeric):
                    z, I, m = compute_all_collisions_time_integrals(
                        fiber, pulse, dgd, gvda, gvdb, use_multiprocessing=True)
                    X0mm_min = get_space_integrals_min(m, z, I)
                    X0mm_max = get_space_integrals_max(m, z, I)
                    X0mm = get_space_integrals(m, z, I)

                    for xx in [X0mm, X0mm_max, X0mm_min]:
                        nonzero = np.real(xx) != 0
                        # lg.warning(
                        #     f"Skipping an assert {np.all(np.imag(xx[nonzero]) < 1e-6 * np.real(xx[nonzero]))}")
                        assert np.all(np.imag(xx[nonzero]) < 1e-6 * np.real(
                            xx[nonzero])), f"Imaginary part too large: {np.max(np.imag(xx[nonzero]) / np.real(xx[nonzero]))}"

                    partial_nlin[idx] = np.sum(np.real(X0mm)**2)
                    partial_nlin_min[idx] = np.sum(np.real(X0mm_min)**2)
                    partial_nlin_max[idx] = np.sum(np.real(X0mm_max)**2)

                if px == 0:
                    np.save(
                        f"results/partial_nlin_gaussian_perfect{gvd}B2.npy", partial_nlin)
                    np.save(
                        f"results/partial_nlin_gaussian_min{gvd}B2.npy", partial_nlin_min)
                    np.save(
                        f"results/partial_nlin_gaussian_max{gvd}B2.npy", partial_nlin_max)
                else:
                    np.save(
                        f"results/partial_nlin_nyquist_perfect{gvd}B2.npy", partial_nlin)
                    np.save(
                        f"results/partial_nlin_nyquist_min{gvd}B2.npy", partial_nlin_min)
                    np.save(
                        f"results/partial_nlin_nyquist_max{gvd}B2.npy", partial_nlin_max)

    T = 1 / cf.baud_rate
    L = fiber.length
    assert (T == pulse.T0)
    # LD_eff = pulse.T0**2 / np.abs(gvd)
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

    # ----------------------------------
    # plotting the threshold
    # ----------------------------------
    plt.figure(figsize=(3.6, 3))
    color_modes = [adjust_luminosity(
        'magenta', 0.8), adjust_luminosity('cyan', 0.8), 'green']

    ps_g, ps_n = get_fit_coefficients(
        fB_simple_interpolation=fB_simple_interpolation, gvd=0.0)
    # FINALIZING FIXME
    lg.trace("Fit coefficients (gauss):", ps_g)
    lg.trace("Fit coefficients (nyquist):", ps_n)
    # exit()
    for im, mode in enumerate(modes):
        na_nlin = L / (T * dgds_numeric_g)
        if mode == "max":
            na_nlin = na_nlin * rcal_hi_max
        elif mode == "min":
            na_nlin = na_nlin * rcal_hi_min

        # gauss = np.ones_like(dgds_analytic) * np.sqrt(np.pi) * (LD_eff / (T * np.sqrt(2 * np.pi)) * np.arcsinh(L / LD_eff))**2
        gauss = np.ones_like(dgds_analytic) * (L / T)**2 / np.sqrt(2 * np.pi)
        nyquist = np.ones_like(dgds_analytic) * 4 / 9 / y_norm
        if use_fB:
            rcal_hi = rcal_hi_max if mode == "max" else rcal_hi_min
            rcal_lo = rcal_lo_max if mode == "max" else rcal_lo_min
            gauss *= rcal_lo
            nyquist *= rcal_lo

        plt.plot(dgds_numeric_g * x_norm, na_nlin * y_norm,
                 lw=1, color=adjust_luminosity('orange', 0.9))
        if use_fB:
            # plt.plot(dgds_analytic * x_norm, gauss * y_norm, color=color_modes[im], lw=1, ls=":", label=r'$N^>$')
            # plt.plot(dgds_analytic * x_norm, nyquist * y_norm, color=color_modes[im], ls="--", lw=1, label='Marco')
            pass
        else:
            plt.plot(dgds_analytic * x_norm, gauss * y_norm,
                     color="blue", lw=1, ls=":", label=r'$N^>$')
            plt.plot(dgds_analytic * x_norm, nyquist * y_norm,
                     color="green", ls="--", lw=1, label='Marco')
        lowest_dgd = 0.0
        lw = 1
        ss = 20

        for ix, gvd in enumerate(gvds):
            partial_B2g = np.load(
                f"results/partial_nlin_gaussian_{mode}{gvd}B2.npy")
            partial_B2n = np.load(
                f"results/partial_nlin_nyquist_{mode}{gvd}B2.npy")

            if fB_simple_interpolation and use_fB:
                d_lo = dgds_analytic[0]
                d_hi = dgds_analytic[-1]
                dgd_span = d_hi - d_lo
                fitted_data_g = softplus2(dgds_analytic * x_norm, *ps_g[0, :]) * (
                    (dgds_analytic - d_lo) * rcal_hi - (dgds_analytic - d_hi) * rcal_lo) / dgd_span
                fitted_data_n = softplus2(dgds_analytic * x_norm, *ps_n[0, :]) * (
                    (dgds_analytic - d_lo) * rcal_hi - (dgds_analytic - d_hi) * rcal_lo) / dgd_span
                # fitted_data_g_flat = softplus2(
                #     dgds_analytic * x_norm, *ps_g[0, :])
                # fitted_data_n_flat = softplus2(
                #     dgds_analytic * x_norm, *ps_n[0, :])
            else:
                # here we have two fitting choices: fit max and min, or fit only min and just substitute the max shifting the min (JLT).
                fitting_method = "shift_min_to_max"  # "fit_both" or "shift_min_to_max"
                if fitting_method == "shift_min_to_max":
                    modified_ps_g = ps_g[0, :].copy()
                    modified_ps_g[0] *= (ps_g[im, 0] / ps_g[0, 0])
                    modified_ps_n = ps_n[0, :].copy()
                    modified_ps_n[0] *= (ps_n[im, 0] / ps_n[0, 0])
                    fitted_data_g = softplus2(
                        dgds_analytic * x_norm, *modified_ps_g)
                    fitted_data_n = softplus2(
                        dgds_analytic * x_norm, *modified_ps_n)
                else:
                    fitted_data_g = softplus2(
                        dgds_analytic * x_norm, *ps_g[im, :])
                    fitted_data_n = softplus2(
                        dgds_analytic * x_norm, *ps_n[im, :])

            if ix == 0:
                lowest_dgd = partial_B2g[0]
            plt.plot(dgds_analytic * x_norm,
                     fitted_data_g, color="gray", lw=0.6)
            plt.plot(dgds_analytic * x_norm, fitted_data_n,
                     color="gray", lw=0.6, ls="-.")

            if use_fB:
                plt.scatter(dgds_numeric_g * x_norm, partial_B2g * y_norm, label='Gauss.' +
                            str(gvd), color=color_modes[im], marker="x", s=ss, lw=lw)
                plt.scatter(dgds_numeric_n * x_norm, partial_B2n * y_norm, label='Nyq.' +
                            str(gvd), color=color_modes[im], marker="*", s=ss, lw=lw)
            else:
                plt.scatter(dgds_numeric_g * x_norm, partial_B2g * y_norm,
                            label='Gauss.' + str(gvd), color="blue", marker="x", s=ss, lw=lw)
                plt.scatter(dgds_numeric_n * x_norm, partial_B2n * y_norm,
                            label='Nyq.' + str(gvd), color="green", marker="*", s=ss, lw=lw)

        plt.xscale('log')
        plt.yscale('log')
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, 1.0)
        if use_fB:
            plt.ylim([0.5e-3, 0.11])
        else:
            plt.ylim([0.7e-2, 1])
        plt.xlabel(r'$L/L_W$')
        plt.ylabel(r'$\mathcal{N} \, T^2 / L^2$')
        plt.tight_layout()
        out_pdf = "media/2-threshold_raman.pdf" if use_fB else "media/2-threshold.pdf"
        plt.savefig(out_pdf, dpi=dpi)
        lg.info(f"Saved figure to {out_pdf}")

        # ----------------------------------
        # plotting the error (only without Raman)
        # ----------------------------------
        if not use_fB:
            plt.clf()
            plt.figure(figsize=(3.6, 2))
            lw = 1
            ss = 5

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
                partial_B2g = np.load(
                    f"results/partial_nlin_gaussian_perfect{gvd}B2.npy")
                if ix == 0:
                    lowest_dgd = partial_B2g[0]
                plt.plot(dgds_numeric_g * x_norm, np.abs(partial_B2g - gauss_sampled) /
                         gauss_sampled, color="blue", marker="x", markersize=ss, lw=lw)
                plt.plot(dgds_numeric_g * x_norm, np.abs(partial_B2g - analytic_nlin_sampled) /
                         analytic_nlin_sampled, color="blue", marker="x", markersize=ss, lw=lw, ls="-.")

                plt.plot(dgds_numeric_n * x_norm, np.abs(partial_B2g * y_norm - softplus2(dgds_numeric_g * x_norm, *ps_g[0, :])) / softplus2(
                    dgds_numeric_g * x_norm, *ps_g[0, :]), color="gray", ls=":", lw=lw, marker="x", markerfacecolor='none', markersize=ss)

                partial_B2n = np.load(
                    f"results/partial_nlin_nyquist_perfect{gvd}B2.npy")
                plt.plot(dgds_numeric_n * x_norm, np.abs(partial_B2n - nyquist_sampled) /
                         nyquist_sampled, color="green", marker="*", markersize=ss, lw=lw)
                plt.plot(dgds_numeric_n * x_norm, np.abs(partial_B2n - analytic_nlin_sampled) /
                         analytic_nlin_sampled, color="green", marker="*", markersize=ss, lw=lw, ls="-.")
                plt.plot(dgds_numeric_n * x_norm, np.abs(partial_B2n * y_norm - softplus2(dgds_numeric_n * x_norm, *ps_n[0, :])) / softplus2(
                    dgds_numeric_n * x_norm, *ps_n[0, :]), color="gray", lw=lw, ls=":", marker="*", markerfacecolor='none', markersize=ss)

            plt.xscale('log')
            plt.ylim([-0.05, 0.3])
            plt.xlabel(r'$L/L_W$')
            plt.ylabel(r'$\\varepsilon$')
            plt.tight_layout()
            err_pdf = "media/2-error.pdf"
            plt.savefig(err_pdf, dpi=dpi)
            lg.info(f"Saved figure to {err_pdf}")


if __name__ == "__main__":
    # plot the theoretical figure
    get_nlin_threshold(recompute=True,
                       use_fB=False,
                       fB_simple_interpolation=False)

    # plot the case-study figure
    # get_nlin_threshold(recompute=True, use_fB=True, fB_simple_interpolation=True)

    # Example utilities (disabled by default):
    # logger.info("Raman corrections: %s", _safe_repr(get_raman_corrections()))
    # logger.info("Fit coefficients shapes: %s", _safe_repr([x.shape for x in get_fit_coefficients()]))
