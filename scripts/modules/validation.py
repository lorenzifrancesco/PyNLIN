from scripts.modules.nlin_estimator import ideal_fit_coefficients, softplus2, fit_nlin, build_lookup_integral_table_with_raman, LLW_MAX, LLW_MIN, ideal_fit_coefficients, softplus2, load_fB
from scripts.modules.load_fiber_values import load_group_delay, load_rms_gvd
from scripts.modules.log_init import init_logging
import matplotlib.colors as mcolors
from pynlin.collisions import get_m_values, get_collision_location
from scipy.interpolate import interp1d
import scripts.modules.cfg as cfg
from pynlin.nlin import compute_all_collisions_time_integrals, X0mm_space_integral
from pynlin.pulses import *
from pynlin.fiber import *
from matplotlib import rc
import matplotlib.pyplot as plt
import os
import numpy as np
from loguru import logger as lg
init_logging()


# ==========================
# Core functions
# ==========================

def adjust_luminosity(color, factor):
    rgb = np.array(mcolors.to_rgb(color))  # Convert to RGB
    return np.clip(rgb * factor, 0, 1)  # Scale and clip values


def compute_numeric_nlin(gvda: float,
                         gvdb: float,
                         ipulse: int,
                         recompute: bool = False,):
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

    if ipulse == 0:
        dgd2 = nc.dgd2_g
        n_samples_numeric = nc.n_samples_numeric_g
    else:
        dgd2 = nc.dgd2_n
        n_samples_numeric = nc.n_samples_numeric_n

    dgds_numeric = np.logspace(
        np.log10(nc.dgd1), np.log10(dgd2), n_samples_numeric)

    partial_nlin = np.zeros(n_samples_numeric)
    partial_nlin_min = np.zeros(n_samples_numeric)
    partial_nlin_max = np.zeros(n_samples_numeric)

    if ipulse == 0:
        pulse = GaussianPulse(
            baud_rate=cf.baud_rate, num_symbols=1e2, samples_per_symbol=2**5)
    else:
        pulse = NyquistPulse(
            baud_rate=cf.baud_rate, num_symbols=1e3, samples_per_symbol=2**5, rolloff=0.0)

    # does the file already exist?
    filename = f"results/partial_nlin_{'gaussian' if ipulse == 0 else 'nyquist'}_perfect_{gvda}_{gvdb}.npy"

    _, _, _, fB_min_func, fB_max_func = load_fB(cf)
    if not os.path.exists(filename) or recompute:
        for idx, dgd in enumerate(dgds_numeric):
            z, I, m = compute_all_collisions_time_integrals(
                fiber, pulse, dgd, gvda, gvdb, 
                use_multiprocessing=True, 
                partial_collisions_margin=5)

            X0mm     = X0mm_space_integral(z, I, amplification_function=lambda x: 1)
            X0mm_max = X0mm_space_integral(z, I, amplification_function=fB_max_func)
            X0mm_min = X0mm_space_integral(z, I, amplification_function=fB_min_func)

            for xx in [X0mm, X0mm_max, X0mm_min]:
                nonzero = np.real(xx) != 0
                assert np.all(np.imag(xx[nonzero]) < 1e-6 * np.real(
                    xx[nonzero])), f"Imaginary part too large: {np.max(np.imag(xx[nonzero]) / np.real(xx[nonzero]))}"

            partial_nlin[idx] = np.sum(np.real(X0mm)**2)
            partial_nlin_min[idx] = np.sum(np.real(X0mm_min)**2)
            partial_nlin_max[idx] = np.sum(np.real(X0mm_max)**2)

        if ipulse == 0:
            np.save(
                f"results/partial_nlin_gaussian_perfect_{gvda}_{gvdb}.npy", partial_nlin)
            np.save(
                f"results/partial_nlin_gaussian_min_{gvda}_{gvdb}.npy", partial_nlin_min)
            np.save(
                f"results/partial_nlin_gaussian_max_{gvda}_{gvdb}.npy", partial_nlin_max)
        else:
            np.save(
                f"results/partial_nlin_nyquist_perfect_{gvda}_{gvdb}.npy", partial_nlin)
            np.save(
                f"results/partial_nlin_nyquist_min_{gvda}_{gvdb}.npy", partial_nlin_min)
            np.save(
                f"results/partial_nlin_nyquist_max_{gvda}_{gvdb}.npy", partial_nlin_max)
        lg.info(f"Saved numeric results to {filename} and similar with _min and _max")
    return


def compute_asymptotic_nlin(ipulse) -> Tuple[np.ndarray, np.ndarray]:
    cf_path = "./input/mmf.toml"
    nc_path = "./input/numerical_config.toml"
    cf = cfg.load_toml_to_struct(cf_path)
    nc = cfg.load_nc_toml_to_struct(nc_path)

    n_samples_analytic = 500
    nc.dgd1 = LLW_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = LLW_MAX / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_g = nc.dgd2_n
    dgd2 = nc.dgd2_n
    dgds_analytic = np.geomspace(nc.dgd1, dgd2, n_samples_analytic)

    L = cf.fiber_length
    T = 1 / cf.baud_rate
    nlin_hi = L/(T * dgds_analytic)
    nlin_lo = np.ones_like(nlin_hi)
    if ipulse == 0:  # Gaussian
        nlin_lo *= (L/T)**2 * 1/(np.sqrt(np.pi) * 2)
    else:  # Nyquist
        nlin_lo *= (L/T)**2 * 4/9

    nlin_hi[nlin_hi > 1e100] = np.nan
    nlin_lo[nlin_lo > 1e100] = np.nan
    return nlin_hi, nlin_lo


# def compute_fitted_nlin(gvda: float,
#                         gvdb: float,
#                         fB_mode: str,
#                         ipulse: int,
#                         recompute: bool = False,):
#     # this should use the methods from the nlin_estimator
#     cf_path = "./input/mmf.toml"
#     nc_path = "./input/numerical_config.toml"
#     cf = cfg.load_toml_to_struct(cf_path)
#     nc = cfg.load_nc_toml_to_struct(nc_path)

#     n_samples_analytic = 500
#     nc.dgd1 = LLW_MIN / (cf.fiber_length * cf.baud_rate)
#     nc.dgd2_n = LLW_MAX / (cf.fiber_length * cf.baud_rate)
#     dgd2 = nc.dgd2_n
#     dgds_analytic = np.geomspace(nc.dgd1, dgd2, n_samples_analytic)

#     x_norm = cf.fiber_length * cf.baud_rate
#     y_norm = x_norm**(-2)

#     # this is the smart fit.
#     ps = ideal_fit_coefficients(gvda=0.0, gvdb=0.0, ipulse=ipulse)
#     lda = 1 / (gvda * cf.baud_rate**2) if gvda != 0 else 1e30
#     ldb = 1 / (gvdb * cf.baud_rate**2) if gvdb != 0 else 1e30
    
#     lo_value = gvd_correction(lda, ldb, cf.fiber_length, ipulse=ipulse)
#     ps = apply_fit_correction(ps, lo_value)
#     fitted_nlin = softplus2(dgds_analytic * x_norm, *ps) / y_norm
#     return fitted_nlin


def simple_plot_threshold(gvda: float = 0.0,
                          gvdb: float = 0.0,
                          fB_mode: str = "perfect",
                          recompute: bool = False,
                          ipulse: int = 1,
                          m_lo_truncation: int = 0):
    cf_path = "./input/mmf.toml"  # FIXME repeated code
    nc_path = "./input/numerical_config.toml"
    cf = cfg.load_toml_to_struct(cf_path)
    nc = cfg.load_nc_toml_to_struct(nc_path)

    x_norm = cf.fiber_length * cf.baud_rate
    y_norm = 1/(cf.fiber_length * cf.baud_rate)**2
    n_samples_analytic = 500
    nc.dgd1   = LLW_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = LLW_MAX / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_g = nc.dgd2_n
    dgd2 = nc.dgd2_n
    dgds_analytic = np.geomspace(nc.dgd1, dgd2, n_samples_analytic)

    if ipulse == 0:
        pulse_shape = "gaussian"
    else:
        pulse_shape = "nyquist"
    
    if ipulse == 0:
        dgd2 = nc.dgd2_g
        n_samples_numeric = nc.n_samples_numeric_g
    else:
        dgd2 = nc.dgd2_n
        n_samples_numeric = nc.n_samples_numeric_n

    dgds_numeric = np.logspace(
        np.log10(nc.dgd1), np.log10(dgd2), n_samples_numeric)
    # assert (fB_mode == "perfect")
    
    _, fB_min, fB_max, _, _ = load_fB(cf)
    if fB_mode == "perfect":
        fB = np.array([1.0]*100)  # dummy, not used since fB_mode is perfect
    elif fB_mode == "max": 
        fB = fB_max
    elif fB_mode == "min":
        fB = fB_min
    else:
        raise ValueError(f"fB_mode {fB_mode} not recognized")    
    
    # ----- call the computation functions -----
    # -- numeric (compute_numeric computes all the fB_modes)
    compute_numeric_nlin(gvda=gvda, gvdb=gvdb, ipulse=ipulse, recompute=recompute)
    nlin_numeric = np.load(
        f"results/partial_nlin_{pulse_shape}_{fB_mode}_{gvda}_{gvdb}.npy")
    lg.info(nlin_numeric)
    lg.info(nlin_numeric * y_norm)
    
    raman_gvd_correction_min, raman_gvd_correction_max = build_lookup_integral_table_with_raman(cf, m_lo_truncation=m_lo_truncation)
    
    # -- analytic and fitted
    nlin_fitted = fit_nlin(cf,
                           gvda,
                           gvdb,
                           fB,
                           raman_gvd_correction_min, 
                           raman_gvd_correction_max,
                           ipulse=ipulse, m_lo_truncation=m_lo_truncation)
    nlin_fitted_data = nlin_fitted(dgds_analytic) / y_norm
    nlin_hi, nlin_lo = compute_asymptotic_nlin(ipulse=ipulse)

    rc('text', usetex=True)
    dpi = 300
    plt.figure(figsize=(3.6, 3))
    # plt.plot(dgds_analytic * x_norm, nlin_lo * y_norm,
    #          color="green", lw=1, ls="--", label='Fit')
    # plt.plot(dgds_analytic * x_norm, nlin_hi * y_norm,
    #          color="green", lw=1, ls="--", label='Fit')
    plt.plot(dgds_analytic * x_norm, nlin_fitted_data * y_norm,
             color="green", lw=1, ls="-", label='Fit')
    plt.scatter(dgds_numeric * x_norm, nlin_numeric * y_norm,
                label='Numeric', color="green", marker="*", s=20, lw=1)
    llda = cf.fiber_length * (gvda * cf.baud_rate**2)
    lldb = cf.fiber_length * (gvdb * cf.baud_rate**2)
    plt.title(fr"$L/L_{{DA}}=${llda:.2f}, $L/L_{{DB}}=${lldb:.2f}")
    plt.xscale('log')
    plt.yscale('log')
    ymin, ymax = plt.ylim()
    # plt.ylim(ymin, 1.0)
    if fB_mode == "perfect":
        pass
        # plt.ylim([0.5e-3, 0.11])
    else:
        pass
        # plt.ylim([0.7e-2, 1])
    plt.xlabel(r'$L/L_W$')
    plt.ylabel(r'$\mathcal{N} \, T^2 / L^2$')
    plt.tight_layout()
    plt.savefig(f"media/simple_threshold_{pulse_shape}_{llda:.2f}_{lldb:.2f}.pdf", dpi=dpi)
    lg.info(f"Saved figure to media/simple_threshold_{pulse_shape}_{llda:.2f}_{lldb:.2f}.pdf")
    return


# FIXME need to apply Raman
def plot_threshold(
    recompute: bool = False,
    use_fB: bool = False,
    fB_simple_interpolation: bool = False,
    gvda: float = 0.0,
    gvdb: float = 0.0,
):
    if not use_fB:
        fB_simple_interpolation = True
        lg.trace("Disabling fB_simple_interpolation since use_fB is False")

    rc('text', usetex=True)
    dpi = 300

    compute_numeric_nlin(gvda=gvda, gvdb=gvdb, ipulse=0, recompute=recompute)
    compute_fitted_nlin(gvda=gvda, gvdb=gvdb, fB_mode="perfect",
                        ipulse=0, recompute=recompute)

    # ----------------------------------
    # plotting the threshold
    # ----------------------------------
    cf_path = "./input/mmf.toml"  # FIXME repeated code
    nc_path = "./input/numerical_config.toml"
    cf = cfg.load_toml_to_struct(cf_path)
    nc = cfg.load_nc_toml_to_struct(nc_path)

    x_norm = cf.fiber_length * cf.baud_rate
    y_norm = 1/(cf.fiber_length * cf.baud_rate)**2
    n_samples_analytic = 500
    nc.dgd1   = LLW_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = LLW_MAX / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_g = nc.dgd2_n
    dgd2 = nc.dgd2_n
    dgds_analytic = np.geomspace(nc.dgd1, dgd2, n_samples_analytic)

    dgds_numeric_n = np.logspace(
        np.log10(nc.dgd1), np.log10(dgd2), nc.n_samples_numeric_n)
    dgds_numeric_g = np.logspace(
        np.log10(nc.dgd1), np.log10(dgd2), nc.n_samples_numeric_g)
    
    x_norm = cf.fiber_length * cf.baud_rate
    y_norm = (cf.fiber_length * cf.baud_rate)**2
    plt.figure(figsize=(3.6, 3))
    color_modes = [adjust_luminosity(
        'magenta', 0.8), adjust_luminosity('cyan', 0.8), 'green']

    ps_g, ps_n = ideal_fit_coefficients(
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
                f"results/partial_nlin_gaussian_{mode}_{gvda}_{gvdb}.npy")
            partial_B2n = np.load(
                f"results/partial_nlin_nyquist_{mode}_{gvda}_{gvdb}.npy")

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
                    f"results/partial_nlin_gaussian_perfect_{gvda}_{gvdb}.npy")
                if ix == 0:
                    lowest_dgd = partial_B2g[0]
                plt.plot(dgds_numeric_g * x_norm, np.abs(partial_B2g - gauss_sampled) /
                         gauss_sampled, color="blue", marker="x", markersize=ss, lw=lw)
                plt.plot(dgds_numeric_g * x_norm, np.abs(partial_B2g - analytic_nlin_sampled) /
                         analytic_nlin_sampled, color="blue", marker="x", markersize=ss, lw=lw, ls="-.")

                plt.plot(dgds_numeric_n * x_norm, np.abs(partial_B2g * y_norm - softplus2(dgds_numeric_g * x_norm, *ps_g[0, :])) / softplus2(
                    dgds_numeric_g * x_norm, *ps_g[0, :]), color="gray", ls=":", lw=lw, marker="x", markerfacecolor='none', markersize=ss)

                partial_B2n = np.load(
                    f"results/partial_nlin_nyquist_perfect_{gvda}_{gvdb}.npy")
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
    simple_plot_threshold(
        gvda = 30.0e-27,
        gvdb = 30.0e-27,
        fB_mode="max",
        recompute=False,
        ipulse=1,
        m_lo_truncation=3)
    exit()

    # plot the theoretical figure
    plot_threshold(recompute=True,
                   use_fB=False,
                   fB_simple_interpolation=False)

    # plot the case-study figure
    # get_nlin_threshold(recompute=True, use_fB=True, fB_simple_interpolation=True)

    # Example utilities (disabled by default):
    # logger.info("Raman corrections: %s", _safe_repr(get_raman_corrections()))
    # logger.info("Fit coefficients shapes: %s", _safe_repr([x.shape for x in get_fit_coefficients()]))