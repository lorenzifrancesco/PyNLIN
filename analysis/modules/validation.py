from analysis.modules.nlin_estimator import fit_nlin, LLW_MAX, LLW_MIN
from analysis.modules.nlin_estimation.raman_integrals import load_fB, raman_integral
from analysis.modules.nlin_estimation.ideal_fits import ideal_fit_coefficients
from analysis.modules.nlin_estimation.lo_correction import build_lookup_integral_table_with_raman, build_I_low_interpolator, build_lookup_integral_table_with_raman_custom
from analysis.modules.load_fiber_values import load_group_delay, load_rms_gvd
from analysis.modules.log_init import init_logging
import matplotlib.colors as mcolors
from pynlin.collisions import get_m_values, get_collision_location
from scipy.interpolate import interp1d
import analysis.modules.cfg as cfg
from pynlin.nlin import compute_all_collisions_time_integrals, X0mm_space_integral
from pynlin.pulses import *
from pynlin.fiber import *
from matplotlib import rc
import matplotlib.pyplot as plt
import os
import numpy as np
from loguru import logger as lg
init_logging()
from scipy.interpolate import interp1d
from scipy.optimize import brentq


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
                partial_collisions_margin=10)

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
    nlin_fitted, ps_ramanful = fit_nlin(cf,
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
    plt.plot(dgds_analytic * x_norm, nlin_hi * y_norm,
             color="green", lw=1, ls="--", label='Fit')
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
        plt.ylim([0.5e-3, 0.11])
    else:
        plt.ylim([0.7e-3, 0.1])
    plt.xlabel(r'$L/L_W$')
    plt.ylabel(r'$\mathcal{N} \, T^2 / L^2$')
    plt.tight_layout()
    plt.axvline(ps_ramanful[1], color='red', lw=1, ls='--')
    plt.axhline(ps_ramanful[0], color='blue', lw=1, ls='--')
    lg.info(f"threshold at L/LW = {1/ps_ramanful[2]:.2f}")
    lg.debug(f"(ramanful) Fitted parameters: a={ps_ramanful[0]:.4f}, b={ps_ramanful[1]:.4f}, c={ps_ramanful[2]:.4f}")
    
    plt.savefig(f"media/simple_threshold_{pulse_shape}_{llda:.2f}_{lldb:.2f}.pdf", dpi=dpi)
    lg.info(f"Saved figure to media/simple_threshold_{pulse_shape}_{llda:.2f}_{lldb:.2f}.pdf")
    return


def fB_undepleted(z):
    """
    f_B(z) = exp[-α_s z + (g * Pp_in * e^{-α_p L} / α_p) * (e^{α_p z} - 1)]
    Parameters are tuned so f_B(70 km) ≈ f_B(0) = 1.
    """
    z = z/ 1000
    # --- Parameters giving approximate gain-loss balance ---
    alpha_s = 0.023   # signal loss [1/km]
    alpha_p = 0.12    # pump loss [1/km]
    g       = 0.95    # gain coefficient [1/(W·km)]
    Pp_in   = 0.20    # pump input power [W]
    L       = 70.0    # fiber length [km]

    z = np.asarray(z, dtype=float)
    if np.any((z < 0) | (z > L)):
        raise ValueError(f"z must be between 0 and {L} km")

    exp_neg_apL = np.exp(-alpha_p * L)
    expm1_apz   = np.expm1(alpha_p * z)
    exponent = -alpha_s * z + (g * Pp_in * exp_neg_apL / alpha_p) * expm1_apz
    return np.exp(exponent)


def plot_threshold(
    recompute: bool = False,
    use_fB: bool = False,
    fB_simple_interpolation: bool = False,
    gvda: float = 0.0,
    gvdb: float = 0.0,
    m_lo_truncation: int = 0
):
    if not use_fB:
        fB_simple_interpolation = True
        lg.trace("Disabling fB_simple_interpolation since use_fB is False")

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
    n_samples_numeric = nc.n_samples_numeric_n
    lg.warning(n_samples_numeric)
    dgd2 = nc.dgd2_n
    dgds_analytic = np.geomspace(nc.dgd1, dgd2, n_samples_analytic)
    dgds_numeric = np.logspace(
            np.log10(nc.dgd1), np.log10(dgd2), n_samples_numeric)
    # undepleted fB
    z_axis = np.linspace(0.0, cf.fiber_length, 100)
    # fB = fB_undepleted(z_axis)
    
    # nlin_fitted = []
    # nlin_fitted_data = []
    # nlin_hi = []
    # nlin_lo = []
    _, fB_min, fB_max, _, _ = load_fB(cf)
    
    fB = fB_max
    fB_mode = "max"
    
    for ipulse in [0, 1]:
        ipulse=1
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
        
        
        # ----- call the computation functions -----
        # -- numeric (compute_numeric computes all the fB_modes)
        compute_numeric_nlin(gvda=gvda, gvdb=gvdb, ipulse=ipulse, recompute=recompute)
        nlin_numeric_bare = np.load(
            f"results/partial_nlin_{pulse_shape}_perfect_0.0_0.0.npy")
        nlin_numeric_gvd = np.load(
            f"results/partial_nlin_{pulse_shape}_perfect_{gvda}_{gvdb}.npy")
        nlin_numeric_gvd_raman = np.load(
            f"results/partial_nlin_{pulse_shape}_{fB_mode}_{gvda}_{gvdb}.npy")
        nlin_numeric_raman = np.load(
            f"results/partial_nlin_{pulse_shape}_{fB_mode}_0.0_0.0.npy")
        # lg.info(nlin_numeric)
        # lg.info(nlin_numeric * y_norm)
        
        raman_gvd_correction_min, raman_gvd_correction_max = build_lookup_integral_table_with_raman(cf, m_lo_truncation=m_lo_truncation, recompute=False)
        
        # -- analytic and fitted
        nlin_fit_bare = (fit_nlin(cf,
                            0.0,
                            0.0,
                            np.ones_like(fB),
                            raman_gvd_correction_min, 
                            raman_gvd_correction_max,
                            ipulse=ipulse, m_lo_truncation=m_lo_truncation))
        nlin_fit_gvd = (fit_nlin(cf,
                            gvda,
                            gvdb,
                            np.ones_like(fB),
                            raman_gvd_correction_min, 
                            raman_gvd_correction_max,
                            ipulse=ipulse, m_lo_truncation=m_lo_truncation))
        nlin_fit_gvd_raman = (fit_nlin(cf,
                            gvda,
                            gvdb,
                            fB, # FIXME adapt
                            raman_gvd_correction_min, 
                            raman_gvd_correction_max,
                            ipulse=ipulse, m_lo_truncation=m_lo_truncation))
        nlin_fit_raman = (fit_nlin(cf,
                            0.0,
                            0.0,
                            fB, # FIXME adapt
                            raman_gvd_correction_min, 
                            raman_gvd_correction_max,
                            ipulse=ipulse, m_lo_truncation=m_lo_truncation))
        # nlin_fitted_data = (nlin_fitted(dgds_analytic) / y_norm)
        n_hi, n_lo = compute_asymptotic_nlin(ipulse=ipulse)
        nlin_hi = (n_hi)
        nlin_hi_correct = nlin_hi * raman_integral(cf, "HI", fB)
        nlin_lo = (n_lo)
    
        rc('text', usetex=True)
        dpi = 300
        # ----------------------------------
        # plotting the threshold
        # ----------------------------------
        plt.figure(figsize=(3.6, 2.5))
        color_modes = [adjust_luminosity(
            'magenta', 0.8), adjust_luminosity('cyan', 0.8), 'green']
        # exit()
        plt.plot(dgds_analytic * x_norm, nlin_lo * y_norm,
                lw=1, color=adjust_luminosity('green', 0.9), ls='--')
        plt.plot(dgds_analytic * x_norm, nlin_hi * y_norm,
                lw=1, color=adjust_luminosity('orange', 0.9))
        plt.plot(dgds_analytic * x_norm, nlin_hi_correct * y_norm,
                lw=1, color=adjust_luminosity('orange', 0.9))

        lw = 1
        ss = 20
        plt.plot(dgds_analytic * x_norm, nlin_fit_bare(dgds_analytic), color="gray", lw=0.6)
        plt.plot(dgds_analytic * x_norm, nlin_fit_gvd(dgds_analytic), color="gray", lw=0.6)
        plt.plot(dgds_analytic * x_norm, nlin_fit_gvd_raman(dgds_analytic), color="gray", lw=0.6)
        plt.plot(dgds_analytic * x_norm, nlin_fit_raman(dgds_analytic), color="gray", lw=0.6)
        plt.scatter(dgds_numeric * x_norm, nlin_numeric_bare * y_norm, label='Gauss.', color="green", marker=".", s=ss, lw=lw)
        plt.scatter(dgds_numeric * x_norm, nlin_numeric_gvd * y_norm, label='Gauss.', color="green", marker="+", s=ss, lw=lw)
        plt.scatter(dgds_numeric * x_norm, nlin_numeric_gvd_raman * y_norm, label='Gauss.', color="green", marker=(6, 2, 0), s=ss, lw=lw)
        plt.scatter(dgds_numeric * x_norm, nlin_numeric_raman * y_norm, label='Gauss.', color="green", marker="x", s=ss, lw=lw)
        
        plt.xscale('log')
        plt.yscale('log')
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, 1.0)
        # if use_fB:
            # plt.ylim([0.5e-3, 0.11])
        # else:
            # plt.ylim([0.7e-2, 1])
        plt.xlabel(r'$L/L_W$')
        plt.ylabel(r'$\mathcal{N} \, T^2 / L^2$')
        plt.tight_layout()
        out_pdf = "media/2-threshold_raman-all.pdf"
        plt.savefig(out_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0)
        lg.info(f"Saved figure to {out_pdf}")
    
        # ----------------------------------
        # plotting the error (only without Raman)
        # ----------------------------------
        if not use_fB:
            plt.clf()
            plt.figure(figsize=(3.6, 2))
            lw = 1
            ss = 8

            # Numeric (no Raman) in normalized units
            numeric_bare_norm = nlin_numeric_bare * y_norm

            # Sample the fitted curves on the numeric grid (fits already return normalized values)
            fit_bare_on_numeric = nlin_fit_bare(dgds_numeric)
            fit_gvd_on_numeric  = nlin_fit_gvd(dgds_numeric)

            # Interpolate asymptotics (normalized) onto the numeric grid
            lo_interp = interp1d(
                dgds_analytic,
                nlin_lo * y_norm,
                kind="linear",
                bounds_error=False,
                fill_value=(nlin_lo[0] * y_norm, nlin_lo[-1] * y_norm),
            )
            hi_interp = interp1d(
                dgds_analytic,
                nlin_hi * y_norm,
                kind="linear",
                bounds_error=False,
                fill_value=(nlin_hi[0] * y_norm, nlin_hi[-1] * y_norm),
            )
            lo_on_numeric = lo_interp(dgds_numeric)
            hi_on_numeric = hi_interp(dgds_numeric)

            # Prevent division by zero
            eps = 1e-30

            # Relative errors
            relerr_fit_bare = np.abs(numeric_bare_norm - fit_bare_on_numeric) / np.maximum(numeric_bare_norm, eps)
            relerr_fit_gvd  = np.abs(numeric_bare_norm - fit_gvd_on_numeric)  / np.maximum(numeric_bare_norm,  eps)
            relerr_lo       = np.abs(numeric_bare_norm - lo_on_numeric)       / np.maximum(numeric_bare_norm,       eps)
            relerr_hi       = np.abs(numeric_bare_norm - hi_on_numeric)       / np.maximum(numeric_bare_norm,       eps)

            # Plots
            plt.plot(dgds_numeric * x_norm, relerr_fit_bare, marker="x",  lw=lw, markersize=ss, linestyle="-",  label="vs fit (bare)")
            plt.plot(dgds_numeric * x_norm, relerr_fit_gvd,  marker="*",  lw=lw, markersize=ss, linestyle="-.", label="vs fit (GVD)")
            plt.plot(dgds_numeric * x_norm, relerr_lo,       marker=None, lw=lw,               linestyle=":",  label="vs asympt. LO")
            plt.plot(dgds_numeric * x_norm, relerr_hi,       marker=None, lw=lw,               linestyle="--", label="vs asympt. HI")
            plt.axhline(0.1, color="grey", lw=0.8, ls="-")
            plt.axhline(0.2, color="grey", lw=0.8, ls="-")
            plt.xscale('log')
            plt.ylim([-0.05, 0.5])
            plt.xlabel(r'$L/L_W$')
            plt.ylabel(r'$\varepsilon$')
            plt.tight_layout()
            err_pdf = f"media/2-error_{pulse_shape}.pdf"
            plt.savefig(err_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0)

            # # --- simple linear sweep for 10% crossings vs asymptotics ---
            # def sweep_crossing_logx(x, y, target=0.1, side="left"):
            #     """
            #     Find x* where y(x*) ~ target by scanning for the first sign change
            #     of (y - target), then doing 2-point linear interpolation in log(x).
            #     Returns np.nan if no crossing is found.
            #     """
            #     x = np.asarray(x)
            #     y = np.asarray(y)
            #     ydiff = y - target

            #     # choose sweep direction
            #     if side == "left":
            #         idx_range = range(len(x) - 1)
            #     else:  # "right"
            #         idx_range = range(len(x) - 2, -1, -1)

            #     for i in idx_range:
            #         j = i + 1
            #         a, b = ydiff[i], ydiff[j]
            #         lg.debug(f"sweep_crossing_logx: i={i}, j={j}, x[i]={x[i]:.3e}, x[j]={x[j]:.3e}, a={a:.3e}, b={b:.3e}")
            #         if np.isnan(a) or np.isnan(b) or np.isnan(x[i]) or np.isnan(x[j]):
            #             continue
            #         # exact hit
            #         if a == 0:
            #             return x[i]
            #         if b == 0:
            #             return x[j]
            #         # sign change between i and j?
            #         if a * b < 0:
            #             # interpolate in log(x) between the two points
            #             lx0, lx1 = np.log(x[i]), np.log(x[j])
            #             # linear interpolation parameter t in y-space
            #             t = -a / (b - a)
            #             t = 0.0 if t < 0 else (1.0 if t > 1.0 else t)
            #             lx_star = lx0 + t * (lx1 - lx0)
            #             return np.exp(lx_star)
            #     return np.nan

            # lo_cross_dgds = sweep_crossing_logx(dgds_numeric, relerr_lo, target=0.1, side="left")
            # hi_cross_dgds = sweep_crossing_logx(dgds_numeric, relerr_hi, target=0.1, side="right")

            # lo_cross_LW = lo_cross_dgds * x_norm if np.isfinite(lo_cross_dgds) else np.nan
            # hi_cross_LW = hi_cross_dgds * x_norm if np.isfinite(hi_cross_dgds) else np.nan

            # if np.isfinite(lo_cross_dgds) and np.isfinite(hi_cross_dgds):
            #     lg.info(f"[{pulse_shape}] 10% error thresholds vs asymptotics: "
            #             f"LO @ dgds={lo_cross_dgds:.6e} (L/L_W={lo_cross_LW:.6e}), "
            #             f"HI @ dgds={hi_cross_dgds:.6e} (L/L_W={hi_cross_LW:.6e})")
            # else:
            #     if not np.isfinite(lo_cross_dgds):
            #         lg.warning(f"[{pulse_shape}] Could not find 10% crossing for LO asymptotic.")
            #     if not np.isfinite(hi_cross_dgds):
            #         lg.warning(f"[{pulse_shape}] Could not find 10% crossing for HI asymptotic.")

            
            lg.info(f"Saved figure to {err_pdf}")



def plot_threshold_fb_extremes(
    recompute: bool = False,
    gvda: float = 0.0,
    gvdb: float = 0.0,
    m_lo_truncation: int = 4,
    out_pdf: str = "media/case-study-threshold.pdf",
):
    # ---- config / normalization ---------------------------------------------
    cf_path = "./input/mmf.toml"
    nc_path = "./input/numerical_config.toml"
    cf = cfg.load_toml_to_struct(cf_path)
    nc = cfg.load_nc_toml_to_struct(nc_path)

    x_norm = cf.fiber_length * cf.baud_rate
    y_norm_fit = 1.0 / (cf.fiber_length * cf.baud_rate) ** 2
    
    x_norm = 1e12
    y_norm = 1e-30
    
    # x-grids (analytic vs numeric sampling)
    n_samples_analytic = 500
    nc.dgd1   = LLW_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = LLW_MAX / (cf.fiber_length * cf.baud_rate)
    lg.info(f"nc.dgd1 = {nc.dgd1:.2e}, nc.dgd2_n = {nc.dgd2_n:.2e}")
    nc.dgd2_g = nc.dgd2_n

    dgds_analytic = np.geomspace(nc.dgd1, nc.dgd2_n, n_samples_analytic)
    
    T = 1 / cf.baud_rate
    L = cf.fiber_length
    # ---- fB (independent of pulse) ------------------------------------------
    _, fB_min, fB_max, _, _ = load_fB(cf)

    # cyan = fB_max; magenta = fB_min
    fB_refs = dict(max=fB_max, min=fB_min)
    colors  = dict(max="turquoise", min="magenta")

    # Raman correction lookup tables (shared)
    raman_corr_min_n, raman_corr_max_n = build_lookup_integral_table_with_raman(
        cf, m_lo_truncation=m_lo_truncation, recompute=False, ipulse = 1
    )
    raman_corr_min_g, raman_corr_max_g = build_lookup_integral_table_with_raman(
        cf, m_lo_truncation=m_lo_truncation, recompute=False, ipulse = 0
    )

    # ---- plotting ------------------------------------------------------------
    rc("text", usetex=True)
    plt.figure(figsize=(3.6, 3.2), dpi=300)

    # loop pulses only for *numeric/fitted/asymptotic* series
    pulse_style = ["-", "-."]
    for ipulse, pulse_shape in ((0, "gaussian"), (1, "nyquist")):
        # numeric x-grid for this pulse
        if ipulse == 0:
            dgd2 = nc.dgd2_g
            n_samples_numeric = nc.n_samples_numeric_g
        else:
            dgd2 = nc.dgd2_n
            n_samples_numeric = nc.n_samples_numeric_n

        dgds_numeric = np.logspace(np.log10(nc.dgd1), np.log10(dgd2), n_samples_numeric)

        compute_numeric_nlin(gvda=gvda, gvdb=gvdb, ipulse=ipulse, recompute=recompute)
        compute_numeric_nlin(gvda=0.0, gvdb=0.0, ipulse=ipulse,   recompute=recompute)

        # load numeric series
        nlin_numeric_max = np.load(f"results/partial_nlin_{'gaussian' if ipulse == 0 else 'nyquist'}_max_{gvda}_{gvdb}.npy")
        nlin_numeric_min = np.load(f"results/partial_nlin_{'gaussian' if ipulse == 0 else 'nyquist'}_min_{gvda}_{gvdb}.npy")
        if ipulse == 0: # not really super nice, but that's the way we do it
            raman_corr_min = raman_corr_min_g
            raman_corr_max = raman_corr_max_g
        else:
            raman_corr_min = raman_corr_min_n
            raman_corr_max = raman_corr_max_n
        nlin_fit_max = fit_nlin(cf, gvda, gvdb, fB_max,
                                      raman_corr_min, raman_corr_max,
                                      ipulse=ipulse, m_lo_truncation=m_lo_truncation)
        nlin_fit_min = fit_nlin(cf, gvda, gvdb, fB_min,
                                  raman_corr_min, raman_corr_max,
                                  ipulse=ipulse, m_lo_truncation=m_lo_truncation)
        lg.debug(f"Fitted coefficients for {'Gaussian' if ipulse == 0 else 'Nyquist'} pulse, ")
        lg.debug(f"Some values of fit: nlin_fit(dgds_analytic[:5]) = {nlin_fit_max(dgds_analytic[:5])}")
        lg.debug(f"Some values of fit: nlin_fit(dgds_analytic[:5]) = {nlin_fit_min(dgds_analytic[:5])}")
        # asymptotics
        n_hi, n_lo = compute_asymptotic_nlin(ipulse=ipulse)
        nlin_hi = n_hi
        nlin_lo = n_lo
        nlin_hi_correct_max = nlin_hi * raman_integral(cf, "HI", fB_max)
        nlin_hi_correct_min = nlin_hi * raman_integral(cf, "HI", fB_min)

        # draw asymptotes and fits (thin gray/orange/green) — once per pulse
        # plt.plot(dgds_analytic * x_norm, nlin_lo * y_norm, lw=1.0,
        #          color="green", ls="--")
        # plt.plot(dgds_analytic * x_norm, nlin_hi * y_norm, lw=1.0,
        #          color="orange")
        plt.plot(dgds_analytic * x_norm, nlin_hi_correct_max * y_norm, lw=1.0,
                 color="orange")
        plt.plot(dgds_analytic * x_norm, nlin_hi_correct_min * y_norm, lw=1.0,
                 color="orange")

        lw = 0.7
        plt.plot(dgds_analytic * x_norm, nlin_fit_max(dgds_analytic) / y_norm_fit * y_norm , color="gray",  lw=lw, ls=pulse_style[ipulse])
        plt.plot(dgds_analytic * x_norm, nlin_fit_min(dgds_analytic) / y_norm_fit * y_norm ,  color="gray", lw=lw, ls=pulse_style[ipulse])

        ms = 24
        if ipulse == 1:  # Nyquist (stars)
            marker = "*"
        else:            # Gaussian (crosses)
            marker = "x"

        plt.scatter(dgds_numeric * x_norm, nlin_numeric_max * y_norm,
                    s=ms, lw=1.0, marker=marker, color=colors["max"])
        plt.scatter(dgds_numeric * x_norm, nlin_numeric_min * y_norm,
                    s=ms, lw=1.0, marker=marker, color=colors["min"])

    # axes/labels
    plt.xscale("log")
    plt.yscale("log")
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, 0.3)
    plt.xlabel(r"$\Delta \beta_1$ [ps/m]")
    plt.ylabel(r"$\mathcal{N}$ [km$^2$/ps$^2$]")
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.02)
    lg.info(f"Saved figure to {out_pdf}")


if __name__ == "__main__":
    
    simple_plot_threshold(
        gvda = 0.0e-27,
        gvdb = 0.0e-27,
        fB_mode="max",
        recompute=False,
        ipulse=1,
        m_lo_truncation=3)
    simple_plot_threshold(
        gvda = 30.0e-27,
        gvdb = 0.0e-27,
        fB_mode="min",
        recompute=False,
        ipulse=1,
        m_lo_truncation=3)
    simple_plot_threshold(
        gvda = 30.0e-27,
        gvdb = 30.0e-27,
        fB_mode="min",
        recompute=False,
        ipulse=1,
        m_lo_truncation=3)
    exit()
    plot_threshold(recompute=False,
                   use_fB=False,
                   fB_simple_interpolation=True,
                   gvda = 30.0e-27,
                   gvdb = 0.0e-27,
                   m_lo_truncation=4)
    exit()
    plot_threshold_fb_extremes(recompute=False,
                #    use_fB=False,
                #    fB_simple_interpolation=True,
                   gvda = 20.0e-27,
                   gvdb = 20.0e-27,
                   m_lo_truncation=4)
    exit()
    simple_plot_threshold(
        gvda = 30.0e-27,
        gvdb = 0.0e-27,
        fB_mode="max",
        recompute=False,
        ipulse=1,
        m_lo_truncation=3)

    # plot the theoretical figure
    plot_threshold(recompute=True,
                   use_fB=False,
                   fB_simple_interpolation=False)