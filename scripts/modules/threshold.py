import logging
import os
import sys
import time
import warnings
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
from typing import Tuple, Dict, Any
from logging.handlers import TimedRotatingFileHandler
from contextlib import contextmanager
from functools import wraps

DGD_MIN = 0.01  # target L/LW
DGD_MAX = 100.0

# ==========================
# Logging utilities
# ==========================

def _resolve_log_level(default: str = "INFO") -> int:
    lvl = getattr(cfg, "LOG_LEVEL", None) if "cfg" in globals() else None
    lvl = os.getenv("LOG_LEVEL", lvl or default)
    try:
        return getattr(logging, str(lvl).upper())
    except Exception:
        return logging.INFO


def setup_logging(
    name: str = __name__,
    log_dir: str = "logs",
    log_filename: str | None = None,
    level: int | None = None,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Configure console + rotating file logging and return a logger."""
    level = level or _resolve_log_level()
    os.makedirs(log_dir, exist_ok=True)
    if log_filename is None:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "run"
        log_filename = f"{base}.log"
    logfile_path = os.path.join(log_dir, log_filename)

    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(level)

        console = logging.StreamHandler(stream=sys.stderr)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

        fileh = TimedRotatingFileHandler(
            logfile_path, when="midnight", backupCount=14, utc=False, encoding="utf-8"
        )
        fileh.setLevel(level)
        fileh.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | pid=%(process)d th=%(threadName)s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

        root.addHandler(console)
        root.addHandler(fileh)

    # Tone down noisy libs unless debugging
    for noisy in ("matplotlib", "urllib3", "numexpr"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    if capture_warnings:
        logging.captureWarnings(True)
        warnings.filterwarnings("default")

    logger = logging.getLogger(name)
    logger.debug("Logging initialized: level=%s, file=%s", logging.getLevelName(level), logfile_path)
    return logger


@contextmanager
def log_duration(logger: logging.Logger, msg: str, level: int = logging.INFO):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.log(level, "%s (%.3f s)", msg, dt)


def log_calls(logger: logging.Logger, level: int = logging.DEBUG):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            logger.log(level, "→ %s(%s)", fn.__name__, _fmt_args_kwargs(args, kwargs))
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
                logger.log(level, "← %s returned %s in %.3f s", fn.__name__, _safe_repr(out), time.perf_counter() - t0)
                return out
            except Exception:
                logger.exception("✖ %s raised", fn.__name__)
                raise
        return wrapped
    return deco


def _safe_repr(obj: Any, maxlen: int = 200) -> str:
    try:
        r = repr(obj)
        return (r[: maxlen - 1] + "…") if len(r) > maxlen else r
    except Exception:
        return f"<{type(obj).__name__}>"


def _fmt_args_kwargs(args: tuple, kwargs: dict) -> str:
    parts = []
    if args:
        parts.extend(_safe_repr(a) for a in args)
    if kwargs:
        parts.extend(f"{k}={_safe_repr(v)}" for k, v in kwargs.items())
    return ", ".join(parts)


def _log_params(logger: logging.Logger, title: str, where: str, params: Dict[str, Any]):
    # Homogeneous, nicely aligned key: value list
    if not logger.isEnabledFor(logging.DEBUG):
        return
    maxk = max((len(k) for k in params), default=0)
    lines = [f"{title} [from {where}] ▶"]
    for k in sorted(params):
        v = params[k]
        lines.append(f"  • {k.ljust(maxk)} : {_safe_repr(v)}")
    logger.debug("\n" + "\n".join(lines))


# Initialize module logger
logger = setup_logging(__name__)
# Convert NumPy floating warnings into logged warnings (don’t raise)
np.seterr(all="warn")


# ==========================
# Core functions
# ==========================

@log_calls(logger)
def softplus2(x, a, b, c):
    # _log_params(logger, "softplus2 params", "function args", {
    #     "x.shape": getattr(np.asarray(x), "shape", None),
    #     "a": a, "b": b, "c": c,
    # })
    return a * (1 + (x / b)**(1 / c))**(-c)


@log_calls(logger)
def get_raman_corrections(smf: bool = False) -> Tuple[float, float, float, float]:
    """
    Given the optimized signal profiles, calculate max and min asymptotic values
    for the correction coefficients f_B.
    Requires: results/oi_fit.npy, input/mmf.toml or smf.toml,
              results/ct_solution-6_gain_0.0[_SMF].npy
    """
    toml_path = "./input/smf.toml" if smf else "./input/mmf.toml"
    select_string = "_SMF" if smf else ""
    cf = cfg.load_toml_to_struct(toml_path)
    _log_params(logger, "get_raman_corrections inputs", "config + args", {
        "smf": smf,
        "toml_path": toml_path,
        "fiber_length": getattr(cf, "fiber_length", None),
        "baud_rate": getattr(cf, "baud_rate", None),
        "launch_power": getattr(cf, "launch_power", None),
        "raman_gain": getattr(cf, "raman_gain", None),
    })

    assert (cf.launch_power == -6.0 and cf.raman_gain == 0.0)

    sol_path = f"results/ct_solution-6_gain_0.0{select_string}.npy"
    solutions = np.load(sol_path, allow_pickle=True).item()
    logger.info("Loaded Raman solutions from %s", sol_path)

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
    rcal_lo_min = (np.sum(fB_min) * dz / cf.fiber_length)**2
    rcal_lo_max = (np.sum(fB_max) * dz / cf.fiber_length)**2
    rcal_hi_min = np.sum(fB_min**2) * dz / cf.fiber_length
    rcal_hi_max = np.sum(fB_max**2) * dz / cf.fiber_length

    # Sanity checks
    rcal_hi = np.sum(fB**2, axis=0) * dz / cf.fiber_length
    rcal_lo = (np.sum(fB, axis=0) * dz / cf.fiber_length)**2
    assert ((rcal_hi >= rcal_hi_min - 1e-15).all())
    assert ((rcal_hi <= rcal_hi_max + 1e-15).all())
    assert ((rcal_lo >= rcal_lo_min - 1e-15).all())
    assert ((rcal_lo <= rcal_lo_max + 1e-15).all())

    _log_params(logger, "get_raman_corrections outputs", "computed", {
        "rcal_lo_min": rcal_lo_min,
        "rcal_lo_max": rcal_lo_max,
        "rcal_hi_min": rcal_hi_min,
        "rcal_hi_max": rcal_hi_max,
    })

    return rcal_lo_min, rcal_lo_max, rcal_hi_min, rcal_hi_max


@log_calls(logger)
def get_fit_coefficients(fB_simple_interpolation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the numerical noise results for a representative GVD, find the best
    fit coefficients in the Nyquist and Gaussian cases. Only consider max and min profiles.
    Requires: results/partial_nlin_gaussian_*.npy, results/partial_nlin_nyquist_*.npy,
              input/mmf.toml, input/numerical_config.toml, results/oi_fit.npy
    """
    cf = cfg.load_toml_to_struct("./input/mmf.toml")
    nc = cfg.load_nc_toml_to_struct("./input/numerical_config.toml")

    # Override DGD ranges using global targets
    old = {"dgd1": getattr(nc, "dgd1", None), "dgd2_n": getattr(nc, "dgd2_n", None), "dgd2_g": getattr(nc, "dgd2_g", None)}
    nc.dgd1 = DGD_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = DGD_MAX / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_g = nc.dgd2_n

    _log_params(logger, "get_fit_coefficients inputs", "config + args", {
        "fB_simple_interpolation": fB_simple_interpolation,
        "mmf.toml": "./input/mmf.toml",
        "numerical_config.toml": "./input/numerical_config.toml",
        "overrode_from": old,
        "overrode_to": {"dgd1": nc.dgd1, "dgd2_n": nc.dgd2_n, "dgd2_g": nc.dgd2_g},
        "baud_rate": cf.baud_rate,
        "fiber_length": cf.fiber_length,
    })

    oi_fit = np.load('results/oi_fit.npy')
    beta1_params = load_group_delay()
    fiber = MMFiber(
        effective_area=cf.effective_area,
        overlap_integrals=oi_fit,
        group_delay=beta1_params,
        length=cf.fiber_length,
        n_modes=cf.n_modes
    )

    modes = ["perfect"] if fB_simple_interpolation else ["min", "max"]
    dgds_numeric_g = np.logspace(np.log10(nc.dgd1), np.log10(nc.dgd2_g), nc.n_samples_numeric_g)
    dgds_numeric_n = np.logspace(np.log10(nc.dgd1), np.log10(nc.dgd2_n), nc.n_samples_numeric_n)
    T = 1 / cf.baud_rate
    L = fiber.length
    x_norm = L / T
    y_norm = x_norm**(-2)
    p0 = [0.2, 4.5, 0.5]

    ps_g = np.zeros((len(modes), 3))
    ps_n = np.zeros((len(modes), 3))

    for mode in modes:
        partial_B2g = np.load(f"results/partial_nlin_gaussian_{mode}{nc.gvd}B2.npy")
        partial_B2n = np.load(f"results/partial_nlin_nyquist_{mode}{nc.gvd}B2.npy")
        _log_params(logger, f"curve_fit datasets [{mode}]", "npy files", {
            "gaussian_file": f"results/partial_nlin_gaussian_{mode}{nc.gvd}B2.npy",
            "nyquist_file": f"results/partial_nlin_nyquist_{mode}{nc.gvd}B2.npy",
            "len_gaussian": len(partial_B2g),
            "len_nyquist": len(partial_B2n),
        })
        popt_n, _ = curve_fit(softplus2, dgds_numeric_n * x_norm, partial_B2n * y_norm, p0=p0)
        popt_g, _ = curve_fit(softplus2, dgds_numeric_g * x_norm, partial_B2g * y_norm, p0=p0)
        ps_g[modes.index(mode), :] = popt_g
        ps_n[modes.index(mode), :] = popt_n
        _log_params(logger, f"fitted coefficients [{mode}]", "scipy.curve_fit", {
            "ps_g(a,b,c)": popt_g,
            "ps_n(a,b,c)": popt_n,
        })

    return ps_g, ps_n


def adjust_luminosity(color, factor):
    _log_params(logger, "adjust_luminosity inputs", "function args", {"color": color, "factor": factor})
    rgb = np.array(mcolors.to_rgb(color))  # Convert to RGB
    return np.clip(rgb * factor, 0, 1)  # Scale and clip values


@log_calls(logger)
def get_space_integrals(m, z, I):
    _log_params(logger, "get_space_integrals inputs", "function args", {
        "m.shape": getattr(np.asarray(m), "shape", None),
        "z.shape": getattr(np.asarray(z), "shape", None),
        "I.shape": getattr(np.asarray(I), "shape", None),
    })
    X0mm = X0mm_space_integral(z, I, amplification_function=None)
    return X0mm


@log_calls(logger)
def get_nlin_threshold(
    recompute: bool = False,
    use_fB: bool = False,
    fB_simple_interpolation: bool = False,
):
    if not use_fB:
        assert fB_simple_interpolation
        
    rc('text', usetex=True)

    cf_path = "./input/mmf.toml"
    nc_path = "./input/numerical_config.toml"
    cf = cfg.load_toml_to_struct(cf_path)
    nc = cfg.load_nc_toml_to_struct(nc_path)

    old = {"dgd1": getattr(nc, "dgd1", None), "dgd2_n": getattr(nc, "dgd2_n", None), "dgd2_g": getattr(nc, "dgd2_g", None)}
    nc.dgd1 = DGD_MIN / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_n = DGD_MAX / (cf.fiber_length * cf.baud_rate)
    nc.dgd2_g = nc.dgd2_n

    _log_params(logger, "get_nlin_threshold inputs", "config + args", {
        "recompute": recompute,
        "use_fB": use_fB,
        "fB_simple_interpolation": fB_simple_interpolation,
        "mmf.toml": cf_path,
        "numerical_config.toml": nc_path,
        "overrode_from": old,
        "overrode_to": {"dgd1": nc.dgd1, "dgd2_n": nc.dgd2_n, "dgd2_g": nc.dgd2_g},
        "baud_rate": cf.baud_rate,
        "fiber_length": cf.fiber_length,
        "n_channels": cf.n_channels,
        "channel_spacing": cf.channel_spacing,
        "center_frequency": cf.center_frequency,
        "gvd": getattr(nc, 'gvd', None),
    })

    logger.info("Within L/LD = %.1f", cf.fiber_length * np.abs(nc.gvd) * cf.baud_rate**2)

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
    gvds = [nc.gvd]

    logger.info(
        "Computing NLIN coeff. in L/LW ∈ [%.1e, %.1e] (ps/m range)",
        nc.dgd1 * 1e12, nc.dgd2_g * 1e12,
    )

    if use_fB:
        rcal_lo_min, rcal_lo_max, rcal_hi_min, rcal_hi_max = get_raman_corrections()
        modes = ["min", "max"]
        assert (cf.launch_power == -6.0 and cf.raman_gain == 0.0)
        sol_path = "results/ct_solution-6_gain_0.0.npy"
        solutions = np.load(sol_path, allow_pickle=True).item()
        logger.info("Loaded case-study Raman solutions from %s", sol_path)

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

    def antonio_rescale_max(dgd):
        acc = 0.0
        for m in get_m_values(fiber, wdm, a_chan, b_chan, T, 0, dgd):
            acc += fB_max_function(get_collision_location(m, fiber, wdm, a_chan, b_chan, pulse, dgd))**2
        return acc

    def antonio_rescale_min(dgd):
        acc = 0.0
        for m in get_m_values(fiber, wdm, a_chan, b_chan, T, 0, dgd):
            acc += fB_min_function(get_collision_location(m, fiber, wdm, a_chan, b_chan, pulse, dgd))**2
        return acc

    # ----------------------------------
    # Computation of the NLIN coefficient
    # ----------------------------------
    for gvd in gvds:
        for px in [0, 1]:
            if px == 0:
                pulse = GaussianPulse(baud_rate=cf.baud_rate, num_symbols=1e2, samples_per_symbol=2**5)
            else:
                pulse = NyquistPulse(baud_rate=cf.baud_rate, num_symbols=1e3, samples_per_symbol=2**5, rolloff=0.0)

            n_samples_analytic = 500
            if px == 0:
                dgd2 = nc.dgd2_g
                n_samples_numeric = nc.n_samples_numeric_g
            else:
                dgd2 = nc.dgd2_n
                n_samples_numeric = nc.n_samples_numeric_n

            dgds_numeric = np.logspace(np.log10(nc.dgd1), np.log10(dgd2), n_samples_numeric)
            dgds_analytic = np.linspace(nc.dgd1, dgd2, n_samples_analytic)

            partial_nlin = np.zeros(n_samples_numeric)
            partial_nlin_min = np.zeros(n_samples_numeric)
            partial_nlin_max = np.zeros(n_samples_numeric)
            a_chan = (1, 1)
            b_chan = (1, 2)

            if recompute:
                with log_duration(logger, f"Recompute partial_nlin (px={px}, gvd={gvd})"):
                    for idx, dgd in enumerate(dgds_numeric):
                        z, I, m = compute_all_collisions_time_integrals(a_chan, b_chan, fiber, wdm, pulse, dgd, gvd)
                        X0mm_min = get_space_integrals_min(m, z, I)
                        X0mm_max = get_space_integrals_max(m, z, I)
                        X0mm = get_space_integrals(m, z, I)

                        for xx in [X0mm, X0mm_max, X0mm_min]:
                            nonzero = np.real(xx) != 0
                            assert (np.all(np.imag(xx[nonzero]) < 1e-6 * np.real(xx[nonzero])))

                        partial_nlin[idx] = np.sum(np.real(X0mm)**2)
                        partial_nlin_min[idx] = np.sum(np.real(X0mm_min)**2)
                        partial_nlin_max[idx] = np.sum(np.real(X0mm_max)**2)

                if px == 0:
                    np.save(f"results/partial_nlin_gaussian_perfect{gvd}B2.npy", partial_nlin)
                    np.save(f"results/partial_nlin_gaussian_min{gvd}B2.npy", partial_nlin_min)
                    np.save(f"results/partial_nlin_gaussian_max{gvd}B2.npy", partial_nlin_max)
                else:
                    np.save(f"results/partial_nlin_nyquist_perfect{gvd}B2.npy", partial_nlin)
                    np.save(f"results/partial_nlin_nyquist_min{gvd}B2.npy", partial_nlin_min)
                    np.save(f"results/partial_nlin_nyquist_max{gvd}B2.npy", partial_nlin_max)

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
    dgds_numeric_g = np.logspace(np.log10(nc.dgd1), np.log10(nc.dgd2_g), nc.n_samples_numeric_g)
    dgds_numeric_n = np.logspace(np.log10(nc.dgd1), np.log10(nc.dgd2_n), nc.n_samples_numeric_n)

    x_norm = L / T
    y_norm = x_norm**(-2)

    dpi = 300

    # ----------------------------------
    # plotting the threshold
    # ----------------------------------
    plt.figure(figsize=(3.6, 3))
    color_modes = [adjust_luminosity('magenta', 0.8), adjust_luminosity('cyan', 0.8), 'green']

    ps_g, ps_n = get_fit_coefficients(fB_simple_interpolation=fB_simple_interpolation)
    for im, mode in enumerate(modes):
        na_nlin = L / (T * dgds_numeric_g)
        if mode == "max":
            na_nlin = na_nlin * rcal_hi_max
        elif mode == "min":
            na_nlin = na_nlin * rcal_hi_min

        gauss = np.ones_like(dgds_analytic) * np.sqrt(np.pi) * (LD_eff / (T * np.sqrt(2 * np.pi)) * np.arcsinh(L / LD_eff))**2
        nyquist = np.ones_like(dgds_analytic) * 4 / 9 / y_norm
        if use_fB:
            rcal_hi = rcal_hi_max if mode == "max" else rcal_hi_min
            rcal_lo = rcal_lo_max if mode == "max" else rcal_lo_min
            gauss *= rcal_lo
            nyquist *= rcal_lo

        plt.plot(dgds_numeric_g * x_norm, na_nlin * y_norm, lw=1, color=adjust_luminosity('orange', 0.9))
        if use_fB:
            plt.plot(dgds_analytic * x_norm, gauss * y_norm, color=color_modes[im], lw=1, ls=":", label=r'$N^>$')
            plt.plot(dgds_analytic * x_norm, nyquist * y_norm, color=color_modes[im], ls="--", lw=1, label='Marco')
        else:
            plt.plot(dgds_analytic * x_norm, gauss * y_norm, color="blue", lw=1, ls=":", label=r'$N^>$')
            plt.plot(dgds_analytic * x_norm, nyquist * y_norm, color="green", ls="--", lw=1, label='Marco')

        lowest_dgd = 0.0
        lw = 1
        ss = 20

        for ix, gvd in enumerate(gvds):
            partial_B2g = np.load(f"results/partial_nlin_gaussian_{mode}{gvd}B2.npy")
            partial_B2n = np.load(f"results/partial_nlin_nyquist_{mode}{gvd}B2.npy")

            if fB_simple_interpolation and use_fB:
                d_lo = dgds_analytic[0]
                d_hi = dgds_analytic[-1]
                dgd_span = d_hi - d_lo
                fitted_data_g = softplus2(dgds_analytic * x_norm, *ps_g[0, :]) * ((dgds_analytic - d_lo) * rcal_hi - (dgds_analytic - d_hi) * rcal_lo) / dgd_span
                fitted_data_n = softplus2(dgds_analytic * x_norm, *ps_n[0, :]) * ((dgds_analytic - d_lo) * rcal_hi - (dgds_analytic - d_hi) * rcal_lo) / dgd_span
                fitted_data_g_flat = softplus2(dgds_analytic * x_norm, *ps_g[0, :])
                fitted_data_n_flat = softplus2(dgds_analytic * x_norm, *ps_n[0, :])
            else:
                fitted_data_g = softplus2(dgds_analytic * x_norm, *ps_g[im, :])
                fitted_data_n = softplus2(dgds_analytic * x_norm, *ps_n[im, :])

            if ix == 0:
                lowest_dgd = partial_B2g[0]
            plt.plot(dgds_analytic * x_norm, fitted_data_g, color="gray", lw=0.6)
            plt.plot(dgds_analytic * x_norm, fitted_data_n, color="gray", lw=0.6, ls="-.")

            if use_fB:
                plt.scatter(dgds_numeric_g * x_norm, partial_B2g * y_norm, label='Gauss.' + str(gvd), color=color_modes[im], marker="x", s=ss, lw=lw)
                plt.scatter(dgds_numeric_n * x_norm, partial_B2n * y_norm, label='Nyq.' + str(gvd), color=color_modes[im], marker="*", s=ss, lw=lw)
            else:
                plt.scatter(dgds_numeric_g * x_norm, partial_B2g * y_norm, label='Gauss.' + str(gvd), color="blue", marker="x", s=ss, lw=lw)
                plt.scatter(dgds_numeric_n * x_norm, partial_B2n * y_norm, label='Nyq.' + str(gvd), color="green", marker="*", s=ss, lw=lw)

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
        logger.info("Saved figure: %s (dpi=%d)", out_pdf, dpi)

        logger.debug("\nPlotting %s case\nGAUSSIAN: fitted_data_g[0]=%s\nNYQUIST:  fitted_data_n[0]=%s",
                     mode, _safe_repr(fitted_data_g[0]), _safe_repr(fitted_data_n[0]))

        # ----------------------------------
        # plotting the error (only without Raman)
        # ----------------------------------
        if not use_fB:
            plt.clf()
            plt.figure(figsize=(3.6, 2))
            lw = 1
            ss = 5

            interp_g = interp1d(dgds_analytic, gauss, kind='cubic', bounds_error=False, fill_value=(gauss[0], gauss[-1]))
            interp_n = interp1d(dgds_analytic, nyquist, kind='cubic', bounds_error=False, fill_value=(nyquist[0], nyquist[-1]))
            interp_a = interp1d(dgds_analytic, analytic_nlin, kind='linear', bounds_error=False, fill_value=(analytic_nlin[0], analytic_nlin[-1]))

            gauss_sampled = interp_g(dgds_numeric_g)
            nyquist_sampled = interp_n(dgds_numeric_n)
            analytic_nlin_sampled = interp_a(dgds_numeric_g)

            for ix, gvd in enumerate(gvds):
                partial_B2g = np.load(f"results/partial_nlin_gaussian_perfect{gvd}B2.npy")
                if ix == 0:
                    lowest_dgd = partial_B2g[0]
                plt.plot(dgds_numeric_g * x_norm, np.abs(partial_B2g - gauss_sampled) / gauss_sampled, color="blue", marker="x", markersize=ss, lw=lw)
                plt.plot(dgds_numeric_g * x_norm, np.abs(partial_B2g - analytic_nlin_sampled) / analytic_nlin_sampled, color="blue", marker="x", markersize=ss, lw=lw, ls="-.")

                plt.plot(dgds_numeric_n * x_norm, np.abs(partial_B2g * y_norm - softplus2(dgds_numeric_g * x_norm, *ps_g[0, :])) / softplus2(dgds_numeric_g * x_norm, *ps_g[0, :]), color="gray", ls=":", lw=lw, marker="x", markerfacecolor='none', markersize=ss)

                partial_B2n = np.load(f"results/partial_nlin_nyquist_perfect{gvd}B2.npy")
                plt.plot(dgds_numeric_n * x_norm, np.abs(partial_B2n - nyquist_sampled) / nyquist_sampled, color="green", marker="*", markersize=ss, lw=lw)
                plt.plot(dgds_numeric_n * x_norm, np.abs(partial_B2n - analytic_nlin_sampled) / analytic_nlin_sampled, color="green", marker="*", markersize=ss, lw=lw, ls="-.")
                plt.plot(dgds_numeric_n * x_norm, np.abs(partial_B2n * y_norm - softplus2(dgds_numeric_n * x_norm, *ps_n[0, :])) / softplus2(dgds_numeric_n * x_norm, *ps_n[0, :]), color="gray", lw=lw, ls=":", marker="*", markerfacecolor='none', markersize=ss)

            plt.xscale('log')
            plt.ylim([-0.05, 0.3])
            plt.xlabel(r'$L/L_W$')
            plt.ylabel(r'$\\varepsilon$')
            plt.tight_layout()
            err_pdf = "media/2-error.pdf"
            plt.savefig(err_pdf, dpi=dpi)
            logger.info("Saved figure: %s (dpi=%d)", err_pdf, dpi)


if __name__ == "__main__":
    logger.info("Starting threshold script with LOG_LEVEL=%s", os.getenv('LOG_LEVEL', getattr(cfg, 'LOG_LEVEL', 'DEBUG')))
    # plot the theoretical figure
    get_nlin_threshold(recompute=False, use_fB=False, fB_simple_interpolation=True)

    # plot the case-study figure
    # get_nlin_threshold(recompute=True, use_fB=True, fB_simple_interpolation=True)

    # Example utilities (disabled by default):
    # logger.info("Raman corrections: %s", _safe_repr(get_raman_corrections()))
    # logger.info("Fit coefficients shapes: %s", _safe_repr([x.shape for x in get_fit_coefficients()]))