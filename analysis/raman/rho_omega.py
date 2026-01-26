#!/usr/bin/env python3
"""Plot the undepleted-pump efficiency rho(Omega) around the beta2 zero crossing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.constants import c as c0

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.log_init import init_logging
from loguru import logger as lg
from analysis.raman.rho_utils import (
    effective_length,
    load_fwm_config,
    rho_attenuation,
    rho_undepleted,
)
from pynlin.raman.undepleted import effective_raman_gain
from pynlin.system import System
from pynlin.utils import alpha_to_linear, dBm2watt


def _fmt_c(val: complex) -> str:
    return f"{val.real:.3e}{val.imag:+.3e}j"


def _log_complex_range(name: str, values: np.ndarray) -> None:
    vals = np.asarray(values, dtype=complex)
    lg.debug(
        "{name} real range [{rmin:.3e}, {rmax:.3e}] imag range [{imin:.3e}, {imax:.3e}]",
        name=name,
        rmin=float(np.min(vals.real)),
        rmax=float(np.max(vals.real)),
        imin=float(np.min(vals.imag)),
        imax=float(np.max(vals.imag)),
    )


def _beta_of_omega(
    omega: np.ndarray,
    omega_prof: np.ndarray,
    beta_prof: np.ndarray,
) -> np.ndarray:
    """Return beta(omega) via interpolation of the fiber profile."""
    omega = np.asarray(omega, dtype=float)
    omega_prof = np.asarray(omega_prof, dtype=float)
    beta_prof = np.asarray(beta_prof, dtype=float)
    order = np.argsort(omega_prof)
    return np.interp(omega, omega_prof[order], beta_prof[order])


def _beta_eval(
    omega: np.ndarray | float,
    omega_prof: np.ndarray,
    beta_prof: np.ndarray,
    spline,
) -> np.ndarray:
    if spline is not None:
        return np.asarray(spline(omega), dtype=float)
    return _beta_of_omega(np.asarray(omega, dtype=float), omega_prof, beta_prof)


def _beta1_at(
    omega_center: float,
    wl_center: float,
    fiber,
    omega_prof: np.ndarray,
    beta_prof: np.ndarray,
    beta_spline,
) -> float:
    """Return beta1=d beta/d omega evaluated at the center frequency."""
    beta1 = None
    if hasattr(fiber, "beta1_at"):
        try:
            beta1 = fiber.beta1_at(wl_center)
        except Exception as exc:
            lg.warning("beta1_at failed; falling back to beta derivative: {}", exc)
    if beta1 is None:
        beta1 = getattr(fiber, "beta1", None)
        if beta1 is not None:
            beta1 = float(beta1)
    if beta1 is None and beta_spline is not None:
        try:
            beta1 = float(beta_spline.derivative()(omega_center))
        except Exception as exc:
            lg.warning("beta spline derivative failed; falling back to finite diff: {}", exc)
    if beta1 is None:
        order = np.argsort(omega_prof)
        omega_sorted = omega_prof[order]
        beta_sorted = beta_prof[order]
        deriv = np.gradient(beta_sorted, omega_sorted)
        beta1 = float(np.interp(omega_center, omega_sorted, deriv))
    return float(beta1)


def _signal_launch_power(system: System, signal_wl: float) -> float:
    freqs = system.wdm.frequency_grid()
    if freqs.size == 0:
        raise ValueError("No WDM channels available to infer signal launch power.")
    wl_grid = c0 / freqs
    if hasattr(system, "_initial_signal_powers_dbm"):
        powers_dbm = system._initial_signal_powers_dbm()
        idx = int(np.argmin(np.abs(wl_grid - signal_wl)))
        p_dbm = float(powers_dbm[idx])
        lg.info(
            "Signal launch power from nearest channel: {:.3f} dBm at {:.2f} nm.",
            p_dbm,
            wl_grid[idx] * 1e9,
        )
        return float(dBm2watt(p_dbm))
    launch_dbm = system.launch_power if system.launch_power is not None else -5.0
    lg.info("Signal launch power from system.launch_power: {:.3f} dBm.", launch_dbm)
    return float(dBm2watt(launch_dbm))


def _pump_profile(
    system: System,
    fiber,
    z: np.ndarray,
    pump_wl: float,
    override_dbm: float | None,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    pump_specs = system.pump_specs or []
    total = np.zeros_like(z)
    if override_dbm is not None or not pump_specs:
        direction = -1
        if pump_specs:
            direction = -1 if np.any([p.direction < 0 for p in pump_specs]) else 1
        p_launch = float(dBm2watt(override_dbm if override_dbm is not None else 0.0))
        alpha_p_db_m = fiber.attenuation_at(pump_wl)
        alpha_p = alpha_to_linear(alpha_p_db_m)
        if direction < 0:
            total = p_launch * np.exp(-alpha_p * (fiber.length - z))
        else:
            total = p_launch * np.exp(-alpha_p * z)
        lg.info(
            "Using single pump profile (override): dir={} P_launch={:.3e} W.",
            direction,
            p_launch,
        )
        return total

    for pump in pump_specs:
        p_launch = float(dBm2watt(pump.power_dbm))
        alpha_p_db_m = fiber.attenuation_at(pump.wavelength)
        alpha_p = alpha_to_linear(alpha_p_db_m)
        if pump.direction < 0:
            profile = p_launch * np.exp(-alpha_p * (fiber.length - z))
        else:
            profile = p_launch * np.exp(-alpha_p * z)
        total += profile
    return total


def _clip_omega_range(omega_center: float, omega_max: float, omega_prof: np.ndarray) -> float:
    omega_min = float(np.min(omega_prof))
    omega_max_prof = float(np.max(omega_prof))
    max_allowed = min(omega_center - omega_min, omega_max_prof - omega_center)
    if max_allowed <= 0.0:
        raise ValueError("omega_center falls outside the beta-profile frequency range.")
    if omega_max > max_allowed:
        lg.warning(
            "omega_max clipped to fit beta-profile range ({old:.3e} -> {new:.3e} rad/s).",
            old=omega_max,
            new=max_allowed,
        )
        omega_max = max_allowed
    return omega_max


def _report_inputs(alpha_s: float, alpha_p: float, length: float, A: complex, delta_beta: np.ndarray) -> None:
    exp_arg = 2.0 * alpha_p * length
    abs_A = abs(A)
    lg.info(
        "alpha_s={alpha_s:.3e}  alpha_p={alpha_p:.3e}  L={length:.3e}  exp_arg={exp_arg:.3e}",
        alpha_s=alpha_s,
        alpha_p=alpha_p,
        length=length,
        exp_arg=exp_arg,
    )
    lg.info(
        "A={A}  |A|={abs_A:.3e}  delta_beta range [{db_min:.3e}, {db_max:.3e}]",
        A=_fmt_c(complex(A)),
        abs_A=abs_A,
        db_min=float(np.min(delta_beta)),
        db_max=float(np.max(delta_beta)),
    )
    if not np.isfinite(abs_A) or abs_A == 0.0:
        lg.warning("A is non-finite or zero; closed-form expression may be ill-conditioned.")
    if exp_arg > 200.0:
        lg.warning("exp(2*alpha_p*L) is extremely large; numerical stability may be limited.")


def _select_beta2_zero(
    wl: np.ndarray,
    beta2: np.ndarray,
    target_wl_nm: float | None,
) -> tuple[float, float, float]:
    """Return (omega0, wl0, beta3) for the beta2 zero nearest target wavelength."""
    wl = np.asarray(wl, dtype=float)
    beta2 = np.asarray(beta2, dtype=float)
    omega = 2.0 * np.pi * c0 / wl

    sign = np.sign(beta2)
    idx = np.where(sign[:-1] * sign[1:] < 0.0)[0]
    if idx.size == 0:
        raise ValueError("No beta2 sign change found in the fiber profile.")

    if target_wl_nm is not None:
        target_wl = target_wl_nm * 1e-9
        mid_wl = 0.5 * (wl[idx] + wl[idx + 1])
        pick = int(np.argmin(np.abs(mid_wl - target_wl)))
    else:
        pick = 0

    i = idx[pick]
    omega0 = omega[i] - beta2[i] * (omega[i + 1] - omega[i]) / (beta2[i + 1] - beta2[i])
    beta3 = (beta2[i + 1] - beta2[i]) / (omega[i + 1] - omega[i])
    wl0 = 2.0 * np.pi * c0 / omega0
    return omega0, wl0, beta3


def _resolve_pump_power(system: System, fiber, override_dbm: float | None) -> float:
    if override_dbm is not None:
        lg.info("Pump power override: {:.3f} dBm ({:.3e} W).", override_dbm, dBm2watt(override_dbm))
        return float(dBm2watt(override_dbm))
    pump_specs = system.pump_specs or []
    if not pump_specs:
        return 0.0
    pump_p0 = []
    backward_count = 0
    for pump in pump_specs:
        p_launch = float(dBm2watt(pump.power_dbm))
        direction = getattr(pump, "direction", 1)
        p_corr = p_launch
        corr_factor = 1.0
        if direction < 0:
            alpha_p_db_m = fiber.attenuation_at(pump.wavelength)
            alpha_p = alpha_to_linear(alpha_p_db_m)
            corr_factor = float(np.exp(-alpha_p * fiber.length))
            p_corr = p_launch * corr_factor
            backward_count += 1
        pump_p0.append(p_corr)
        lg.info(
            "Pump {:.1f} nm dir={} P_launch={:.3e} W ({:.2f} dBm) Pp0={:.3e} W corr={:.3e}",
            pump.wavelength * 1e9,
            direction,
            p_launch,
            pump.power_dbm,
            p_corr,
            corr_factor,
        )
    total = float(np.sum(pump_p0))
    if backward_count:
        lg.info(
            "Applied backward pump correction to {} pumps; total Pp0={:.3e} W.",
            backward_count,
            total,
        )
    return total


def main() -> None:
    init_logging()
    cfg = load_fwm_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=cfg.config)
    ap.add_argument(
        "--out",
        type=str,
        default=cfg.out if cfg.out is not None else "media/raman/rho_omega.pdf",
    )
    ap.add_argument(
        "--f-min-ghz",
        type=float,
        default=cfg.f_min_ghz if cfg.f_min_ghz is not None else 10.0 / (2.0 * np.pi),
    )
    ap.add_argument(
        "--f-max-ghz",
        type=float,
        default=cfg.f_max_ghz if cfg.f_max_ghz is not None else 100.0 / (2.0 * np.pi),
    )
    ap.add_argument(
        "--f-points",
        type=int,
        default=cfg.f_points if cfg.f_points is not None else 400,
    )
    ap.add_argument("--f-offset-ghz", type=float, default=cfg.f_offset_ghz)
    ap.add_argument(
        "--f-both-signs",
        action="store_true",
        help="Include negative f values (symmetric about 0).",
        default=cfg.f_both_signs,
    )
    ap.add_argument("--omega-min-ghz", dest="f_min_ghz", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--omega-max-ghz", dest="f_max_ghz", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--omega-points", dest="f_points", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--omega-offset-ghz", dest="f_offset_ghz", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--omega-both-signs", dest="f_both_signs", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--target-wl-nm", type=float, default=cfg.target_wl_nm)
    ap.add_argument("--signal-wl-nm", type=float, default=cfg.signal_wl_nm)
    ap.add_argument("--pump-wl-nm", type=float, default=cfg.pump_wl_nm)
    ap.add_argument("--pump-power-dbm", type=float, default=cfg.pump_power_dbm)
    ap.add_argument("--rho-pol", type=float, default=cfg.rho_pol)
    ap.add_argument(
        "--rho-model",
        choices=("raman", "attenuation"),
        default=cfg.rho_model,
        help="Select rho model: undepleted Raman or attenuation-only.",
    )
    ap.add_argument(
        "--logy",
        action="store_true",
        default=cfg.logy if cfg.logy is not None else False,
        help="Use log scale on rho axis.",
    )
    ap.add_argument(
        "--band",
        type=str,
        default=cfg.band,
        help="Spectral band for contour maps (e.g., O, C).",
    )
    args = ap.parse_args()
    band_map = {
        "o": (1260.0, 1360.0),
        "c": (1530.0, 1565.0),
    }
    band_key = (args.band or "O").strip().lower()
    if band_key not in band_map:
        raise ValueError(f"Unsupported band '{args.band}'. Use one of: {', '.join(sorted(band_map))}.")
    band_min_nm, band_max_nm = band_map[band_key]
    band_label = band_key.upper()
    band_center_nm = 0.5 * (band_min_nm + band_max_nm)
    lg.info(
        "rho_omega start: config={} out={} f_min_ghz={} f_max_ghz={} f_points={} f_offset_ghz={}",
        args.config,
        args.out,
        args.f_min_ghz,
        args.f_max_ghz,
        args.f_points,
        args.f_offset_ghz,
    )
    lg.info("f_both_signs={}", args.f_both_signs)
    lg.info("rho_model={}", args.rho_model)

    system = System.from_toml(args.config)
    fiber = system.fiber
    lg.debug("Loaded system; fiber length={:.3e} m.", fiber.length)

    wl_prof, beta2_prof = fiber.beta2_profile() or (None, None)
    if wl_prof is None or beta2_prof is None:
        raise ValueError("Fiber beta2 profile is required to locate the zero-dispersion point.")

    if band_key == "c" and args.target_wl_nm == cfg.target_wl_nm:
        args.target_wl_nm = band_center_nm
        lg.info("C-band selected; using default target_wl_nm={:.1f} nm.", band_center_nm)

    omega0, wl0, beta3 = _select_beta2_zero(wl_prof, beta2_prof, args.target_wl_nm)
    lg.info("ZDW near {:.2f} nm (omega0={:.3e} rad/s); beta3={:.3e} s^3/m.", wl0 * 1e9, omega0, beta3)

    signal_wl = wl0 if args.signal_wl_nm is None else args.signal_wl_nm * 1e-9
    if args.pump_wl_nm is not None:
        pump_wl = args.pump_wl_nm * 1e-9
    else:
        pump_specs = system.pump_specs or []
        if pump_specs:
            pump_wl = float(np.mean([p.wavelength for p in pump_specs]))
        else:
            pump_wl = signal_wl
    lg.debug("signal_wl={:.3e} m pump_wl={:.3e} m", signal_wl, pump_wl)

    alpha_s_db_m = fiber.attenuation_at(signal_wl)
    alpha_p_db_m = fiber.attenuation_at(pump_wl)
    alpha_s = alpha_to_linear(alpha_s_db_m)
    alpha_p = alpha_to_linear(alpha_p_db_m)
    if alpha_p <= 0.0:
        raise ValueError("alpha_p must be positive to evaluate the closed-form expression.")
    lg.debug("alpha_s_db_m={:.3e} alpha_p_db_m={:.3e}", alpha_s_db_m, alpha_p_db_m)

    aeff = fiber.effective_area_at(signal_wl)
    c_r = effective_raman_gain(fiber.raman_coefficient, args.rho_pol, aeff)
    p_p0 = _resolve_pump_power(system, fiber, args.pump_power_dbm)
    lg.info("Aeff={:.3e} m^2  C_R={:.3e}  Pp0={:.3e} W", aeff, c_r, p_p0)

    A = -c_r * p_p0 / (2.0 * alpha_p)

    beta_profile = fiber.beta_profile()
    if beta_profile is None:
        raise ValueError("Fiber beta profile is required to evaluate DeltaBeta.")
    wl_beta, beta_vals = beta_profile
    omega_prof = 2.0 * np.pi * c0 / np.asarray(wl_beta, dtype=float)
    beta_spline = None
    if hasattr(fiber, "beta_spline_omega"):
        try:
            beta_spline = fiber.beta_spline_omega()
            lg.info("Using beta spline for dispersion evaluation.")
        except Exception as exc:
            lg.warning("Beta spline unavailable; falling back to linear interpolation: {}", exc)

    f_min_ghz = abs(args.f_min_ghz)
    f_max_ghz = abs(args.f_max_ghz)
    if args.f_min_ghz < 0.0 or args.f_max_ghz < 0.0:
        lg.warning("f-min/f-max should be positive; using absolute values.")
    f_offset_hz = args.f_offset_ghz * 1e9
    omega_offset = 2.0 * np.pi * f_offset_hz
    omega_center = omega0 + omega_offset
    wl_center = 2.0 * np.pi * c0 / omega_center
    f_center_thz = omega_center / (2.0 * np.pi * 1e12)
    lg.info("Center absolute frequency: {:.6f} THz.", f_center_thz)
    f_min_hz = f_min_ghz * 1e9
    f_max_hz = f_max_ghz * 1e9
    omega_min = 2.0 * np.pi * f_min_hz
    omega_max = 2.0 * np.pi * f_max_hz
    omega_max = _clip_omega_range(omega_center, omega_max, omega_prof)
    f_max_hz = omega_max / (2.0 * np.pi)
    f_max_ghz = f_max_hz / 1e9
    if f_min_hz > f_max_hz:
        raise ValueError("f_min exceeds f_max after clipping.")
    lg.debug("omega_center={:.3e} omega_min={:.3e} omega_max={:.3e}", omega_center, omega_min, omega_max)
    f_pos = np.linspace(f_min_hz, f_max_hz, args.f_points)
    if args.f_both_signs:
        if f_min_hz == 0.0:
            f_neg = -f_pos[1:][::-1]
        else:
            f_neg = -f_pos[::-1]
        f_grid = np.concatenate([f_neg, f_pos])
        lg.debug("f grid includes negative values; total points={}.", f_grid.size)
    else:
        f_grid = f_pos
    omega_grid = 2.0 * np.pi * f_grid

    beta_c = _beta_eval(omega_center, omega_prof, beta_vals, beta_spline)
    beta_p = _beta_eval(omega_center + omega_grid, omega_prof, beta_vals, beta_spline)
    beta_m = _beta_eval(omega_center - omega_grid, omega_prof, beta_vals, beta_spline)
    beta1 = _beta1_at(omega_center, wl_center, fiber, omega_prof, beta_vals, beta_spline)
    if not np.isfinite(beta1):
        lg.warning("beta1 is non-finite; skipping linear beta reference.")
        beta1 = None
    else:
        lg.info("beta1 at omega_center={:.3e} rad/s (wl={:.2f} nm): {:.3e} s/m.", omega_center, wl_center * 1e9, beta1)
        beta_line_p = beta_c + beta1 * omega_grid
        beta_line_m = beta_c - beta1 * omega_grid

    delta_beta_3_exact = beta_p + beta_m - 2.0 * beta_c
    beta2_center = float(np.interp(wl_center, wl_prof, beta2_prof))
    delta_beta_3_quad = beta2_center * omega_grid ** 2
    delta_beta_2 = np.zeros_like(delta_beta_3_exact)

    wl_plus = 2.0 * np.pi * c0 / (omega_center + omega_grid)
    wl_minus = 2.0 * np.pi * c0 / (omega_center - omega_grid)
    alpha_plus_db_m = fiber.loss_profile(wl_plus)
    alpha_minus_db_m = fiber.loss_profile(wl_minus)
    alpha_eff_db_m = 0.5 * (alpha_plus_db_m + alpha_minus_db_m)
    alpha_plus = alpha_to_linear(alpha_plus_db_m)
    alpha_minus = alpha_to_linear(alpha_minus_db_m)
    alpha_eff = 0.5 * (alpha_plus + alpha_minus)
    ratio_alpha = np.where(
        alpha_eff > 0.0,
        np.abs(delta_beta_3_exact) / (2.0 * alpha_eff),
        np.nan,
    )

    mu3 = alpha_s / alpha_p - 1j * delta_beta_3_exact / (2.0 * alpha_p)
    mu2 = alpha_s / alpha_p
    _report_inputs(alpha_s, alpha_p, fiber.length, A, delta_beta_3_exact)
    _log_complex_range("mu3", mu3)
    _log_complex_range("mu2", mu2)
    ratio_3 = np.abs(delta_beta_3_exact) / (2.0 * alpha_s)
    ratio_2 = np.abs(delta_beta_2) / (2.0 * alpha_s)
    lg.info(
        "|Δβ_3|/(2α) range [{:.3e}, {:.3e}]  |Δβ_2|/(2α) range [{:.3e}, {:.3e}]",
        float(np.min(ratio_3)),
        float(np.max(ratio_3)),
        float(np.min(ratio_2)),
        float(np.max(ratio_2)),
    )

    rho_att_3 = rho_attenuation(alpha_s, delta_beta_3_exact, fiber.length)
    rho_att_2 = rho_attenuation(alpha_s, delta_beta_2, fiber.length)
    l_eff = effective_length(alpha_s, fiber.length)
    l_eff_sq = l_eff ** 2
    lg.info("L_eff={:.3e} m  L_eff^2={:.3e}", l_eff, l_eff_sq)
    rho_att_3_norm = rho_att_3 / l_eff_sq
    rho_att_2_norm = rho_att_2 / l_eff_sq
    if args.rho_model == "attenuation":
        rho3 = rho_att_3_norm
        rho2 = rho_att_2_norm
        lg.info("Using attenuation-only rho (normalized by L_eff^2).")
    else:
        rho3 = rho_undepleted(mu3, A, alpha_p, fiber.length, log_points=True)
        rho2 = rho_undepleted(mu2, A, alpha_p, fiber.length, log_points=True)
        if l_eff_sq > 0.0:
            rho3 = rho3 / l_eff_sq
            rho2 = rho2 / l_eff_sq
            lg.info("Normalized rho by L_eff^2.")
    if np.any(~np.isfinite(rho3)):
        lg.warning("rho3 contains non-finite values ({} / {}).", np.sum(~np.isfinite(rho3)), rho3.size)
    if np.any(~np.isfinite(rho2)):
        lg.warning("rho2 contains non-finite values ({} / {}).", np.sum(~np.isfinite(rho2)), rho2.size)

    z_grid = np.linspace(0.0, fiber.length, 600)
    pump_profile = _pump_profile(system, fiber, z_grid, pump_wl, args.pump_power_dbm)
    integrand = c_r * pump_profile
    gain_int = np.zeros_like(z_grid)
    if z_grid.size > 1:
        dz = np.diff(z_grid)
        trap = 0.5 * (integrand[1:] + integrand[:-1]) * dz
        gain_int[1:] = np.cumsum(trap)
    signal_in = _signal_launch_power(system, signal_wl)
    signal_profile = signal_in * np.exp(-alpha_s * z_grid + gain_int)
    lg.info(
        "Signal profile: P_in={:.3e} W, min={:.3e} W, max={:.3e} W.",
        signal_in,
        float(np.min(signal_profile)),
        float(np.max(signal_profile)),
    )

    f_ghz = f_grid / 1e9
    f_thz = f_grid / 1e12
    lg.info(
        "f range for plotting: [{:.6f}, {:.6f}] THz",
        float(np.min(f_thz)),
        float(np.max(f_thz)),
    )
    if args.f_both_signs:
        f_xlim_min_ghz = -f_max_ghz
        f_xlim_max_ghz = f_max_ghz
        f_xlim_min_thz = -f_max_ghz / 1000.0
        f_xlim_max_thz = f_max_ghz / 1000.0
    else:
        f_xlim_min_ghz = f_min_ghz
        f_xlim_max_ghz = f_max_ghz
        f_xlim_min_thz = f_min_ghz / 1000.0
        f_xlim_max_thz = f_max_ghz / 1000.0
    plt.figure()
    if args.rho_model == "attenuation":
        label3 = r"$\mathnormal \rho_{3\mathrm{-waves},\mathrm{att}}$"
    else:
        label3 = r"$\mathnormal \rho_{3\mathrm{-waves}}$"
    if args.logy:
        plt.semilogy(f_ghz, rho3, label=label3)
    else:
        plt.plot(f_ghz, rho3, label=label3)
    plt.axvline(0.0, color="0.7", lw=1.0, ls="--")
    plt.xlabel(r"$\mathnormal \Delta f_1 = \Delta f_2$ [GHz]")
    plt.ylabel(r"$\mathnormal \rho$")
    plt.xlim(f_xlim_min_ghz, f_xlim_max_ghz)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path = out_path.with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    lg.info("Saved plot to {}", out_path)
    lg.info("alpha_s = {:.3e} dB/m, alpha_p = {:.3e} dB/m", alpha_s_db_m, alpha_p_db_m)
    lg.info("C_R = {:.3e} 1/W/m, Pp0 = {:.3e} W, A = {:.3e}", c_r, p_p0, A)

    fig2, (ax2, ax2_alpha, ax2_ratio) = plt.subplots(
        nrows=3,
        sharex=False,
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]},
        figsize=(3.5, 4.2),
    )
    ax2.plot(f_thz, delta_beta_3_exact, label=r"$\mathnormal \Delta\beta_{3\mathrm{-waves}}$ (exact)")
    ax2.plot(
        f_thz,
        delta_beta_3_quad,
        lw=1.0,
        ls="--",
        label=r"$\mathnormal \Delta\beta_{3\mathrm{-waves}}$ ($\beta_2\Omega^2$)",
    )
    ax2.axhline(0.0, color="0.7", lw=1.0, ls="--")
    ax2.set_xlabel(r"$\mathnormal f$ [THz]")
    ax2.set_ylabel(r"$\mathnormal \Delta\beta$ [1/m]")
    ax2.set_xlim(f_xlim_min_thz, f_xlim_max_thz)
    ax2.legend(fontsize=8)

    ax2_alpha.plot(f_thz, alpha_eff_db_m)
    ax2_alpha.set_xlabel(r"$\mathnormal f$ [THz]")
    ax2_alpha.set_ylabel(r"$\mathnormal \alpha$ [dB/m]")
    ax2_alpha.set_xlim(f_xlim_min_thz, f_xlim_max_thz)

    ratio_mask = np.isfinite(f_thz) & np.isfinite(ratio_alpha)
    if np.any(ratio_mask):
        ax2_ratio.plot(f_thz[ratio_mask], ratio_alpha[ratio_mask])
    ax2_ratio.set_xlabel(r"$\mathnormal f$ [THz]")
    ax2_ratio.set_ylabel(r"$|\mathnormal \Delta\beta|/(2\alpha(f))$")
    ax2_ratio.set_xlim(f_xlim_min_thz, f_xlim_max_thz)
    fig2.tight_layout()
    out_beta = out_path.with_name(out_path.stem + "_deltabeta3-waves" + out_path.suffix)
    fig2.savefig(out_beta, dpi=200, bbox_inches="tight")
    lg.info("Saved delta-beta plot to {}", out_beta)

    beta1_zdw = _beta1_at(omega0, wl0, fiber, omega_prof, beta_vals, beta_spline)
    if not np.isfinite(beta1_zdw):
        lg.warning("beta1 at ZDW is non-finite; skipping linear O-band comparison.")
        beta1_zdw = None
    beta_zdw = _beta_eval(omega0, omega_prof, beta_vals, beta_spline)

    wl_nm = np.asarray(wl_beta, dtype=float) * 1e9
    beta_vals_nm = np.asarray(beta_vals, dtype=float)
    order = np.argsort(wl_nm)
    wl_nm = wl_nm[order]
    beta_vals_nm = beta_vals_nm[order]
    ob_mask = (wl_nm >= band_min_nm) & (wl_nm <= band_max_nm)
    if np.count_nonzero(ob_mask) < 2:
        lg.warning("{}-band range not covered by beta profile; plotting full profile instead.", band_label)
        wl_plot = wl_nm
        beta_real = beta_vals_nm
        x_limits = None
    else:
        wl_plot = wl_nm[ob_mask]
        beta_real = beta_vals_nm[ob_mask]
        x_limits = (band_min_nm, band_max_nm)

    freq_plot = c0 / (wl_plot * 1e-9)
    omega_plot = 2.0 * np.pi * freq_plot
    if beta1_zdw is not None:
        beta_lin = beta_zdw + beta1_zdw * (omega_plot - omega0)
        denom = np.where(beta_real != 0.0, beta_real, np.nan)
        rel_diff = (beta_real - beta_lin) / denom
    else:
        beta_lin = None
        rel_diff = None

    freq_thz = freq_plot * 1e-12
    order_f = np.argsort(freq_thz)
    freq_thz = freq_thz[order_f]
    beta_real = beta_real[order_f]
    if beta_lin is not None:
        beta_lin = beta_lin[order_f]
    if rel_diff is not None:
        rel_diff = rel_diff[order_f]
    f0_thz = c0 / wl0 * 1e-12
    if x_limits is not None:
        f_lo = c0 / (band_max_nm * 1e-9) * 1e-12
        f_hi = c0 / (band_min_nm * 1e-9) * 1e-12
        x_limits = (f_lo, f_hi)

    fig_ob, (ax_ob, ax_ob_diff) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    ax_ob.plot(freq_thz, beta_real, label=r"$\mathnormal \beta(\omega)$")
    if beta_lin is not None:
        ax_ob.plot(freq_thz, beta_lin, lw=1.0, ls="--", label=r"$\mathnormal \beta_{\mathrm{lin}}$")
    ax_ob.axvline(f0_thz, color="0.5", lw=1.0, ls=":", label="ZDW")
    chan_freqs_thz = None
    try:
        chan_freqs_hz = system.wdm.frequency_grid()
        chan_freqs_thz = np.asarray(chan_freqs_hz, dtype=float) * 1e-12
    except Exception as exc:
        lg.warning("Failed to load WDM channel grid: {}", exc)
    if chan_freqs_thz is not None and chan_freqs_thz.size:
        f_min = float(np.min(freq_thz))
        f_max = float(np.max(freq_thz))
        ch_mask = (chan_freqs_thz >= f_min) & (chan_freqs_thz <= f_max)
        ch_in = chan_freqs_thz[ch_mask]
        if ch_in.size:
            beta_ch = np.interp(ch_in, freq_thz, beta_real)
            ax_ob.plot(
                ch_in,
                beta_ch,
                marker="o",
                ls="none",
                ms=2.5,
                mfc="none",
                mec="0.2",
                mew=0.5,
                alpha=0.1,
                label="Channels",
            )
        ax_ob.text(
            0.98,
            0.95,
            f"Channels: {ch_in.size}",
            transform=ax_ob.transAxes,
            va="top",
            ha="right",
            fontsize=8,
        )
        lg.info("O-band plot includes {} channels.", ch_in.size)
    ax_ob.set_ylabel(r"$\mathnormal \beta$ [1/m]")
    if x_limits is not None:
        ax_ob.set_xlim(*x_limits)
    ax_ob.legend(fontsize=8)

    if rel_diff is not None:
        ax_ob_diff.plot(freq_thz, rel_diff)
        ax_ob_diff.axhline(0.0, color="0.7", lw=1.0, ls="--")
    ax_ob_diff.set_xlabel("Frequency [THz]")
    ax_ob_diff.set_ylabel(r"$(\beta-\beta_{\mathrm{lin}})/\beta$")
    fig_ob.tight_layout()
    out_ob = out_path.with_name(out_path.stem + f"_{band_label.lower()}band_beta" + out_path.suffix)
    fig_ob.savefig(out_ob, dpi=200, bbox_inches="tight")
    lg.info("Saved {}-band beta plot to {}", band_label, out_ob)

    beta1_profile = fiber.beta1_profile()
    beta2_profile = fiber.beta2_profile()
    beta1_prof = None
    beta2_prof = None
    if beta1_profile is not None:
        wl_b1, b1_vals = beta1_profile
        beta1_prof = np.interp(wl_beta, wl_b1, b1_vals)
    if beta2_profile is not None:
        wl_b2, b2_vals = beta2_profile
        beta2_prof = np.interp(wl_beta, wl_b2, b2_vals)

    omega_full = 2.0 * np.pi * c0 / np.asarray(wl_beta, dtype=float)
    order_full = np.argsort(omega_full)
    omega_full = omega_full[order_full]
    beta_full = np.asarray(beta_vals, dtype=float)[order_full]
    beta1_num = np.gradient(beta_full, omega_full)
    beta2_num = np.gradient(beta1_num, omega_full)
    freq_full_thz = omega_full / (2.0 * np.pi * 1e12)

    if beta2_profile is not None:
        wl_b2, b2_vals = beta2_profile
        freq_b2_thz = (c0 / np.asarray(wl_b2, dtype=float)) * 1e-12
        order_b2 = np.argsort(freq_b2_thz)
        freq_b2_thz = freq_b2_thz[order_b2]
        b2_vals = np.asarray(b2_vals, dtype=float)[order_b2]
    else:
        freq_b2_thz = freq_full_thz
        b2_vals = beta2_num

    f_lo = c0 / (band_max_nm * 1e-9) * 1e-12
    f_hi = c0 / (band_min_nm * 1e-9) * 1e-12
    f_ref_thz = np.linspace(f_lo, f_hi, 250)
    f_other_thz = np.linspace(f_lo, f_hi, 250)
    beta2_ref = np.interp(f_ref_thz, freq_b2_thz, b2_vals)
    f_ref_grid, f_other_grid = np.meshgrid(f_ref_thz, f_other_thz, indexing="xy")
    delta_f_hz = (f_other_grid - f_ref_grid) * 1e12
    z_beta2 = beta2_ref[np.newaxis, :] * delta_f_hz**2

    cmap_ob = "coolwarm"
    fig_b2, ax_b2map = plt.subplots()
    finite_b2 = z_beta2[np.isfinite(z_beta2)]
    norm_b2 = None
    if finite_b2.size:
        abs_max_b2 = float(np.max(np.abs(finite_b2)))
        if abs_max_b2 > 0.0:
            norm_b2 = TwoSlopeNorm(vmin=-abs_max_b2, vcenter=0.0, vmax=abs_max_b2)
    cf = ax_b2map.contourf(
        f_ref_grid,
        f_other_grid,
        z_beta2,
        levels=80,
        cmap=cmap_ob,
        norm=norm_b2,
    )
    ax_b2map.set_xlabel(r"$\mathnormal f_{\mathrm{ref}}$ [THz]")
    ax_b2map.set_ylabel(r"$\mathnormal f_{\mathrm{other}}$ [THz]")
    ax_b2map.set_title(r"$\mathnormal \beta_2(f_{\mathrm{ref}})\,(f_{\mathrm{other}}-f_{\mathrm{ref}})^2$")
    ax_b2map.set_aspect("equal", adjustable="box")
    fig_b2.colorbar(cf, ax=ax_b2map, label=r"$\mathnormal \beta_2(\Delta f)^2$ [1/m]")
    fig_b2.tight_layout()
    out_b2 = out_path.with_name(out_path.stem + f"_{band_label.lower()}band_beta2_deltaf" + out_path.suffix)
    fig_b2.savefig(out_b2, dpi=200, bbox_inches="tight")
    lg.info("Saved {}-band beta2 delta-f contour to {}", band_label, out_b2)

    omega_ref_grid = 2.0 * np.pi * f_ref_grid * 1e12
    omega_other_grid = 2.0 * np.pi * f_other_grid * 1e12
    omega_sym_grid = 2.0 * omega_ref_grid - omega_other_grid
    beta_ref_grid = _beta_eval(omega_ref_grid, omega_prof, beta_vals, beta_spline)
    beta_other_grid = _beta_eval(omega_other_grid, omega_prof, beta_vals, beta_spline)
    beta_sym_grid = _beta_eval(omega_sym_grid, omega_prof, beta_vals, beta_spline)
    omega_min_prof = float(np.min(omega_prof))
    omega_max_prof = float(np.max(omega_prof))
    mask_beta = (
        (omega_ref_grid >= omega_min_prof)
        & (omega_ref_grid <= omega_max_prof)
        & (omega_other_grid >= omega_min_prof)
        & (omega_other_grid <= omega_max_prof)
        & (omega_sym_grid >= omega_min_prof)
        & (omega_sym_grid <= omega_max_prof)
    )
    delta_beta_exact_map = beta_other_grid + beta_sym_grid - 2.0 * beta_ref_grid
    delta_beta_exact_map = np.where(mask_beta, delta_beta_exact_map, np.nan)

    fig_db3, ax_db3 = plt.subplots()
    finite_db3 = delta_beta_exact_map[np.isfinite(delta_beta_exact_map)]
    norm_db3 = None
    if finite_db3.size:
        abs_max = float(np.max(np.abs(finite_db3)))
        if abs_max > 0.0:
            norm_db3 = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    cf_db3 = ax_db3.contourf(
        f_ref_grid,
        f_other_grid,
        delta_beta_exact_map,
        levels=80,
        cmap=cmap_ob,
        norm=norm_db3,
    )
    ax_db3.set_xlabel(r"$\mathnormal f_{\mathrm{ref}}$ [THz]")
    ax_db3.set_ylabel(r"$\mathnormal f_{\mathrm{other}}$ [THz]")
    ax_db3.set_title(r"$\mathnormal \Delta\beta_{3\mathrm{-waves}}$ (exact)")
    ax_db3.set_aspect("equal", adjustable="box")
    fig_db3.colorbar(cf_db3, ax=ax_db3, label=r"$\mathnormal \Delta\beta$ [1/m]")
    fig_db3.tight_layout()
    out_db3 = out_path.with_name(out_path.stem + f"_{band_label.lower()}band_deltabeta3_exact" + out_path.suffix)
    fig_db3.savefig(out_db3, dpi=200, bbox_inches="tight")
    lg.info("Saved {}-band delta-beta3 exact contour to {}", band_label, out_db3)

    fig_diag, (ax_b1, ax_b2) = plt.subplots(nrows=2, sharex=True)
    ax_b1.plot(freq_full_thz, beta1_num * 1e12, label=r"$\mathnormal d\beta/d\omega$")
    if beta1_prof is not None:
        ax_b1.plot(
            freq_full_thz,
            np.asarray(beta1_prof, dtype=float)[order_full] * 1e12,
            lw=1.0,
            ls="--",
            label=r"$\mathnormal \beta_1$ profile",
        )
    ax_b1.set_ylabel(r"$\mathnormal \beta_1$ [ps/m]")
    ax_b1.legend(fontsize=8)

    ax_b2.plot(freq_full_thz, beta2_num * 1e27, label=r"$\mathnormal d^2\beta/d\omega^2$")
    if beta2_prof is not None:
        ax_b2.plot(
            freq_full_thz,
            np.asarray(beta2_prof, dtype=float)[order_full] * 1e27,
            lw=1.0,
            ls="--",
            label=r"$\mathnormal \beta_2$ profile",
        )
    ax_b2.axhline(0.0, color="0.7", lw=1.0, ls="--")
    ax_b2.set_xlabel("Frequency [THz]")
    ax_b2.set_ylabel(r"$\mathnormal \beta_2$ [ps$^2$/km]")
    ax_b2.legend(fontsize=8)
    fig_diag.tight_layout()
    out_diag = out_path.with_name(out_path.stem + "_dispersion_diag" + out_path.suffix)
    fig_diag.savefig(out_diag, dpi=200, bbox_inches="tight")
    lg.info("Saved dispersion diagnostic plot to {}", out_diag)

    fig_sig, ax_sig = plt.subplots()
    ax_sig.plot(z_grid, signal_profile)
    ax_sig.set_xlabel(r"$\mathnormal z$ [m]")
    ax_sig.set_ylabel(r"$\mathnormal P_s$ [W]")
    fig_sig.tight_layout()
    out_sig = out_path.with_name(out_path.stem + "_signal_profile" + out_path.suffix)
    fig_sig.savefig(out_sig, dpi=200, bbox_inches="tight")
    lg.info("Saved signal profile plot to {}", out_sig)


if __name__ == "__main__":
    main()
