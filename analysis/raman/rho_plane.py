#!/usr/bin/env python3
"""Plot undepleted-pump efficiency over the (omega1, omega2) plane."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c0

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.log_init import init_logging
from analysis.raman.rho_utils import effective_length, load_fwm_config, rho_attenuation, rho_undepleted
from loguru import logger as lg
from pynlin.raman.undepleted import effective_raman_gain
from pynlin.system import System
from pynlin.utils import alpha_to_linear, dBm2watt


def _beta_of_omega(
    omega: np.ndarray,
    omega_prof: np.ndarray,
    beta_prof: np.ndarray,
) -> np.ndarray:
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


def _select_beta2_zero(
    wl: np.ndarray,
    beta2: np.ndarray,
    target_wl_nm: float | None,
) -> tuple[float, float]:
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
    wl0 = 2.0 * np.pi * c0 / omega0
    return omega0, wl0


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


def _clip_omega_bounds(
    omega_center: float,
    omega_min: float,
    omega_max: float,
    omega_prof: np.ndarray,
) -> tuple[float, float]:
    omega_min_prof = float(np.min(omega_prof))
    omega_max_prof = float(np.max(omega_prof))
    min_allowed = omega_min_prof - omega_center
    max_allowed = omega_max_prof - omega_center
    if omega_min < min_allowed:
        lg.warning("omega_min clipped ({:.3e} -> {:.3e}) rad/s.", omega_min, min_allowed)
        omega_min = min_allowed
    if omega_max > max_allowed:
        lg.warning("omega_max clipped ({:.3e} -> {:.3e}) rad/s.", omega_max, max_allowed)
        omega_max = max_allowed
    if omega_min >= omega_max:
        raise ValueError("omega_min exceeds omega_max after clipping.")
    return omega_min, omega_max


def _log_axis_phase_match(
    axis_name: str,
    omega_axis: np.ndarray,
    delta_beta_slice: np.ndarray,
) -> None:
    omega_axis = np.asarray(omega_axis, dtype=float)
    delta_beta_slice = np.asarray(delta_beta_slice, dtype=float)
    if omega_axis.size == 0 or delta_beta_slice.size == 0:
        return
    idx0 = int(np.argmin(np.abs(omega_axis)))
    omega0 = float(omega_axis[idx0])
    step = float(np.abs(omega_axis[1] - omega_axis[0])) if omega_axis.size > 1 else 0.0
    f0_thz = omega0 / (2.0 * np.pi * 1e12)
    step_thz = step / (2.0 * np.pi * 1e12) if step > 0.0 else 0.0
    if step_thz > 0.0 and abs(f0_thz) > 0.5 * step_thz:
        lg.warning(
            "{} slice: f=0 not on grid; nearest at {:.6f} THz.",
            axis_name,
            f0_thz,
        )
    min_idx = int(np.argmin(np.abs(delta_beta_slice)))
    min_abs = float(np.abs(delta_beta_slice[min_idx]))
    omega_at_min = float(omega_axis[min_idx]) / (2.0 * np.pi * 1e12)
    sign = np.sign(delta_beta_slice)
    cross = np.where(sign[:-1] * sign[1:] < 0.0)[0]
    if cross.size:
        j = int(cross[0])
        v0 = float(delta_beta_slice[j])
        v1 = float(delta_beta_slice[j + 1])
        if v1 != v0:
            t = -v0 / (v1 - v0)
            omega_cross = float(omega_axis[j] + t * (omega_axis[j + 1] - omega_axis[j]))
        else:
            omega_cross = float(omega_axis[j])
        omega_cross_thz = omega_cross / (2.0 * np.pi * 1e12)
        lg.info(
            "{} slice: sign change at {:.6f} THz; min|Δβ|={:.3e} 1/m at {:.6f} THz.",
            axis_name,
            omega_cross_thz,
            min_abs,
            omega_at_min,
        )
    else:
        lg.info(
            "{} slice: no sign change; min|Δβ|={:.3e} 1/m at {:.6f} THz.",
            axis_name,
            min_abs,
            omega_at_min,
        )


def main() -> None:
    init_logging()
    cfg = load_fwm_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=cfg.config)
    ap.add_argument(
        "--out",
        type=str,
        default=cfg.out if cfg.out is not None else "media/raman/rho_plane.pdf",
    )
    ap.add_argument(
        "--f-min-ghz",
        type=float,
        default=cfg.f_min_ghz if cfg.f_min_ghz is not None else -100.0 / (2.0 * np.pi),
    )
    ap.add_argument(
        "--f-max-ghz",
        type=float,
        default=cfg.f_max_ghz if cfg.f_max_ghz is not None else 100.0 / (2.0 * np.pi),
    )
    ap.add_argument(
        "--f-points",
        type=int,
        default=cfg.f_points if cfg.f_points is not None else 201,
    )
    ap.add_argument("--f-offset-ghz", type=float, default=cfg.f_offset_ghz)
    ap.add_argument("--f-grid-spacing-ghz", type=float, default=cfg.f_grid_spacing_ghz)
    ap.add_argument(
        "--f-both-signs",
        action="store_true",
        help="Force a symmetric f range about 0 (uses max(|min|,|max|)).",
        default=cfg.f_both_signs,
    )
    ap.add_argument("--omega-min-ghz", dest="f_min_ghz", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--omega-max-ghz", dest="f_max_ghz", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--omega-points", dest="f_points", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--omega-offset-ghz", dest="f_offset_ghz", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--omega-both-signs", dest="f_both_signs", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--target-wl-nm", type=float, default=cfg.target_wl_nm)
    ap.add_argument("--center-wl-nm", type=float, default=cfg.center_wl_nm)
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
        "--normalize-leff",
        action=argparse.BooleanOptionalAction,
        default=cfg.normalize_leff if cfg.normalize_leff is not None else True,
        help="Normalize by L_eff^2.",
    )
    ap.add_argument(
        "--db",
        action="store_true",
        default=cfg.db if cfg.db is not None else False,
        help="Plot in dB.",
    )
    ap.add_argument(
        "--db-floor",
        type=float,
        default=cfg.db_floor if cfg.db_floor is not None else 1e-30,
    )
    ap.add_argument("--band", type=str, default=cfg.band)
    ap.add_argument(
        "--animate-band",
        action="store_true",
        default=cfg.animate_band if cfg.animate_band is not None else False,
        help="Save a GIF sweeping the center frequency across the selected band.",
    )
    ap.add_argument("--animate-frames", type=int, default=cfg.animate_frames)
    ap.add_argument("--animate-fps", type=int, default=cfg.animate_fps)
    ap.add_argument("--animate-dpi", type=int, default=cfg.animate_dpi)
    args = ap.parse_args()

    lg.info(
        "rho_plane start: config={} out={} f_min_ghz={} f_max_ghz={} f_points={} f_offset_ghz={} f_grid_spacing_ghz={}",
        args.config,
        args.out,
        args.f_min_ghz,
        args.f_max_ghz,
        args.f_points,
        args.f_offset_ghz,
        args.f_grid_spacing_ghz,
    )
    lg.info("f_both_signs={}", args.f_both_signs)
    lg.info("rho_model={}", args.rho_model)
    band_map = {
        "o": (1260.0, 1360.0),
        "c": (1530.0, 1565.0),
    }
    band_key = (args.band or "O").strip().lower()
    if band_key not in band_map:
        raise ValueError(f"Unsupported band '{args.band}'. Use one of: {', '.join(sorted(band_map))}.")
    band_min_nm, band_max_nm = band_map[band_key]
    band_label = band_key.upper()

    system = System.from_toml(args.config)
    fiber = system.fiber

    wl_prof, beta2_prof = fiber.beta2_profile() or (None, None)
    if wl_prof is None or beta2_prof is None:
        raise ValueError("Fiber beta2 profile is required to locate the zero-dispersion point.")

    omega0, wl0 = _select_beta2_zero(wl_prof, beta2_prof, args.target_wl_nm)
    if args.center_wl_nm is not None:
        signal_wl = args.center_wl_nm * 1e-9 if args.signal_wl_nm is None else args.signal_wl_nm * 1e-9
    else:
        signal_wl = wl0 if args.signal_wl_nm is None else args.signal_wl_nm * 1e-9
    pump_wl = signal_wl
    if args.rho_model == "raman":
        if args.pump_wl_nm is not None:
            pump_wl = args.pump_wl_nm * 1e-9
        else:
            pump_specs = system.pump_specs or []
            if pump_specs:
                pump_wl = float(np.mean([p.wavelength for p in pump_specs]))

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

    alpha_s_db_m = fiber.attenuation_at(signal_wl)
    alpha_s = alpha_to_linear(alpha_s_db_m)
    l_eff = effective_length(alpha_s, fiber.length)
    lg.info("alpha_s={:.3e} Np/m  L_eff={:.3e} m", alpha_s, l_eff)
    A = None
    alpha_p = None
    if args.rho_model == "raman":
        alpha_p_db_m = fiber.attenuation_at(pump_wl)
        alpha_p = alpha_to_linear(alpha_p_db_m)
        if alpha_p <= 0.0:
            raise ValueError("alpha_p must be positive to evaluate the closed-form expression.")
        aeff = fiber.effective_area_at(signal_wl)
        c_r = effective_raman_gain(fiber.raman_coefficient, args.rho_pol, aeff)
        p_p0 = _resolve_pump_power(system, fiber, args.pump_power_dbm)
        # FIXME: temporary power reduction for debugging undepleted vs attenuation-only.
        p_p0 = p_p0 * 0.1
        lg.info("Aeff={:.3e} m^2  C_R={:.3e}  Pp0={:.3e} W", aeff, c_r, p_p0)
        A = -c_r * p_p0 / (2.0 * alpha_p)

    f_min_ghz = args.f_min_ghz
    f_max_ghz = args.f_max_ghz
    if args.f_both_signs:
        f_span = max(abs(f_min_ghz), abs(f_max_ghz))
        if f_span == 0.0:
            raise ValueError("f-both-signs requires a non-zero f range.")
        f_min_ghz = -f_span
        f_max_ghz = f_span
        lg.debug(
            "f-both-signs enabled; using symmetric range [{:.3f}, {:.3f}] GHz.",
            f_min_ghz,
            f_max_ghz,
        )
    f_offset_hz = args.f_offset_ghz * 1e9
    if args.center_wl_nm is not None:
        omega_center = 2.0 * np.pi * c0 / (args.center_wl_nm * 1e-9)
    else:
        omega_center = omega0
    omega_center = omega_center + 2.0 * np.pi * f_offset_hz
    wl_center = 2.0 * np.pi * c0 / omega_center
    f_min_hz = f_min_ghz * 1e9
    f_max_hz = f_max_ghz * 1e9

    def _compute_plane(
        omega_center_val: float,
        omega1_fixed: np.ndarray | None = None,
        omega2_fixed: np.ndarray | None = None,
        *,
        clip_bounds: bool = True,
        extent_mode: str = "absolute",
        normalize_max: bool = False,
    ) -> tuple[np.ndarray, list[float], float]:
        omega_min_val = 2.0 * np.pi * f_min_hz
        omega_max_val = 2.0 * np.pi * f_max_hz
        if omega1_fixed is None or omega2_fixed is None:
            if clip_bounds:
                omega_min_val, omega_max_val = _clip_omega_bounds(
                    omega_center_val, omega_min_val, omega_max_val, omega_prof
                )
            omega1_val = np.linspace(omega_min_val, omega_max_val, args.f_points)
            omega2_val = np.linspace(omega_min_val, omega_max_val, args.f_points)
        else:
            omega1_val = np.asarray(omega1_fixed, dtype=float)
            omega2_val = np.asarray(omega2_fixed, dtype=float)
        omega1_grid_val, omega2_grid_val = np.meshgrid(omega1_val, omega2_val, indexing="xy")
        omega1_abs = omega_center_val + omega1_grid_val
        omega2_abs = omega_center_val + omega2_grid_val
        omega3_abs = omega1_abs + omega2_grid_val
        omega_min_prof = float(np.min(omega_prof))
        omega_max_prof = float(np.max(omega_prof))
        mask = (
            (omega1_abs >= omega_min_prof)
            & (omega1_abs <= omega_max_prof)
            & (omega2_abs >= omega_min_prof)
            & (omega2_abs <= omega_max_prof)
            & (omega3_abs >= omega_min_prof)
            & (omega3_abs <= omega_max_prof)
        )
        beta0_val = _beta_eval(omega_center_val, omega_prof, beta_vals, beta_spline)
        beta1_val = _beta_eval(omega1_abs, omega_prof, beta_vals, beta_spline)
        beta2_val = _beta_eval(omega2_abs, omega_prof, beta_vals, beta_spline)
        beta3_val = _beta_eval(
            omega3_abs,
            omega_prof,
            beta_vals,
            beta_spline,
        )
        delta_beta_val = beta1_val + beta2_val - beta3_val - beta0_val
        delta_beta_val = np.where(mask, delta_beta_val, np.nan)
        if args.rho_model == "attenuation":
            rho_val = rho_attenuation(alpha_s, delta_beta_val, fiber.length)
        else:
            mu_val = alpha_s / alpha_p - 1j * delta_beta_val / (2.0 * alpha_p)
            rho_val = rho_undepleted(mu_val, A, alpha_p, fiber.length, log_points=False)
        rho_val = np.where(mask, rho_val, np.nan)
        if args.normalize_leff:
            rho_val = rho_val / (l_eff ** 2)
        if normalize_max:
            max_val = np.nanmax(rho_val)
            if np.isfinite(max_val) and max_val > 0.0:
                rho_val = rho_val / max_val
        if args.db:
            rho_plot_val = 10.0 * np.log10(np.maximum(rho_val, args.db_floor))
        else:
            rho_plot_val = rho_val
        if extent_mode == "delta":
            f1_thz_val = omega1_val / (2.0 * np.pi * 1e12)
            f2_thz_val = omega2_val / (2.0 * np.pi * 1e12)
            extent_val = [
                float(np.min(f1_thz_val)),
                float(np.max(f1_thz_val)),
                float(np.min(f2_thz_val)),
                float(np.max(f2_thz_val)),
            ]
        else:
            f1_abs_thz_val = (omega_center_val + omega1_val) / (2.0 * np.pi * 1e12)
            f2_abs_thz_val = (omega_center_val + omega2_val) / (2.0 * np.pi * 1e12)
            extent_val = [
                float(np.min(f1_abs_thz_val)),
                float(np.max(f1_abs_thz_val)),
                float(np.min(f2_abs_thz_val)),
                float(np.max(f2_abs_thz_val)),
            ]
        f_center_thz_val = omega_center_val / (2.0 * np.pi * 1e12)
        return rho_plot_val, extent_val, f_center_thz_val

    omega_min_base = 2.0 * np.pi * f_min_hz
    omega_max_base = 2.0 * np.pi * f_max_hz
    omega_min = omega_min_base
    omega_max = omega_max_base
    omega_min, omega_max = _clip_omega_bounds(omega_center, omega_min, omega_max, omega_prof)
    omega1_fixed = np.linspace(omega_min_base, omega_max_base, args.f_points)
    omega2_fixed = np.linspace(omega_min_base, omega_max_base, args.f_points)

    omega1 = np.linspace(omega_min, omega_max, args.f_points)
    omega2 = np.linspace(omega_min, omega_max, args.f_points)
    omega1_grid, omega2_grid = np.meshgrid(omega1, omega2, indexing="xy")

    beta0 = _beta_eval(omega_center, omega_prof, beta_vals, beta_spline)
    beta1 = _beta_eval(omega_center + omega1_grid, omega_prof, beta_vals, beta_spline)
    beta2 = _beta_eval(omega_center + omega2_grid, omega_prof, beta_vals, beta_spline)
    omega3_grid = omega_center + omega1_grid + omega2_grid
    beta3 = _beta_eval(omega3_grid, omega_prof, beta_vals, beta_spline)

    omega_min_prof = float(np.min(omega_prof))
    omega_max_prof = float(np.max(omega_prof))
    out_mask = (omega3_grid < omega_min_prof) | (omega3_grid > omega_max_prof)
    if np.any(out_mask):
        lg.warning(
            "omega1+omega2-omega outside beta profile: {} of {} points.",
            int(np.sum(out_mask)),
            int(out_mask.size),
        )

    # beta2_center = float(np.interp(wl_center, wl_prof, beta2_prof))
    # beta2_scale = 1.0  # FIXME: temporary curvature scaling for sensitivity check.
    # beta2_scaled = beta2_center * beta2_scale
    # delta_beta = -beta2_scaled * omega1_grid * omega2_grid
    # lg.info(
    #     "Using quadratic approximation for Δβ with beta2={:.3e} s^2/m (scaled x{:.1f}).",
    #     beta2_scaled,
    #     beta2_scale,
    # )
    delta_beta = beta1 + beta2 - beta3 - beta0
    i1 = int(np.argmin(np.abs(omega1)))
    i2 = int(np.argmin(np.abs(omega2)))
    _log_axis_phase_match(r"$ f_1=0$", omega2, delta_beta[:, i1])
    _log_axis_phase_match(r"$ f_2=0$", omega1, delta_beta[i2, :])
    if args.rho_model == "attenuation":
        rho = rho_attenuation(alpha_s, delta_beta, fiber.length)
    else:
        mu = alpha_s / alpha_p - 1j * delta_beta / (2.0 * alpha_p)
        rho = rho_undepleted(mu, A, alpha_p, fiber.length, log_points=False)
    if args.normalize_leff:
        rho = rho / (l_eff ** 2)
    if np.any(~np.isfinite(rho)):
        lg.warning("rho contains non-finite values ({} / {}).", np.sum(~np.isfinite(rho)), rho.size)

    base_label = r"$ \rho$"
    if args.db:
        rho_plot = 10.0 * np.log10(np.maximum(rho, args.db_floor))
        cbar_label = base_label + " [dB]"
    else:
        rho_plot = rho
        cbar_label = base_label

    out_path = Path(args.out)
    out_path = out_path.with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.animate_band:
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except Exception as exc:
            raise ImportError("Animation requires matplotlib with Pillow support.") from exc
        wl_centers_nm = np.linspace(band_min_nm, band_max_nm, max(2, args.animate_frames))
        omega_centers = 2.0 * np.pi * c0 / (wl_centers_nm * 1e-9)
        fig_anim, ax_anim = plt.subplots()
        rho0, extent0, f_center0 = _compute_plane(
            float(omega_centers[0]),
            omega1_fixed=omega1_fixed,
            omega2_fixed=omega2_fixed,
            clip_bounds=False,
            extent_mode="delta",
            normalize_max=True,
        )
        im_anim = ax_anim.imshow(
            rho0,
            origin="lower",
            extent=extent0,
            aspect="auto",
            cmap="nipy_spectral",
        )
        ax_anim.set_aspect("equal", adjustable="box")
        ax_anim.set_xlabel(r"$ \Delta f_1$ [THz]")
        ax_anim.set_ylabel(r"$ \Delta f_2$ [THz]")
        txt_anim = ax_anim.text(
            0.02,
            0.98,
            rf"${{f}} = {f_center0:.0f}\,\mathrm{{THz}}$",
            transform=ax_anim.transAxes,
            ha="left",
            va="top",
            color="white",
        )
        cbar_anim = fig_anim.colorbar(im_anim, ax=ax_anim)
        cbar_anim.set_label(cbar_label)
        if args.f_grid_spacing_ghz:
            lg.info("Skipping channel grid overlay during animation.")

        def _update(frame_idx: int):
            rho_frame, extent_frame, f_center_frame = _compute_plane(
                float(omega_centers[frame_idx]),
                omega1_fixed=omega1_fixed,
                omega2_fixed=omega2_fixed,
                clip_bounds=False,
                extent_mode="delta",
                normalize_max=True,
            )
            im_anim.set_data(rho_frame)
            im_anim.set_extent(extent_frame)
            ax_anim.set_xlim(extent_frame[0], extent_frame[1])
            ax_anim.set_ylim(extent_frame[2], extent_frame[3])
            txt_anim.set_text(rf"${{f}} = {f_center_frame:.0f}\,\mathrm{{THz}}$")
            if args.db:
                finite_frame = rho_frame[np.isfinite(rho_frame)]
                if finite_frame.size:
                    im_anim.set_clim(np.min(finite_frame), 0.0)
            else:
                im_anim.set_clim(0.0, 1.0)
            return im_anim, txt_anim

        fig_anim.tight_layout()
        out_gif = out_path.with_name(out_path.stem + f"_{band_label.lower()}band.gif")
        anim = FuncAnimation(fig_anim, _update, frames=omega_centers.size, blit=False)
        anim.save(
            out_gif,
            writer=PillowWriter(fps=max(1, args.animate_fps)),
            dpi=int(max(50, args.animate_dpi)),
        )
        lg.info("Saved {}-band rho plane animation to {}", band_label, out_gif)

    f1_abs_thz = (omega_center + omega1) / (2.0 * np.pi * 1e12)
    f2_abs_thz = (omega_center + omega2) / (2.0 * np.pi * 1e12)
    extent = [
        float(np.min(f1_abs_thz)),
        float(np.max(f1_abs_thz)),
        float(np.min(f2_abs_thz)),
        float(np.max(f2_abs_thz)),
    ]
    f1_min_thz, f1_max_thz, f2_min_thz, f2_max_thz = extent
    fig, ax = plt.subplots()
    im = ax.imshow(rho_plot, origin="lower", extent=extent, aspect="auto", cmap="nipy_spectral")
    if args.f_grid_spacing_ghz:
        spacing_ghz = abs(args.f_grid_spacing_ghz)
        if spacing_ghz > 0.0:
            spacing_thz = spacing_ghz / 1000.0
            f1_min = min(f1_min_thz, f1_max_thz)
            f1_max = max(f1_min_thz, f1_max_thz)
            f2_min = min(f2_min_thz, f2_max_thz)
            f2_max = max(f2_min_thz, f2_max_thz)
            f1_start = np.ceil(f1_min / spacing_thz) * spacing_thz
            f2_start = np.ceil(f2_min / spacing_thz) * spacing_thz
            f1_vals = np.arange(f1_start, f1_max + 0.5 * spacing_thz, spacing_thz)
            f2_vals = np.arange(f2_start, f2_max + 0.5 * spacing_thz, spacing_thz)
            n_points = int(f1_vals.size * f2_vals.size)
            if n_points > 200_000:
                factor = int(np.ceil((n_points / 200_000) ** 0.5))
                spacing_thz = spacing_thz * factor
                f1_start = np.ceil(f1_min / spacing_thz) * spacing_thz
                f2_start = np.ceil(f2_min / spacing_thz) * spacing_thz
                f1_vals = np.arange(f1_start, f1_max + 0.5 * spacing_thz, spacing_thz)
                f2_vals = np.arange(f2_start, f2_max + 0.5 * spacing_thz, spacing_thz)
                n_points = int(f1_vals.size * f2_vals.size)
                lg.warning(
                    "Grid spacing too fine; increased to {:.3f} GHz ({} points).",
                    spacing_thz * 1000.0,
                    n_points,
                )
            f1_grid, f2_grid = np.meshgrid(f1_vals, f2_vals, indexing="xy")
            ax.scatter(
                f1_grid.ravel(),
                f2_grid.ravel(),
                s=0.6,
                c="white",
                alpha=0.2,
                linewidths=0.0,
                zorder=3,
            )
            lg.info("Plotted channel grid with {:.3f} GHz spacing ({} points).", spacing_thz * 1000.0, n_points)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$ f_1$ [THz]")
    ax.set_ylabel(r"$ f_2$ [THz]")
    f_center_thz = omega_center / (2.0 * np.pi * 1e12)
    lg.info("Center absolute frequency at f={:.0f} THz.", f_center_thz)
    # line_color = "cyan"
    # line_width = 0.3
    # line_style = ":"
    # if f1_min_thz <= delta_f_thz <= f1_max_thz:
    #     ax.axvline(delta_f_thz, color=line_color, lw=line_width, ls=line_style)
    # if f2_min_thz <= delta_f_thz <= f2_max_thz:
    #     ax.axhline(delta_f_thz, color=line_color, lw=line_width, ls=line_style)
    # x_line = np.linspace(f1_min_thz, f1_max_thz, 400)
    # y_line = delta_f_thz - x_line
    # mask = (y_line >= f2_min_thz) & (y_line <= f2_max_thz)
    # if np.any(mask):
    #     ax.plot(x_line[mask], y_line[mask], color=line_color, lw=line_width, ls=line_style)
    ax.text(
        0.02,
        0.98,
        rf"${{f}} = {f_center_thz:.0f}\,\mathrm{{THz}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    lg.info("Saved rho plane plot to {}", out_path)

    omega1_axis = omega1
    omega2_axis = omega2
    omega_min_prof = float(np.min(omega_prof))
    omega_max_prof = float(np.max(omega_prof))

    def _delta_beta_line(omega1_line: np.ndarray, omega2_line: np.ndarray) -> np.ndarray:
        beta1_line = _beta_eval(omega_center + omega1_line, omega_prof, beta_vals, beta_spline)
        beta2_line = _beta_eval(omega_center + omega2_line, omega_prof, beta_vals, beta_spline)
        beta3_line = _beta_eval(
            omega_center + omega1_line + omega2_line,
            omega_prof,
            beta_vals,
            beta_spline,
        )
        delta_line = beta1_line + beta2_line - beta3_line - beta0
        mask = (
            (omega_center + omega1_line >= omega_min_prof)
            & (omega_center + omega1_line <= omega_max_prof)
            & (omega_center + omega2_line >= omega_min_prof)
            & (omega_center + omega2_line <= omega_max_prof)
            & (omega_center + omega1_line + omega2_line >= omega_min_prof)
            & (omega_center + omega1_line + omega2_line <= omega_max_prof)
        )
        delta_line = np.where(mask, delta_line, np.nan)
        return delta_line

    def _rho_line(delta_line: np.ndarray) -> np.ndarray:
        if args.rho_model == "attenuation":
            rho_line = rho_attenuation(alpha_s, delta_line, fiber.length)
        else:
            mu_line = alpha_s / alpha_p - 1j * delta_line / (2.0 * alpha_p)
            rho_line = rho_undepleted(mu_line, A, alpha_p, fiber.length, log_points=False)
        if args.normalize_leff:
            rho_line = rho_line / (l_eff ** 2)
        if args.db:
            rho_line = 10.0 * np.log10(np.maximum(rho_line, args.db_floor))
        return rho_line

    omega_diag = omega1_axis
    omega2_diag = -omega_diag
    delta_diag = _delta_beta_line(omega_diag, omega2_diag)
    rho_diag = _rho_line(delta_diag)
    valid = np.isfinite(delta_diag)
    if np.any(valid):
        delta_diag_valid = delta_diag[valid]
        omega_diag_valid = omega_diag[valid]
        min_idx = int(np.argmin(np.abs(delta_diag_valid)))
        min_abs = float(np.abs(delta_diag_valid[min_idx]))
        f_at_min = float(omega_diag_valid[min_idx] / (2.0 * np.pi * 1e12))
        sign = np.sign(delta_diag_valid)
        cross = np.where(sign[:-1] * sign[1:] < 0.0)[0]
        if cross.size:
            j = int(cross[0])
            v0 = float(delta_diag_valid[j])
            v1 = float(delta_diag_valid[j + 1])
            if v1 != v0:
                t = -v0 / (v1 - v0)
                omega_cross = float(omega_diag_valid[j] + t * (omega_diag_valid[j + 1] - omega_diag_valid[j]))
            else:
                omega_cross = float(omega_diag_valid[j])
            f_cross = omega_cross / (2.0 * np.pi * 1e12)
            lg.info(
                "Diagonal (f1+f2=0): sign change at {:.6f} THz; min|Δβ|={:.3e} 1/m at {:.6f} THz.",
                f_cross,
                min_abs,
                f_at_min,
            )
        else:
            lg.info(
                "Diagonal (f1+f2=0): no sign change; min|Δβ|={:.3e} 1/m at {:.6f} THz.",
                min_abs,
                f_at_min,
            )
        f_diag_thz = omega_diag_valid / (2.0 * np.pi * 1e12)
        f1_thz = omega1_axis / (2.0 * np.pi * 1e12)
        f2_thz = omega2_axis / (2.0 * np.pi * 1e12)
        rho_f1 = _rho_line(_delta_beta_line(np.zeros_like(omega2_axis), omega2_axis))
        rho_f2 = _rho_line(_delta_beta_line(omega1_axis, np.zeros_like(omega1_axis)))
        fig_diag, ax_diag = plt.subplots()
        ax_diag.plot(f_diag_thz, rho_diag[valid], label=r"$ f_1+f_2=0$")
        ax_diag.plot(f2_thz, rho_f1, label=r"$ f_1=0$")
        ax_diag.plot(f1_thz, rho_f2, label=r"$ f_2=0$")
        ax_diag.set_xlabel(r"$ f$ [THz]")
        ax_diag.set_ylabel(cbar_label)
        ax_diag.legend(fontsize=8)
        fig_diag.tight_layout()
        out_diag = out_path.with_name(out_path.stem + "_rho_diag_sum0" + out_path.suffix)
        fig_diag.savefig(out_diag, dpi=200, bbox_inches="tight")
        lg.info("Saved diagonal rho plot to {}", out_diag)

        f_min_hz_abs = abs(args.f_min_ghz) * 1e9
        f_max_hz_abs = abs(args.f_max_ghz) * 1e9
        max_allowed = min(omega_center - omega_min_prof, omega_max_prof - omega_center)
        omega_max_check = min(2.0 * np.pi * f_max_hz_abs, max_allowed)
        f_max_hz_abs = omega_max_check / (2.0 * np.pi)
        if f_min_hz_abs > f_max_hz_abs:
            lg.warning("Comparison grid clipped; f_min exceeds f_max after bounds.")
        f_pos_check = np.linspace(f_min_hz_abs, f_max_hz_abs, args.f_points)
        if args.f_both_signs:
            if f_min_hz_abs == 0.0:
                f_neg_check = -f_pos_check[1:][::-1]
            else:
                f_neg_check = -f_pos_check[::-1]
            f_grid_check = np.concatenate([f_neg_check, f_pos_check])
        else:
            f_grid_check = f_pos_check
        omega_grid_check = 2.0 * np.pi * f_grid_check
        delta_threewave = (
            _beta_eval(omega_center + omega_grid_check, omega_prof, beta_vals, beta_spline)
            + _beta_eval(omega_center - omega_grid_check, omega_prof, beta_vals, beta_spline)
            - 2.0 * beta0
        )
        mask_three = (
            (omega_center + omega_grid_check >= omega_min_prof)
            & (omega_center + omega_grid_check <= omega_max_prof)
            & (omega_center - omega_grid_check >= omega_min_prof)
            & (omega_center - omega_grid_check <= omega_max_prof)
        )
        delta_threewave = np.where(mask_three, delta_threewave, np.nan)
        diag_interp = np.interp(
            omega_grid_check,
            omega_diag_valid,
            delta_diag_valid,
        )
        mask_interp = (omega_grid_check >= omega_diag_valid[0]) & (omega_grid_check <= omega_diag_valid[-1])
        diag_interp = np.where(mask_interp, diag_interp, np.nan)
        cmp_mask = np.isfinite(diag_interp) & np.isfinite(delta_threewave)
        if np.any(cmp_mask):
            diff = diag_interp[cmp_mask] - delta_threewave[cmp_mask]
            max_abs = float(np.max(np.abs(diff)))
            denom = np.maximum(np.abs(delta_threewave[cmp_mask]), 1e-30)
            max_rel = float(np.max(np.abs(diff) / denom))
            lg.info(
                "Δβ compare (plane diag vs omega 3-wave): max_abs={:.3e} 1/m, max_rel={:.3e}.",
                max_abs,
                max_rel,
            )

            f_check_thz = omega_grid_check / (2.0 * np.pi * 1e12)
            fig_cmp, ax_cmp = plt.subplots()
            ax_cmp.plot(f_diag_thz, delta_diag_valid, label=r"$ \Delta\beta$ (diag)")
            ax_cmp.plot(f_check_thz, delta_threewave, ls="--", label=r"$ \Delta\beta$ (3-wave)")
            ax_cmp.set_xlabel(r"$ f$ [THz]")
            ax_cmp.set_ylabel(r"$ \Delta\beta$ [1/m]")
            ax_cmp.legend(fontsize=8)
            fig_cmp.tight_layout()
            out_cmp = out_path.with_name(out_path.stem + "_deltabeta_diag_compare" + out_path.suffix)
            fig_cmp.savefig(out_cmp, dpi=200, bbox_inches="tight")
            lg.info("Saved delta-beta comparison plot to {}", out_cmp)

    try:
        import plotly.graph_objects as go
    except ImportError:
        lg.warning("plotly is not installed; skipping 3D rho plot.")
        return

    f1_thz = omega1 / (2.0 * np.pi * 1e12)
    f2_thz = omega2 / (2.0 * np.pi * 1e12)
    surface = go.Surface(
        x=f1_thz,
        y=f2_thz,
        z=rho_plot,
        colorscale="Magma",
        colorbar=dict(title=cbar_label),
    )
    fig3d = go.Figure(data=[surface])
    fig3d.update_layout(
        scene=dict(
            xaxis_title=r"$f_1$ [THz]",
            yaxis_title=r"$f_2$ [THz]",
            zaxis_title=cbar_label,
        )
    )
    out_html = out_path.with_name(out_path.stem + "_3d.html")
    fig3d.write_html(out_html, include_plotlyjs="cdn")
    lg.info("Saved 3D rho plot to {}", out_html)


if __name__ == "__main__":
    main()
