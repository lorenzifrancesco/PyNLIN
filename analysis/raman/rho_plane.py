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
from analysis.raman.rho_utils import effective_length, rho_attenuation, rho_undepleted
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


def main() -> None:
    init_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="input/uwb_struct.toml")
    ap.add_argument("--out", type=str, default="media/raman/rho_plane.png")
    ap.add_argument("--omega-min-ghz", type=float, default=-100.0)
    ap.add_argument("--omega-max-ghz", type=float, default=100.0)
    ap.add_argument("--omega-points", type=int, default=201)
    ap.add_argument("--omega-offset-ghz", type=float, default=0.0)
    ap.add_argument(
        "--omega-both-signs",
        action="store_true",
        help="Force a symmetric Omega range about 0 (uses max(|min|,|max|)).",
    )
    ap.add_argument("--target-wl-nm", type=float, default=1310.0)
    ap.add_argument("--signal-wl-nm", type=float, default=None)
    ap.add_argument("--pump-wl-nm", type=float, default=None)
    ap.add_argument("--pump-power-dbm", type=float, default=None)
    ap.add_argument("--rho-pol", type=float, default=2 / 3)
    ap.add_argument(
        "--rho-model",
        choices=("raman", "attenuation"),
        default="raman",
        help="Select rho model: undepleted Raman or attenuation-only.",
    )
    ap.add_argument(
        "--normalize-leff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize by L_eff^2.",
    )
    ap.add_argument("--db", action="store_true", help="Plot in dB.")
    ap.add_argument("--db-floor", type=float, default=1e-30)
    args = ap.parse_args()

    lg.info(
        "rho_plane start: config={} out={} omega_min_ghz={} omega_max_ghz={} omega_points={} omega_offset_ghz={}",
        args.config,
        args.out,
        args.omega_min_ghz,
        args.omega_max_ghz,
        args.omega_points,
        args.omega_offset_ghz,
    )
    lg.info("omega_both_signs={}", args.omega_both_signs)
    lg.info("rho_model={}", args.rho_model)

    system = System.from_toml(args.config)
    fiber = system.fiber

    wl_prof, beta2_prof = fiber.beta2_profile() or (None, None)
    if wl_prof is None or beta2_prof is None:
        raise ValueError("Fiber beta2 profile is required to locate the zero-dispersion point.")

    omega0, wl0 = _select_beta2_zero(wl_prof, beta2_prof, args.target_wl_nm)
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
        # FIXME: temporary low pump power for debugging undepleted vs attenuation-only.
        p_p0 = float(dBm2watt(-100.0))
        lg.info("Aeff={:.3e} m^2  C_R={:.3e}  Pp0={:.3e} W", aeff, c_r, p_p0)
        A = -c_r * p_p0 / (2.0 * alpha_p)

    omega_min_ghz = args.omega_min_ghz
    omega_max_ghz = args.omega_max_ghz
    if args.omega_both_signs:
        omega_span = max(abs(omega_min_ghz), abs(omega_max_ghz))
        if omega_span == 0.0:
            raise ValueError("omega-both-signs requires a non-zero omega range.")
        omega_min_ghz = -omega_span
        omega_max_ghz = omega_span
        lg.debug(
            "omega-both-signs enabled; using symmetric range [{:.3f}, {:.3f}] GHz.",
            omega_min_ghz,
            omega_max_ghz,
        )
    omega_offset = args.omega_offset_ghz * 1e9
    omega_center = omega0 + omega_offset
    omega_min = omega_min_ghz * 1e9
    omega_max = omega_max_ghz * 1e9
    omega_min, omega_max = _clip_omega_bounds(omega_center, omega_min, omega_max, omega_prof)

    omega1 = np.linspace(omega_min, omega_max, args.omega_points)
    omega2 = np.linspace(omega_min, omega_max, args.omega_points)
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

    delta_beta = beta1 + beta2 - beta3 - beta0
    if args.rho_model == "attenuation":
        rho = rho_attenuation(alpha_s, delta_beta, fiber.length)
    else:
        mu = alpha_s / alpha_p - 1j * delta_beta / (2.0 * alpha_p)
        rho = rho_undepleted(mu, A, alpha_p, fiber.length, log_points=False)
    if args.normalize_leff:
        rho = rho / (l_eff ** 2)
    if np.any(~np.isfinite(rho)):
        lg.warning("rho contains non-finite values ({} / {}).", np.sum(~np.isfinite(rho)), rho.size)

    if args.normalize_leff:
        base_label = r"$\mathnormal \rho/L_{\mathrm{eff}}^2$"
    else:
        base_label = r"$\mathnormal \rho$"
    if args.db:
        rho_plot = 10.0 * np.log10(np.maximum(rho, args.db_floor))
        cbar_label = base_label + " [dB]"
    else:
        rho_plot = rho
        cbar_label = base_label

    extent = [
        omega1.min() / 1e9,
        omega1.max() / 1e9,
        omega2.min() / 1e9,
        omega2.max() / 1e9,
    ]
    omega1_min_ghz, omega1_max_ghz, omega2_min_ghz, omega2_max_ghz = extent
    fig, ax = plt.subplots()
    im = ax.imshow(rho_plot, origin="lower", extent=extent, aspect="auto", cmap="magma")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\mathnormal \Omega_1$ [GHz]")
    ax.set_ylabel(r"$\mathnormal \Omega_2$ [GHz]")
    delta_omega_ghz = (omega0 - omega_center) / 1e9
    lg.info("ZDW overlay offsets at Omega={:.3f} GHz.", delta_omega_ghz)
    line_color = "cyan"
    line_width = 0.4
    if omega1_min_ghz <= delta_omega_ghz <= omega1_max_ghz:
        ax.axvline(delta_omega_ghz, color=line_color, lw=line_width, ls="--")
    if omega2_min_ghz <= delta_omega_ghz <= omega2_max_ghz:
        ax.axhline(delta_omega_ghz, color=line_color, lw=line_width, ls="--")
    x_line = np.linspace(omega1_min_ghz, omega1_max_ghz, 400)
    y_line = delta_omega_ghz - x_line
    mask = (y_line >= omega2_min_ghz) & (y_line <= omega2_max_ghz)
    if np.any(mask):
        ax.plot(x_line[mask], y_line[mask], color=line_color, lw=line_width, ls="--")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    lg.info("Saved rho plane plot to {}", out_path)


if __name__ == "__main__":
    main()
