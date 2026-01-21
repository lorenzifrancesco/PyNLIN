#!/usr/bin/env python3
"""Plot the undepleted-pump efficiency rho(Omega) around the beta2 zero crossing."""

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
from loguru import logger as lg
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


def _rho_undepleted(mu: np.ndarray, A: complex, alpha_p: float, length: float) -> np.ndarray:
    """Compute rho from the closed-form undepleted-pump expression."""
    try:
        import mpmath as mp
    except ImportError as exc:
        raise ImportError(
            "mpmath is required for complex incomplete Gamma evaluation. "
            "Install it or provide real-valued inputs."
        ) from exc
    mp.mp.dps = 80
    lg.debug("mpmath precision set to {dps} digits.", dps=mp.mp.dps)
    mu_arr = np.asarray(mu, dtype=complex)
    A_mp = mp.mpc(A)
    exp_factor = mp.e ** (2.0 * alpha_p * length)
    lg.debug(
        "rho inputs: alpha_p={alpha_p:.3e}, L={length:.3e}, A={A}, exp_factor={exp_factor}",
        alpha_p=alpha_p,
        length=length,
        A=_fmt_c(complex(A)),
        exp_factor=str(exp_factor),
    )
    out = np.empty(mu_arr.shape, dtype=float)
    fail_count = 0
    for idx, mu_val in np.ndenumerate(mu_arr):
        mu_mp = mp.mpc(mu_val)
        lg.debug(
            "gammainc call idx={} mu={} A={} Aexp={}",
            idx,
            _fmt_c(complex(mu_val)),
            _fmt_c(complex(A_mp)),
            _fmt_c(complex(A_mp * exp_factor)),
        )
        try:
            term = mp.power(A_mp, mu_mp) * (
                mp.gammainc(-mu_mp, A_mp, mp.inf)
                - mp.gammainc(-mu_mp, A_mp * exp_factor, mp.inf)
            )
            rho_val = mp.e ** (A_mp) / (2.0 * alpha_p) * abs(term) ** 2
            out_val = float(mp.re(rho_val))
            out[idx] = out_val
            if not np.isfinite(out_val):
                lg.debug(
                    "non-finite rho at idx={} mu={} rho={}",
                    idx,
                    _fmt_c(complex(mu_val)),
                    _fmt_c(complex(rho_val)),
                )
        except Exception:
            fail_count += 1
            lg.debug("gammainc failed at idx={} mu={}", idx, _fmt_c(complex(mu_val)))
            out[idx] = np.nan
    if fail_count:
        lg.warning("gammainc failed for {} of {} points; values set to NaN.", fail_count, out.size)
    return out


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


def _rho_attenuation(alpha: float, delta_beta: np.ndarray, length: float) -> np.ndarray:
    """Attenuation-only efficiency with full dispersion-based delta_beta."""
    alpha = float(alpha)
    delta_beta = np.asarray(delta_beta, dtype=float)
    denom = 2.0 * alpha - 1j * delta_beta
    numer = 1.0 - np.exp(-(2.0 * alpha - 1j * delta_beta) * length)
    return np.abs(numer / denom) ** 2


def _effective_length(alpha: float, length: float) -> float:
    alpha = float(alpha)
    length = float(length)
    if alpha <= 0.0:
        return length
    return (1.0 - np.exp(-2.0 * alpha * length)) / (2.0 * alpha)


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


def _resolve_pump_power(system: System, override_dbm: float | None) -> float:
    if override_dbm is not None:
        return float(dBm2watt(override_dbm))
    pump_specs = system.pump_specs or []
    if not pump_specs:
        return 0.0
    pump_dbm = np.array([p.power_dbm for p in pump_specs], dtype=float)
    return float(np.sum(dBm2watt(pump_dbm)))


def main() -> None:
    init_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="input/uwb_struct.toml")
    ap.add_argument("--out", type=str, default="media/raman/rho_omega.png")
    ap.add_argument("--omega-min-ghz", type=float, default=10.0)
    ap.add_argument("--omega-max-ghz", type=float, default=100.0)
    ap.add_argument("--omega-points", type=int, default=400)
    ap.add_argument("--omega-offset-ghz", type=float, default=0.0)
    ap.add_argument("--target-wl-nm", type=float, default=1310.0)
    ap.add_argument("--signal-wl-nm", type=float, default=None)
    ap.add_argument("--pump-wl-nm", type=float, default=None)
    ap.add_argument("--pump-power-dbm", type=float, default=None)
    ap.add_argument("--rho-pol", type=float, default=2 / 3)
    ap.add_argument("--logy", action="store_true", help="Use log scale on rho axis.")
    args = ap.parse_args()
    lg.info(
        "rho_omega start: config={} out={} omega_min_ghz={} omega_max_ghz={} omega_points={} omega_offset_ghz={}",
        args.config,
        args.out,
        args.omega_min_ghz,
        args.omega_max_ghz,
        args.omega_points,
        args.omega_offset_ghz,
    )

    system = System.from_toml(args.config)
    fiber = system.fiber
    lg.debug("Loaded system; fiber length={:.3e} m.", fiber.length)

    wl_prof, beta2_prof = fiber.beta2_profile() or (None, None)
    if wl_prof is None or beta2_prof is None:
        raise ValueError("Fiber beta2 profile is required to locate the zero-dispersion point.")

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
    p_p0 = _resolve_pump_power(system, args.pump_power_dbm)
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

    omega_offset = 2.0 * np.pi * args.omega_offset_ghz * 1e9
    omega_center = omega0 + omega_offset
    omega_min = 2.0 * np.pi * args.omega_min_ghz * 1e9
    omega_max = 2.0 * np.pi * args.omega_max_ghz * 1e9
    omega_max = _clip_omega_range(omega_center, omega_max, omega_prof)
    if omega_min > omega_max:
        raise ValueError("omega_min exceeds omega_max after clipping.")
    lg.debug("omega_center={:.3e} omega_min={:.3e} omega_max={:.3e}", omega_center, omega_min, omega_max)
    omega_grid = np.linspace(omega_min, omega_max, args.omega_points)

    beta_c = _beta_eval(omega_center, omega_prof, beta_vals, beta_spline)
    beta_p = _beta_eval(omega_center + omega_grid, omega_prof, beta_vals, beta_spline)
    beta_m = _beta_eval(omega_center - omega_grid, omega_prof, beta_vals, beta_spline)

    delta_beta_3 = beta_p + beta_m - 2.0 * beta_c
    delta_beta_2 = beta_p - beta_c

    mu3 = alpha_s / alpha_p - 1j * delta_beta_3 / (2.0 * alpha_p)
    mu2 = alpha_s / alpha_p - 1j * delta_beta_2 / (2.0 * alpha_p)
    _report_inputs(alpha_s, alpha_p, fiber.length, A, delta_beta_3)
    _log_complex_range("mu3", mu3)
    _log_complex_range("mu2", mu2)
    ratio_3 = np.abs(delta_beta_3) / (2.0 * alpha_s)
    ratio_2 = np.abs(delta_beta_2) / (2.0 * alpha_s)
    lg.info(
        "|Δβ_3|/(2α) range [{:.3e}, {:.3e}]  |Δβ_2|/(2α) range [{:.3e}, {:.3e}]",
        float(np.min(ratio_3)),
        float(np.max(ratio_3)),
        float(np.min(ratio_2)),
        float(np.max(ratio_2)),
    )

    rho3 = _rho_undepleted(mu3, A, alpha_p, fiber.length)
    rho2 = _rho_undepleted(mu2, A, alpha_p, fiber.length)
    rho_att_3 = _rho_attenuation(alpha_s, delta_beta_3, fiber.length)
    rho_att_2 = _rho_attenuation(alpha_s, delta_beta_2, fiber.length)
    l_eff = _effective_length(alpha_s, fiber.length)
    l_eff_sq = l_eff ** 2
    lg.info("L_eff={:.3e} m  L_eff^2={:.3e}", l_eff, l_eff_sq)
    rho_att_3_norm = rho_att_3 / l_eff_sq
    rho_att_2_norm = rho_att_2 / l_eff_sq
    if np.any(~np.isfinite(rho3)):
        lg.warning("rho3 contains non-finite values ({} / {}).", np.sum(~np.isfinite(rho3)), rho3.size)
    if np.any(~np.isfinite(rho2)):
        lg.warning("rho2 contains non-finite values ({} / {}).", np.sum(~np.isfinite(rho2)), rho2.size)

    omega_ghz = omega_grid / (2.0 * np.pi * 1e9)
    lg.info(
        "Omega range for plotting: [{:.3f}, {:.3f}] GHz",
        float(np.min(omega_ghz)),
        float(np.max(omega_ghz)),
    )
    plt.figure()
    if args.logy:
        plt.semilogy(omega_ghz, rho3, label=r"$\mathnormal \rho_3$")
        plt.semilogy(omega_ghz, rho2, label=r"$\mathnormal \rho_2$")
    else:
        plt.plot(omega_ghz, rho3, label=r"$\mathnormal \rho_3$")
        plt.plot(omega_ghz, rho2, label=r"$\mathnormal \rho_2$")
    plt.axvline(0.0, color="0.7", lw=1.0, ls="--")
    plt.xlabel(r"$\mathnormal \Omega$ [GHz]")
    plt.ylabel(r"$\mathnormal \rho$")
    plt.xlim(args.omega_min_ghz, args.omega_max_ghz)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    lg.info("Saved plot to {}", out_path)
    lg.info("alpha_s = {:.3e} dB/m, alpha_p = {:.3e} dB/m", alpha_s_db_m, alpha_p_db_m)
    lg.info("C_R = {:.3e} 1/W/m, Pp0 = {:.3e} W, A = {:.3e}", c_r, p_p0, A)

    fig2, ax2 = plt.subplots()
    ax2.plot(omega_ghz, delta_beta_3, label=r"$\mathnormal \Delta\beta_3$")
    ax2.axhline(0.0, color="0.7", lw=1.0, ls="--")
    ax2.set_xlabel(r"$\mathnormal \Omega$ [GHz]")
    ax2.set_ylabel(r"$\mathnormal \Delta\beta$ [1/m]")
    ax2.set_xlim(args.omega_min_ghz, args.omega_max_ghz)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    out_beta = out_path.with_name(out_path.stem + "_deltabeta3" + out_path.suffix)
    fig2.savefig(out_beta, dpi=200, bbox_inches="tight")
    lg.info("Saved delta-beta plot to {}", out_beta)

    fig_att, axes_att = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes_att = np.atleast_1d(axes_att)
    axes_att[0].plot(omega_ghz, rho_att_3_norm, label=r"$\mathnormal \rho_{3,\mathrm{att}}/L_{\mathrm{eff}}^2$")
    axes_att[0].set_ylabel(r"$\mathnormal \rho/L_{\mathrm{eff}}^2$")
    axes_att[0].legend(fontsize=8)
    axes_att[1].plot(omega_ghz, rho_att_2_norm, label=r"$\mathnormal \rho_{2,\mathrm{att}}/L_{\mathrm{eff}}^2$")
    axes_att[1].set_xlabel(r"$\mathnormal \Omega$ [GHz]")
    axes_att[1].set_ylabel(r"$\mathnormal \rho/L_{\mathrm{eff}}^2$")
    axes_att[1].set_xlim(args.omega_min_ghz, args.omega_max_ghz)
    axes_att[1].legend(fontsize=8)
    fig_att.tight_layout()
    out_att = out_path.with_name(out_path.stem + "_atten" + out_path.suffix)
    fig_att.savefig(out_att, dpi=200, bbox_inches="tight")
    lg.info("Saved attenuation-only rho plot to {}", out_att)

    fig3, ax3 = plt.subplots()
    ax3.plot(omega_ghz, beta_p, label=r"$\mathnormal \beta(\omega+\Omega)$")
    ax3.plot(omega_ghz, beta_m, label=r"$\mathnormal \beta(\omega-\Omega)$")
    ax3.axhline(beta_c, color="0.5", lw=1.0, ls="--", label=r"$\mathnormal \beta(\omega)$")
    ax3.set_xlabel(r"$\mathnormal \Omega$ [GHz]")
    ax3.set_ylabel(r"$\mathnormal \beta$ [1/m]")
    ax3.set_xlim(args.omega_min_ghz, args.omega_max_ghz)
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    out_beta3 = out_path.with_name(out_path.stem + "_beta_threewave" + out_path.suffix)
    fig3.savefig(out_beta3, dpi=200, bbox_inches="tight")
    lg.info("Saved three-wave beta plot to {}", out_beta3)

    fig4, ax4 = plt.subplots()
    ax4.plot(omega_ghz, beta_p, label=r"$\mathnormal \beta(\omega+\Omega)$")
    ax4.axhline(beta_c, color="0.5", lw=1.0, ls="--", label=r"$\mathnormal \beta(\omega)$")
    ax4.set_xlabel(r"$\mathnormal \Omega$ [GHz]")
    ax4.set_ylabel(r"$\mathnormal \beta$ [1/m]")
    ax4.set_xlim(args.omega_min_ghz, args.omega_max_ghz)
    ax4.legend(fontsize=8)
    fig4.tight_layout()
    out_beta2 = out_path.with_name(out_path.stem + "_beta_twowave" + out_path.suffix)
    fig4.savefig(out_beta2, dpi=200, bbox_inches="tight")
    lg.info("Saved two-wave beta plot to {}", out_beta2)


if __name__ == "__main__":
    main()
