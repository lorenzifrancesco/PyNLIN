#!/usr/bin/env python3
"""Fit an undepleted-pump model to Raman power profiles (O-band focus)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.constants import speed_of_light as c0
from scipy.optimize import least_squares

from pynlin.fiber_data.response import scale_eq39
from pynlin.raman.solvers import SMWidebandRamanAmplifier
from pynlin.raman.undepleted import (
    pump_power_coprop,
    pump_power_counterprop,
    signal_power_undepleted_counterprop,
    signal_power_undepleted_coprop,
)
from pynlin.system import System
from pynlin.utils import alpha_to_linear, dBm2watt, watt2dBm
from pynlin.wdm import IrregularWDM


def _load_profile(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        payload = {key: data[key] for key in data.files}
    else:
        payload = data.item()
    return payload


def _select_band_indices(system: System, signal_wavelengths: np.ndarray, band: str) -> np.ndarray:
    wdm = system.wdm
    if isinstance(wdm, IrregularWDM):
        slc = wdm._band_slices.get(band)
        if slc is not None:
            return np.arange(slc.start, slc.stop)
    if band.upper() == "O":
        wl_nm = signal_wavelengths * 1e9
        return np.where((wl_nm >= 1260.0) & (wl_nm <= 1360.0))[0]
    return np.array([], dtype=int)


def _select_highest_freq_pumps(pump_wavelengths: np.ndarray, count: int) -> np.ndarray:
    freqs = c0 / pump_wavelengths
    idx = np.argsort(freqs)[-count:]
    return np.sort(idx)


def _build_aeff_of_f(fiber):
    profile = getattr(fiber, "_effective_area_profile", None)
    if profile is None:
        aeff_const = float(getattr(fiber, "effective_area", 1.0))

        def _aeff(freq_hz):
            freq = np.asarray(freq_hz, dtype=float)
            return np.full_like(freq, aeff_const, dtype=float)

        return _aeff

    wl, values = profile
    wl = np.asarray(wl, dtype=float)
    values = np.asarray(values, dtype=float)

    def _aeff(freq_hz):
        freq = np.asarray(freq_hz, dtype=float)
        wl_q = c0 / freq
        return np.interp(wl_q, wl, values)

    return _aeff


def _alpha_db_per_km(alpha_np_per_m: float) -> float:
    return alpha_np_per_m * 10.0 / np.log(10.0) * 1e3


def _to_dbm(power_w: np.ndarray | float, floor_w: float = 1e-30) -> np.ndarray:
    power = np.maximum(np.asarray(power_w, dtype=float), floor_w)
    return watt2dBm(power)


def _exp_approx_profile(
    z: np.ndarray,
    length: float,
    a: float,
    a2: float,
    b2: float,
    p0: float,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return p0 * (np.exp(-a * z) + b2 * np.exp(-a2 * (length - z)))


def _map_pumps_to_profile(
    pump_wl_cfg: np.ndarray,
    pump_wl_profile: np.ndarray,
    tol_nm: float = 0.05,
) -> np.ndarray:
    if pump_wl_profile is None:
        raise ValueError("Profile file missing pump_wavelengths.")
    pump_wl_profile = np.asarray(pump_wl_profile, dtype=float)
    idx = []
    for wl in pump_wl_cfg:
        diffs = np.abs(pump_wl_profile - wl)
        j = int(np.argmin(diffs))
        if diffs[j] * 1e9 > tol_nm:
            raise ValueError(
                f"Pump wavelength {wl*1e9:.3f} nm not found in profile (min diff {diffs[j]*1e9:.3f} nm)."
            )
        idx.append(j)
    return np.array(idx, dtype=int)


def _pump_powers_to_watt(pump_powers: np.ndarray) -> np.ndarray:
    pump_powers = np.asarray(pump_powers, dtype=float)
    if pump_powers.size == 0:
        return pump_powers
    if np.any(pump_powers < 0.0) or np.nanmax(pump_powers) > 1.0:
        # Heuristic to handle payloads storing pump powers in dBm.
        pump_powers = dBm2watt(pump_powers)
    if pump_powers.ndim > 1:
        pump_powers = np.sum(pump_powers, axis=1)
    return pump_powers


def _compute_geff_avg(
    fiber,
    pump_wavelengths: np.ndarray,
    signal_wavelengths: np.ndarray,
    rho_pol: float,
    pump_ref_wl: float,
) -> np.ndarray:
    pump_wavelengths = np.asarray(pump_wavelengths, dtype=float)
    signal_wavelengths = np.asarray(signal_wavelengths, dtype=float)
    nu_p = c0 / pump_wavelengths
    nu_s = c0 / signal_wavelengths

    aeff_of_f = _build_aeff_of_f(fiber)

    amp = SMWidebandRamanAmplifier(fiber)
    resolution = amp._gain_resolution(np.concatenate([nu_p, nu_s]))
    max_shift = np.max(np.abs(nu_p[:, None] - nu_s[None, :]))
    shifts, gains = amp._build_gain_spectrum(max_shift, resolution)

    nu_p_ref = c0 / pump_ref_wl
    Aeff_p_ref = float(np.atleast_1d(aeff_of_f(nu_p_ref))[0])
    Aeff_s = aeff_of_f(nu_s)
    Aps_ref = 0.5 * (Aeff_p_ref + Aeff_s)

    g_eff_pumps = []
    for wl_p, nu_p_i in zip(pump_wavelengths, nu_p):
        dnu = nu_p_i - nu_s
        gain_shape = np.interp(np.abs(dnu), shifts, gains, left=0.0, right=0.0)
        g_ref_material = fiber.raman_coefficient * gain_shape
        g_ref_eff = np.where(Aps_ref > 0, g_ref_material / Aps_ref, 0.0)
        g_scaled, _, valid = scale_eq39(
            dnu_hz=np.abs(dnu),
            g_ref=g_ref_eff,
            lam_p_ref_m=pump_ref_wl,
            lam_p_new_m=wl_p,
            Aeff_of_f=aeff_of_f,
        )
        g_scaled = np.where(valid & (dnu > 0.0), g_scaled, 0.0)
        g_eff_pumps.append(g_scaled)

    if not g_eff_pumps:
        return np.zeros_like(signal_wavelengths)

    g_eff_avg = np.mean(np.stack(g_eff_pumps, axis=0), axis=0)
    return rho_pol * g_eff_avg


def _fit_channel(
    z: np.ndarray,
    signal: np.ndarray,
    g_eff: float,
    length: float,
    model: str,
    alpha_s: float,
    alpha_p: float,
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    fit_gr: bool,
    fit_log: bool,
    exp_approx: bool,
) -> tuple[np.ndarray, float]:
    if exp_approx:
        if fit_gr:
            raise ValueError("exp-approximation is incompatible with --fit-gr.")
    else:
        if model not in ("counter", "co"):
            raise ValueError(f"Unknown model '{model}'")

    def _residual(params: np.ndarray) -> np.ndarray:
        if exp_approx:
            a, a2, b2, p0 = params
            pred = _exp_approx_profile(z, length, a, a2, b2, p0)
        else:
            if fit_gr:
                pump_in, sig_in, g_scale = params
            else:
                pump_in, sig_in = params
                g_scale = 1.0
            g_eff_fit = g_eff * g_scale
            if model == "counter":
                pred = signal_power_undepleted_counterprop(
                    z, sig_in, alpha_s, g_eff_fit, pump_in, alpha_p, length
                )
            else:
                pred = signal_power_undepleted_coprop(
                    z, sig_in, alpha_s, g_eff_fit, pump_in, alpha_p
                )
        if fit_log:
            return _to_dbm(pred) - _to_dbm(signal)
        return pred - signal

    result = least_squares(_residual, x0=x0, bounds=bounds, max_nfev=8000)
    mmse = float(np.mean(result.fun ** 2)) if result.fun.size else float("nan")
    return result.x, mmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="input/studies.toml")
    ap.add_argument("--profile", type=str, default="results/uwb_power_profiles.npy")
    ap.add_argument("--band", type=str, default="O")
    ap.add_argument("--pump-count", type=int, default=4)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--rho", type=float, default=2 / 3)
    ap.add_argument("--pump-ref-nm", type=float, default=1550.0)
    ap.add_argument("--fit-gr", action="store_true", help="Fit a global g_r scaling per channel.")
    ap.add_argument("--fit-log", action="store_true", help="Fit in dB (log) scale.")
    ap.add_argument(
        "--exp-approximation",
        action="store_true",
        help="Fit an exponential approximation profile instead of the undepleted model.",
    )
    ap.add_argument("--out", type=str, default="media/undepleted/undepleted_mmse_o_band.png")
    args = ap.parse_args()
    use_exp_approx = args.exp_approximation

    system = System.from_toml(args.config)
    payload = _load_profile(Path(args.profile))

    signal_sol = payload.get("signal_sol")
    if signal_sol is None:
        signal_sol = payload.get("signal_solution")
    pump_sol = payload.get("pump_sol")
    if pump_sol is None:
        pump_sol = payload.get("pump_solution")
    pump_powers = payload.get("pump_powers")
    if pump_powers is None:
        pump_powers = payload.get("pump_power")
    if pump_powers is None:
        pump_powers = payload.get("pump_powers_w")
    if pump_powers is None:
        pump_powers = payload.get("pump_power_w")
    z = payload.get("z")
    pump_wl = payload.get("pump_wavelengths")
    signal_wl = payload.get("signal_wavelengths")

    if signal_sol is None or z is None or signal_wl is None:
        raise ValueError("Profile file missing signal_sol/signal_solution, z, or signal_wavelengths.")
    if (not use_exp_approx) and pump_wl is None:
        raise ValueError("Profile file missing pump_wavelengths.")

    signal_sol = np.asarray(signal_sol, dtype=float)
    if signal_sol.ndim == 3:
        signal_sol = np.sum(signal_sol, axis=2)
    z = np.asarray(z, dtype=float)
    if pump_wl is not None:
        pump_wl = np.asarray(pump_wl, dtype=float)
    signal_wl = np.asarray(signal_wl, dtype=float)
    if pump_powers is not None:
        pump_powers = np.asarray(pump_powers, dtype=float)

    o_idx = _select_band_indices(system, signal_wl, args.band)
    if o_idx.size == 0:
        raise ValueError(f"No {args.band}-band channels found in the profile.")

    if use_exp_approx and args.fit_gr:
        raise ValueError("--exp-approximation cannot be combined with --fit-gr.")

    g_eff = None
    model = "counter"
    pump_wl_cfg = None
    pump_idx_cfg = None
    pump_idx_profile = None
    pump_in_guess = 0.0
    alpha_p0 = 0.0
    alpha_p_file = 0.0
    pump_in_file = None
    alpha_s_fixed = np.zeros(o_idx.size, dtype=float)
    if not use_exp_approx:
        if not system.pump_specs:
            raise ValueError("No pumps found in the config file.")
        pump_wl_cfg = np.array([p.wavelength for p in system.pump_specs], dtype=float)
        pump_dirs_cfg = np.array([p.direction for p in system.pump_specs], dtype=int)
        pump_powers_cfg_dbm = np.array([p.power_dbm for p in system.pump_specs], dtype=float)

        pump_idx_cfg = _select_highest_freq_pumps(pump_wl_cfg, args.pump_count)
        dir_subset = pump_dirs_cfg[pump_idx_cfg]
        if np.any(dir_subset != dir_subset[0]):
            raise ValueError("Selected pumps have mixed directions; single-pump model is ambiguous.")
        model = "counter" if dir_subset[0] < 0 else "co"

        pump_ref_wl = args.pump_ref_nm * 1e-9
        g_eff = _compute_geff_avg(
            system.fiber,
            pump_wl_cfg[pump_idx_cfg],
            signal_wl[o_idx],
            rho_pol=args.rho,
            pump_ref_wl=pump_ref_wl,
        )

        alpha_p0 = []
        for wl in pump_wl_cfg[pump_idx_cfg]:
            if hasattr(system.fiber, "attenuation_at"):
                alpha_db_m = system.fiber.attenuation_at(wl)
                alpha_p0.append(alpha_to_linear(alpha_db_m))
        alpha_p0 = float(np.mean(alpha_p0)) if alpha_p0 else 0.0
        alpha_p_file = alpha_p0

        if pump_sol is not None:
            pump_sol = np.asarray(pump_sol, dtype=float)
            if pump_wl is not None:
                pump_idx_profile = _map_pumps_to_profile(pump_wl_cfg[pump_idx_cfg], pump_wl)
            pump_in_guess = (
                float(np.sum(pump_sol[-1, pump_idx_profile]))
                if model == "counter"
                else float(np.sum(pump_sol[0, pump_idx_profile]))
            )
            pump_in_file = pump_in_guess
        else:
            pump_in_guess = float(np.sum(dBm2watt(pump_powers_cfg_dbm[pump_idx_cfg])))
            if pump_powers is not None:
                pump_powers_w = _pump_powers_to_watt(pump_powers)
                if pump_powers_w.size:
                    if pump_wl is not None:
                        pump_idx_profile = _select_highest_freq_pumps(pump_wl, args.pump_count)
                    if pump_idx_profile is not None:
                        pump_in_file = float(np.sum(pump_powers_w[pump_idx_profile]))
                    else:
                        pump_in_file = float(np.sum(pump_powers_w))

    num_params = 4 if use_exp_approx else (3 if args.fit_gr else 2)
    fit_params = np.zeros((o_idx.size, num_params), dtype=float)
    fit_mmse = np.zeros(o_idx.size, dtype=float)
    if use_exp_approx:
        bounds = (np.zeros(4), np.array([1e-2, 1e-2, 100.0, 100.0]))
    elif args.fit_gr:
        bounds = (np.zeros(3), np.array([100.0, 10.0, 10.0]))
    else:
        bounds = (np.zeros(2), np.array([100.0, 10.0]))

    for n, ch in enumerate(o_idx):
        signal = signal_sol[:, ch]
        alpha_s0 = 0.0
        if hasattr(system.fiber, "attenuation_at"):
            alpha_db_m = system.fiber.attenuation_at(signal_wl[ch])
            alpha_s0 = alpha_to_linear(alpha_db_m)
        alpha_s_fixed[n] = alpha_s0
        if use_exp_approx:
            a_guess = max(alpha_s0, 1e-6)
            a2_guess = a_guess
            p0_guess = max(signal[0], 1e-15)
            tail_ratio = float(signal[-1] / p0_guess) if p0_guess > 0 else 0.0
            b2_guess = max(tail_ratio - float(np.exp(-a_guess * system.fiber.length)), 0.0)
            x0 = np.array([a_guess, a2_guess, b2_guess, p0_guess])
        else:
            x0 = np.array([max(pump_in_guess, 1e-12), max(signal[0], 1e-15)])
            if args.fit_gr:
                x0 = np.append(x0, 1.0)
        params, mmse = _fit_channel(
            z,
            signal,
            float(g_eff[n]) if g_eff is not None else 0.0,
            system.fiber.length,
            model,
            alpha_s0,
            alpha_p0,
            x0,
            bounds,
            args.fit_gr,
            args.fit_log,
            use_exp_approx,
        )
        fit_params[n] = params
        fit_mmse[n] = mmse

    def _print_fit_table() -> None:
        if use_exp_approx:
            print("Fitted exp approximation parameters:")
            print("ch  lambda_nm    a(1/m)    a2(1/m)     b2    P0(dBm)     MMSE")
            for local_idx, ch in enumerate(o_idx):
                a, a2, b2, p0 = fit_params[local_idx][:4]
                wl_nm = signal_wl[ch] * 1e9
                print(
                    f"{ch:2d}  {wl_nm:9.1f}  {a:9.3e}  {a2:9.3e}  {b2:7.3f}  {_to_dbm(p0):8.2f}  {fit_mmse[local_idx]:9.2e}"
                )
            return

        print("Fitted undepleted parameters:")
        header = (
            "ch  lambda_nm  alpha_s(dB/km)  alpha_p(dB/km)  Pp_in(dBm)  Ps0(dBm)  g_eff(W^-1 m^-1)"
        )
        if args.fit_gr:
            header = header + "   g_r(m/W)"
        header = header + "      MMSE"
        print(header)
        for local_idx, ch in enumerate(o_idx):
            alpha_s = float(alpha_s_fixed[local_idx])
            alpha_p = float(alpha_p0)
            pump_in = float(fit_params[local_idx][0])
            sig_in = float(fit_params[local_idx][1])
            g_scale = float(fit_params[local_idx][2]) if args.fit_gr else 1.0
            g_eff_fit = float(g_eff[local_idx]) * g_scale if g_eff is not None else 0.0
            wl_nm = signal_wl[ch] * 1e9
            row = (
                f"{ch:2d}  {wl_nm:9.1f}  {_alpha_db_per_km(alpha_s):14.3f}  {_alpha_db_per_km(alpha_p):14.3f}"
                f"  {_to_dbm(pump_in):9.2f}  {_to_dbm(sig_in):8.2f}  {g_eff_fit:14.3e}"
            )
            if args.fit_gr:
                g_r_fit = g_scale * system.fiber.raman_coefficient
                row = row + f"  {g_r_fit:8.3e}"
            row = row + f"  {fit_mmse[local_idx]:9.2e}"
            print(row)

    _print_fit_table()

    def _save_fit_data(out_path: Path) -> None:
        if use_exp_approx:
            param_names = ["a", "a2", "b2", "p0"]
        elif args.fit_gr:
            param_names = ["pump_in", "sig_in", "g_scale"]
        else:
            param_names = ["pump_in", "sig_in"]
        g_eff_out = np.asarray(g_eff) if g_eff is not None else np.array([])
        pump_wl_cfg_out = pump_wl_cfg if pump_wl_cfg is not None else np.array([])
        pump_idx_cfg_out = pump_idx_cfg if pump_idx_cfg is not None else np.array([], dtype=int)
        pump_idx_profile_out = (
            pump_idx_profile if pump_idx_profile is not None else np.array([], dtype=int)
        )
        data_path = out_path.with_suffix(".npz")
        np.savez(
            data_path,
            z=z,
            signal_wavelengths=signal_wl,
            band=args.band,
            band_indices=o_idx,
            fit_params=fit_params,
            fit_mmse=fit_mmse,
            param_names=np.array(param_names, dtype=object),
            use_exp_approx=use_exp_approx,
            fit_gr=args.fit_gr,
            fit_log=args.fit_log,
            model=model,
            pump_count=args.pump_count,
            pump_wavelengths_cfg=pump_wl_cfg_out,
            pump_indices_cfg=pump_idx_cfg_out,
            pump_indices_profile=pump_idx_profile_out,
            g_eff=g_eff_out,
            pump_in_file=float(pump_in_file) if pump_in_file is not None else np.nan,
            alpha_p_file=float(alpha_p_file),
            alpha_s_fixed=alpha_s_fixed,
            alpha_p_fixed=float(alpha_p0),
        )
        print(f"Saved fit data to {data_path}")

    def _save_fit_csv(out_path: Path) -> None:
        if use_exp_approx:
            param_names = ["a_1_per_m", "a2_1_per_m", "b2", "p0_w"]
        elif args.fit_gr:
            param_names = ["pump_in_w", "sig_in_w", "g_scale"]
        else:
            param_names = ["pump_in_w", "sig_in_w"]

        columns = ["channel", "wavelength_nm", "alpha_s_np_per_m", "alpha_p_np_per_m"] + param_names
        g_eff_valid = g_eff is not None and np.size(g_eff)
        if g_eff_valid:
            columns.append("g_eff_1_per_w_m")
            if args.fit_gr:
                columns.append("g_eff_fit_1_per_w_m")
        columns.append("mmse")

        rows = []
        for local_idx, ch in enumerate(o_idx):
            row = {
                "channel": int(ch),
                "wavelength_nm": float(signal_wl[ch] * 1e9),
                "alpha_s_np_per_m": float(alpha_s_fixed[local_idx]),
                "alpha_p_np_per_m": float(alpha_p0),
            }
            for idx, name in enumerate(param_names):
                row[name] = float(fit_params[local_idx][idx])
            if g_eff_valid:
                row["g_eff_1_per_w_m"] = float(g_eff[local_idx])
                if args.fit_gr:
                    g_scale = float(fit_params[local_idx][-1])
                    row["g_eff_fit_1_per_w_m"] = float(g_eff[local_idx] * g_scale)
            row["mmse"] = float(fit_mmse[local_idx])
            rows.append(row)

        data_path = out_path.with_suffix(".csv")
        with data_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row.get(name, float("nan")) for name in columns])
        print(f"Saved fit data to {data_path}")

    out_path = Path(args.out)
    if args.fit_gr:
        out_path = out_path.with_name(out_path.stem + "_fitgr" + out_path.suffix)
    if args.fit_log:
        out_path = out_path.with_name(out_path.stem + "_fitlog" + out_path.suffix)
    if use_exp_approx:
        out_path = out_path.with_name(out_path.stem + "_expapprox" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_fit_data(out_path)
    _save_fit_csv(out_path)

    def _plot_channel(
        ax,
        local_idx: int,
        label_prefix: str = "ch",
        annotate_xy: tuple[float, float] = (0.98, 0.90),
        annotate_ha: str = "right",
        label_xy: tuple[float, float] = (0.98, 0.98),
        show_label: bool = True,
        label_suffix: str | None = None,
        legend_labels: bool = True,
    ) -> None:
        ch = o_idx[local_idx]
        signal = signal_sol[:, ch]
        if use_exp_approx:
            a, a2, b2, p0 = fit_params[local_idx][:4]
            pred = _exp_approx_profile(z, system.fiber.length, a, a2, b2, p0)
        else:
            alpha_s = float(alpha_s_fixed[local_idx])
            alpha_p = float(alpha_p0)
            pump_in = float(fit_params[local_idx][0])
            sig_in = float(fit_params[local_idx][1])
            g_scale = float(fit_params[local_idx][2]) if args.fit_gr else 1.0
            g_eff_fit = float(g_eff[local_idx]) * g_scale
            if model == "counter":
                pred = signal_power_undepleted_counterprop(
                    z, sig_in, alpha_s, g_eff_fit, pump_in, alpha_p, system.fiber.length
                )
            else:
                pred = signal_power_undepleted_coprop(
                    z, sig_in, alpha_s, g_eff_fit, pump_in, alpha_p
                )
        signal_dbm = _to_dbm(signal)
        pred_dbm = _to_dbm(pred)
        z_km = z / 1e3
        if legend_labels:
            sig_label = "signal"
            fit_label = "exp approximation" if use_exp_approx else "undepleted fit"
            file_label = "undepleted"
            if label_suffix:
                sig_label = f"{sig_label} ({label_suffix})"
                fit_label = f"{fit_label} ({label_suffix})"
                file_label = f"{file_label} ({label_suffix})"
        else:
            sig_label = "_nolegend_"
            fit_label = "_nolegend_"
            file_label = "_nolegend_"
        line_sig = ax.plot(z_km, signal_dbm, lw=0.8, label=sig_label)[0]
        ax.plot(z_km, pred_dbm, lw=1.2, ls="--", color=line_sig.get_color(), label=fit_label)
        if (not use_exp_approx) and (pump_in_file is not None) and (g_eff is not None):
            alpha_s_file = 0.0
            if hasattr(system.fiber, "attenuation_at"):
                alpha_db_m = system.fiber.attenuation_at(signal_wl[ch])
                alpha_s_file = alpha_to_linear(alpha_db_m)
            sig_in_file = float(signal[0])
            g_eff_file = float(g_eff[local_idx])
            if model == "counter":
                pred_file = signal_power_undepleted_counterprop(
                    z,
                    sig_in_file,
                    alpha_s_file,
                    g_eff_file,
                    pump_in_file,
                    alpha_p_file,
                    system.fiber.length,
                )
            else:
                pred_file = signal_power_undepleted_coprop(
                    z, sig_in_file, alpha_s_file, g_eff_file, pump_in_file, alpha_p_file
                )
            ax.plot(
                z_km,
                _to_dbm(pred_file),
                lw=1.1,
                ls=":",
                color=line_sig.get_color(),
                label=file_label,
            )
        if show_label:
            ax.text(
                label_xy[0],
                label_xy[1],
                f"{label_prefix} {ch}",
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=9,
                clip_on=True,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="none"),
            )

    def _apply_ylim_padding(ax, pad_top: float = 0.18, pad_bottom: float = 0.06) -> None:
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min
        if y_span <= 0:
            return
        ax.set_ylim(y_min - pad_bottom * y_span, y_max + pad_top * y_span)

    sel = np.unique(np.clip([0, o_idx.size // 2, o_idx.size - 1], 0, o_idx.size - 1))
    n_sel = sel.size
    ncols = 1
    nrows = n_sel if n_sel else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)
    if nrows > 1:
        fig.set_size_inches(fig.get_size_inches()[0], 3.2 * nrows)
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, local_idx in zip(axes, sel):
        _plot_channel(ax, local_idx, label_prefix="ch")
        _apply_ylim_padding(ax)

    for ax in axes[n_sel:]:
        ax.axis("off")

    axes[0].set_ylabel(r"$ P$ [dBm]")
    for ax in axes[-ncols:]:
        ax.set_xlabel(r"$ z$ [km]")
    if n_sel:
        axes[0].legend(loc="lower center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    if o_idx.size:
        fig2, ax2 = plt.subplots()
        _plot_channel(
            ax2,
            0,
            label_prefix="first",
            annotate_xy=(0.98, 0.90),
            annotate_ha="right",
            label_xy=(0.98, 0.98),
            show_label=False,
            label_suffix="first",
            legend_labels=False,
        )
        if o_idx.size > 1:
            _plot_channel(
                ax2,
                o_idx.size - 1,
                label_prefix="last",
                annotate_xy=(0.98, 0.62),
                annotate_ha="right",
                label_xy=(0.98, 0.70),
                label_suffix="last",
                show_label=False,
                legend_labels=False,
            )
        _apply_ylim_padding(ax2, pad_top=0.22, pad_bottom=0.08)
        ax2.set_ylabel(r"$ P$ [dBm]")
        ax2.set_xlabel(r"$ z$ [km]")
        legend_handles = [
            Line2D([0], [0], color="0.5", lw=1.2, ls="-"),
            Line2D([0], [0], color="0.5", lw=1.2, ls="--"),
        ]
        legend_labels = ["signal", "exp approximation" if use_exp_approx else "undepleted fit"]
        if (not use_exp_approx) and (pump_in_file is not None):
            legend_handles.append(Line2D([0], [0], color="0.5", lw=1.1, ls=":"))
            legend_labels.append("undepleted")
        ax2.legend(handles=legend_handles, labels=legend_labels, loc="lower center", fontsize=8)
        out_edge = out_path.with_name(out_path.stem + "_edges" + out_path.suffix)
        fig2.tight_layout()
        fig2.savefig(out_edge, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_edge}")

    if (not use_exp_approx) and pump_sol is not None and system.pump_specs:
        pump_sol = np.asarray(pump_sol, dtype=float)
        pump_idx_profile = _map_pumps_to_profile(pump_wl_cfg[pump_idx_cfg], pump_wl)
        fig3, ax3 = plt.subplots()
        pump_freqs = c0 / pump_wl_cfg[pump_idx_cfg]
        f_min = float(np.min(pump_freqs))
        f_max = float(np.max(pump_freqs))
        f_den = f_max - f_min + 1e-30
        cmap = plt.get_cmap("gnuplot2")
        for idx, wl in zip(pump_idx_profile, pump_wl_cfg[pump_idx_cfg]):
            freq = c0 / wl
            norm = (freq - f_min) / f_den
            color = cmap(0.05 + 0.55 * norm)
            ax3.plot(z / 1e3, _to_dbm(pump_sol[:, idx]), lw=0.45, color=color, label="_nolegend_")

        mid_local = o_idx.size // 2
        alpha_p_eq = float(alpha_p0)
        pump_in_eq = float(fit_params[mid_local][0])
        if model == "counter":
            pump_eq = pump_power_counterprop(z, pump_in_eq, alpha_p_eq, system.fiber.length)
        else:
            pump_eq = pump_power_coprop(z, pump_in_eq, alpha_p_eq)
        ax3.plot(z / 1e3, _to_dbm(pump_eq), lw=0.55, ls="--", color="black", label="equiv pump (mid ch)")

        ax3.set_xlabel(r"$ z$ [km]")
        ax3.set_ylabel(r"$ P$ [dBm]")
        ax3.legend(loc="lower center", fontsize=8)
        out_pump = out_path.with_name(out_path.stem + "_pumps" + out_path.suffix)
        fig3.tight_layout()
        fig3.savefig(out_pump, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_pump}")


if __name__ == "__main__":
    main()
