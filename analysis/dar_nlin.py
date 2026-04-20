"""
Simplified Dar et al. 2014 benchmark: single-span, 5-channel SMF.

Runs TD (collision-coefficient) and PCFM NLIN using a minimal configuration
stored in input/dar_struct.toml, and saves per-channel NLIN power + GSNR to CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger as lg

from pynlin.system import System
from pynlin.utils import dBm2watt
import pynlin.nlin.nlin_estimator_uwb as nlin_uwb
from pynlin.nlin.nlin_estimator_uwb import collision_coeffs_system_uwb, total_nlin_uwb
from pynlin.nlin.pcfm_gn import PcfmConfig, compute_pcfm_nlin
from pynlin.constellation_stats import gaussian_mu0, qam_mu0
from scipy.constants import c

from analysis.uwb_nlin import compute_raman_profiles, _load_profile_launch_powers, _nlin_cache_path


PROFILE_MAX_W = 10.0


def _profile_needs_recompute(profile_path: Path | str, max_power_w: float = PROFILE_MAX_W) -> bool:
    path = Path(profile_path)
    if not path.exists():
        return True
    try:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            sig = data.get("signal_sol")
        else:
            data = data.item() if isinstance(data, np.ndarray) and data.shape == () else data
            sig = data.get("signal_sol") if isinstance(data, dict) else None
    except Exception as exc:
        lg.warning(f"Failed to read profile for validation: {exc}")
        return True
    if sig is None:
        lg.warning("Profile missing signal_sol; recompute required.")
        return True
    sig = np.asarray(sig, dtype=float)
    if not np.all(np.isfinite(sig)):
        lg.warning("Profile contains non-finite values; recompute required.")
        return True
    max_val = float(np.nanmax(sig))
    if max_val > max_power_w:
        lg.warning(f"Profile max power {max_val:.2e} W exceeds {max_power_w:.2e} W; recompute required.")
        return True
    return False


def _flat_profiles_enabled(system: System) -> bool:
    raw = getattr(system, "raw_config", None)
    if not isinstance(raw, Mapping):
        return False
    nlin_section = raw.get("nlin")
    if not isinstance(nlin_section, Mapping):
        return False
    return bool(nlin_section.get("flat_profiles") or nlin_section.get("flat_profile"))


def _write_flat_profile(profile_path: Path | str,
                        system: System,
                        launch_powers_w: np.ndarray | None = None,
                        n_z: int = 200) -> None:
    path = Path(profile_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    freqs = system.wdm.frequency_grid()
    n_channels = freqs.size
    n_z = max(int(n_z), 2)
    z_axis = np.linspace(0.0, float(system.fiber_length), n_z)
    if launch_powers_w is None:
        launch_powers_w = np.ones(n_channels, dtype=float)
    else:
        launch_powers_w = np.asarray(launch_powers_w, dtype=float).reshape(-1)
        if launch_powers_w.size != n_channels:
            raise ValueError(
                f"Flat profile launch powers size {launch_powers_w.size} != n_channels {n_channels}"
            )
    signal_sol = np.tile(launch_powers_w[None, :], (n_z, 1))
    payload = {
        "signal_sol": signal_sol,
        "signal_wavelengths": (c / freqs),
        "z": z_axis,
    }
    np.save(path, payload)
    lg.info(f"Saved flat SPP profile to {path}")


def _qam_mu0(order: int) -> float:
    return qam_mu0(order)


def _td_prefactor_coeffs(mode_a: int, mode_b: int, n_modes: int) -> tuple[float, float]:
    if n_modes == 1:
        mode_a = mode_b = 0
    if mode_a == mode_b:
        a = 2.0 * nlin_uwb.SPATIAL_MODES[mode_a] + 3.0
        b = -4.0
    else:
        a = 2.0 * nlin_uwb.SPATIAL_MODES[mode_b]
        b = -2.0 * nlin_uwb.SPATIAL_MODES[mode_b]
    return a, b


def _td_modulation_components(system: System,
                              collision_coeffs: np.ndarray,
                              launch_powers_w: np.ndarray | None,
                              use_kappa: bool = True,
                              use_x_mode: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = float(system.fiber_length)
    br = float(system.pulse.baud_rate)
    y_norm = 1.0 / (L * br) ** 2
    collision_coeffs_si = collision_coeffs / y_norm

    n_modes, n_freqs, _, _ = collision_coeffs_si.shape
    if launch_powers_w is None:
        power_dbm = system.launch_power if system.launch_power is not None else -5.0
        P_in_arr = np.full((n_modes, n_freqs), dBm2watt(power_dbm))
    else:
        P_raw = np.asarray(launch_powers_w, dtype=float)
        if P_raw.ndim == 1:
            if P_raw.size != n_freqs:
                raise ValueError(f"launch_powers_w length {P_raw.size} != n_freqs {n_freqs}")
            P_in_arr = np.broadcast_to(P_raw[None, :], (n_modes, n_freqs))
        elif P_raw.shape == (n_modes, n_freqs):
            P_in_arr = P_raw
        else:
            raise ValueError(
                f"launch_powers_w shape {P_raw.shape} incompatible with (n_modes,n_freqs)=({n_modes},{n_freqs})"
            )

    freqs = system.wdm.frequency_grid()
    n2 = 2.6e-20
    aeff = nlin_uwb._effective_area_array(system, freqs)
    gamma = n2 * (2.0 * np.pi * freqs) / (aeff * c)
    gamma = gamma[None, :]
    constant_prefactor = (P_in_arr ** 3) * (gamma ** 2) / (br ** 2)

    kappa2 = nlin_uwb.get_kappa2_matrix_uwb(system, use_kappa, use_x_mode)

    sum_a = np.zeros((n_modes, n_freqs), dtype=float)
    sum_b = np.zeros_like(sum_a)
    for mA in range(n_modes):
        for nuA in range(n_freqs):
            for mB in range(n_modes):
                a, b = _td_prefactor_coeffs(mA, mB, n_modes)
                weight = kappa2[mA, mB]
                coeff_sum = float(np.sum(collision_coeffs_si[mA, nuA, mB, :]))
                sum_a[mA, nuA] += weight * coeff_sum * a
                sum_b[mA, nuA] += weight * coeff_sum * b
    return constant_prefactor, sum_a, sum_b


def _resolve_launch_powers(system: System, profile_path: Path | str | None) -> np.ndarray:
    freqs = system.wdm.frequency_grid()
    launch = None
    if profile_path is not None:
        launch = _load_profile_launch_powers(profile_path, system.n_channels)
    if launch is None:
        power_dbm = system.launch_power if system.launch_power is not None else -5.0
        launch = np.full_like(freqs, dBm2watt(power_dbm), dtype=float)
        lg.info("Using uniform launch powers from TOML.")
    else:
        lg.info("Using per-channel launch powers from Raman profile (z≈0).")
    if (not np.all(np.isfinite(launch)) or np.any(launch <= 0) or np.max(launch) > PROFILE_MAX_W):
        raise ValueError("Resolved launch powers are unreasonable.")
    return launch


def _save_nlin_csv(path: Path | str,
                   freqs_hz: np.ndarray,
                   nlin_w: np.ndarray,
                   signal_power_w: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nlin_flat = np.asarray(nlin_w, dtype=float).reshape(-1)
    if nlin_flat.size != freqs_hz.size:
        raise ValueError(f"NLIN array size {nlin_flat.size} != freq size {freqs_hz.size}")
    signal_power_w = np.asarray(signal_power_w, dtype=float).reshape(-1)
    if signal_power_w.size != freqs_hz.size:
        raise ValueError(f"Signal power size {signal_power_w.size} != freq size {freqs_hz.size}")
    denom = np.maximum(nlin_flat, 1e-18)
    gsnr_db = 10.0 * np.log10(signal_power_w / denom)
    data = np.column_stack([freqs_hz * 1e-12, nlin_flat, gsnr_db])
    header = "frequency_THz,nlin_W,gsnr_nli_dB"
    np.savetxt(out, data, delimiter=",", header=header, comments="")
    lg.success(f"Saved NLIN CSV to {out}")


def _print_comparison_table(freqs_hz: np.ndarray,
                            nlin_td_w: np.ndarray,
                            nlin_pcfm_w: np.ndarray,
                            signal_power_w: np.ndarray) -> None:
    freqs_thz = np.asarray(freqs_hz, dtype=float).reshape(-1) * 1e-12
    nlin_td = np.asarray(nlin_td_w, dtype=float).reshape(-1)
    nlin_pcfm = np.asarray(nlin_pcfm_w, dtype=float).reshape(-1)
    sig = np.asarray(signal_power_w, dtype=float).reshape(-1)
    if freqs_thz.size != nlin_td.size or nlin_td.size != nlin_pcfm.size or sig.size != nlin_td.size:
        raise ValueError("Comparison table arrays must have the same size.")
    denom_td = np.maximum(nlin_td, 1e-18)
    denom_pcfm = np.maximum(nlin_pcfm, 1e-18)
    gsnr_td = 10.0 * np.log10(sig / denom_td)
    gsnr_pcfm = 10.0 * np.log10(sig / denom_pcfm)
    ratio = nlin_pcfm / np.maximum(nlin_td, 1e-30)
    delta_db = 10.0 * np.log10(np.maximum(ratio, 1e-30))
    dgsnr = gsnr_pcfm - gsnr_td

    print("Dar TD vs PCFM comparison:")
    print("ch  f_THz   NLIN_TD(W)   NLIN_PCFM(W)   ratio  dNLIN(dB)  GSNR_TD(dB)  GSNR_PCFM(dB)  dGSNR(dB)")
    for idx, f_thz in enumerate(freqs_thz):
        print(
            f"{idx:2d}  {f_thz:6.3f}  {nlin_td[idx]:11.3e}  {nlin_pcfm[idx]:13.3e}"
            f"  {ratio[idx]:7.3f}  {delta_db[idx]:9.3f}  {gsnr_td[idx]:11.3f}"
            f"  {gsnr_pcfm[idx]:13.3f}  {dgsnr[idx]:9.3f}"
        )


def run_dar_workflow(cfg_path: Path | str = Path("input/dar_struct.toml"),
                     profile_path: Path | str = Path("results/dar_power_profiles.npy"),
                     recompute_profiles: bool = False,
                     recompute_td: bool = False,
                     recompute_pcfm: bool = False) -> None:
    """Run simplified Dar 2014 benchmark: TD + PCFM in a 1-span SMF case."""
    system = System.from_toml(cfg_path)
    freqs = system.wdm.frequency_grid()

    flat_profiles = _flat_profiles_enabled(system)
    if flat_profiles:
        if recompute_profiles:
            lg.warning("flat_profiles enabled; ignoring recompute_profiles.")
        launch_powers = _resolve_launch_powers(system, None)
        _write_flat_profile(profile_path, system, launch_powers_w=launch_powers)
    else:
        # Generate (or reuse) Raman profiles; with raman_gain=0 and no pumps, this is a passive span.
        if Path(profile_path).exists():
            if recompute_profiles:
                lg.info("Recomputing Raman profiles.")
                compute_raman_profiles(system, save_path=profile_path, recompute=True, max_power_w=PROFILE_MAX_W)
            else:
                compute_raman_profiles(system, save_path=profile_path, recompute=False)
        else:
            lg.info("Computing Raman profiles (missing cache).")
            compute_raman_profiles(system, save_path=profile_path, recompute=True, max_power_w=PROFILE_MAX_W)

        if _profile_needs_recompute(profile_path):
            raise ValueError(
                f"Raman profile at {profile_path} appears invalid; rerun with recompute_profiles=True."
            )

        launch_powers = _resolve_launch_powers(system, profile_path)
    signal_power = launch_powers.copy()

    # TD NLIN
    ccfs = collision_coeffs_system_uwb(
        system,
        ipulse=1,
        recompute=recompute_td,
        profile_path=profile_path,
    )
    nlin_td = total_nlin_uwb(
        system,
        ccfs,
        use_kappa=True,
        use_x_mode=True,
        launch_powers_w=launch_powers,
        cache_path=_nlin_cache_path(profile_path, use_kappa=True, use_x_mode=True),
        recompute=recompute_td,
    )
    nlin_td_flat = np.asarray(nlin_td, dtype=float).reshape(-1)
    _save_nlin_csv(
        Path("results") / f"s3_chan_nlin_td_{Path(profile_path).stem}_dar.csv",
        freqs,
        nlin_td_flat,
        signal_power,
    )

    # PCFM NLIN (MCI disabled)
    cfg = PcfmConfig(
        degree=9,
        include_mci=False,
        use_numeric_sci=True,
        use_numeric_xci=False,
    )
    pcfm_path = Path("results") / f"total_nlin_{Path(profile_path).stem}_dar_pcfm.npy"
    if pcfm_path.exists() and not recompute_pcfm:
        lg.info(f"Loading cached PCFM NLIN from {pcfm_path}")
        nlin_pcfm = np.load(pcfm_path)
    else:
        nlin_pcfm = compute_pcfm_nlin(
            system,
            profile_path=profile_path,
            launch_powers_w=launch_powers,
            config=cfg,
        )
        pcfm_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pcfm_path, nlin_pcfm)
        lg.success(f"Saved PCFM NLIN to {pcfm_path}")

    nlin_pcfm_flat = np.asarray(nlin_pcfm, dtype=float).reshape(-1)
    _save_nlin_csv(
        Path("results") / f"total_nlin_{Path(profile_path).stem}_dar_pcfm.csv",
        freqs,
        nlin_pcfm_flat,
        signal_power,
    )

    _print_comparison_table(freqs, nlin_td_flat, nlin_pcfm_flat, signal_power)
    lg.info("Dar benchmark finished.")


def run_dar_fig3(cfg_path: Path | str = Path("input/dar_struct.toml"),
                 lengths_km: np.ndarray | None = None,
                 profile_dir: Path | str = Path("results"),
                 recompute_profiles: bool = False,
                 recompute_td: bool = False,
                 recompute: bool = False) -> None:
    """Reproduce Dar Fig. 3 sweep: GN vs QPSK vs 16-QAM vs length at fixed power."""
    system = System.from_toml(cfg_path)
    flat_profiles = _flat_profiles_enabled(system)
    if lengths_km is None:
        lengths_km = np.linspace(100.0, 1000.0, 3)
    lengths_km = np.asarray(lengths_km, dtype=float)
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    mu0_gn = gaussian_mu0()
    mu0_qpsk = _qam_mu0(4)
    mu0_16qam = _qam_mu0(16)

    out_path = Path("results") / "dar_fig3_length_sweep.csv"
    if out_path.exists() and not recompute:
        lg.info(f"Loading cached Dar Fig.3 sweep from {out_path}")
        out = np.loadtxt(out_path, delimiter=",", skiprows=1)
        if out.ndim == 1:
            out = out[None, :]
    else:
        rows = []
        for L_km in lengths_km:
            system.fiber.length = float(L_km) * 1e3
            profile_path = profile_dir / f"dar_power_profiles_L{int(L_km)}km.npy"

            if flat_profiles:
                launch_powers = _resolve_launch_powers(system, None)
                _write_flat_profile(profile_path, system, launch_powers_w=launch_powers)
            else:
                if profile_path.exists():
                    if recompute_profiles:
                        lg.info(f"Recomputing Raman profiles for L={L_km:.0f} km.")
                        compute_raman_profiles(system, save_path=profile_path, recompute=True, max_power_w=PROFILE_MAX_W)
                    else:
                        compute_raman_profiles(system, save_path=profile_path, recompute=False)
                else:
                    lg.info(f"Computing Raman profiles for L={L_km:.0f} km.")
                    compute_raman_profiles(system, save_path=profile_path, recompute=True, max_power_w=PROFILE_MAX_W)
                if _profile_needs_recompute(profile_path):
                    raise ValueError(
                        f"Raman profile at {profile_path} appears invalid; rerun with recompute_profiles=True."
                    )
                launch_powers = _resolve_launch_powers(system, profile_path)

            ccfs = collision_coeffs_system_uwb(
                system,
                ipulse=1,
                recompute=recompute_td,
                profile_path=profile_path,
            )
            const_pref, sum_a, sum_b = _td_modulation_components(
                system,
                ccfs,
                launch_powers,
                use_kappa=True,
                use_x_mode=True,
            )
            nlin_gn = const_pref * (mu0_gn * sum_a + sum_b)
            nlin_qpsk = const_pref * (mu0_qpsk * sum_a + sum_b)
            nlin_16qam = const_pref * (mu0_16qam * sum_a + sum_b)

            nlin_gn = np.asarray(nlin_gn, dtype=float).reshape(-1)
            nlin_qpsk = np.asarray(nlin_qpsk, dtype=float).reshape(-1)
            nlin_16qam = np.asarray(nlin_16qam, dtype=float).reshape(-1)
            launch_flat = np.asarray(launch_powers, dtype=float).reshape(-1)
            cut_idx = int(launch_flat.size // 2)
            p_sig = float(launch_flat[cut_idx])
            gn = float(nlin_gn[cut_idx])
            qpsk = float(nlin_qpsk[cut_idx])
            qam16 = float(nlin_16qam[cut_idx])
            gsnr_gn = 10.0 * np.log10(p_sig / max(gn, 1e-18))
            gsnr_qpsk = 10.0 * np.log10(p_sig / max(qpsk, 1e-18))
            gsnr_16qam = 10.0 * np.log10(p_sig / max(qam16, 1e-18))
            rows.append([L_km, gn, qpsk, qam16, gsnr_gn, gsnr_qpsk, gsnr_16qam])
            lg.info(
                f"Fig3 L={L_km:.0f} km: GN={gn:.3e} W, QPSK={qpsk:.3e} W, "
                f"16QAM={qam16:.3e} W (cut ch {cut_idx})"
            )

        out = np.asarray(rows, dtype=float)
        header = "length_km,nlin_gn_W,nlin_qpsk_W,nlin_16qam_W,gsnr_gn_dB,gsnr_qpsk_dB,gsnr_16qam_dB"
        np.savetxt(out_path, out, delimiter=",", header=header, comments="")
        lg.success(f"Saved Dar Fig.3 sweep to {out_path}")

    lengths = out[:, 0]
    nlin_gn = out[:, 1]
    nlin_qpsk = out[:, 2]
    nlin_16qam = out[:, 3]
    nlin_gn_ratio_db = -out[:, 4]
    nlin_qpsk_ratio_db = -out[:, 5]
    nlin_16qam_ratio_db = -out[:, 6]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(lengths, nlin_gn_ratio_db, marker="o", ms=3, lw=0.9, label="GN (Gaussian)")
    ax.plot(lengths, nlin_qpsk_ratio_db, marker="s", ms=3, lw=0.9, label="QPSK")
    ax.plot(lengths, nlin_16qam_ratio_db, marker="^", ms=3, lw=0.9, label="16-QAM")
    ax.set_xlabel("Length [km]")
    ax.set_ylabel(r"$P_{NLI}/P_{sig}(L)\;[\mathrm{dB}]$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    plot_path = Path("media") / "dar" / "dar_fig3_length_sweep.pdf"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    lg.success(f"Saved Dar Fig.3 plot to {plot_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simplified Dar 2014 TD+PCFM benchmark.")
    parser.add_argument("--cfg", type=str, default="input/dar_struct.toml", help="Path to dar_struct.toml")
    parser.add_argument("--profile", type=str, default="results/dar_power_profiles.npy",
                        help="Path to Raman profile .npy")
    parser.add_argument("--recompute-profiles", action="store_true", help="Recompute Raman profiles")
    parser.add_argument("--recompute-td", action="store_true", help="Recompute TD collision coefficients/NLIN")
    parser.add_argument("--recompute-pcfm", action="store_true", help="Recompute PCFM NLIN")
    parser.add_argument("--single-span", action="store_true", help="Run the single-span TD+PCFM workflow.")
    parser.add_argument("--recompute", action="store_true", help="Recompute the length sweep CSV.")
    parser.add_argument("--len-min-km", type=float, default=100.0, help="Length sweep start (km).")
    parser.add_argument("--len-max-km", type=float, default=1000.0, help="Length sweep end (km).")
    parser.add_argument("--len-step-km", type=float, default=450.0, help="Length sweep step (km).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.single_span:
        run_dar_workflow(
            cfg_path=Path(args.cfg),
            profile_path=Path(args.profile),
            recompute_profiles=args.recompute_profiles,
            recompute_td=args.recompute_td,
            recompute_pcfm=args.recompute_pcfm,
        )
    else:
        lengths = np.arange(args.len_min_km, args.len_max_km + 1e-9, args.len_step_km)
        run_dar_fig3(
            cfg_path=Path(args.cfg),
            lengths_km=lengths,
            recompute_profiles=args.recompute_profiles,
            recompute_td=args.recompute_td,
            recompute=args.recompute,
        )
