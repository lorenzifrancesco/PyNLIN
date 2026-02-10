"""
Simplified Dar et al. 2014 benchmark: single-span, 5-channel SMF.

Runs TD (collision-coefficient) and PCFM NLIN using a minimal configuration
stored in input/dar.toml, and saves per-channel NLIN power + GSNR to CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger as lg

from pynlin.system import System
from pynlin.utils import dBm2watt
from pynlin.nlin.nlin_estimator_uwb import collision_coeffs_system_uwb, total_nlin_uwb
from pynlin.nlin.pcfm_gn import PcfmConfig, compute_pcfm_nlin

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


def run_dar_workflow(cfg_path: Path | str = Path("input/dar.toml"),
                     profile_path: Path | str = Path("results/dar_power_profiles.npy"),
                     recompute_profiles: bool = False,
                     recompute_td: bool = False,
                     recompute_pcfm: bool = False) -> None:
    """Run simplified Dar 2014 benchmark: TD + PCFM in a 1-span SMF case."""
    system = System.from_toml(cfg_path)
    freqs = system.wdm.frequency_grid()

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
        Path("results") / f"total_nlin_{Path(profile_path).stem}_dar_td.csv",
        freqs,
        nlin_td_flat,
        signal_power,
    )

    # PCFM NLIN (no lumped losses, no MCI)
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
            lumped_losses=None,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simplified Dar 2014 TD+PCFM benchmark.")
    parser.add_argument("--cfg", type=str, default="input/dar.toml", help="Path to dar.toml")
    parser.add_argument("--profile", type=str, default="results/dar_power_profiles.npy",
                        help="Path to Raman profile .npy")
    parser.add_argument("--recompute-profiles", action="store_true", help="Recompute Raman profiles")
    parser.add_argument("--recompute-td", action="store_true", help="Recompute TD collision coefficients/NLIN")
    parser.add_argument("--recompute-pcfm", action="store_true", help="Recompute PCFM NLIN")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_dar_workflow(
        cfg_path=Path(args.cfg),
        profile_path=Path(args.profile),
        recompute_profiles=args.recompute_profiles,
        recompute_td=args.recompute_td,
        recompute_pcfm=args.recompute_pcfm,
    )
