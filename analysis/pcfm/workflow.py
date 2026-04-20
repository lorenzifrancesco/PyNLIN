import hashlib
from pathlib import Path

import numpy as np
from loguru import logger as lg

from pynlin.constellation_stats import gaussian_mu0
from pynlin.nlin.nlin_estimator_uwb import collision_coeffs_system_uwb, total_nlin_uwb
from pynlin.nlin.pcfm_gn import PcfmConfig, load_signal_profiles
from pynlin.raman.solvers_jiang import JiangIterativeConfig
from pynlin.system import System
from pynlin.utils import watt2dBm

from .config import (
    PROFILE_MAX_W,
    _flat_profiles_enabled,
    _load_pcfm_runtime_config,
    _normalize_mode,
    _to_optional_path,
)
from .io import (
    _launch_referenced_nlin_to_output_power,
    _output_over_launch_signal_power_ratio,
    _power_profile_hash,
    _resolve_launch_powers,
    _resolve_signal_power,
    _save_nlin_csv,
    _write_flat_profile,
)
from .models import _load_or_compute_gn, _load_or_compute_gn_direct, _load_or_compute_pcfm
from .models import _load_or_compute_flat_analytic_xci
from .plotting import (
    plot_pcfm_diagnostics,
    plot_pcfm_gsnr,
    plot_pcfm_nlin_power,
)
from .reporting import _log_td_pcfm_parameters
from .td import _qam_mu0, _td_modulation_components

from analysis.uwb_nlin import _nlin_cache_path, compute_raman_profiles, plot_power_profiles

MANAKOV_SCALE_PCFM = np.nan # this legacy thing must explode
PCFM_MEDIA_DIR = Path("media") / "PCFM"

def _apply_pcfm_manakov_scaling(values: np.ndarray) -> np.ndarray:
    """Apply manual 16/9 Manakov scaling for PCFM TD comparisons."""
    raise("No more using this")
    return np.asarray(values, dtype=float) * MANAKOV_SCALE_PCFM


def _log_td_gn_vs_pcfm_xci_diff_stats(
    td_modulations: dict[str, np.ndarray],
    nlin_pcfm_xci: dict[str, np.ndarray],
) -> None:
    """Print TD(Gaussian) minus PCFM(XCI) summary stats in linear and dB domains."""
    td_gn = np.asarray(td_modulations["Gaussian"], dtype=float).reshape(-1)
    for label, pcfm_xci in nlin_pcfm_xci.items():
        pcfm_xci_flat = np.asarray(pcfm_xci, dtype=float).reshape(-1)
        diff_w = td_gn - pcfm_xci_flat
        ratio = td_gn / np.maximum(pcfm_xci_flat, 1e-30)
        td_db = watt2dBm(np.maximum(td_gn, 1e-30))
        xci_db = watt2dBm(np.maximum(pcfm_xci_flat, 1e-30))
        diff_db = td_db - xci_db
        lg.info(
            "TD(GN)-PCFM(XCI) [{}] linear diff [W]: max={:.3e}, min={:.3e}, avg={:.3e}; "
            "ratio TD/PCFM_XCI [-]: max={:.3f}, min={:.3f}, avg={:.3f}; "
            "dB diff [dB]: max={:.3f}, min={:.3f}, avg={:.3f}".format(
                label,
                float(np.max(diff_w)),
                float(np.min(diff_w)),
                float(np.mean(diff_w)),
                float(np.max(ratio)),
                float(np.min(ratio)),
                float(np.mean(ratio)),
                float(np.max(diff_db)),
                float(np.min(diff_db)),
                float(np.mean(diff_db)),
            )
        )


def run_pcfm_workflow(
    cfg_path: Path | str = Path("./input/pcfm_struct.toml"),
    profile_path: Path | str | None = None,
    launch_csv_path: Path | str | None = None,
    power_profiles_mode: str | None = None,
    td_mode: str | None = None,
    pcfm_mode: str | None = None,
    gn_mode: str | None = None,
    gn_direct_mode: str | None = None,
    pcfm_numeric_xci: bool | None = None,
    td_exclude_self_channel: bool | None = None,
    plot_mode: str | None = None,
) -> None:
    """Run TD + PCFM (+ optional GN) workflow and plot GSNR overlays."""
    system = System.from_toml(cfg_path)
    runtime_cfg = _load_pcfm_runtime_config(system)

    profile_path = _to_optional_path(
        profile_path if profile_path is not None else runtime_cfg["profile_path"]
    )
    if profile_path is None:
        raise ValueError("PCFM profile path is required.")
    launch_csv_path = _to_optional_path(
        launch_csv_path if launch_csv_path is not None else runtime_cfg["launch_csv_path"]
    )

    power_profiles_mode = _normalize_mode(
        "power_profiles_mode",
        runtime_cfg["power_profiles_mode"]
        if power_profiles_mode is None
        else power_profiles_mode,
        {"flat", "cached", "recompute", "cached_no_profile_launch", "recompute_no_profile_launch"},
    )
    td_mode = _normalize_mode(
        "td_mode",
        runtime_cfg["td_mode"] if td_mode is None else td_mode,
        {"cached", "recompute"},
    )
    pcfm_mode = _normalize_mode(
        "pcfm_mode",
        runtime_cfg["pcfm_mode"] if pcfm_mode is None else pcfm_mode,
        {"cached", "recompute"},
    )
    gn_mode = _normalize_mode(
        "gn_mode",
        runtime_cfg["gn_mode"] if gn_mode is None else gn_mode,
        {"off", "cached", "recompute"},
    )
    gn_direct_mode = _normalize_mode(
        "gn_direct_mode",
        runtime_cfg["gn_direct_mode"] if gn_direct_mode is None else gn_direct_mode,
        {"off", "cached", "recompute"},
    )
    plot_mode = _normalize_mode(
        "plot_mode",
        runtime_cfg["plot_mode"] if plot_mode is None else plot_mode,
        {"off", "on"},
    )

    flat_profiles_mode = power_profiles_mode == "flat"
    recompute_profiles = power_profiles_mode in {"recompute", "recompute_no_profile_launch"}
    use_profile_launch_powers = power_profiles_mode in {"cached", "recompute"}
    recompute_td = td_mode == "recompute"
    recompute_pcfm = pcfm_mode == "recompute"
    compute_gn = gn_mode != "off"
    recompute_gn = gn_mode == "recompute"
    compute_gn_direct = gn_direct_mode != "off"
    recompute_gn_direct = gn_direct_mode == "recompute"
    do_plot = plot_mode == "on"

    pcfm_numeric_xci = (
        bool(runtime_cfg["pcfm_numeric_xci"])
        if pcfm_numeric_xci is None
        else bool(pcfm_numeric_xci)
    )
    pcfm_eq18_xci = bool(runtime_cfg["pcfm_eq18_xci"])
    td_exclude_self_channel = (
        bool(runtime_cfg["td_exclude_self_channel"])
        if td_exclude_self_channel is None
        else bool(td_exclude_self_channel)
    )
    plot_pcfm_total_and_sci = bool(runtime_cfg["plot_pcfm_total_and_sci"])
    cfg = PcfmConfig(
        degree=9,
        include_mci=False,
        use_numeric_sci=True,
        use_numeric_xci=pcfm_numeric_xci,
    )

    freqs = system.wdm.frequency_grid()
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    dispersion_signature = np.ascontiguousarray(
        np.stack([beta1_grid, beta2_grid], axis=0),
        dtype=np.float64,
    ).view(np.uint8)
    dispersion_tag = hashlib.sha1(dispersion_signature).hexdigest()[:12]
    lg.info(
        "Dispersion cache tag={} (beta2 min/max: {:.3e}/{:.3e} s^2/m).".format(
            dispersion_tag,
            float(np.min(beta2_grid)),
            float(np.max(beta2_grid)),
        )
    )
    flat_profiles = flat_profiles_mode or _flat_profiles_enabled(system)

    if flat_profiles:
        use_profile_launch_powers = False
        launch_powers = _resolve_launch_powers(system, None, launch_csv_path, use_profile=False)
        _write_flat_profile(profile_path, system, launch_powers_w=launch_powers)
    else:
        if Path(profile_path).exists():
            if recompute_profiles:
                jiang_cfg = JiangIterativeConfig(
                    iterative_steps=120,
                    pump_scale_start=1e-6,
                    early_stop_rtol=1e-4,
                )
                lg.info("Recomputing Raman profiles with Jiang solver configuration.")
                compute_raman_profiles(
                    system,
                    save_path=profile_path,
                    recompute=True,
                    jiang_cfg=jiang_cfg,
                    max_power_w=PROFILE_MAX_W,
                )
            else:
                compute_raman_profiles(system, save_path=profile_path, recompute=False)
        else:
            if not recompute_profiles:
                raise FileNotFoundError(
                    f"Raman profile missing at {profile_path}. "
                    "Set [pcfm.run].power_profiles_mode='recompute'."
                )
            jiang_cfg = JiangIterativeConfig(
                iterative_steps=120,
                pump_scale_start=1e-6,
                early_stop_rtol=1e-4,
            )
            lg.info("Computing Raman profiles with Jiang solver configuration.")
            compute_raman_profiles(
                system,
                save_path=profile_path,
                recompute=True,
                jiang_cfg=jiang_cfg,
                max_power_w=PROFILE_MAX_W,
            )
        launch_powers = _resolve_launch_powers(
            system,
            profile_path,
            launch_csv_path,
            use_profile=use_profile_launch_powers,
        )

    profile_power_tag = _power_profile_hash(system, profile_path)
    model_cache_tag = f"disp{dispersion_tag}_prof{profile_power_tag}"
    lg.info(f"Power-profile cache tag={profile_power_tag}.")

    sig_ch_z, z_axis = load_signal_profiles(profile_path, system)
    span = float(z_axis[-1] - z_axis[0]) if z_axis.size else 0.0
    avg_power = np.trapezoid(sig_ch_z, z_axis, axis=1) / max(span, 1.0)
    out_power = sig_ch_z[:, -1]
    lg.info(
        "Profile power summary (per-channel): "
        f"avg_W min/med/max = {float(np.min(avg_power)):.3e} / "
        f"{float(np.median(avg_power)):.3e} / {float(np.max(avg_power)):.3e}; "
        f"out_W min/med/max = {float(np.min(out_power)):.3e} / "
        f"{float(np.median(out_power)):.3e} / {float(np.max(out_power)):.3e}"
    )

    if do_plot:
        plot_power_profiles(system, profile_path, output_dir=PCFM_MEDIA_DIR)
        plot_pcfm_diagnostics(
            system=system,
            profile_path=profile_path,
            launch_powers_w=launch_powers,
            out_dir=PCFM_MEDIA_DIR,
            cfg=cfg,
        )

    launch_dbm = watt2dBm(np.maximum(launch_powers, 1e-18))
    lg.info(
        "Launch power summary (per-channel): dBm min/med/max = "
        f"{float(np.min(launch_dbm)):.2f} / {float(np.median(launch_dbm)):.2f} / "
        f"{float(np.max(launch_dbm)):.2f}"
    )
    if hasattr(system.wdm, "band_specs") and system.wdm.band_specs:
        for name, slc in system.wdm._band_slices.items():
            band_dbm = launch_dbm[slc]
            lg.info(
                f"Band {name}: {slc.stop - slc.start} ch, "
                f"launch_dBm min/med/max = {float(np.min(band_dbm)):.2f} / "
                f"{float(np.median(band_dbm)):.2f} / {float(np.max(band_dbm)):.2f}"
            )

    output_signal_power_w = _resolve_signal_power(system, profile_path, launch_powers)
    output_over_launch_signal_power_ratio = _output_over_launch_signal_power_ratio(
        system=system,
        profile_path=profile_path,
        launch_powers_w=launch_powers,
    )
    lg.info(
        "TD/PCFM/GN producers return launch-referenced NLIN powers; "
        "applying P_signal,out/P_signal,launch to obtain output NLIN powers."
    )
    lg.info(
        "Output/launch signal-power ratio summary [-]: min/med/max = "
        f"{float(np.min(output_over_launch_signal_power_ratio)):.3e} / "
        f"{float(np.median(output_over_launch_signal_power_ratio)):.3e} / "
        f"{float(np.max(output_over_launch_signal_power_ratio)):.3e}"
    )
    _log_td_pcfm_parameters(
        system=system,
        launch_powers_w=launch_powers,
        profile_path=profile_path,
        cfg=cfg,
    )

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
        exclude_self_channel=td_exclude_self_channel,
        cache_path=_nlin_cache_path(
            profile_path,
            use_kappa=True,
            use_x_mode=True,
            extra_tag=f"{model_cache_tag}_{'xci' if td_exclude_self_channel else 'all'}",
        ),
        recompute=recompute_td,
    )
    nlin_td_output_w = _launch_referenced_nlin_to_output_power(
        nlin_td,
        output_over_launch_signal_power_ratio,
    )

    qam_orders = [16, 64, 256]
    const_pref, sum_a, sum_b = _td_modulation_components(
        system,
        ccfs,
        launch_powers,
        use_kappa=True, # see above comment
        use_x_mode=True,
        exclude_self_channel=td_exclude_self_channel,
    )
    # const_pref = _apply_pcfm_manakov_scaling(const_pref)

    td_modulations: dict[str, np.ndarray] = {}
    for order in qam_orders:
        mu0 = _qam_mu0(order)
        nlin_mod = const_pref * (mu0 * sum_a + sum_b)
        td_modulations[f"{order}-QAM"] = _launch_referenced_nlin_to_output_power(
            nlin_mod,
            output_over_launch_signal_power_ratio,
        )
    mu0_gaussian = gaussian_mu0()
    nlin_gaussian = const_pref * (mu0_gaussian * sum_a + sum_b)
    td_modulations["Gaussian"] = _launch_referenced_nlin_to_output_power(
        nlin_gaussian,
        output_over_launch_signal_power_ratio,
    )

    _save_nlin_csv(
        Path("results") / f"s3_chan_nlin_td_{Path(profile_path).stem}_k1_x1.csv",
        freqs,
        nlin_td_output_w,
        output_signal_power_w,
    )

    gsnr_pcfm = {}
    gsnr_gn = {} if compute_gn else None
    gsnr_gn_direct = {} if compute_gn_direct else None
    nlin_pcfm = {}
    nlin_gn = {} if compute_gn else None
    nlin_gn_direct = {} if compute_gn_direct else None
    nlin_gn_direct_ratio = {} if compute_gn_direct else None
    nlin_pcfm_xci = {}
    nlin_gn_xci = {} if compute_gn else None
    nlin_gn_direct_xci = {} if compute_gn_direct else None
    nlin_gn_direct_xci_ratio = {} if compute_gn_direct else None

    for label in ("no_loss",):
        pcfm_path = (
            Path("results")
            / f"total_nlin_{Path(profile_path).stem}_{model_cache_tag}_pcfm_{label}.npy"
        )
        nlin_pcfm_arr, _, nlin_pcfm_xci_arr = _load_or_compute_pcfm(
            system,
            profile_path=profile_path,
            launch_powers_w=launch_powers,
            output_path=pcfm_path,
            cfg=cfg,
            recompute=recompute_pcfm,
            return_components=True,
        )
        nlin_pcfm_output_w = _launch_referenced_nlin_to_output_power(
            nlin_pcfm_arr,
            output_over_launch_signal_power_ratio,
        )
        nlin_pcfm_xci_output_w = _launch_referenced_nlin_to_output_power(
            nlin_pcfm_xci_arr,
            output_over_launch_signal_power_ratio,
        )
        _save_nlin_csv(
            Path("results") / f"total_nlin_{Path(profile_path).stem}_pcfm_{label}.csv",
            freqs,
            nlin_pcfm_output_w,
            output_signal_power_w,
        )
        _save_nlin_csv(
            Path("results") / f"total_nlin_{Path(profile_path).stem}_pcfm_{label}_xci.csv",
            freqs,
            nlin_pcfm_xci_output_w,
            output_signal_power_w,
        )
        gsnr_pcfm[label] = 10.0 * np.log10(
            output_signal_power_w / np.maximum(nlin_pcfm_output_w, 1e-18)
        )
        nlin_pcfm[label] = nlin_pcfm_output_w
        nlin_pcfm_xci[label] = nlin_pcfm_xci_output_w

        if pcfm_eq18_xci:
            if not flat_profiles:
                lg.info(
                    "PCFM Eq. 18 XCI requested with non-flat power profiles: "
                    "adding the flat-analytic Eq. 18 reference curve to the plot/output."
                )
                lg.warning("This multiplication for a SPP function is heuristic.")
            eq18_path = (
                Path("results")
                / f"total_nlin_{Path(profile_path).stem}_{model_cache_tag}_pcfm_{label}_xci_eq18.npy"
            )
            nlin_pcfm_eq18_xci_output_w = _launch_referenced_nlin_to_output_power(
                _load_or_compute_flat_analytic_xci(
                    system,
                    launch_powers_w=launch_powers,
                    output_path=eq18_path,
                    profile_path=profile_path,
                    degree=cfg.degree,
                    xci_model="eq18",
                    recompute=recompute_pcfm,
                ),
                output_over_launch_signal_power_ratio,
            )
            _save_nlin_csv(
                Path("results") / f"total_nlin_{Path(profile_path).stem}_pcfm_{label}_xci_eq18.csv",
                freqs,
                nlin_pcfm_eq18_xci_output_w,
                output_signal_power_w,
            )
            eq18_label = "eq18" if label == "no_loss" else f"{label} eq18"
            nlin_pcfm_xci[eq18_label] = nlin_pcfm_eq18_xci_output_w

        if compute_gn:
            gn_path = (
                Path("results")
                / f"total_nlin_{Path(profile_path).stem}_{model_cache_tag}_gn_{label}.npy"
            )
            nlin_gn_arr, _, nlin_gn_xci_arr = _load_or_compute_gn(
                system,
                profile_path=profile_path,
                launch_powers_w=launch_powers,
                output_path=gn_path,
                recompute=recompute_gn,
                return_components=True,
            )
            nlin_gn_output_w = _launch_referenced_nlin_to_output_power(
                nlin_gn_arr,
                output_over_launch_signal_power_ratio,
            )
            nlin_gn_xci_output_w = _launch_referenced_nlin_to_output_power(
                nlin_gn_xci_arr,
                output_over_launch_signal_power_ratio,
            )
            _save_nlin_csv(
                Path("results") / f"total_nlin_{Path(profile_path).stem}_gn_{label}.csv",
                freqs,
                nlin_gn_output_w,
                output_signal_power_w,
            )
            _save_nlin_csv(
                Path("results") / f"total_nlin_{Path(profile_path).stem}_gn_{label}_xci.csv",
                freqs,
                nlin_gn_xci_output_w,
                output_signal_power_w,
            )
            if gsnr_gn is not None:
                gsnr_gn[label] = 10.0 * np.log10(
                    output_signal_power_w / np.maximum(nlin_gn_output_w, 1e-18)
                )
            if nlin_gn is not None:
                nlin_gn[label] = nlin_gn_output_w
            if nlin_gn_xci is not None:
                nlin_gn_xci[label] = nlin_gn_xci_output_w

        if compute_gn_direct:
            gn_direct_path = (
                Path("results")
                / f"total_nlin_{Path(profile_path).stem}_{model_cache_tag}_gn_direct_{label}.npy"
            )
            nlin_gn_direct_arr, _, nlin_gn_direct_xci_arr = _load_or_compute_gn_direct(
                system,
                profile_path=profile_path,
                launch_powers_w=launch_powers,
                output_path=gn_direct_path,
                recompute=recompute_gn_direct,
                return_components=True,
            )
            nlin_gn_direct_output_w = _launch_referenced_nlin_to_output_power(
                nlin_gn_direct_arr,
                output_over_launch_signal_power_ratio,
            )
            nlin_gn_direct_xci_output_w = _launch_referenced_nlin_to_output_power(
                nlin_gn_direct_xci_arr,
                output_over_launch_signal_power_ratio,
            )
            _save_nlin_csv(
                Path("results") / f"total_nlin_{Path(profile_path).stem}_gn_direct_{label}.csv",
                freqs,
                nlin_gn_direct_output_w,
                output_signal_power_w,
            )
            _save_nlin_csv(
                Path("results") / f"total_nlin_{Path(profile_path).stem}_gn_direct_{label}_xci.csv",
                freqs,
                nlin_gn_direct_xci_output_w,
                output_signal_power_w,
            )
            if gsnr_gn_direct is not None:
                gsnr_gn_direct[label] = 10.0 * np.log10(
                    output_signal_power_w / np.maximum(nlin_gn_direct_output_w, 1e-18)
                )
            if nlin_gn_direct is not None:
                nlin_gn_direct[label] = nlin_gn_direct_output_w
            if nlin_gn_direct_xci is not None:
                nlin_gn_direct_xci[label] = nlin_gn_direct_xci_output_w
            if nlin_gn_direct_ratio is not None:
                denom = np.maximum(output_signal_power_w, 1e-18)
                nlin_gn_direct_ratio[label] = nlin_gn_direct_output_w / denom
            if nlin_gn_direct_xci_ratio is not None:
                denom = np.maximum(output_signal_power_w, 1e-18)
                nlin_gn_direct_xci_ratio[label] = nlin_gn_direct_xci_output_w / denom

    _log_td_gn_vs_pcfm_xci_diff_stats(td_modulations=td_modulations, nlin_pcfm_xci=nlin_pcfm_xci)

    gsnr_td = 10.0 * np.log10(output_signal_power_w / np.maximum(nlin_td_output_w, 1e-18))
    if do_plot:
        plot_pcfm_gsnr(
            freqs_hz=freqs,
            gsnr_td=gsnr_td,
            gsnr_pcfm=gsnr_pcfm,
            gsnr_gn=gsnr_gn,
            out_path=PCFM_MEDIA_DIR / "fig14c.pdf",
            gsnr_gn_direct=gsnr_gn_direct,
            plot_pcfm_total_and_sci=plot_pcfm_total_and_sci,
        )
        plot_pcfm_nlin_power(
            freqs_hz=freqs,
            signal_power_w=output_signal_power_w,
            nlin_td_w=nlin_td_output_w,
            nlin_pcfm_w=nlin_pcfm,
            nlin_gn_w=nlin_gn,
            nlin_td_mod_w=td_modulations,
            nlin_pcfm_xci_w=nlin_pcfm_xci,
            nlin_gn_xci_w=nlin_gn_xci,
            nlin_gn_direct_w=nlin_gn_direct,
            nlin_gn_direct_xci_w=nlin_gn_direct_xci,
            gn_direct_is_ratio=False,
            gn_direct_xci_is_ratio=False,
            out_path=PCFM_MEDIA_DIR / "nlin_power.pdf",
            plot_pcfm_total_and_sci=plot_pcfm_total_and_sci,
        )
