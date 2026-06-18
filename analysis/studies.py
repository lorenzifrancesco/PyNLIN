import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as lg

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.config import (  # noqa: E402
    StudyConfig,
    StudiesRuntimeConfig,
    _select_scaling_channel,
    load_studies_runtime_config,
)
from analysis.methods.io import _save_nlin_csv  # noqa: E402
from analysis.runners.methods import TDResult, run_mc, run_pcfm, run_td  # noqa: E402
from analysis.runtime.cache import safe_tag  # noqa: E402
from analysis.runtime.context import build_run_context  # noqa: E402
from analysis.subset import resolve_subset  # noqa: E402
from pynlin.system import System  # noqa: E402


def _require_td_for_mc(study: StudyConfig, cfg: StudiesRuntimeConfig) -> None:
    if "td" not in study.methods:
        raise ValueError(f"MC method requires 'td' in study.methods for {study.name!r}.")
    if cfg.methods.td.mode == "off":
        raise ValueError(f"MC method requires TD method enabled for {study.name!r}.")


def _save_subset_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fields = [
        "channel_idx",
        "method",
        "nlin_w",
        "chi1",
        "chi2",
        "prefactor",
        "nlin_16qam_w",
    ]
    data = np.array(
        [[row.get(field, float("nan")) for field in fields] for row in rows],
        dtype=object,
    )
    np.savetxt(path, data, delimiter=",", fmt="%s", header=",".join(fields), comments="")
    lg.success("Saved subset study summary to {}", path)


def _run_full_system(config_path: Path, cfg: StudiesRuntimeConfig, study: StudyConfig) -> None:
    from analysis.methods.workflow import run_pcfm_workflow

    lg.info(
        "Study: {}\n"
        "  type: {}\n"
        "  methods: {}\n"
        "  td_mode: {}\n"
        "  pcfm_mode: {}\n"
        "  profiles_mode: {}",
        study.name, study.type, ", ".join(study.methods),
        cfg.methods.td.mode, cfg.methods.pcfm.mode, cfg.profiles.mode,
    )

    td = cfg.methods.td
    pcfm = cfg.methods.pcfm
    gn = cfg.methods.gn
    td_mode = td.mode if "td" in study.methods else "off"
    pcfm_mode = pcfm.mode if "pcfm" in study.methods else "off"
    gn_mode = gn.mode if "gn" in study.methods else "off"
    gn_direct_mode = gn.direct_mode if "gn" in study.methods else "off"
    run_pcfm_workflow(
        cfg_path=config_path,
        profile_path=cfg.profiles.path,
        launch_csv_path=cfg.profiles.launch_csv,
        power_profiles_mode=cfg.profiles.mode,
        td_mode=td_mode,
        pcfm_mode=pcfm_mode,
        gn_mode=gn_mode,
        gn_direct_mode=gn_direct_mode,
        pcfm_numeric_sci=pcfm.numeric_sci,
        pcfm_numeric_xci=pcfm.numeric_xci,
        pcfm_degree=pcfm.degree,
        pcfm_include_mci=pcfm.include_mci,
        td_exclude_self_channel=td.exclude_self_channel,
        plot_mode="on" if study.plot else "off",
    )


def _run_subset(config_path: Path, system: System, cfg: StudiesRuntimeConfig, study: StudyConfig) -> None:
    lg.info(
        "Study: {}\n"
        "  type: {}\n"
        "  methods: {}\n"
        "  subset_mode: {}\n"
        "  subset_center: {}\n"
        "  subset_half_width: {}",
        study.name, study.type, ", ".join(study.methods),
        study.subset.mode if study.subset else "N/A",
        study.subset.center if study.subset and study.subset.center else "auto",
        study.subset.half_width if study.subset else "N/A",
    )
    subset = resolve_subset(system, study.subset)
    context = build_run_context(
        system=system,
        config_path=config_path,
        out_dir=study.out_dir,
        profiles=cfg.profiles,
        cache_prefix=f"study_{study.name}_{subset.tag}",
    )
    cut = list(subset.cut_indices)
    rows: list[dict[str, float | int | str]] = []
    td_result: TDResult | None = None

    if "td" in study.methods and cfg.methods.td.mode != "off":
        td_result = run_td(context, cfg.methods.td, cache_scope=f"{study.name}_{subset.tag}")
        td_flat = np.asarray(td_result.output_nlin_w).reshape(-1)
        _save_nlin_csv(
            context.out_dir / "td_subset.csv",
            context.freqs_hz[cut],
            td_flat[cut],
            context.output_signal_power_w[cut],
        )
        rows.extend(
            {"channel_idx": idx, "method": "td", "nlin_w": float(td_flat[idx])}
            for idx in cut
        )

    if "pcfm" in study.methods and cfg.methods.pcfm.mode != "off":
        pcfm_result = run_pcfm(context, cfg.methods.pcfm, cache_scope=f"{study.name}_{subset.tag}")
        pcfm_flat = np.asarray(pcfm_result.output_nlin_w).reshape(-1)
        _save_nlin_csv(
            context.out_dir / "pcfm_subset.csv",
            context.freqs_hz[cut],
            pcfm_flat[cut],
            context.output_signal_power_w[cut],
        )
        rows.extend(
            {"channel_idx": idx, "method": "pcfm", "nlin_w": float(pcfm_flat[idx])}
            for idx in cut
        )
        if pcfm_result.eq18_xci_output_w is not None:
            eq18_flat = np.asarray(pcfm_result.eq18_xci_output_w).reshape(-1)
            _save_nlin_csv(
                context.out_dir / "pcfm_eq18_xci_subset.csv",
                context.freqs_hz[cut],
                eq18_flat[cut],
                context.output_signal_power_w[cut],
            )
            rows.extend(
                {"channel_idx": idx, "method": "pcfm_eq18_xci", "nlin_w": float(eq18_flat[idx])}
                for idx in cut
            )

    if "mc" in study.methods and cfg.methods.mc.mode != "off":
        _require_td_for_mc(study, cfg)
        if td_result is None:
            td_result = run_td(context, cfg.methods.td, cache_scope=f"{study.name}_{subset.tag}")
        mc_result = run_mc(context, td_result, cfg.methods.td, cfg.methods.mc)
        for idx in cut:
            rows.append(
                {
                    "channel_idx": idx,
                    "method": "mc",
                    "nlin_w": float("nan"),
                    "chi1": float(np.asarray(mc_result.chi1).reshape(-1)[idx]),
                    "chi2": float(np.asarray(mc_result.chi2).reshape(-1)[idx]),
                    "prefactor": float(np.asarray(mc_result.prefactor).reshape(-1)[idx]),
                    "nlin_16qam_w": float(np.asarray(mc_result.nlin_16qam_output_w).reshape(-1)[idx]),
                }
            )

    _save_subset_summary(context.out_dir / "subset_summary.csv", rows)

    if study.plot:
        fig, ax = plt.subplots(figsize=(5, 3))
        freqs_thz = np.asarray(context.freqs_hz[cut]) * 1e-12
        methods_in_rows = {r["method"] for r in rows}
        for method in sorted(methods_in_rows):
            vals = np.array([r["nlin_w"] for r in rows if r["method"] == method])
            marker = "o" if method == "td" else "^"
            label = method.upper()
            ax.plot(freqs_thz, vals, marker=marker, label=label, lw=0.8, markersize=4)
        ax.set_xlabel("Frequency [THz]")
        ax.set_ylabel("NLIN power [W]")
        ax.legend()
        fig.tight_layout()
        out_path = context.out_dir / "subset_nlin.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        lg.success("Saved subset NLIN plot to {}", out_path)
        plt.close(fig)


_REWRITE_ATTRS = {"length", "fiber.length", "baud", "pulse.baud_rate", "spacing", "wdm.spacing"}


def _set_sweep_variable(system: System, variable: str, value: float) -> str:
    if variable in {"length", "fiber.length"}:
        system.fiber.length = float(value)
        return f"{float(value) * 1e-3:.1f} km"
    if variable in {"baud", "pulse.baud_rate"}:
        system.pulse.baud_rate = float(value)
        if system.pulse_config is not None:
            system.pulse_config.baud_rate = float(value)
        return f"{float(value) * 1e-9:.2f} GBd"
    if variable in {"spacing", "wdm.spacing"}:
        from pynlin.wdm import RegularWDM

        if not isinstance(system.wdm, RegularWDM):
            raise ValueError(
                f"Sweep variable 'spacing' requires a RegularWDM grid. "
                f"The current system uses {type(system.wdm).__name__}."
            )
        system.wdm.spacing = float(value)
        return f"{float(value) * 1e-9:.2f} GHz"
    raise ValueError(
        f"Sweep variable {variable!r} is not supported. "
        f"Expected one of: {sorted(_REWRITE_ATTRS)}."
    )


def _run_sweep(config_path: Path, cfg: StudiesRuntimeConfig, study: StudyConfig) -> None:
    sweep = study.sweep
    if sweep is None:
        raise ValueError(f"Sweep study '{study.name}' has no [sweep] section.")
    values = list(sweep.values)
    if not values:
        raise ValueError(f"Sweep study '{study.name}' has no sweep values.")
    lg.info(
        "Study: {}\n"
        "  type: {}\n"
        "  methods: {}\n"
        "  sweep_variable: {}\n"
        "  sweep_values: {}",
        study.name, study.type, ", ".join(study.methods),
        sweep.variable, ", ".join(f"{v:.6g}" for v in values),
    )

    out_dir = Path(study.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_system = System.from_toml(config_path)
    center_idx, band_label = _select_scaling_channel(base_system)

    rows: list[dict[str, float | int | str]] = []
    sweep_variable = sweep.variable.strip().lower()
    sweep_file_token = sweep_variable.replace(".", "_")
    for val in values:
        system = System.from_toml(config_path)
        sweep_label = _set_sweep_variable(system, sweep_variable, float(val))
        sweep_tag = safe_tag(float(val))
        profile_tag = f"sweep_{sweep_file_token}_{sweep_tag}"
        context = build_run_context(
            system=system,
            config_path=config_path,
            out_dir=out_dir,
            profiles=cfg.profiles,
            profile_path=out_dir / f"flat_profile_{profile_tag}.npy",
            profile_mode="flat",
            cache_prefix=profile_tag,
        )

        td_val = float("nan")
        td_result: TDResult | None = None
        if "td" in study.methods and cfg.methods.td.mode != "off":
            td_result = run_td(context, cfg.methods.td, cache_scope=profile_tag)
            td_val = float(np.asarray(td_result.output_nlin_w).reshape(-1)[center_idx])

        pcfm_val = float("nan")
        pcfm_eq18_xci_val = float("nan")
        if "pcfm" in study.methods and cfg.methods.pcfm.mode != "off":
            pcfm_result = run_pcfm(context, cfg.methods.pcfm, cache_scope=profile_tag)
            pcfm_val = float(np.asarray(pcfm_result.output_nlin_w).reshape(-1)[center_idx])
            if pcfm_result.eq18_xci_output_w is not None:
                pcfm_eq18_xci_val = float(np.asarray(pcfm_result.eq18_xci_output_w).reshape(-1)[center_idx])

        chi1_val = float("nan")
        chi2_val = float("nan")
        prefactor_val = float("nan")
        nlin_16qam_w = float("nan")
        if "mc" in study.methods and cfg.methods.mc.mode != "off":
            _require_td_for_mc(study, cfg)
            if td_result is None:
                td_result = run_td(context, cfg.methods.td, cache_scope=profile_tag)
            mc_result = run_mc(context, td_result, cfg.methods.td, cfg.methods.mc)
            chi1_val = float(np.asarray(mc_result.chi1).reshape(-1)[center_idx])
            chi2_val = float(np.asarray(mc_result.chi2).reshape(-1)[center_idx])
            prefactor_val = float(np.asarray(mc_result.prefactor).reshape(-1)[center_idx])
            nlin_16qam_w = float(np.asarray(mc_result.nlin_16qam_output_w).reshape(-1)[center_idx])

        rows.append(
            {
                "sweep_value": float(val),
                "sweep_label": str(sweep_label),
                "channel_idx": int(center_idx),
                "band_label": str(band_label),
                "td_nlin_w": td_val,
                "pcfm_nlin_w": pcfm_val,
                "pcfm_eq18_xci_w": pcfm_eq18_xci_val,
                "chi1": chi1_val,
                "chi2": chi2_val,
                "prefactor": prefactor_val,
                "output_ratio": float(context.output_over_launch_ratio[center_idx]),
                "nlin_16qam_w": nlin_16qam_w,
            }
        )

    path = out_dir / f"sweep_{sweep_file_token}.csv"
    fields = [
        "sweep_value",
        "sweep_label",
        "channel_idx",
        "band_label",
        "td_nlin_w",
        "pcfm_nlin_w",
        "pcfm_eq18_xci_w",
        "chi1",
        "chi2",
        "prefactor",
        "output_ratio",
        "nlin_16qam_w",
    ]
    data = np.array([[row[k] for k in fields] for row in rows], dtype=object)
    np.savetxt(path, data, delimiter=",", fmt="%s", header=",".join(fields), comments="")
    lg.success("Saved sweep study '{}' to {}", study.name, path)

    if study.plot:
        sweep_labels = [r["sweep_label"] for r in rows]
        sweep_values = [float(r["sweep_value"]) for r in rows]
        xlabel = {"length": "Fiber length [km]", "fiber.length": "Fiber length [km]",
                  "baud": "Baud rate [GBd]", "pulse.baud_rate": "Baud rate [GBd]",
                  "spacing": "Channel spacing [GHz]", "wdm.spacing": "Channel spacing [GHz]"}.get(sweep_variable, sweep_variable)
        fig, ax = plt.subplots(figsize=(5, 3))
        td_vals = [float(r["td_nlin_w"]) for r in rows]
        pcfm_vals = [float(r["pcfm_nlin_w"]) for r in rows]
        pcfm_eq18_vals = [float(r["pcfm_eq18_xci_w"]) for r in rows]
        ax.plot(sweep_values, td_vals, marker="o", label="TD", lw=0.8)
        ax.plot(sweep_values, pcfm_vals, marker="^", label="PCFM", lw=0.8)
        if np.any(np.isfinite(pcfm_eq18_vals)):
            ax.plot(sweep_values, pcfm_eq18_vals, marker="s", label="PCFM Eq. 18 XCI", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("NLIN power [W]")
        ax.legend()
        fig.tight_layout()
        plot_path = out_dir / f"sweep_{sweep_file_token}.pdf"
        fig.savefig(plot_path, dpi=300)
        lg.success("Saved sweep NLIN plot to {}", plot_path)
        plt.close(fig)


def run_studies(config_path: Path | str) -> None:
    config_path = Path(config_path)
    system = System.from_toml(config_path)
    cfg = load_studies_runtime_config(system)
    for study in cfg.studies:
        lg.info("--- Starting study: {} ---", study.name)
        if study.type == "full_system":
            _run_full_system(config_path, cfg, study)
        elif study.type == "subset":
            _run_subset(config_path, System.from_toml(config_path), cfg, study)
        elif study.type == "sweep":
            _run_sweep(config_path, cfg, study)
        else:
            raise NotImplementedError(f"Study type {study.type!r} is parsed but not implemented yet.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run named studies from a system TOML.")
    parser.add_argument("--config", type=str, default="input/studies.toml")
    args = parser.parse_args()
    run_studies(Path(args.config))


if __name__ == "__main__":
    main()
