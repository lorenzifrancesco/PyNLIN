import argparse
from pathlib import Path

try:
    from analysis.poggiolini.config import (
        PROFILE_MAX_W,
        _flat_profiles_enabled,
        _load_poggiolini_runtime_config,
        _to_optional_path,
    )
    from analysis.poggiolini.io import (
        _load_launch_powers_csv,
        _profile_needs_recompute,
        _resolve_launch_powers,
        _resolve_signal_power,
        _save_nlin_csv,
        _write_flat_profile,
    )
    from analysis.poggiolini.models import (
        _load_or_compute_gn,
        _load_or_compute_gn_direct,
        _load_or_compute_pcfm,
    )
    from analysis.poggiolini.plotting import (
        plot_poggiolini_diagnostics,
        plot_poggiolini_gsnr,
        plot_poggiolini_nlin_power,
    )
    from analysis.poggiolini.reporting import (
        _format_array_snippet,
        _format_param_table,
        _log_td_pcfm_parameters,
        _summarize_array,
    )
    from analysis.poggiolini.td import (
        _qam_mu0,
        _td_modulation_components,
        _td_prefactor_coeffs,
    )
    from analysis.poggiolini.workflow import run_poggiolini_workflow
except ModuleNotFoundError:
    from poggiolini.config import (  # type: ignore[no-redef]
        PROFILE_MAX_W,
        _flat_profiles_enabled,
        _load_poggiolini_runtime_config,
        _to_optional_path,
    )
    from poggiolini.io import (  # type: ignore[no-redef]
        _load_launch_powers_csv,
        _profile_needs_recompute,
        _resolve_launch_powers,
        _resolve_signal_power,
        _save_nlin_csv,
        _write_flat_profile,
    )
    from poggiolini.models import (  # type: ignore[no-redef]
        _load_or_compute_gn,
        _load_or_compute_gn_direct,
        _load_or_compute_pcfm,
    )
    from poggiolini.plotting import (  # type: ignore[no-redef]
        plot_poggiolini_diagnostics,
        plot_poggiolini_gsnr,
        plot_poggiolini_nlin_power,
    )
    from poggiolini.reporting import (  # type: ignore[no-redef]
        _format_array_snippet,
        _format_param_table,
        _log_td_pcfm_parameters,
        _summarize_array,
    )
    from poggiolini.td import (  # type: ignore[no-redef]
        _qam_mu0,
        _td_modulation_components,
        _td_prefactor_coeffs,
    )
    from poggiolini.workflow import run_poggiolini_workflow  # type: ignore[no-redef]


__all__ = [
    "PROFILE_MAX_W",
    "_format_array_snippet",
    "_summarize_array",
    "_format_param_table",
    "_log_td_pcfm_parameters",
    "_qam_mu0",
    "_td_prefactor_coeffs",
    "_td_modulation_components",
    "_profile_needs_recompute",
    "_flat_profiles_enabled",
    "_load_poggiolini_runtime_config",
    "_to_optional_path",
    "_write_flat_profile",
    "_load_launch_powers_csv",
    "_resolve_launch_powers",
    "_resolve_signal_power",
    "_save_nlin_csv",
    "_load_or_compute_pcfm",
    "_load_or_compute_gn",
    "_load_or_compute_gn_direct",
    "plot_poggiolini_gsnr",
    "plot_poggiolini_nlin_power",
    "plot_poggiolini_diagnostics",
    "run_poggiolini_workflow",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poggiolini PCFM/TD workflow runner")
    parser.add_argument(
        "--config",
        type=str,
        default="./input/poggiolini_struct.toml",
        help="Path to system TOML (includes [poggiolini] runtime settings).",
    )
    args = parser.parse_args()

    run_poggiolini_workflow(cfg_path=Path(args.config))
