import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from analysis.pcfm.config import (
        PROFILE_MAX_W,
        _flat_profiles_enabled,
        _load_pcfm_runtime_config,
        _to_optional_path,
    )
    from analysis.pcfm.io import (
        _load_launch_powers_csv,
        _profile_needs_recompute,
        _resolve_launch_powers,
        _resolve_signal_power,
        _save_nlin_csv,
        _write_flat_profile,
    )
    from analysis.pcfm.models import (
        _load_or_compute_gn,
        _load_or_compute_gn_direct,
        _load_or_compute_pcfm_I,
    )
    from analysis.pcfm.plotting import (
        plot_pcfm_diagnostics,
        plot_pcfm_gsnr,
        plot_pcfm_nlin_power,
    )
    from analysis.pcfm.reporting import (
        _format_array_snippet,
        _format_param_table,
        _log_td_pcfm_parameters,
        _summarize_array,
    )
    from analysis.pcfm.td import (
        _qam_mu0,
        _td_modulation_components,
        _td_prefactor_coeffs,
    )
    from analysis.pcfm.workflow import run_pcfm_workflow
except ModuleNotFoundError:
    from pcfm.config import (  # type: ignore[no-redef]
        PROFILE_MAX_W,
        _flat_profiles_enabled,
        _load_pcfm_runtime_config,
        _to_optional_path,
    )
    from pcfm.io import (  # type: ignore[no-redef]
        _load_launch_powers_csv,
        _profile_needs_recompute,
        _resolve_launch_powers,
        _resolve_signal_power,
        _save_nlin_csv,
        _write_flat_profile,
    )
    from pcfm.models import (  # type: ignore[no-redef]
        _load_or_compute_gn,
        _load_or_compute_gn_direct,
        _load_or_compute_pcfm_I,
    )
    from pcfm.plotting import (  # type: ignore[no-redef]
        plot_pcfm_diagnostics,
        plot_pcfm_gsnr,
        plot_pcfm_nlin_power,
    )
    from pcfm.reporting import (  # type: ignore[no-redef]
        _format_array_snippet,
        _format_param_table,
        _log_td_pcfm_parameters,
        _summarize_array,
    )
    from pcfm.td import (  # type: ignore[no-redef]
        _qam_mu0,
        _td_modulation_components,
        _td_prefactor_coeffs,
    )
    from pcfm.workflow import run_pcfm_workflow  # type: ignore[no-redef]


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
    "_load_pcfm_runtime_config",
    "_to_optional_path",
    "_write_flat_profile",
    "_load_launch_powers_csv",
    "_resolve_launch_powers",
    "_resolve_signal_power",
    "_save_nlin_csv",
    "_load_or_compute_pcfm_I",
    "_load_or_compute_gn",
    "_load_or_compute_gn_direct",
    "plot_pcfm_gsnr",
    "plot_pcfm_nlin_power",
    "plot_pcfm_diagnostics",
    "run_pcfm_workflow",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCFM TD/GN workflow runner")
    parser.add_argument(
        "--config",
        type=str,
        default="./input/pcfm_struct.toml",
        help="Path to system TOML (includes [pcfm] runtime settings).",
    )
    args = parser.parse_args()

    run_pcfm_workflow(cfg_path=Path(args.config))
