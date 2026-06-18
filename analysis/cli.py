import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from analysis.config import (
        PROFILE_MAX_W,
        _flat_profiles_enabled,
        _load_pcfm_runtime_config,
        _to_optional_path,
    )
    from analysis.methods.io import (
        _load_launch_powers_csv,
        _profile_needs_recompute,
        _resolve_launch_powers,
        _resolve_signal_power,
        _save_nlin_csv,
        _write_flat_profile,
    )
    from analysis.methods.models import (
        _load_or_compute_gn,
        _load_or_compute_gn_direct,
        _load_or_compute_pcfm_I,
    )
    from analysis.methods.plotting import (
        plot_pcfm_diagnostics,
        plot_pcfm_gsnr,
        plot_pcfm_nlin_power,
    )
    from analysis.methods.reporting import (
        _format_array_snippet,
        _format_param_table,
        _log_td_pcfm_parameters,
        _summarize_array,
    )
    from pynlin.methods.td import (
        _qam_mu0,
        _td_modulation_components,
        _td_prefactor_coeffs,
    )
    from analysis.methods.workflow import run_pcfm_workflow
except ModuleNotFoundError:
    from analysis.config import (  # type: ignore[no-redef]
        PROFILE_MAX_W,
        _flat_profiles_enabled,
        _load_pcfm_runtime_config,
        _to_optional_path,
    )
    from methods.io import (  # type: ignore[no-redef]
        _load_launch_powers_csv,
        _profile_needs_recompute,
        _resolve_launch_powers,
        _resolve_signal_power,
        _save_nlin_csv,
        _write_flat_profile,
    )
    from methods.models import (  # type: ignore[no-redef]
        _load_or_compute_gn,
        _load_or_compute_gn_direct,
        _load_or_compute_pcfm_I,
    )
    from methods.plotting import (  # type: ignore[no-redef]
        plot_pcfm_diagnostics,
        plot_pcfm_gsnr,
        plot_pcfm_nlin_power,
    )
    from methods.reporting import (  # type: ignore[no-redef]
        _format_array_snippet,
        _format_param_table,
        _log_td_pcfm_parameters,
        _summarize_array,
    )
    from pynlin.methods.td import (  # type: ignore[no-redef]
        _qam_mu0,
        _td_modulation_components,
        _td_prefactor_coeffs,
    )
    from methods.workflow import run_pcfm_workflow  # type: ignore[no-redef]


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
    parser = argparse.ArgumentParser(description="Named studies workflow runner")
    parser.add_argument(
        "--config",
        type=str,
        default="./input/studies.toml",
        help="Path to system TOML (includes [profiles], [methods.*], and [studies.*]).",
    )
    args = parser.parse_args()

    from analysis.studies import run_studies

    run_studies(Path(args.config))
