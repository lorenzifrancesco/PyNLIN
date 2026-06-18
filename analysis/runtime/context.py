from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.config import ProfilesConfig
from analysis.methods.io import (
    _output_over_launch_signal_power_ratio,
    _power_profile_hash,
    _resolve_signal_power,
)
from analysis.runtime.cache import dispersion_hash, launch_hash, method_cache_tag
from analysis.runtime.profiles import ensure_profile
from pynlin.system import System


@dataclass(frozen=True)
class RunContext:
    system: System
    config_path: Path
    out_dir: Path
    profile_path: Path
    launch_powers_w: np.ndarray
    output_signal_power_w: np.ndarray
    output_over_launch_ratio: np.ndarray
    freqs_hz: np.ndarray
    profile_hash: str
    dispersion_hash: str
    launch_hash: str
    cache_tag: str


def build_run_context(
    *,
    system: System,
    config_path: Path | str,
    out_dir: Path | str,
    profiles: ProfilesConfig,
    profile_path: Path | str | None = None,
    profile_mode: str | None = None,
    cache_prefix: str | None = None,
) -> RunContext:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_profile_path, launch = ensure_profile(
        system,
        profiles,
        profile_path=profile_path,
        profile_mode=profile_mode,
    )
    output_signal_power = _resolve_signal_power(
        system,
        resolved_profile_path,
        launch_override=launch,
    )
    output_ratio = _output_over_launch_signal_power_ratio(
        system,
        resolved_profile_path,
        launch,
    )
    profile_tag = _power_profile_hash(system, resolved_profile_path)
    disp_tag = dispersion_hash(system)
    launch_tag = launch_hash(launch)
    cache_tag = method_cache_tag(cache_prefix, f"disp{disp_tag}", f"prof{profile_tag}", f"launch{launch_tag}")
    return RunContext(
        system=system,
        config_path=Path(config_path),
        out_dir=out_dir,
        profile_path=resolved_profile_path,
        launch_powers_w=np.asarray(launch, dtype=float).reshape(-1),
        output_signal_power_w=np.asarray(output_signal_power, dtype=float).reshape(-1),
        output_over_launch_ratio=np.asarray(output_ratio, dtype=float).reshape(-1),
        freqs_hz=np.asarray(system.wdm.frequency_grid(), dtype=float).reshape(-1),
        profile_hash=profile_tag,
        dispersion_hash=disp_tag,
        launch_hash=launch_tag,
        cache_tag=cache_tag,
    )
