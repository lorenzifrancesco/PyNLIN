from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.config import ProfilesConfig
from analysis.methods.io import _resolve_launch_powers, _write_flat_profile
from analysis.uwb_nlin import compute_raman_profiles
from pynlin.system import System


def ensure_profile(
    system: System,
    profiles: ProfilesConfig,
    *,
    profile_path: Path | str | None = None,
    profile_mode: str | None = None,
) -> tuple[Path, np.ndarray]:
    path = Path(profile_path) if profile_path is not None else Path(profiles.path)
    mode = profile_mode or profiles.mode
    recompute_profiles = mode in {"recompute", "recompute_no_profile_launch"}
    flat_profiles = mode == "flat"
    use_profile_launch = mode in {"cached", "recompute"}

    if flat_profiles:
        launch = _resolve_launch_powers(
            system,
            profile_path=None,
            launch_csv_path=profiles.launch_csv,
            use_profile=False,
        )
        _write_flat_profile(path, system, launch_powers_w=launch)
        return path, launch

    if path.exists():
        compute_raman_profiles(system, save_path=path, recompute=recompute_profiles)
    else:
        if not recompute_profiles:
            raise FileNotFoundError(
                f"Profile missing at {path}; set [profiles].mode='recompute'."
            )
        compute_raman_profiles(system, save_path=path, recompute=True)

    launch = _resolve_launch_powers(
        system,
        path,
        profiles.launch_csv,
        use_profile=use_profile_launch,
    )
    return path, launch
