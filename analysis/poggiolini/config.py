from pathlib import Path
from typing import Mapping

from pynlin.system import System

PROFILE_MAX_W = 10.0
_BINARY_MODES = {"off", "on"}
_CACHE_MODES = {"cached", "recompute"}
_MODEL_MODES = {"off", "cached", "recompute"}
_POWER_PROFILE_MODES = {
    "flat",
    "cached",
    "recompute",
    "cached_no_profile_launch",
    "recompute_no_profile_launch",
}


def _flat_profiles_enabled(system: System) -> bool:
    raw = getattr(system, "raw_config", None)
    if not isinstance(raw, Mapping):
        return False
    pog = raw.get("poggiolini")
    if isinstance(pog, Mapping):
        run = pog.get("run")
        if isinstance(run, Mapping):
            mode = run.get("power_profiles_mode")
            if isinstance(mode, str):
                mode = mode.strip().lower()
                if mode == "flat":
                    return True
                if mode in _POWER_PROFILE_MODES:
                    return False
    nlin_section = raw.get("nlin")
    if not isinstance(nlin_section, Mapping):
        return False
    return bool(nlin_section.get("flat_profiles") or nlin_section.get("flat_profile"))


def _load_poggiolini_runtime_config(system: System) -> dict[str, object]:
    """Load workflow/runtime flags from [poggiolini] in the system TOML."""
    defaults: dict[str, object] = {
        "profile_path": "results/poggiolini_power_profiles.npy",
        "launch_csv_path": "results/poggiolini_launch_power.csv",
        "pcfm_numeric_xci": False,
        "include_lumped_losses": False,
        "power_profiles_mode": "recompute",
        "td_mode": "cached",
        "pcfm_mode": "cached",
        "gn_mode": "off",
        "gn_direct_mode": "off",
        "plot_mode": "on",
    }
    raw = getattr(system, "raw_config", None)
    if not isinstance(raw, Mapping):
        return defaults
    pog = raw.get("poggiolini")
    if not isinstance(pog, Mapping):
        return defaults
    paths = pog.get("paths")
    run = pog.get("run")
    recompute = pog.get("recompute")
    if isinstance(paths, Mapping):
        if "profile" in paths:
            defaults["profile_path"] = paths.get("profile")
        if "launch_csv" in paths:
            defaults["launch_csv_path"] = paths.get("launch_csv")
    if isinstance(run, Mapping):
        legacy_run_keys = [
            key
            for key in (
                "compute_gn",
                "compute_gn_direct",
                "plot",
                "recompute_all",
                "recompute_profiles",
                "recompute_td",
                "recompute_pcfm",
                "recompute_gn",
                "recompute_gn_direct",
                "profiles_mode",
                "use_profile_launch_powers",
            )
            if key in run
        ]
        if legacy_run_keys:
            raise ValueError(
                "Legacy Poggiolini run keys are no longer supported: "
                f"{legacy_run_keys}. Use *_mode keys in [poggiolini.run]."
            )
        for key in (
            "pcfm_numeric_xci",
            "include_lumped_losses",
            "power_profiles_mode",
            "td_mode",
            "pcfm_mode",
            "gn_mode",
            "gn_direct_mode",
            "plot_mode",
        ):
            if key in run:
                defaults[key] = run.get(key)
    if isinstance(recompute, Mapping):
        raise ValueError(
            "Legacy [poggiolini.recompute] is no longer supported. "
            "Use *_mode keys in [poggiolini.run]."
        )

    defaults["power_profiles_mode"] = _normalize_mode(
        "power_profiles_mode", defaults["power_profiles_mode"], _POWER_PROFILE_MODES
    )
    defaults["td_mode"] = _normalize_mode("td_mode", defaults["td_mode"], _CACHE_MODES)
    defaults["pcfm_mode"] = _normalize_mode("pcfm_mode", defaults["pcfm_mode"], _CACHE_MODES)
    defaults["gn_mode"] = _normalize_mode("gn_mode", defaults["gn_mode"], _MODEL_MODES)
    defaults["gn_direct_mode"] = _normalize_mode(
        "gn_direct_mode", defaults["gn_direct_mode"], _MODEL_MODES
    )
    defaults["plot_mode"] = _normalize_mode("plot_mode", defaults["plot_mode"], _BINARY_MODES)
    return defaults


def _to_optional_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return Path(text)


def _normalize_mode(name: str, value: object, allowed: set[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string in {sorted(allowed)}; got {value!r}.")
    mode = value.strip().lower()
    if mode not in allowed:
        raise ValueError(f"Invalid {name}={value!r}; expected one of {sorted(allowed)}.")
    return mode
