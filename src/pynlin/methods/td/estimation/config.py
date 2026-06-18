"""Shared NLIN estimation configuration helpers."""


def flat_profiles_enabled(system) -> bool:
    """Return whether the run requests flat-profile Raman shortcuts."""
    raw = getattr(system, "raw_config", None)
    if not isinstance(raw, dict):
        return False

    pcfm = raw.get("pcfm")
    if isinstance(pcfm, dict):
        run = pcfm.get("run")
        if isinstance(run, dict):
            mode = run.get("power_profiles_mode")
            if isinstance(mode, str):
                mode = mode.strip().lower()
                if mode == "flat":
                    return True
                if mode in {
                    "cached",
                    "recompute",
                    "cached_no_profile_launch",
                    "recompute_no_profile_launch",
                }:
                    return False

    nlin_section = raw.get("nlin")
    if not isinstance(nlin_section, dict):
        return False
    return bool(nlin_section.get("flat_profiles") or nlin_section.get("flat_profile"))


__all__ = ["flat_profiles_enabled"]
