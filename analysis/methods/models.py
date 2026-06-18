from pathlib import Path

import numpy as np
from loguru import logger as lg

from .analytics import pcfm_general
from pynlin.methods.pcfm import (
    PcfmConfig,
    compute_gn_direct,
    compute_gn_numeric,
    compute_pcfm_nlin,
)
from pynlin.system import System

_NLIN_CACHE_VERSION = 2


def _component_paths(output_path: Path) -> tuple[Path, Path]:
    return (
        output_path.with_name(f"{output_path.stem}_sci.npy"),
        output_path.with_name(f"{output_path.stem}_xci.npy"),
    )


def _version_path(output_path: Path) -> Path:
    return output_path.with_suffix(".npy.ver")


def _load_npy_with_version(path: Path) -> np.ndarray:
    """Load a .npy file and verify its cache version."""
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        ver = int(np.asarray(loaded.get("cache_version", 0)).item())
        if ver != _NLIN_CACHE_VERSION:
            lg.warning(f"Cache version mismatch ({ver} != {_NLIN_CACHE_VERSION}) at {path}; treating as stale.")
            raise FileNotFoundError(f"Stale cache version at {path}")
        return loaded["nlin"]
    ver_path = _version_path(path)
    if not ver_path.exists():
        lg.warning(f"Cache version missing at {path}; treating as stale.")
        raise FileNotFoundError(f"Unversioned cache at {path}")
    ver = int(ver_path.read_text().strip())
    if ver != _NLIN_CACHE_VERSION:
        lg.warning(f"Cache version mismatch ({ver} != {_NLIN_CACHE_VERSION}) at {path}; treating as stale.")
        raise FileNotFoundError(f"Stale cache version at {path}")
    return loaded


def _save_with_version(path: Path, arr: np.ndarray) -> None:
    """Save a .npy file with a sidecar version marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    _version_path(path).write_text(str(_NLIN_CACHE_VERSION))


def _load_cached(
    output_path: Path,
    return_components: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    nlin = _load_npy_with_version(output_path)
    if not return_components:
        return nlin
    sci_path, xci_path = _component_paths(output_path)
    sci = xci = None
    try:
        sci = _load_npy_with_version(sci_path)
    except Exception:
        lg.warning(f"SCI component missing or stale at {sci_path}; will recompute.")
    try:
        xci = _load_npy_with_version(xci_path)
    except Exception:
        lg.warning(f"XCI component missing or stale at {xci_path}; will recompute.")
    if sci is not None and xci is not None:
        return nlin, sci, xci
    raise FileNotFoundError(f"Incomplete cached components at {output_path}")


def _save(
    output_path: Path,
    nlin: np.ndarray,
    components: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_with_version(output_path, nlin)
    if components is None:
        return
    sci_path, xci_path = _component_paths(output_path)
    _save_with_version(sci_path, components[0])
    _save_with_version(xci_path, components[1])


def _load_or_compute_pcfm_I(
    system: System,
    profile_path: Path | str,
    launch_powers_w: np.ndarray,
    output_path: Path,
    cfg: PcfmConfig,
    recompute: bool = False,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load PCFM NLIN and persist to .npy."""
    if output_path.exists() and not recompute:
        lg.debug(f"Loading cached PCFM NLIN from {output_path}")
        try:
            return _load_cached(output_path, return_components)
        except Exception:
            lg.warning(f"Stale or invalid cache at {output_path}; recomputing.")

    lg.info("Computing PCFM NLIN")
    out = compute_pcfm_nlin(
        system=system,
        profile_path=profile_path,
        launch_powers_w=launch_powers_w,
        config=cfg,
        return_components=return_components,
    )
    if return_components:
        nlin, sci, xci = out
        _save(output_path, nlin, (sci, xci))
        return nlin, sci, xci
    _save(output_path, out)
    return out


def _load_or_compute_pcfm_general(
    system: System,
    launch_powers_w: np.ndarray,
    output_path: Path,
    *,
    profile_path: Path | str | None = None,
    degree: int = 9,
    xci_model: str,
    recompute: bool = False,
) -> np.ndarray:
    """Compute or load an analytic XCI vector and persist to .npy."""
    if output_path.exists() and not recompute:
        lg.debug(f"Loading cached analytic XCI from {output_path}")
        try:
            return _load_npy_with_version(output_path)
        except Exception:
            lg.warning(f"Stale or invalid cache at {output_path}; recomputing.")

    lg.info(f"Computing analytic XCI (model={xci_model})")
    if xci_model == "eq18":
        lg.warning("PCFM-II analytic XCI requested.")
    values = np.array(
        [
            pcfm_general(
                system,
                channel_idx=idx,
                launch_powers_w=launch_powers_w,
                profile_path=str(profile_path) if profile_path is not None else None,
                degree=int(degree),
                xci_model=xci_model,
            )
            for idx in range(system.wdm.frequency_grid().size)
        ],
        dtype=float,
    )
    _save_with_version(output_path, values)
    return values


def _load_or_compute_gn(
    system: System,
    profile_path: Path | str,
    launch_powers_w: np.ndarray,
    output_path: Path,
    recompute: bool = False,
    n_f: int = 40,
    n_z: int = 200,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load numeric GN NLIN and persist to .npy."""
    if output_path.exists() and not recompute:
        lg.debug(f"Loading cached numeric GN NLIN from {output_path}")
        try:
            return _load_cached(output_path, return_components)
        except Exception:
            lg.warning(f"Stale or invalid cache at {output_path}; recomputing.")

    lg.info("Computing numeric GN NLIN")
    out = compute_gn_numeric(
        system=system,
        profile_path=profile_path,
        launch_powers_w=launch_powers_w,
        n_f=n_f,
        n_z=n_z,
        return_components=return_components,
    )
    if return_components:
        nlin, sci, xci = out
        _save(output_path, nlin, (sci, xci))
        return nlin, sci, xci
    _save(output_path, out)
    return out


def _load_or_compute_gn_direct(
    system: System,
    profile_path: Path | str,
    launch_powers_w: np.ndarray,
    output_path: Path,
    recompute: bool = False,
    n_f: int = 40,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load direct GN NLIN and persist to .npy."""
    if output_path.exists() and not recompute:
        lg.debug(f"Loading cached direct GN NLIN from {output_path}")
        try:
            return _load_cached(output_path, return_components)
        except Exception:
            lg.warning(f"Stale or invalid cache at {output_path}; recomputing.")

    lg.info("Computing direct GN NLIN")
    out = compute_gn_direct(
        system=system,
        profile_path=profile_path,
        launch_powers_w=launch_powers_w,
        n_f=n_f,
        return_components=return_components,
    )
    if return_components:
        nlin, sci, xci = out
        _save(output_path, nlin, (sci, xci))
        return nlin, sci, xci
    _save(output_path, out)
    return out
