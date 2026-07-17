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
from pynlin.methods.td.fullband_mc import (
    FULLBAND_MC_CACHE_VERSION,
    FullbandMCDiagnostic,
    compute_fullband_prefactor_free_mc,
    decimated_frequency_grid,
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
        lg.warning(f"PCFM-II analytic XCI requested with SPP fit degree={int(degree)}.")
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


from pynlin.methods.td.fullband_mc import FWMPruningStats  # noqa: E402


def _fullband_mc_target_indices(
    system: System,
    *,
    channel_decimation: int = 1,
    target_decimation: int = 1,
    target_offset: int = 0,
    target_limit: int | None = None,
) -> np.ndarray:
    """Select target indices on the decimated grid."""
    indices, _ = decimated_frequency_grid(system, channel_decimation)
    base = np.arange(indices.size, dtype=int)[target_offset::target_decimation]
    if target_limit is not None:
        base = base[:max(int(target_limit), 0)]
    return base


def _load_or_compute_fullband_mc(
    system: System,
    output_path: Path,
    *,
    channel_decimation: int = 1,
    target_decimation: int = 1,
    target_offset: int = 0,
    target_limit: int | None = None,
    target_indices: np.ndarray | None = None,
    xpm_samples: int = 10000,
    fwm_samples: int = 5000,
    fwm_frequency_samples: int = 50,
    seed: int = 1234,
    max_fwm_tuples_per_target: int | None = None,
    fwm_tuple_selection: str = "joint_reservoir",
    workers: int = 1,
    recompute: bool = False,
) -> FullbandMCDiagnostic:
    """Compute or load a prefactor-free fullband MC diagnostic NPZ."""
    if target_indices is None:
        target_indices = _fullband_mc_target_indices(
            system,
            channel_decimation=channel_decimation,
            target_decimation=target_decimation,
            target_offset=target_offset,
            target_limit=target_limit,
        )
    target_indices = np.asarray(target_indices, dtype=int).reshape(-1)
    kept_indices, _ = decimated_frequency_grid(system, channel_decimation)
    expected_targets = kept_indices[target_indices]
    expected_meta = {
        "cache_version": FULLBAND_MC_CACHE_VERSION,
        "decimation": int(channel_decimation),
        "xpm_samples": int(xpm_samples),
        "fwm_samples": int(fwm_samples),
        "fwm_frequency_samples": int(fwm_frequency_samples),
        "seed": int(seed),
        "max_fwm_tuples_per_target": max_fwm_tuples_per_target,
        "fwm_tuple_selection": str(fwm_tuple_selection),
    }
    if output_path.exists() and not recompute:
        lg.debug(f"Loading cached fullband MC diagnostic from {output_path}")
        try:
            data = np.load(output_path, allow_pickle=True)
            meta = {k[len("meta_"):]: data[k].item() for k in data.files if k.startswith("meta_")}
            mismatches = [key for key, value in expected_meta.items() if meta.get(key) != value]
            if mismatches or not np.array_equal(data["target_indices"], expected_targets):
                raise FileNotFoundError(f"Stale fullband MC cache parameters: {', '.join(mismatches)}")
            return FullbandMCDiagnostic(
                target_indices=data["target_indices"],
                target_frequencies=data["target_frequencies"],
                xpm=data["xpm"],
                fwm=data["fwm"],
                total=data["total"],
                fwm_tuple_count=data["fwm_tuple_count"],
                fwm_support_count=data["fwm_support_count"],
                pruning=FWMPruningStats(
                    naive_tuples=int(data["naive_tuples"].item()),
                    support_survivors=int(data["support_survivors"].item()),
                    evaluated_tuples=int(data["evaluated_tuples"].item()),
                ),
                metadata=meta,
            )
        except Exception:
            lg.warning(f"Stale or invalid cache at {output_path}; recomputing.")

    lg.info("Computing fullband prefactor-free MC diagnostic")
    diagnostic = compute_fullband_prefactor_free_mc(
        system,
        decimation=channel_decimation,
        target_indices=target_indices,
        include_xpm=True,
        include_fwm=True,
        xpm_samples=xpm_samples,
        fwm_samples=fwm_samples,
        fwm_frequency_samples=fwm_frequency_samples,
        seed=seed,
        max_fwm_tuples_per_target=max_fwm_tuples_per_target,
        fwm_tuple_selection=fwm_tuple_selection,
        n_workers=workers,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        target_indices=diagnostic.target_indices,
        target_frequencies=diagnostic.target_frequencies,
        xpm=diagnostic.xpm,
        fwm=diagnostic.fwm,
        total=diagnostic.total,
        fwm_tuple_count=diagnostic.fwm_tuple_count,
        fwm_support_count=diagnostic.fwm_support_count,
        naive_tuples=np.array([diagnostic.pruning.naive_tuples]),
        support_survivors=np.array([diagnostic.pruning.support_survivors]),
        evaluated_tuples=np.array([diagnostic.pruning.evaluated_tuples]),
        **{f"meta_{k}": np.array([v]) for k, v in diagnostic.metadata.items() if v is not None},
    )
    lg.success(
        "Fullband MC diagnostic saved to {} ({} targets, {} XPM samples, {} FWM samples)".format(
            output_path, len(diagnostic.target_indices), xpm_samples, fwm_samples
        )
    )
    return diagnostic
