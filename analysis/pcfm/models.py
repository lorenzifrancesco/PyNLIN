from pathlib import Path

import numpy as np
from loguru import logger as lg

from .analytics import flat_profile_pcfm_xci_channel_power
from pynlin.nlin.pcfm_gn import (
    PcfmConfig,
    compute_gn_direct,
    compute_gn_numeric,
    compute_pcfm_nlin,
)
from pynlin.system import System


def _component_paths(output_path: Path) -> tuple[Path, Path]:
    return (
        output_path.with_name(f"{output_path.stem}_sci.npy"),
        output_path.with_name(f"{output_path.stem}_xci.npy"),
    )


def _load_cached(
    output_path: Path,
    return_components: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    nlin = np.load(output_path)
    if not return_components:
        return nlin
    sci_path, xci_path = _component_paths(output_path)
    return nlin, np.load(sci_path), np.load(xci_path)


def _save(
    output_path: Path,
    nlin: np.ndarray,
    components: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, nlin)
    if components is None:
        return
    sci_path, xci_path = _component_paths(output_path)
    np.save(sci_path, components[0])
    np.save(xci_path, components[1])


def _load_or_compute_pcfm(
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
        lg.info(f"Loading cached PCFM NLIN from {output_path}")
        return _load_cached(output_path, return_components)

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


def _load_or_compute_flat_analytic_xci(
    system: System,
    launch_powers_w: np.ndarray,
    output_path: Path,
    *,
    profile_path: Path | str | None = None,
    degree: int = 9,
    xci_model: str,
    recompute: bool = False,
) -> np.ndarray:
    """Compute or load a flat-profile analytic XCI vector and persist to .npy."""
    if output_path.exists() and not recompute:
        lg.info(f"Loading cached flat analytic XCI from {output_path}")
        return np.load(output_path)

    lg.info(f"Computing flat analytic XCI (model={xci_model})")
    if xci_model == "eq18":
        lg.warning("Eq. 18 flat analytic XCI requested.")
        lg.warning("This path evaluates the special-function kernel for all channels.")
        lg.warning("Use cached results when possible; recomputation is still heavier than closed-form XCI.")
    values = np.array(
        [
            flat_profile_pcfm_xci_channel_power(
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, values)
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
        lg.info(f"Loading cached numeric GN NLIN from {output_path}")
        return _load_cached(output_path, return_components)

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
        lg.info(f"Loading cached direct GN NLIN from {output_path}")
        return _load_cached(output_path, return_components)

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
