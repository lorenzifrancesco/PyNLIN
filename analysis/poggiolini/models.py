from pathlib import Path

import numpy as np
from loguru import logger as lg

from pynlin.nlin.pcfm_gn import (
    PcfmConfig,
    compute_gn_direct,
    compute_gn_numeric,
    compute_pcfm_nlin,
)
from pynlin.system import System


def _load_or_compute_pcfm(
    system: System,
    profile_path: Path | str,
    launch_powers_w: np.ndarray,
    output_path: Path,
    cfg: PcfmConfig,
    lumped_losses: list[tuple[float, float]] | None = None,
    recompute: bool = False,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load PCFM NLIN and persist to .npy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sci_path = output_path.with_name(f"{output_path.stem}_sci.npy") if return_components else None
    xci_path = output_path.with_name(f"{output_path.stem}_xci.npy") if return_components else None
    if output_path.exists() and not recompute:
        lg.info(f"Loading cached PCFM NLIN from {output_path}")
        cached = np.load(output_path)
        if np.all(np.isfinite(cached)):
            if return_components:
                if sci_path is None or xci_path is None:
                    raise ValueError("PCFM component paths not configured.")
                if sci_path.exists() and xci_path.exists():
                    sci = np.load(sci_path)
                    xci = np.load(xci_path)
                    if np.all(np.isfinite(sci)) and np.all(np.isfinite(xci)):
                        return cached, sci, xci
                    raise ValueError(
                        "Cached PCFM components contain non-finite values; "
                        "set [poggiolini.run].pcfm_mode='recompute'."
                    )
                lg.warning("PCFM component cache missing; recomputing.")
            else:
                return cached
        raise ValueError(
            f"Cached PCFM NLIN at {output_path} contains non-finite values; "
            "set [poggiolini.run].pcfm_mode='recompute'."
        )
    lg.info(f"Computing PCFM NLIN (losses={lumped_losses})")
    if return_components:
        nlin, nlin_sci, nlin_xci = compute_pcfm_nlin(
            system,
            profile_path=profile_path,
            launch_powers_w=launch_powers_w,
            config=cfg,
            lumped_losses=lumped_losses,
            return_components=True,
        )
        np.save(output_path, nlin)
        if sci_path is not None and xci_path is not None:
            np.save(sci_path, nlin_sci)
            np.save(xci_path, nlin_xci)
            lg.success(f"Saved PCFM SCI to {sci_path}")
            lg.success(f"Saved PCFM XCI to {xci_path}")
        lg.success(f"Saved PCFM NLIN to {output_path}")
        return nlin, nlin_sci, nlin_xci
    nlin = compute_pcfm_nlin(
        system,
        profile_path=profile_path,
        launch_powers_w=launch_powers_w,
        config=cfg,
        lumped_losses=lumped_losses,
    )
    np.save(output_path, nlin)
    lg.success(f"Saved PCFM NLIN to {output_path}")
    return nlin


def _load_or_compute_gn(
    system: System,
    profile_path: Path | str,
    launch_powers_w: np.ndarray,
    output_path: Path,
    lumped_losses: list[tuple[float, float]] | None = None,
    recompute: bool = False,
    n_f: int = 40,
    n_z: int = 200,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load numeric GN NLIN and persist to .npy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sci_path = output_path.with_name(f"{output_path.stem}_sci.npy") if return_components else None
    xci_path = output_path.with_name(f"{output_path.stem}_xci.npy") if return_components else None
    if output_path.exists() and not recompute:
        lg.info(f"Loading cached GN NLIN from {output_path}")
        cached = np.load(output_path)
        if np.all(np.isfinite(cached)):
            if return_components:
                if sci_path is None or xci_path is None:
                    raise ValueError("GN component paths not configured.")
                if sci_path.exists() and xci_path.exists():
                    sci = np.load(sci_path)
                    xci = np.load(xci_path)
                    if np.all(np.isfinite(sci)) and np.all(np.isfinite(xci)):
                        return cached, sci, xci
                    raise ValueError(
                        "Cached GN components contain non-finite values; "
                        "set [poggiolini.run].gn_mode='recompute'."
                    )
                lg.warning("GN component cache missing; recomputing.")
            else:
                return cached
        raise ValueError(
            f"Cached GN NLIN at {output_path} contains non-finite values; "
            "set [poggiolini.run].gn_mode='recompute'."
        )
    lg.info(f"Computing numeric GN NLIN (losses={lumped_losses})")
    if return_components:
        nlin, nlin_sci, nlin_xci = compute_gn_numeric(
            system,
            profile_path=profile_path,
            launch_powers_w=launch_powers_w,
            lumped_losses=lumped_losses,
            n_f=n_f,
            n_z=n_z,
            return_components=True,
        )
        np.save(output_path, nlin)
        if sci_path is not None and xci_path is not None:
            np.save(sci_path, nlin_sci)
            np.save(xci_path, nlin_xci)
            lg.success(f"Saved numeric GN SCI to {sci_path}")
            lg.success(f"Saved numeric GN XCI to {xci_path}")
        lg.success(f"Saved numeric GN NLIN to {output_path}")
        return nlin, nlin_sci, nlin_xci
    nlin = compute_gn_numeric(
        system,
        profile_path=profile_path,
        launch_powers_w=launch_powers_w,
        lumped_losses=lumped_losses,
        n_f=n_f,
        n_z=n_z,
    )
    np.save(output_path, nlin)
    lg.success(f"Saved numeric GN NLIN to {output_path}")
    return nlin


def _load_or_compute_gn_direct(
    system: System,
    profile_path: Path | str,
    launch_powers_w: np.ndarray,
    output_path: Path,
    lumped_losses: list[tuple[float, float]] | None = None,
    recompute: bool = False,
    n_f: int = 40,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load direct GN NLIN and persist to .npy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sci_path = output_path.with_name(f"{output_path.stem}_sci.npy") if return_components else None
    xci_path = output_path.with_name(f"{output_path.stem}_xci.npy") if return_components else None
    if output_path.exists() and not recompute:
        lg.info(f"Loading cached direct GN NLIN from {output_path}")
        cached = np.load(output_path)
        if np.all(np.isfinite(cached)):
            if return_components:
                if sci_path is None or xci_path is None:
                    raise ValueError("Direct GN component paths not configured.")
                if sci_path.exists() and xci_path.exists():
                    sci = np.load(sci_path)
                    xci = np.load(xci_path)
                    if np.all(np.isfinite(sci)) and np.all(np.isfinite(xci)):
                        return cached, sci, xci
                    raise ValueError(
                        "Cached direct GN components contain non-finite values; "
                        "set [poggiolini.run].gn_direct_mode='recompute'."
                    )
                lg.warning("Direct GN component cache missing; recomputing.")
            else:
                return cached
        raise ValueError(
            f"Cached direct GN NLIN at {output_path} contains non-finite values; "
            "set [poggiolini.run].gn_direct_mode='recompute'."
        )
    lg.info(f"Computing direct GN NLIN (losses={lumped_losses})")
    if return_components:
        nlin, nlin_sci, nlin_xci = compute_gn_direct(
            system,
            profile_path=profile_path,
            launch_powers_w=launch_powers_w,
            lumped_losses=lumped_losses,
            n_f=n_f,
            return_components=True,
        )
        np.save(output_path, nlin)
        if sci_path is not None and xci_path is not None:
            np.save(sci_path, nlin_sci)
            np.save(xci_path, nlin_xci)
            lg.success(f"Saved direct GN SCI to {sci_path}")
            lg.success(f"Saved direct GN XCI to {xci_path}")
        lg.success(f"Saved direct GN NLIN to {output_path}")
        return nlin, nlin_sci, nlin_xci
    nlin = compute_gn_direct(
        system,
        profile_path=profile_path,
        launch_powers_w=launch_powers_w,
        lumped_losses=lumped_losses,
        n_f=n_f,
    )
    np.save(output_path, nlin)
    lg.success(f"Saved direct GN NLIN to {output_path}")
    return nlin
