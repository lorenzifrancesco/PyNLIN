from dataclasses import dataclass

import numpy as np

from pynlin.system import System

from .config import SubsetConfig, _select_scaling_channel


@dataclass(frozen=True)
class ResolvedSubset:
    cut_indices: tuple[int, ...]
    interferer_indices: tuple[int, ...]
    include_sci: bool
    center_index: int
    tag: str


def _validate_indices(indices: tuple[int, ...], n_channels: int, name: str) -> tuple[int, ...]:
    unique = tuple(dict.fromkeys(int(idx) for idx in indices))
    invalid = [idx for idx in unique if idx < 0 or idx >= n_channels]
    if invalid:
        raise IndexError(f"{name} contains out-of-range channel indices {invalid}; n_channels={n_channels}")
    return unique


def _auto_center_index(system: System) -> int:
    center_idx, _ = _select_scaling_channel(system)
    return int(center_idx)


def resolve_subset(system: System, subset: SubsetConfig | None) -> ResolvedSubset:
    """Resolve a subset config into concrete global channel indices."""
    n_channels = int(system.n_channels)
    if n_channels <= 0:
        raise ValueError("Cannot resolve a subset for an empty WDM grid.")

    cfg = subset or SubsetConfig()
    mode = str(cfg.mode).strip().lower()
    if mode == "center_window":
        if str(cfg.center).strip().lower() == "auto":
            center_idx = _auto_center_index(system)
        else:
            center_idx = int(cfg.center)
        _validate_indices((center_idx,), n_channels, "subset.center")
        start = max(center_idx - int(cfg.half_width), 0)
        stop = min(center_idx + int(cfg.half_width) + 1, n_channels)
        cut_indices = (center_idx,)
        interferer_indices = tuple(range(start, stop))
    elif mode == "explicit":
        cut_indices = _validate_indices(cfg.cut_indices, n_channels, "subset.cut_indices")
        interferer_indices = _validate_indices(
            cfg.interferer_indices, n_channels, "subset.interferer_indices"
        )
        if not cut_indices:
            raise ValueError("subset.cut_indices is required for mode='explicit'.")
        if not interferer_indices:
            raise ValueError("subset.interferer_indices is required for mode='explicit'.")
        center_idx = int(cut_indices[0])
    else:
        raise ValueError("subset.mode must be one of {'center_window', 'explicit'}.")

    if not cfg.include_sci:
        interferer_indices = tuple(idx for idx in interferer_indices if idx not in set(cut_indices))

    cut_tag = "-".join(str(idx) for idx in cut_indices)
    int_tag = "-".join(str(idx) for idx in interferer_indices)
    tag = f"cuts{cut_tag}_ints{int_tag}_sci{int(bool(cfg.include_sci))}"
    return ResolvedSubset(
        cut_indices=cut_indices,
        interferer_indices=interferer_indices,
        include_sci=bool(cfg.include_sci),
        center_index=center_idx,
        tag=tag,
    )


def subset_mask(n_channels: int, indices: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(int(n_channels), dtype=bool)
    mask[list(indices)] = True
    return mask


__all__ = ["ResolvedSubset", "resolve_subset", "subset_mask"]
