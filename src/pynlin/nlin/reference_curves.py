"""Helpers to materialize stage-S1 numeric NLIN reference curves on demand."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
from loguru import logger as lg

from pynlin.nlin.cache_names import pulse_name, s1_ref_nlin_curve_path


_S1_REQUIRED_KEYS = {
    "llw_grid",
    "raw_nlin_curve",
    "ref_nlin_curve",
    "fiber_length",
    "baud_rate",
    "x_norm",
    "pulse_shape",
    "mode",
    "gvda",
    "gvdb",
    "n_samples_numeric",
}

_CANONICAL_PERFECT_S1_PARAMS = {
    "gaussian": (0.28221327, 3.28726640, 0.49259348),
    "nyquist": (0.47125353, 2.11578009, 0.92291662),
}
_CANONICAL_S1_LLW_MIN = 1e-2
_CANONICAL_S1_LLW_MAX = 1e2
_CANONICAL_S1_NPTS = 400
# FIXME this hardcoding of the S1 data is ok, but not nice.

def _scalar_value(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    raise ValueError(f"Expected scalar metadata value, got shape {arr.shape}")


def save_s1_ref_nlin_curve(
    path: str | Path,
    *,
    llw_grid: np.ndarray,
    raw_nlin_curve: np.ndarray,
    fiber_length: float,
    baud_rate: float,
    pulse_shape: str,
    mode: str,
    gvda: float,
    gvdb: float,
    n_samples_numeric: int,
) -> Path:
    """Save a normalized S1 reference dataset atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    llw_grid = np.asarray(llw_grid, dtype=float)
    raw_nlin_curve = np.asarray(raw_nlin_curve, dtype=float)
    if llw_grid.ndim != 1 or raw_nlin_curve.ndim != 1:
        raise ValueError("S1 reference curves must be one-dimensional.")
    if llw_grid.shape != raw_nlin_curve.shape:
        raise ValueError(
            f"llw_grid and raw_nlin_curve shape mismatch: {llw_grid.shape} vs {raw_nlin_curve.shape}"
        )
    x_norm = float(fiber_length) * float(baud_rate)
    if not np.isfinite(x_norm) or x_norm <= 0.0:
        raise ValueError(f"Invalid x_norm derived from fiber_length/baud_rate: {x_norm}")
    ref_nlin_curve = raw_nlin_curve * x_norm ** (-2)

    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez(
            tmp_path,
            llw_grid=llw_grid,
            raw_nlin_curve=raw_nlin_curve,
            ref_nlin_curve=ref_nlin_curve,
            fiber_length=np.array(float(fiber_length)),
            baud_rate=np.array(float(baud_rate)),
            x_norm=np.array(x_norm),
            pulse_shape=np.array(str(pulse_shape)),
            mode=np.array(str(mode)),
            gvda=np.array(float(gvda)),
            gvdb=np.array(float(gvdb)),
            n_samples_numeric=np.array(int(n_samples_numeric)),
        )
        os.replace(tmp_path, target)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
    return target


def _save_canonical_perfect_s1_ref_curve(target: Path, pulse: str) -> Path:
    """Write the dispersionless perfect S1 reference without legacy MMF inputs."""
    raise("This should never be called")
    try:
        a, b, c = _CANONICAL_PERFECT_S1_PARAMS[pulse]
    except KeyError as exc:
        raise ValueError(f"No canonical perfect S1 parameters for pulse {pulse!r}.") from exc

    llw_grid = np.geomspace(
        _CANONICAL_S1_LLW_MIN,
        _CANONICAL_S1_LLW_MAX,
        _CANONICAL_S1_NPTS,
    )
    ref_nlin_curve = a * (1.0 + (llw_grid / b) ** (1.0 / c)) ** (-c)
    return save_s1_ref_nlin_curve(
        target,
        llw_grid=llw_grid,
        raw_nlin_curve=ref_nlin_curve,
        fiber_length=1.0,
        baud_rate=1.0,
        pulse_shape=pulse,
        mode="perfect",
        gvda=0.0,
        gvdb=0.0,
        n_samples_numeric=_CANONICAL_S1_NPTS,
    )


def load_s1_ref_dataset(
    *,
    path: str | Path | None = None,
    ipulse: int | None = None,
    pulse_shape: str | None = None,
    mode: str,
    gvda: float,
    gvdb: float,
) -> dict[str, np.ndarray | float | int | str]:
    """Load and validate a structured S1 reference dataset."""
    target = Path(path) if path is not None else s1_ref_nlin_curve_path(
        ipulse=ipulse,
        pulse_shape=pulse_shape,
        mode=mode,
        gvda=gvda,
        gvdb=gvdb,
    )
    with np.load(target, allow_pickle=False) as dataset:
        missing = _S1_REQUIRED_KEYS.difference(dataset.files)
        if missing:
            raise ValueError(f"S1 dataset {target} is missing keys: {sorted(missing)}")
        llw_grid = np.asarray(dataset["llw_grid"], dtype=float)
        raw_nlin_curve = np.asarray(dataset["raw_nlin_curve"], dtype=float)
        ref_nlin_curve = np.asarray(dataset["ref_nlin_curve"], dtype=float)
        if llw_grid.ndim != 1 or raw_nlin_curve.ndim != 1 or ref_nlin_curve.ndim != 1:
            raise ValueError(f"S1 dataset {target} must store 1D arrays.")
        if not (llw_grid.shape == raw_nlin_curve.shape == ref_nlin_curve.shape):
            raise ValueError(
                f"S1 dataset {target} has inconsistent shapes: "
                f"{llw_grid.shape}, {raw_nlin_curve.shape}, {ref_nlin_curve.shape}"
            )
        fiber_length = float(_scalar_value(dataset["fiber_length"]))
        baud_rate = float(_scalar_value(dataset["baud_rate"]))
        x_norm = float(_scalar_value(dataset["x_norm"]))
        pulse = str(_scalar_value(dataset["pulse_shape"]))
        saved_mode = str(_scalar_value(dataset["mode"]))
        saved_gvda = float(_scalar_value(dataset["gvda"]))
        saved_gvdb = float(_scalar_value(dataset["gvdb"]))
        n_samples_numeric = int(_scalar_value(dataset["n_samples_numeric"]))
    expected_pulse = pulse_name(ipulse=ipulse, pulse_shape=pulse_shape) if (ipulse is not None or pulse_shape is not None) else None
    if expected_pulse is not None and pulse != expected_pulse:
        raise ValueError(f"S1 dataset {target} has pulse_shape={pulse!r}, expected {expected_pulse!r}.")
    if saved_mode != str(mode):
        raise ValueError(f"S1 dataset {target} has mode={saved_mode!r}, expected {mode!r}.")
    if not np.isclose(saved_gvda, float(gvda)) or not np.isclose(saved_gvdb, float(gvdb)):
        raise ValueError(
            f"S1 dataset {target} has (gvda, gvdb)=({saved_gvda}, {saved_gvdb}), "
            f"expected ({gvda}, {gvdb})."
        )
    expected_x_norm = fiber_length * baud_rate
    if not np.isclose(x_norm, expected_x_norm):
        raise ValueError(f"S1 dataset {target} has inconsistent x_norm metadata.")
    if not np.allclose(ref_nlin_curve, raw_nlin_curve * x_norm ** (-2), rtol=1e-12, atol=0.0):
        raise ValueError(f"S1 dataset {target} has inconsistent normalized curve data.")
    return {
        "llw_grid": llw_grid,
        "raw_nlin_curve": raw_nlin_curve,
        "ref_nlin_curve": ref_nlin_curve,
        "fiber_length": fiber_length,
        "baud_rate": baud_rate,
        "x_norm": x_norm,
        "pulse_shape": pulse,
        "mode": saved_mode,
        "gvda": saved_gvda,
        "gvdb": saved_gvdb,
        "n_samples_numeric": n_samples_numeric,
        "path": target,
    }


def ensure_s1_ref_nlin_curve(
    *,
    ipulse: int | None = None,
    pulse_shape: str | None = None,
    mode: str,
    gvda: float,
    gvdb: float,
    timeout_s: float = 3600.0,
    poll_s: float = 0.2,
) -> Path:
    """Ensure the requested S1 reference curve exists on disk.

    With legacy filename compatibility removed, callers must materialize the
    new stage-labelled cache explicitly. This helper does that lazily and uses
    a small lock file so concurrent processes do not race while generating the
    same reference curve.
    """
    pulse = pulse_name(ipulse=ipulse, pulse_shape=pulse_shape)
    target = s1_ref_nlin_curve_path(
        pulse_shape=pulse,
        mode=mode,
        gvda=gvda,
        gvdb=gvdb,
    )
    if target.exists():
        load_s1_ref_dataset(path=target, pulse_shape=pulse, mode=mode, gvda=gvda, gvdb=gvdb)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    deadline = time.monotonic() + float(timeout_s)

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if target.exists():
                load_s1_ref_dataset(path=target, mode=mode, gvda=gvda, gvdb=gvdb)
                return target
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for S1 reference curve {target}")
            time.sleep(poll_s)
            continue

        try:
            os.close(fd)
            if target.exists():
                load_s1_ref_dataset(path=target, mode=mode, gvda=gvda, gvdb=gvdb)
                return target
            lg.info(f"Generating missing S1 reference curve: {target.name}")
            if mode == "perfect" and np.isclose(float(gvda), 0.0) and np.isclose(float(gvdb), 0.0):
                _save_canonical_perfect_s1_ref_curve(target, pulse)
                load_s1_ref_dataset(path=target, pulse_shape=pulse, mode=mode, gvda=gvda, gvdb=gvdb)
                return target
            from pynlin.nlin.validation import compute_numeric_nlin

            compute_numeric_nlin(
                gvda=float(gvda),
                gvdb=float(gvdb),
                ipulse=0 if pulse == "gaussian" else 1,
                recompute=False,
                perfect_only=(mode == "perfect"),
            )
            if not target.exists():
                raise FileNotFoundError(
                    f"S1 reference curve generation completed but {target} was not created."
                )
            load_s1_ref_dataset(path=target, mode=mode, gvda=gvda, gvdb=gvdb)
            return target
        finally:
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass
