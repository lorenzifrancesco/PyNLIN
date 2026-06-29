"""Helpers to materialize stage-S1 numeric NLIN reference curves on demand."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
from loguru import logger as lg

from pynlin.methods.td.cache import pulse_name, s1_ref_nlin_curve_path


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

_TIME_INTEGRAL_BACKENDS = {"direct", "x0mm_fft"}

_XHKM_SCHEMA_VERSION = 2
_XHKM_REQUIRED_KEYS = {
    "llw_grid",
    "raw_n1",
    "raw_n2",
    "raw_n_2pc",
    "raw_n_3pc_total",
    "raw_n_3pca",
    "raw_n_3pcb",
    "raw_n_3pc_other",
    "raw_n_3pc_k_eq_m",
    "raw_n_4pc",
    "raw_n_k_neq_m",
    "ref_n1",
    "ref_n2",
    "ref_n_2pc",
    "ref_n_3pc_total",
    "ref_n_3pca",
    "ref_n_3pcb",
    "ref_n_3pc_other",
    "ref_n_3pc_k_eq_m",
    "ref_n_4pc",
    "ref_n_k_neq_m",
    "n2_over_n1",
    "fiber_length",
    "baud_rate",
    "x_norm",
    "pulse_shape",
    "mode",
    "gvda",
    "gvdb",
    "h_values",
    "r_values",
    "partial_collisions_margin",
    "n_samples_numeric",
    "time_integral_backend",
    "schema_version",
    "calculation",
}


def normalize_time_integral_backend(value: str = "direct") -> str:
    backend = str(value).strip().lower()
    if backend not in _TIME_INTEGRAL_BACKENDS:
        raise ValueError(
            f"Unsupported time_integral_backend={value!r}. "
            f"Expected one of {sorted(_TIME_INTEGRAL_BACKENDS)}."
        )
    return backend

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


def _as_1d_curve(name: str, value: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match expected {shape}")
    return arr


def save_xhkm_sum_reference_curves(
    path: str | Path,
    *,
    llw_grid: np.ndarray,
    raw_n1: np.ndarray,
    raw_n2: np.ndarray,
    raw_n_2pc: np.ndarray,
    raw_n_3pc_total: np.ndarray,
    raw_n_3pca: np.ndarray,
    raw_n_3pcb: np.ndarray,
    raw_n_3pc_other: np.ndarray,
    raw_n_3pc_k_eq_m: np.ndarray,
    raw_n_4pc: np.ndarray,
    raw_n_k_neq_m: np.ndarray,
    fiber_length: float,
    baud_rate: float,
    pulse_shape: str,
    mode: str,
    gvda: float,
    gvdb: float,
    h_values: np.ndarray,
    r_values: np.ndarray,
    partial_collisions_margin: int,
    n_samples_numeric: int,
    schema_version: int = _XHKM_SCHEMA_VERSION,
) -> Path:
    """Save prefactor-free Dar-style ``N1``/``N2`` Xhkm curves atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    llw_grid = _as_1d_curve("llw_grid", llw_grid)
    shape = llw_grid.shape
    raw_n1 = _as_1d_curve("raw_n1", raw_n1, shape)
    raw_n2 = _as_1d_curve("raw_n2", raw_n2, shape)
    raw_n_2pc = _as_1d_curve("raw_n_2pc", raw_n_2pc, shape)
    raw_n_3pc_total = _as_1d_curve("raw_n_3pc_total", raw_n_3pc_total, shape)
    raw_n_3pca = _as_1d_curve("raw_n_3pca", raw_n_3pca, shape)
    raw_n_3pcb = _as_1d_curve("raw_n_3pcb", raw_n_3pcb, shape)
    raw_n_3pc_other = _as_1d_curve("raw_n_3pc_other", raw_n_3pc_other, shape)
    raw_n_3pc_k_eq_m = _as_1d_curve("raw_n_3pc_k_eq_m", raw_n_3pc_k_eq_m, shape)
    raw_n_4pc = _as_1d_curve("raw_n_4pc", raw_n_4pc, shape)
    raw_n_k_neq_m = _as_1d_curve("raw_n_k_neq_m", raw_n_k_neq_m, shape)
    h_values = np.asarray(h_values, dtype=int).reshape(-1)
    r_values = np.asarray(r_values, dtype=int).reshape(-1)

    x_norm = float(fiber_length) * float(baud_rate)
    if not np.isfinite(x_norm) or x_norm <= 0.0:
        raise ValueError(f"Invalid x_norm derived from fiber_length/baud_rate: {x_norm}")
    normalization = x_norm ** (-2)
    ref_n1 = raw_n1 * normalization
    ref_n2 = raw_n2 * normalization
    ref_n_2pc = raw_n_2pc * normalization
    ref_n_3pc_total = raw_n_3pc_total * normalization
    ref_n_3pca = raw_n_3pca * normalization
    ref_n_3pcb = raw_n_3pcb * normalization
    ref_n_3pc_other = raw_n_3pc_other * normalization
    ref_n_3pc_k_eq_m = raw_n_3pc_k_eq_m * normalization
    ref_n_4pc = raw_n_4pc * normalization
    ref_n_k_neq_m = raw_n_k_neq_m * normalization
    n2_over_n1 = np.divide(
        raw_n2,
        raw_n1,
        out=np.full_like(raw_n1, np.nan, dtype=float),
        where=raw_n1 > 0.0,
    )

    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez(
            tmp_path,
            llw_grid=llw_grid,
            raw_n1=raw_n1,
            raw_n2=raw_n2,
            raw_n_2pc=raw_n_2pc,
            raw_n_3pc_total=raw_n_3pc_total,
            raw_n_3pca=raw_n_3pca,
            raw_n_3pcb=raw_n_3pcb,
            raw_n_3pc_other=raw_n_3pc_other,
            raw_n_3pc_k_eq_m=raw_n_3pc_k_eq_m,
            raw_n_4pc=raw_n_4pc,
            raw_n_k_neq_m=raw_n_k_neq_m,
            ref_n1=ref_n1,
            ref_n2=ref_n2,
            ref_n_2pc=ref_n_2pc,
            ref_n_3pc_total=ref_n_3pc_total,
            ref_n_3pca=ref_n_3pca,
            ref_n_3pcb=ref_n_3pcb,
            ref_n_3pc_other=ref_n_3pc_other,
            ref_n_3pc_k_eq_m=ref_n_3pc_k_eq_m,
            ref_n_4pc=ref_n_4pc,
            ref_n_k_neq_m=ref_n_k_neq_m,
            n2_over_n1=n2_over_n1,
            fiber_length=np.array(float(fiber_length)),
            baud_rate=np.array(float(baud_rate)),
            x_norm=np.array(x_norm),
            pulse_shape=np.array(str(pulse_shape)),
            mode=np.array(str(mode)),
            gvda=np.array(float(gvda)),
            gvdb=np.array(float(gvdb)),
            h_values=h_values,
            r_values=r_values,
            partial_collisions_margin=np.array(int(partial_collisions_margin)),
            n_samples_numeric=np.array(int(n_samples_numeric)),
            time_integral_backend=np.array("xhkm_fft"),
            schema_version=np.array(int(schema_version)),
            calculation=np.array("prefactor_free_dar_n1_n2_from_xhkm"),
        )
        os.replace(tmp_path, target)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
    return target


def load_xhkm_sum_reference_curves(
    path: str | Path,
    *,
    pulse_shape: str | None = None,
    mode: str | None = None,
    gvda: float | None = None,
    gvdb: float | None = None,
    h_values: np.ndarray | None = None,
    r_values: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int | str | Path]:
    """Load and validate prefactor-free Xhkm ``N1``/``N2`` curves."""
    target = Path(path)
    with np.load(target, allow_pickle=False) as dataset:
        missing = _XHKM_REQUIRED_KEYS.difference(dataset.files)
        if missing:
            raise ValueError(f"Xhkm sum dataset {target} is missing keys: {sorted(missing)}")
        out = {name: np.asarray(dataset[name]) for name in dataset.files}

    llw_grid = _as_1d_curve("llw_grid", out["llw_grid"])
    shape = llw_grid.shape
    raw_names = [
        "raw_n1",
        "raw_n2",
        "raw_n_2pc",
        "raw_n_3pc_total",
        "raw_n_3pca",
        "raw_n_3pcb",
        "raw_n_3pc_other",
        "raw_n_3pc_k_eq_m",
        "raw_n_4pc",
        "raw_n_k_neq_m",
    ]
    ref_names = [name.replace("raw_", "ref_") for name in raw_names]
    for name in raw_names + ref_names + ["n2_over_n1"]:
        out[name] = _as_1d_curve(name, out[name], shape)

    fiber_length = float(_scalar_value(out["fiber_length"]))
    baud_rate = float(_scalar_value(out["baud_rate"]))
    x_norm = float(_scalar_value(out["x_norm"]))
    saved_pulse = str(_scalar_value(out["pulse_shape"]))
    saved_mode = str(_scalar_value(out["mode"]))
    saved_gvda = float(_scalar_value(out["gvda"]))
    saved_gvdb = float(_scalar_value(out["gvdb"]))
    saved_h = np.asarray(out["h_values"], dtype=int).reshape(-1)
    saved_r = np.asarray(out["r_values"], dtype=int).reshape(-1)
    saved_backend = str(_scalar_value(out["time_integral_backend"]))
    schema_version = int(_scalar_value(out["schema_version"]))
    calculation = str(_scalar_value(out["calculation"]))

    if saved_backend != "xhkm_fft":
        raise ValueError(f"Xhkm sum dataset {target} has backend={saved_backend!r}")
    if schema_version != _XHKM_SCHEMA_VERSION:
        raise ValueError(f"Xhkm sum dataset {target} has schema_version={schema_version}")
    if calculation != "prefactor_free_dar_n1_n2_from_xhkm":
        raise ValueError(f"Xhkm sum dataset {target} has calculation={calculation!r}")
    if pulse_shape is not None and saved_pulse != str(pulse_shape):
        raise ValueError(f"Xhkm sum dataset {target} has pulse_shape={saved_pulse!r}, expected {pulse_shape!r}.")
    if mode is not None and saved_mode != str(mode):
        raise ValueError(f"Xhkm sum dataset {target} has mode={saved_mode!r}, expected {mode!r}.")
    if gvda is not None and not np.isclose(saved_gvda, float(gvda)):
        raise ValueError(f"Xhkm sum dataset {target} has gvda={saved_gvda}, expected {gvda}.")
    if gvdb is not None and not np.isclose(saved_gvdb, float(gvdb)):
        raise ValueError(f"Xhkm sum dataset {target} has gvdb={saved_gvdb}, expected {gvdb}.")
    if h_values is not None and not np.array_equal(saved_h, np.asarray(h_values, dtype=int).reshape(-1)):
        raise ValueError(f"Xhkm sum dataset {target} has incompatible h_values.")
    if r_values is not None and not np.array_equal(saved_r, np.asarray(r_values, dtype=int).reshape(-1)):
        raise ValueError(f"Xhkm sum dataset {target} has incompatible r_values.")
    if not np.isclose(x_norm, fiber_length * baud_rate):
        raise ValueError(f"Xhkm sum dataset {target} has inconsistent x_norm metadata.")
    for raw_name, ref_name in zip(raw_names, ref_names):
        if not np.allclose(out[ref_name], out[raw_name] * x_norm ** (-2), rtol=1e-12, atol=0.0):
            raise ValueError(f"Xhkm sum dataset {target} has inconsistent {ref_name} data.")

    return {
        **out,
        "fiber_length": fiber_length,
        "baud_rate": baud_rate,
        "x_norm": x_norm,
        "pulse_shape": saved_pulse,
        "mode": saved_mode,
        "gvda": saved_gvda,
        "gvdb": saved_gvdb,
        "h_values": saved_h,
        "r_values": saved_r,
        "partial_collisions_margin": int(_scalar_value(out["partial_collisions_margin"])),
        "n_samples_numeric": int(_scalar_value(out["n_samples_numeric"])),
        "time_integral_backend": saved_backend,
        "schema_version": schema_version,
        "calculation": calculation,
        "path": target,
    }


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
    time_integral_backend: str = "direct",
) -> Path:
    """Save a normalized S1 reference dataset atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    llw_grid = np.asarray(llw_grid, dtype=float)
    raw_nlin_curve = np.asarray(raw_nlin_curve, dtype=float)
    time_integral_backend = normalize_time_integral_backend(time_integral_backend)
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
            time_integral_backend=np.array(time_integral_backend),
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
    time_integral_backend: str = "direct",
) -> dict[str, np.ndarray | float | int | str]:
    """Load and validate a structured S1 reference dataset."""
    time_integral_backend = normalize_time_integral_backend(time_integral_backend)
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
        saved_backend = (
            str(_scalar_value(dataset["time_integral_backend"]))
            if "time_integral_backend" in dataset
            else "direct"
        )
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
    if saved_backend != time_integral_backend:
        raise ValueError(
            f"S1 dataset {target} has time_integral_backend={saved_backend!r}, "
            f"expected {time_integral_backend!r}."
        )
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
        "time_integral_backend": saved_backend,
        "path": target,
    }


def ensure_s1_ref_nlin_curve(
    *,
    ipulse: int | None = None,
    pulse_shape: str | None = None,
    mode: str,
    gvda: float,
    gvdb: float,
    time_integral_backend: str = "direct",
    timeout_s: float = 3600.0,
    poll_s: float = 0.2,
) -> Path:
    """Ensure the requested S1 reference curve exists on disk.

    With legacy filename compatibility removed, callers must materialize the
    new stage-labelled cache explicitly. This helper does that lazily and uses
    a small lock file so concurrent processes do not race while generating the
    same reference curve.
    """
    time_integral_backend = normalize_time_integral_backend(time_integral_backend)
    pulse = pulse_name(ipulse=ipulse, pulse_shape=pulse_shape)
    target = s1_ref_nlin_curve_path(
        pulse_shape=pulse,
        mode=mode,
        gvda=gvda,
        gvdb=gvdb,
    )
    regenerate_existing = False
    if target.exists():
        try:
            load_s1_ref_dataset(
                path=target,
                pulse_shape=pulse,
                mode=mode,
                gvda=gvda,
                gvdb=gvdb,
                time_integral_backend=time_integral_backend,
            )
            return target
        except ValueError as exc:
            if "time_integral_backend" not in str(exc):
                raise
            regenerate_existing = True
            lg.info(f"Regenerating S1 reference curve for backend {time_integral_backend}: {target.name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    deadline = time.monotonic() + float(timeout_s)

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if target.exists():
                try:
                    load_s1_ref_dataset(
                        path=target,
                        mode=mode,
                        gvda=gvda,
                        gvdb=gvdb,
                        time_integral_backend=time_integral_backend,
                    )
                    return target
                except ValueError as exc:
                    if "time_integral_backend" not in str(exc):
                        raise
                    regenerate_existing = True
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for S1 reference curve {target}")
            time.sleep(poll_s)
            continue

        try:
            os.close(fd)
            if target.exists():
                try:
                    load_s1_ref_dataset(
                        path=target,
                        mode=mode,
                        gvda=gvda,
                        gvdb=gvdb,
                        time_integral_backend=time_integral_backend,
                    )
                    return target
                except ValueError as exc:
                    if "time_integral_backend" not in str(exc):
                        raise
                    regenerate_existing = True
            lg.info(f"Generating missing S1 reference curve: {target.name}")
            from pynlin.methods.td.validation import compute_numeric_nlin

            compute_numeric_nlin(
                gvda=float(gvda),
                gvdb=float(gvdb),
                ipulse=0 if pulse == "gaussian" else 1,
                recompute=regenerate_existing,
                perfect_only=(mode == "perfect"),
                time_integral_backend=time_integral_backend,
            )
            if not target.exists():
                raise FileNotFoundError(
                    f"S1 reference curve generation completed but {target} was not created."
                )
            load_s1_ref_dataset(
                path=target,
                mode=mode,
                gvda=gvda,
                gvdb=gvdb,
                time_integral_backend=time_integral_backend,
            )
            return target
        finally:
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass
