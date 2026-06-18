from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from pynlin.methods.td.cache import s3_chan_nlin_td_path
from pynlin.system import System


def _fiber_config_bytes(system: System) -> bytes:
    """Serialize fiber configuration relevant to Raman evolution."""
    fiber = system.fiber
    buf = bytearray()
    buf.extend(float(fiber.length).hex().encode("ascii"))
    if hasattr(fiber, "loss_profile"):
        wls = np.linspace(1.4e-6, 1.7e-6, 100, dtype=float)
        losses = np.asarray(fiber.loss_profile(wls), dtype=float)
        buf.extend(b"|")
        buf.extend(np.ascontiguousarray(losses).tobytes())
    if hasattr(fiber, "raman_coefficient"):
        buf.extend(b"|")
        buf.extend(float(fiber.raman_coefficient).hex().encode("ascii"))
    if hasattr(fiber, "effective_area"):
        buf.extend(b"|")
        buf.extend(float(fiber.effective_area).hex().encode("ascii"))
    return bytes(buf)


def _pump_config_bytes(system: System) -> bytes:
    """Serialize pump configuration relevant to Raman evolution."""
    pump_specs = getattr(system, "pump_specs", None) or []
    if not pump_specs:
        return b""
    parts = []
    for p in pump_specs:
        parts.append(f"{float(p.wavelength):.6e},{float(p.power_dbm):.6e}")
    return ",".join(parts).encode("ascii")


def profile_system_hash(system: System, length: int = 12) -> str:
    """Hash of system parameters that affect the computed Raman power profile.

    Changing any of fiber length, loss, Raman coefficient, Aeff, pump
    configuration, or WDM channel grid will produce a different hash.
    """
    h = hashlib.sha1()
    freqs = np.asarray(system.wdm.frequency_grid(), dtype=float)
    h.update(np.ascontiguousarray(freqs).tobytes())
    h.update(_fiber_config_bytes(system))
    h.update(_pump_config_bytes(system))
    return h.hexdigest()[:length]


def safe_tag(value: float) -> str:
    text = f"{value:.6e}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def array_hash(*arrays: np.ndarray, length: int = 12) -> str:
    h = hashlib.sha1()
    for values in arrays:
        arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        h.update(str(arr.shape).encode("ascii"))
        h.update(arr.view(np.uint8))
    return h.hexdigest()[:length]


def dispersion_hash(system: System, length: int = 12) -> str:
    freqs = np.asarray(system.wdm.frequency_grid(), dtype=float)
    beta1, beta2 = system.beta_grids(freqs=freqs)
    return array_hash(freqs, beta1, beta2, length=length)


def launch_hash(launch_powers_w: np.ndarray, length: int = 12) -> str:
    return array_hash(np.asarray(launch_powers_w, dtype=float), length=length)


def method_cache_tag(*parts: object) -> str:
    clean = [str(part) for part in parts if part not in (None, "")]
    return "_".join(clean)


def td_cache_path(
    profile_path: Path | str | None,
    *,
    use_kappa: bool,
    use_x_mode: bool,
    extra_tag: str | None = None,
) -> Path:
    tag = Path(profile_path).stem if profile_path is not None else "default"
    return s3_chan_nlin_td_path(
        tag=tag,
        use_kappa=use_kappa,
        use_x_mode=use_x_mode,
        extra_tag=extra_tag,
    )
