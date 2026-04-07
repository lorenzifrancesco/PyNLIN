"""
System container combining fiber, WDM grid, pulse settings, amplification, and numerics.

Designed to mirror the sectioned TOML layout used by existing configs without
going through the legacy cfg helpers. Leverages the section-aware loaders in
fiber.py and wdm.py, and optionally pulls numerical parameters from a dedicated
TOML (defaults to numerical_config.toml next to the system file).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .fiber import (
    Fiber,
    MMFiber,
    MMFiberConfig,
    SMFiber,
    SMFiberConfig,
    _extract_fiber_section,
)
from .pulses import Pulse, PulseConfig, pulse_from_config
from .utils import NumericalConfig, _toml_load, lambda2nu, nu2lambda
from .wdm import Amplification, BaseWDM, wdm_from_toml


def _interp_with_linear_extrapolation(
    x_query: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> np.ndarray:
    """Interpolate 1D data and linearly extrapolate beyond both endpoints."""
    x = np.asarray(x_data, dtype=float).reshape(-1)
    y = np.asarray(y_data, dtype=float).reshape(-1)
    q = np.asarray(x_query, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"x/y size mismatch: {x.size} != {y.size}")
    if x.size < 2:
        raise ValueError("Need at least two samples for interpolation/extrapolation.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    out = np.interp(q, x, y)

    left = q < x[0]
    if np.any(left):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        out[left] = y[0] + slope * (q[left] - x[0])

    right = q > x[-1]
    if np.any(right):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        out[right] = y[-1] + slope * (q[right] - x[-1])

    return out


def _build_fiber(system_path: Path, data: Mapping) -> Fiber:
    fiber_data = _extract_fiber_section(data)
    if not isinstance(fiber_data, Mapping):
        raise ValueError(f"No fiber section found in {system_path}")
    n_modes = int(fiber_data.get("n_modes", 1))
    if n_modes > 1:
        cfg = MMFiberConfig(**fiber_data)
        return MMFiber.from_config(cfg)
    cfg = SMFiberConfig(**fiber_data)
    return SMFiber.from_config(cfg)


def _load_numerics(system_path: Path, numerical_path: Optional[Path]) -> Optional[NumericalConfig]:
    if numerical_path is None:
        candidate = system_path.with_name("numerical_config.toml")
        if candidate.exists():
            numerical_path = candidate
    if numerical_path is None:
        return None
    return NumericalConfig(**_toml_load(numerical_path))


@dataclass
class System:
    fiber: Fiber
    wdm: BaseWDM
    pulse: Pulse
    amplification: Amplification
    pulse_config: Optional[PulseConfig] = None
    numerics: Optional[NumericalConfig] = None
    source: Optional[Path] = None
    numerics_source: Optional[Path] = None
    launch_power: Optional[float] = None
    store: Optional[bool] = None
    collision_margin: Optional[int] = None
    dispersion: Optional[float] = None
    raw_config: Optional[Mapping] = None
    pumps: Optional[list] = None

    @classmethod
    def from_toml(cls, system_path: Path | str, numerical_path: Path | str | None = None) -> "System":
        system_path = Path(system_path)
        data = _toml_load(system_path)
        wdm = wdm_from_toml(system_path)
        fiber = _build_fiber(system_path, data)
        pulse_cfg = PulseConfig.from_mapping(data)
        pulse = pulse_from_config(pulse_cfg)
        amp_section = data.get("amplification") if isinstance(data, Mapping) else None
        amp = Amplification.from_mapping(amp_section if isinstance(amp_section, Mapping) else {}, root_data=data)
        numerics_path = Path(numerical_path) if numerical_path is not None else None
        numerics = _load_numerics(system_path, numerics_path)
        wdm_section = data.get("wdm") if isinstance(data, Mapping) else {}
        nlin_section = data.get("nlin") if isinstance(data, Mapping) else {}
        fiber_section = data.get("fiber") if isinstance(data, Mapping) else {}
        launch_power = None
        if isinstance(wdm_section, Mapping):
            launch_power = wdm_section.get("launch_power", wdm_section.get("launch_power_dbm"))
        if launch_power is None and isinstance(data, Mapping):
            launch_power = data.get("launch_power", data.get("launch_power_dbm"))
        store = nlin_section.get("store") if isinstance(nlin_section, Mapping) else None
        collision_margin = nlin_section.get("collision_margin") if isinstance(nlin_section, Mapping) else None
        if store is None and isinstance(data, Mapping):
            store = data.get("store")
        if collision_margin is None and isinstance(data, Mapping):
            collision_margin = data.get("collision_margin")
        return cls(
            fiber=fiber,
            wdm=wdm,
            pulse=pulse,
            amplification=amp,
            pulse_config=pulse_cfg,
            numerics=numerics,
            source=system_path,
            numerics_source=numerics_path,
            launch_power=launch_power,
            store=store,
            collision_margin=collision_margin,
            dispersion=fiber_section.get("dispersion") if isinstance(fiber_section, Mapping) else None,
            raw_config=data if isinstance(data, Mapping) else None,
            pumps=amp.pumps if hasattr(amp, "pumps") else None,
        )

    def summary(self) -> str:
        pulse_desc = (
            f"{self.pulse_config.type.name}" if self.pulse_config else self.pulse.__class__.__name__
        )
        pumps_desc = f"{len(self.pump_specs)} pumps" if self.pump_specs else f"{self.n_pumps} pumps"
        lines = [
            f"Fiber     : {self.fiber}",
            f"WDM       : {self.wdm.summary() if hasattr(self.wdm, 'summary') else self.wdm}",
            f"Pulse     : type={pulse_desc}, baud_rate={self.pulse.baud_rate:.3e}",
            f"Amplif.   : {pumps_desc}, raman_gain={self.amplification.raman_gain}",
        ]
        if self.numerics is not None:
            lines.append(f"Numerics  : {self.numerics}")
        return "\n".join(lines)

    # Compatibility properties (formerly on Config)
    @property
    def n_modes(self) -> int:
        return getattr(self.fiber, "n_modes", 1)

    @property
    def n_channels(self) -> int:
        return getattr(self.wdm, "num_channels", None)

    @property
    def channel_spacing(self) -> Optional[float]:
        return getattr(self.wdm, "spacing", None)

    @property
    def center_frequency(self) -> Optional[float]:
        return getattr(self.wdm, "central_frequency", None)

    @property
    def fiber_length(self) -> Optional[float]:
        return getattr(self.fiber, "length", None)

    @property
    def baud_rate(self) -> Optional[float]:
        return getattr(self.pulse, "baud_rate", None)

    @property
    def pulse_shape(self) -> Optional[int]:
        if self.pulse_config:
            return self.pulse_config.pulse_shape
        return None

    @property
    def raman_gain(self) -> Optional[float]:
        return getattr(self.amplification, "raman_gain", None)

    @property
    def n_pumps(self) -> Optional[int]:
        return getattr(self.amplification, "n_pumps", None)

    @property
    def pump_specs(self):
        return getattr(self.amplification, "pumps", None)

    @property
    def effective_area(self) -> Optional[float]:
        return getattr(self.fiber, "effective_area", None)

    def beta_grids(self, freqs: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return beta1/beta2 grids (shape: n_modes x n_freqs) for the system.

        Uses fiber.group_delay when available; otherwise falls back to fiber beta arrays
        or profiles. For multimode without group_delay, a default dataset is used.
        """
        if freqs is None:
            freqs = self.wdm.frequency_grid()
        wavelengths = nu2lambda(freqs)

        force_constant = bool(getattr(self.fiber, "_force_constant_dispersion", False))
        gd = getattr(self.fiber, "group_delay", None)
        if gd is not None and not force_constant:
            beta1 = np.array([gd.evaluate_beta1(m, freqs) for m in range(self.n_modes)])
            beta2 = np.array([gd.evaluate_beta2(m, freqs) for m in range(self.n_modes)])
            return beta1, beta2

        if self.n_modes > 1 and not force_constant:
            from pynlin.fiber_data.load_fiber_values import load_group_delay
            gd = load_group_delay()
            beta1 = np.array([gd.evaluate_beta1(m, freqs) for m in range(self.n_modes)])
            beta2 = np.array([gd.evaluate_beta2(m, freqs) for m in range(self.n_modes)])
            return beta1, beta2

        beta1 = None
        beta2 = None

        # Prefer wavelength-dependent beta1 whenever available. In constant-
        # dispersion mode this may be a synthetic linear profile generated from
        # a constant beta2 and CSV anchor beta1.
        if getattr(self.fiber, "_beta1_profile", None) is not None:
            wl_prof, b1_prof = self.fiber._beta1_profile
            if force_constant and getattr(self.fiber, "_freq_profile", None) is not None:
                beta1 = _interp_with_linear_extrapolation(
                    freqs,
                    np.asarray(self.fiber._freq_profile, dtype=float),
                    np.asarray(b1_prof, dtype=float),
                )[None, :]
            else:
                beta1 = np.interp(wavelengths, wl_prof, b1_prof)[None, :]
        # Keep beta2 profile disabled when constant dispersion is requested.
        if not force_constant and getattr(self.fiber, "_beta2_profile", None) is not None:
            wl_prof, b2_prof = self.fiber._beta2_profile
            beta2 = np.interp(wavelengths, wl_prof, b2_prof)[None, :]

        b1_raw = getattr(self.fiber, "beta1", None)
        if beta1 is None and b1_raw is not None:
            b1_raw = np.array(b1_raw)
            if b1_raw.ndim == 0:
                beta1 = np.full((self.n_modes, freqs.size), float(b1_raw))
            elif b1_raw.ndim == 1 and b1_raw.size == freqs.size:
                beta1 = b1_raw[None, :]
        b2_raw = getattr(self.fiber, "beta2", None)
        if beta2 is None and b2_raw is not None:
            b2_raw = np.array(b2_raw)
            if b2_raw.ndim == 0:
                beta2 = np.full((self.n_modes, freqs.size), float(b2_raw))
            elif b2_raw.ndim == 1 and b2_raw.size == freqs.size:
                beta2 = b2_raw[None, :]

        if beta1 is None and beta2 is not None:
            raise ValueError(
                "beta1 unavailable while beta2 is available. Refusing beta1=0 fallback "
                "because it collapses channel-pair DGD in TD computations."
            )
        if beta1 is None or beta2 is None:
            raise ValueError("beta1/beta2 unavailable for this system.")

        return beta1, beta2

    def report_max_l_normalizations(self) -> Optional[tuple[float, float]]:
        """
        Print the maximum L/LD and L/LW over modes/frequencies for this system.

        LD is based on |beta2| via LD = 1/(baud_rate^2 * |beta2|).
        LW is based on |beta1| via LW = 1/(baud_rate * |beta1|).
        """
        try:
            L = float(self.fiber_length)
            br = float(self.baud_rate)
            beta1, beta2 = self.beta_grids()
        except Exception as exc:
            print(f"Could not compute L/LD or L/LW: missing system parameters ({exc})")
            return None

        if beta2 is None:
            print("Could not compute L/LD: beta2 unavailable.")
            max_l_ld = float("nan")
        else:
            abs_b2 = np.maximum(np.abs(beta2), 0.0)
            max_b2 = np.max(abs_b2)
            if max_b2 == 0:
                print("Could not compute L/LD: beta2 is zero.")
                max_l_ld = float("nan")
            else:
                max_l_ld = float(L * br * br * max_b2)

        if beta1 is None:
            print("Could not compute L/LW: beta1 unavailable.")
            max_l_lw = float("nan")
        else:
            # For wideband SMF, use the spread across channels (per mode, then max)
            b1_max = np.max(beta1, axis=1)
            b1_min = np.min(beta1, axis=1)
            span_b1 = float(np.max(np.abs(b1_max - b1_min)))
            if span_b1 == 0:
                print("Could not compute L/LW: beta1 span is zero.")
                max_l_lw = float("nan")
            else:
                max_l_lw = float(L * br * span_b1)

        print(f"Max L/LD: {max_l_ld:.3e}, Max L/LW: {max_l_lw:.3e}")
        return max_l_ld, max_l_lw

    def model_dump(self):
        """Minimal dict representation for persistence helpers."""
        return {
            "launch_power": self.launch_power,
            "store": self.store,
            "collision_margin": self.collision_margin,
            "dispersion": self.dispersion,
        }

    def _initial_signal_powers_dbm(self) -> np.ndarray:
        """Return launch powers per channel in dBm, honoring band overrides."""
        freqs = self.wdm.frequency_grid()
        launch_dbm = self.launch_power if self.launch_power is not None else -5.0
        powers_dbm = np.full(len(freqs), launch_dbm, dtype=float)
        if hasattr(self.wdm, "band_specs") and hasattr(self.wdm, "_band_slices"):
            for name, slc in self.wdm._band_slices.items():
                spec = self.wdm.band_specs.get(name)
                if spec and spec.launch_power_dbm is not None:
                    powers_dbm[slc] = spec.launch_power_dbm
        return powers_dbm

    def plot_launch_spectrum(self, save_path: Path | str | None = None, tiny_markers: bool = True):
        """
        Plot pumps and signals at fiber start and save to media/debug.

        Scatter of channel powers (dBm) vs frequency (THz) plus pump markers.
        """
        import matplotlib.pyplot as plt  # local import to avoid hard dependency at import time

        freqs = self.wdm.frequency_grid()
        sig_powers_dbm = self._initial_signal_powers_dbm()

        pump_specs = self.pump_specs or []
        pump_freqs = np.array([lambda2nu(p.wavelength) for p in pump_specs]) if pump_specs else np.array([])
        pump_powers_dbm = np.array([p.power_dbm for p in pump_specs]) if pump_specs else np.array([])

        plt.figure()
        plt.scatter(freqs * 1e-12, sig_powers_dbm, s=2 if tiny_markers else 6, alpha=0.6, label=r"signals") # $\mathnormal P(z=0)$")
        if pump_powers_dbm.size:
            plt.scatter(pump_freqs * 1e-12, pump_powers_dbm, marker="x", color="red", s=12, label=r"pumps") # $\mathnormal P(z=L)$")
        plt.xlabel(r"$\mathnormal f$ [THz]")
        plt.ylabel(r" Launch power [dBm]")
        plt.grid(True, alpha=0.2)
        plt.legend(loc="best")
        out_dir = Path(save_path).parent if save_path else Path("media/debug")
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(save_path) if save_path else out_dir / f"launch_spectrum_{self.source.stem if self.source else 'system'}.png"
        plt.tight_layout()
        plt.grid(False, alpha=0.2)
        plt.savefig(filename, dpi=200)
        plt.close()
        return filename
