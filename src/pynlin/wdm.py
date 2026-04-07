import numpy as np
from scipy.constants import nu2lambda
from pathlib import Path
from typing import Dict, Mapping, Optional

import pynlin.utils
from pynlin.utils import BaseModel, ConfigDict, EXTRA_IGNORE_CONFIG, _toml_load


def _extract_wdm_section(data):
    """Return the wdm subsection if present, otherwise the raw mapping."""
    if not isinstance(data, Mapping):
        return data
    return data.get("wdm", data)


def _extract_spacing_hz(wdm_data, root_data=None) -> Optional[float]:
    if not isinstance(wdm_data, Mapping):
        return None
    for key in ("spacing", "channel_spacing", "spacing_hz"):
        if key in wdm_data:
            return wdm_data[key]
    for key in ("spacing_ghz", "spacing_GHz"):
        if key in wdm_data:
            return wdm_data[key] * 1e9
    if isinstance(root_data, Mapping):
        sys = root_data.get("system", {})
        if isinstance(sys, Mapping):
            for key in ("spacing_ghz", "spacing_GHz"):
                if key in sys:
                    return sys[key] * 1e9
    return None


def _extract_center_frequency_hz(wdm_data, root_data=None) -> Optional[float]:
    if isinstance(wdm_data, Mapping):
        for key in ("center_frequency", "central_frequency"):
            if key in wdm_data:
                return wdm_data[key]
        for key in ("center_frequency_thz", "central_frequency_thz"):
            if key in wdm_data:
                return wdm_data[key] * 1e12
    if isinstance(root_data, Mapping):
        sys = root_data.get("system", {})
        if isinstance(sys, Mapping):
            for key in ("center_frequency_thz", "central_frequency_thz"):
                if key in sys:
                    return sys[key] * 1e12
    return None


def _pick(key, *sections):
    for sec in sections:
        if isinstance(sec, Mapping) and key in sec:
            return sec[key]
    return None


def _is_single_band_config(data: Mapping | object) -> bool:
    return isinstance(data, Mapping) and "n_channels" in data and "start_nm" in data


def _infer_band_name(start_nm: float | None, existing_names: set[str]) -> str:
    if start_nm is not None:
        if 1460.0 <= float(start_nm) < 1530.0:
            preferred = "S"
        elif 1530.0 <= float(start_nm) < 1565.0:
            preferred = "C"
        elif 1565.0 <= float(start_nm) < 1625.0:
            preferred = "L"
        else:
            preferred = "band"
    else:
        preferred = "band"

    if preferred not in existing_names:
        return preferred

    counter = 2
    while True:
        candidate = f"{preferred}_{counter}"
        if candidate not in existing_names:
            return candidate
        counter += 1


def _extract_irregular_bands_data(wdm_data: Mapping | object) -> Optional[Dict[str, dict]]:
    if not isinstance(wdm_data, Mapping):
        return None

    merged: Dict[str, dict] = {}

    bands_section = wdm_data.get("bands")
    if isinstance(bands_section, Mapping):
        for name, cfg in bands_section.items():
            if isinstance(cfg, Mapping):
                merged[str(name)] = dict(cfg)

    band_section = wdm_data.get("band")
    if _is_single_band_config(band_section):
        band_name = band_section.get("name")
        if not isinstance(band_name, str) or not band_name.strip():
            band_name = _infer_band_name(band_section.get("start_nm"), set(merged))
        merged[str(band_name)] = dict(band_section)
    elif isinstance(band_section, Mapping):
        for name, cfg in band_section.items():
            if isinstance(cfg, Mapping):
                merged[str(name)] = dict(cfg)

    return merged or None


class WDMConfig(BaseModel):
    spacing: float
    num_channels: int
    center_frequency: float

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"

    @classmethod
    def from_mapping(cls, data: Mapping, root_data=None):
        spacing = _extract_spacing_hz(data, root_data)
        center_frequency = _extract_center_frequency_hz(data, root_data)
        num_channels = data.get("n_channels") if isinstance(data, Mapping) else None
        if spacing is None or center_frequency is None or num_channels is None:
            raise ValueError("Missing spacing/center_frequency/n_channels in WDM config.")
        return cls(
            spacing=spacing,
            num_channels=int(num_channels),
            center_frequency=center_frequency,
        )


class WDMBandConfig(BaseModel):
    n_channels: int
    launch_power_dbm: Optional[float] = None
    start_nm: float
    modulation: Optional[str] = None

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


class PumpSpec(BaseModel):
    wavelength: float
    power_dbm: float
    direction: int = 1  # +1 co-propagating, -1 counter-propagating

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


class Amplification(BaseModel):
    n_pumps: int
    raman_gain: float
    pumps: Optional[list[PumpSpec]] = None

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"

    @classmethod
    def from_mapping(cls, data: Mapping, root_data=None) -> "Amplification":
        amp = data if isinstance(data, Mapping) else {}
        root = root_data if isinstance(root_data, Mapping) else {}
        n_pumps = _pick("n_pumps", amp, root)
        target = amp.get("target") if isinstance(amp, Mapping) else {}
        raman_gain = _pick("raman_gain", amp, root) or (target.get("raman_gain") if isinstance(target, Mapping) else None)
        pumps_data = amp.get("pumps") if isinstance(amp, Mapping) else None
        pumps = None
        if isinstance(pumps_data, list):
            pumps = [PumpSpec(**p) if not isinstance(p, PumpSpec) else p for p in pumps_data]
            if n_pumps is None:
                n_pumps = len(pumps)
        if n_pumps is None or raman_gain is None:
            raise ValueError("Amplification requires n_pumps/raman_gain.")
        return cls(n_pumps=int(n_pumps), raman_gain=float(raman_gain), pumps=pumps)

    @classmethod
    def from_toml(cls, filepath: Path | str) -> "Amplification":
        data = _toml_load(Path(filepath))
        amp = data.get("amplification") if isinstance(data, Mapping) else None
        return cls.from_mapping(amp if isinstance(amp, Mapping) else {}, root_data=data)

# backwards compatibility for earlier name
AmplificationConfig = Amplification


class BaseWDM:
    """Base WDM class."""

    def frequency_grid(self) -> np.ndarray:
        raise NotImplementedError

    def decimate(self, factor: int, rescale_power: bool = False) -> "BaseWDM":
        """Return a sparsified WDM keeping every `factor`-th channel.

        Parameters
        ----------
        factor: int
            Keep one channel out of every `factor` (factor >= 1). factor=1 returns self.
        rescale_power: bool, optional
            If True and band launch powers are available (IrregularWDM), per-channel
            launch_power_dbm is increased to conserve total band power after decimation.
        """
        raise NotImplementedError

    def wavelength_grid(self) -> np.ndarray:
        """Generate the wavelength grid in meters."""
        return nu2lambda(self.frequency_grid())

    def plot(self, ax, xaxis="wavelength", **kwargs):
        pynlin.utils.OpticalBands.plot(ax, xaxis=xaxis)
        if xaxis == "wavelength":
            w = self.wavelength_grid() * 1e9
            ax.set_xlabel("Wavelength [nm]")
        else:
            w = self.frequency_grid() * 1e-12
            ax.set_xlabel("Frequency [THz]")

        ax.stem(w, np.ones_like(w), **kwargs)

    def summary(self) -> str:
        raise NotImplementedError

    def log(self, logger=None, level: str = "trace"):
        """Log or print a summary; defaults to loguru-style trace."""
        summary = self.summary()
        if logger is None:
            print(summary)
            return
        log_fn = getattr(logger, level, None)
        if callable(log_fn):
            log_fn(summary)
        elif hasattr(logger, "log"):
            logger.log(level.upper() if isinstance(level, str) else level, summary)
        else:
            print(summary)


DEFAULT_CENTER_FREQUENCY_HZ = 193.1e12


class RegularWDM(BaseWDM):
    """Uniform-grid WDM."""

    def __init__(self, spacing: float, num_channels: int, center_frequency: float | None = None):
        if center_frequency is None:
            center_frequency = DEFAULT_CENTER_FREQUENCY_HZ
        self.spacing = spacing
        self.num_channels = num_channels
        self.central_frequency = center_frequency

    @classmethod
    def from_config(cls, cfg: WDMConfig) -> "RegularWDM":
        return cls(cfg.spacing, cfg.num_channels, cfg.center_frequency)

    @classmethod
    def from_toml(cls, filepath: Path | str) -> "RegularWDM":
        data = _toml_load(Path(filepath))
        wdm_data = _extract_wdm_section(data)
        cfg = WDMConfig.from_mapping(wdm_data if isinstance(wdm_data, Mapping) else {}, root_data=data)
        return cls.from_config(cfg)

    def frequency_grid(self) -> np.ndarray:
        num_channels = self.num_channels
        if num_channels % 2:
            freqs = np.arange(-(num_channels - 1) / 2, (num_channels - 1) / 2 + 1)
        else:
            freqs = np.arange(-num_channels / 2, num_channels / 2)
        return freqs * self.spacing + self.central_frequency

    def decimate(self, factor: int, rescale_power: bool = False) -> "RegularWDM":
        if factor < 1:
            raise ValueError("decimate factor must be >= 1")
        if factor == 1:
            return self
        idx = np.arange(0, self.num_channels, factor, dtype=int)
        if idx.size < 1:
            raise ValueError("decimation removed all channels")
        freqs = self.frequency_grid()[idx]
        if freqs.size > 1:
            diffs = np.diff(freqs)
            if not np.allclose(diffs, diffs[0]):
                raise ValueError("decimated grid is not uniform; cannot keep RegularWDM")
            new_spacing = float(np.abs(diffs[0]))
        else:
            new_spacing = self.spacing * factor
        center_freq = float(freqs[len(freqs) // 2])
        new_wdm = RegularWDM(
            spacing=new_spacing,
            num_channels=int(idx.size),
            center_frequency=center_freq,
        )
        # Attach a power scaling hint so callers can optionally conserve total power.
        new_wdm.power_scale = (self.num_channels / idx.size) if rescale_power else 1.0
        return new_wdm

    def summary(self) -> str:
        return "\n".join([
            f"Spacing       : {self.spacing:.3e} Hz",
            f"Channels      : {self.num_channels}",
            f"Center freq   : {self.central_frequency:.3e} Hz",
        ])


class IrregularWDM(BaseWDM):
    """Multi-band WDM with arbitrary band start wavelengths."""

    def __init__(
        self,
        spacing: float,
        band_specs: Dict[str, WDMBandConfig],
        band_slices: Dict[str, slice],
        full_grid: np.ndarray,
    ):
        self.spacing = spacing
        self.band_specs = band_specs
        self._band_slices = band_slices
        self._frequency_grid = full_grid
        self.num_channels = len(full_grid)
        self.central_frequency = float(np.mean(full_grid))

    @classmethod
    def from_bands_mapping(cls, bands_data: Mapping, wdm_data=None, root_data=None) -> "IrregularWDM":
        spacing = _extract_spacing_hz(wdm_data, root_data)
        if spacing is None:
            raise ValueError("Spacing is required to build WDM grid from band specs.")

        band_specs: Dict[str, WDMBandConfig] = {}
        for name, cfg in bands_data.items():
            band_specs[name] = WDMBandConfig(**cfg)

        c = 299_792_458
        band_slices: Dict[str, slice] = {}
        grids = []
        current = 0
        intervals = []
        tol = spacing * 1e-6 if spacing else 0.0

        for name, spec in sorted(band_specs.items(), key=lambda kv: kv[1].start_nm):
            f_start = c / (spec.start_nm * 1e-9)
            f_band = f_start - np.arange(spec.n_channels) * spacing

            # Overlap check against existing bands
            f_low, f_high = float(np.min(f_band)), float(np.max(f_band))
            for other_name, o_low, o_high in intervals:
                if max(f_low, o_low) <= min(f_high, o_high) + tol:
                    raise ValueError(
                        f"WDM bands '{name}' and '{other_name}' overlap in frequency "
                        f"ranges [{f_low*1e-12:.3f}, {f_high*1e-12:.3f}] THz and "
                        f"[{o_low*1e-12:.3f}, {o_high*1e-12:.3f}] THz"
                    )
            intervals.append((name, f_low, f_high))

            grids.append(f_band)
            band_slices[name] = slice(current, current + spec.n_channels)
            current += spec.n_channels

        full_grid = np.concatenate(grids)
        return cls(
            spacing=spacing,
            band_specs=band_specs,
            band_slices=band_slices,
            full_grid=full_grid,
        )

    @classmethod
    def from_toml(cls, filepath: Path | str) -> "IrregularWDM":
        data = _toml_load(Path(filepath))
        wdm_data = _extract_wdm_section(data)
        bands_data = _extract_irregular_bands_data(wdm_data)
        if bands_data is None:
            raise ValueError("No bands section found for IrregularWDM.")
        return cls.from_bands_mapping(bands_data, wdm_data, root_data=data)

    def frequency_grid(self) -> np.ndarray:
        return self._frequency_grid

    def decimate(self, factor: int, rescale_power: bool = False) -> "IrregularWDM":
        if factor < 1:
            raise ValueError("decimate factor must be >= 1")
        if factor == 1:
            return self

        new_band_specs: Dict[str, WDMBandConfig] = {}
        new_band_slices: Dict[str, slice] = {}
        new_grids = []
        current = 0

        for name, slc in self._band_slices.items():
            spec = self.band_specs[name]
            band_freqs = self._frequency_grid[slc]
            idx = np.arange(0, band_freqs.size, factor, dtype=int)
            if idx.size < 1:
                raise ValueError(f"decimation removed all channels from band '{name}'")
            decimated = band_freqs[idx]
            power_dbm = spec.launch_power_dbm
            if rescale_power and power_dbm is not None:
                scale = band_freqs.size / idx.size
                power_dbm = float(power_dbm + 10 * np.log10(scale))
            new_band_specs[name] = WDMBandConfig(
                n_channels=int(idx.size),
                launch_power_dbm=power_dbm,
                start_nm=spec.start_nm,
                modulation=spec.modulation,
            )
            new_band_slices[name] = slice(current, current + idx.size)
            current += idx.size
            new_grids.append(decimated)

        full_grid = np.concatenate(new_grids) if new_grids else np.array([])
        if full_grid.size == 0:
            raise ValueError("decimation produced an empty WDM grid")

        new_spacing = self.spacing * factor
        new_wdm = IrregularWDM(
            spacing=new_spacing,
            band_specs=new_band_specs,
            band_slices=new_band_slices,
            full_grid=full_grid,
        )
        new_wdm.power_scale = (self.num_channels / full_grid.size) if rescale_power else 1.0
        return new_wdm

    def summary(self) -> str:
        lines = [
            f"Spacing       : {self.spacing:.3e} Hz",
            f"Channels      : {self.num_channels}",
            f"Center freq   : {self.central_frequency:.3e} Hz",
        ]
        lines.append("Bands:")
        for name, slc in self._band_slices.items():
            spec = self.band_specs.get(name)
            f_band = self.frequency_grid()[slc]
            lines.append(
                f"  {name}: {slc.stop - slc.start} ch, "
                f"start {spec.start_nm if spec else 'n/a'} nm, "
                f"freq {f_band.max()*1e-12:.2f}-{f_band.min()*1e-12:.2f} THz, "
                f"mod {spec.modulation if spec else 'n/a'}"
            )
        return "\n".join(lines)

    def get_band_modulation(self, band_name: str) -> Optional[str]:
        """Return the modulation format for a band, if available."""
        spec = self.band_specs.get(band_name)
        return spec.modulation if spec else None


def wdm_from_toml(filepath: Path | str) -> BaseWDM:
    """Helper to load either RegularWDM or IrregularWDM from TOML."""
    data = _toml_load(Path(filepath))
    wdm_data = _extract_wdm_section(data)

    bands_data = _extract_irregular_bands_data(wdm_data)
    if bands_data:
        return IrregularWDM.from_bands_mapping(bands_data, wdm_data, root_data=data)

    cfg = WDMConfig.from_mapping(wdm_data if isinstance(wdm_data, Mapping) else {}, root_data=data)
    return RegularWDM.from_config(cfg)

# Backwards compatibility: old code using WDM(...) will create a regular grid
WDM = RegularWDM

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        raise SystemExit("Usage: python -m pynlin.wdm <system.toml>")
    cfg_path = Path(sys.argv[1])
    wdm = wdm_from_toml(cfg_path)
    print(f"Loaded WDM from {cfg_path}:")
    print(wdm.summary())
