import numpy as np
from scipy.constants import nu2lambda
from pathlib import Path
from typing import Dict, Mapping, Optional

import pynlin.utils
from pynlin.utils import BaseModel, _toml_load


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


class WDMConfig(BaseModel):
    spacing: float
    num_channels: int
    center_frequency: float

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

    class Config:
        extra = "ignore"


class BaseWDM:
    """Base WDM class."""

    def frequency_grid(self) -> np.ndarray:
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


class RegularWDM(BaseWDM):
    """Uniform-grid WDM."""

    def __init__(self, spacing: float, num_channels: int, center_frequency: float):
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
        for name, spec in sorted(band_specs.items(), key=lambda kv: kv[1].start_nm):
            f_start = c / (spec.start_nm * 1e-9)
            f_band = f_start - np.arange(spec.n_channels) * spacing
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
        bands_data = None
        if isinstance(wdm_data, Mapping):
            for key in ("bands", "band"):
                if key in wdm_data and isinstance(wdm_data[key], Mapping):
                    bands_data = wdm_data[key]
                    break
        if bands_data is None:
            raise ValueError("No bands section found for IrregularWDM.")
        return cls.from_bands_mapping(bands_data, wdm_data, root_data=data)

    def frequency_grid(self) -> np.ndarray:
        return self._frequency_grid

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

    bands_data = None
    if isinstance(wdm_data, Mapping):
        for key in ("bands", "band"):
            if key in wdm_data and isinstance(wdm_data[key], Mapping):
                bands_data = wdm_data[key]
                break
    if bands_data:
        return IrregularWDM.from_bands_mapping(bands_data, wdm_data, root_data=data)

    cfg = WDMConfig.from_mapping(wdm_data if isinstance(wdm_data, Mapping) else {}, root_data=data)
    return RegularWDM.from_config(cfg)

# Backwards compatibility: old code using WDM(...) will create a regular grid
WDM = RegularWDM

if __name__ == "__main__":
    import sys
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input/jlt2.toml")
    wdm = wdm_from_toml(cfg_path)
    print(f"Loaded WDM from {cfg_path}:")
    print(wdm.summary())
