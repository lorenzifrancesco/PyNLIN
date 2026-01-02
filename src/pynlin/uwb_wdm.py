import numpy as np
import tomllib
from pydantic import BaseModel
from typing import Dict
from pynlin.wdm import WDM

class BandSettings(BaseModel):
    n_channels: int
    launch_power_dbm: float
    start_nm: float

class UWBSystemSettings(BaseModel):
    baud_rate_gbaud: float
    spacing_ghz: float
    bands: Dict[str, BandSettings]

class UWBWDM(WDM):
    """
    UWB-aware WDM class compatible with Puttnam (2025) benchmarks.
    Inherits from WDM to maintain backward compatibility.
    """
    def __init__(self, toml_path: str):
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
            self.settings = UWBSystemSettings(
                baud_rate_gbaud=data["system"]["baud_rate_gbaud"],
                spacing_ghz=data["system"]["spacing_ghz"],
                bands=data["bands"]
            )
        
        self.c = 299792458
        self._band_slices = {}
        self._full_grid = []
        
        # Sort bands by wavelength O -> U to build a continuous index map
        sorted_bands = sorted(self.settings.bands.items(), key=lambda x: x[1].start_nm)
        
        current_pos = 0
        for name, b_cfg in sorted_bands:
            f_start = self.c / (b_cfg.start_nm * 1e-9)
            f_band = f_start - np.arange(b_cfg.n_channels) * (self.settings.spacing_ghz * 1e9)
            
            self._full_grid.append(f_band)
            self._band_slices[name] = slice(current_pos, current_pos + b_cfg.n_channels)
            current_pos += b_cfg.n_channels

        self._full_grid = np.concatenate(self._full_grid)

        super().__init__(
            spacing=self.settings.spacing_ghz * 1e9,
            num_channels=len(self._full_grid),
            center_frequency=np.mean(self._full_grid)
        )

    def frequency_grid(self) -> np.ndarray:
        return self._full_grid

    def get_launch_power_per_channel(self, band_name: str) -> float:
        b = self.settings.bands[band_name]
        return b.launch_power_dbm - 10 * np.log10(b.n_channels)

    def print_summary(self):
        """Prints a detailed report of the UWB system configuration."""
        print("="*85)
        print(f"{'UWB SYSTEM SUMMARY':^85}")
        print("="*85)
        print(f"Global Spacing: {self.settings.spacing_ghz} GHz")
        print(f"Baud Rate:      {self.settings.baud_rate_gbaud} GBaud")
        print(f"Total Channels: {len(self._full_grid)}")
        print("-" * 85)
        print(f"{'Band':<6} | {'Channels':<10} | {'Start (nm)':<12} | {'Freq Range (THz)':<22} | {'P_ch (dBm)':<10}")
        print("-" * 85)

        for name, slc in self._band_slices.items():
            b_cfg = self.settings.bands[name]
            f_band = self._full_grid[slc]
            p_ch = self.get_launch_power_per_channel(name)
            
            f_max = f_band.max() * 1e-12
            f_min = f_band.min() * 1e-12
            
            print(f"{name:<6} | {b_cfg.n_channels:<10} | {b_cfg.start_nm:<12.1f} | "
                  f"{f_max:>6.2f} - {f_min:<6.2f} THz     | {p_ch:<10.2f}")
        
        print("="*85)

# --- Usage Example ---
if __name__ == "__main__":
    uwb = UWBWDM("input/uwb_settings.toml")
    uwb.print_summary()