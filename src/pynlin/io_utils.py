import os
from enum import Enum
import numpy as np
from pydantic import BaseModel
from scipy.constants import lambda2nu, nu2lambda
import toml

class OpticalBands(Enum):
    """Class enumerating the optical transmission bandwidths.

    Data taken from: https://www.thefoa.org/tech/ref/basic/SMbands.html
    """

    O = (1260, 1360)
    E = (1360, 1460)
    S = (1460, 1530)
    C = (1530, 1565)
    L = (1565, 1625)
    U = (1625, 1675)

    def __add__(self, other):
        joint = self.value + other.value
        return (min(joint), max(joint))

    @staticmethod
    def plot(ax, xaxis="wavelength", **kwargs):
        for x, band in enumerate(OpticalBands):
            m, M = band.value

            if xaxis == "frequency":
                m = lambda2nu(m * 1e-9) * 1e-12
                M = lambda2nu(M * 1e-9) * 1e-12

            mid_point = (M + m) / 2
            name = band.name
            ax.axvspan(m, M, alpha=0.1, color=f"C{x}")
            ax.text(mid_point, 0.1, name)

        if xaxis == "frequency":
            ax.set_xlabel("Frequency [THz]")
        else:
            ax.set_xlabel("Wavelength [nm]")


"""
O-band 	1260 – 1360 nm 	Original band, PON upstream
E-band 	1360 – 1460 nm 	Water peak band
S-band 	1460 – 1530 nm 	PON downstream
C-band 	1530 – 1565 nm 	Lowest attenuation, original DWDM band, compatible with fiber amplifiers, CATV
L-band 	1565 – 1625 nm 	Low attenuation, expanded DWDM band
U-band 	1625 – 1675 nm 	Ultra-long wavelength
"""

class Config(BaseModel):
  dispersion       : float
  effective_area   : float
  baud_rate        : float
  fiber_length     : float
  n_modes          : int
  n_channels       : int
  launch_power     : float
  raman_gain       : float
  channel_spacing  : float
  center_frequency : float
  store            : bool
  pulse_shape      : int
  collision_margin : int
  n_pumps          : int 

class NumericalConfig(BaseModel):
   gvd             : float
   dgd1            : float
   dgd2_g          : float
   dgd2_n          : float
   n_samples_numeric_g : int
   n_samples_numeric_n : int

# Deserialize TOML file into a Pydantic model
def _toml_load(filepath):
    """
    Load TOML data handling both stdlib tomllib (expects a binary file object)
    and the third-party toml package (accepts paths or file-like objects).
    """
    if getattr(toml, "__name__", "") == "tomllib" and isinstance(filepath, (str, os.PathLike)):
        with open(filepath, "rb") as f:
            return toml.load(f)
    return toml.load(filepath)


def load_toml_to_struct(filepath) -> Config:
    """Load simulation configuration from a TOML file into a Config object."""
    data = _toml_load(filepath)
    return Config(**data)

def load_nc_toml_to_struct(filepath) -> NumericalConfig:
    """Load numerical configuration from TOML into a NumericalConfig object."""
    data = _toml_load(filepath)
    return NumericalConfig(**data)

# Serialize a Pydantic model into a TOML file
def save_struct_to_toml(filepath: str, config: Config):
    with open(filepath, "w") as f:
        toml.dump(config.model_dump(), f)
        

def get_next_filename(
  base_name, 
  extension, 
  use_active_naming=True):
    """
    Generate a unique file name by appending an incrementing numeral
    if a file with the same name already exists.

    Args:
        base_name (str): The base name of the file (without extension).
        extension (str): The file extension (with or without a dot).

    Returns:
        str: A unique file name.
    """
    if use_active_naming:
      if not extension.startswith('.'):
          extension = '.' + extension
      
      filename = f"{base_name}{extension}"
      counter = 1

      # Check if the file already exists and increment until it's unique
      while os.path.exists(filename):
          filename = f"{base_name}_{counter}{extension}"
          counter += 1
    else:
      if not extension.startswith('.'):
          extension = '.' + extension
      filename = f"{base_name}{extension}"
    return filename




class PulseShape(Enum):
    GAU = 0
    NYQ = 1
    def __str__(self):
        return self.name.lower()   # "gaussian" or "nyquist"

    @classmethod
    def from_str(cls, s: str):
        """Parse from a lowercase string."""
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(f"Unknown pulse shape: {s}")
        
    # specify the line style:
    def line_style(self):
        """Matplotlib line style associated with the pulse shape."""
        if self == PulseShape.GAU:
            return "-"
        elif self == PulseShape.NYQ:
            return "--"
        else:
            raise ValueError(f"Unknown pulse shape: {self}")
