from enum import Enum

import numpy as np
import scipy.constants
import torch
from scipy.constants import lambda2nu
from scipy.constants import speed_of_light as c0

import os
import toml
from pydantic import BaseModel

def wavelength_to_frequency(lambdas):
    """Convert wavelength to frequency."""
    return scipy.constants.lambda2nu(lambdas)


def frequency_to_wavelength(freqs):
    """Convert frequency to wavelength."""
    return scipy.constants.nu2lambda(freqs)


def alpha_to_linear(alpha):
    """Convert attenuation constant from dB/m to neper/m."""
    return alpha * np.log(10) / 10


def alpha2linear(alpha):
    return alpha_to_linear(alpha)


def wavelength2frequency(lambdas):
    return wavelength_to_frequency(lambdas)


def frequency2wavelength(freqs):
    return frequency_to_wavelength(freqs)


def watt_to_dBm(power):
    return 10 * np.log10(power) + 30


def dBm_to_watt(power):
    return 10 ** ((power - 30) / 10)


def watt2dBm(power):
    return watt_to_dBm(power)


def dBm2watt(power):
    return dBm_to_watt(power)


def beta2_to_dispersion(beta2, wavelength):
    """Convert GVD (beta2) parameter to dispersion coefficient.
    Values are assumed in SI units.
    """
    return -2 * np.pi * c0 / (wavelength**2) * beta2

def beta2rms(beta2a, beta2b):
    return np.sqrt((beta2a**2 + beta2b**2)/2)


def dispersion_to_beta2(D, wavelength):
    """Convert the dispersion coefficient to GVD (beta2).
    Values are assumed in SI units.
    """
    return D * wavelength**2 / (-2 * np.pi * c0)


def oi_law(l1, l2, params):
    a1, b1, a2, b2, x, c = params
    return (a1 * l1**2 + b1 * l1 + a2 * l2**2 + b2 * l2 + x * l1 * l2 + c)

 
def oi_polynomial_expansion(wl, values):
  
    A1, B1, A2, B2, X, C = values
    batch_size = wl.shape[0]
    M = A1.shape[0] # Number of modes
    N = wl.shape[1] # Number of wavelengths
    # print(f"{A1.shape} {B1.shape} {A2.shape} {B2.shape} {X.shape} {C.shape}")
    # print(f"{wl.shape}")
    OIa1 = torch.kron((wl**2).reshape(batch_size, N, 1), A1).repeat(1, 1, N)
    OIa2 = torch.kron((wl**2).reshape(batch_size, 1, N), A2).repeat(1, N, 1)
    OIb1 = torch.kron(wl.reshape(batch_size, N, 1), B1).repeat(1, 1, N)
    OIb2 = torch.kron(wl.reshape(batch_size, 1, N), B2).repeat(1, N, 1)
    OIx = torch.kron(torch.kron(wl.reshape(batch_size, N, 1), X), wl.reshape(batch_size, 1, N))
    OIc = C.repeat((batch_size, N, N))
    # print(f"{OIa1.shape} {OIa2.shape} {OIb1.shape} {OIb2.shape} {OIx.shape} {OIc.shape}")
    return (OIa1 + OIa2 + OIb1 + OIb2 + OIx + OIc).float()

    # A1, B1, A2, B2, X, C = values
    # batch_size = wl.shape[0]
    # N = wl.shape[1]
    
    # OIa1 = torch.kron((wl**2).reshape(batch_size, N, 1), A1).repeat(1, N, 1)
    # OIa2 = torch.kron(wl**2, A2).repeat(batch_size, N, 1)
    # OIb1 = torch.kron(wl.reshape(batch_size, N, 1), B1).repeat(1, N, N)
    # OIb2 = torch.kron(wl, B2).repeat(batch_size, N, 1)
    # OIx = torch.kron(torch.kron(wl.reshape(batch_size, N, 1), X), wl)
    # OIc = C.repeat(batch_size, N, N)
    
    # return (OIa1 + OIa2 + OIb1 + OIb2 + OIx + OIc).float()



# def beta_n_polynomial_expansion(vals, f):
#   beta1 = vals[0] * f ** 2 + vals[1] * f + vals[2]
#   return beta1


def oi_law_fit(L, a1, b1, a2, b2, x, c):
    l1, l2 = L
    return (a1 * np.square(l1) + b1 * l1 + a2 * np.square(l2) + b2 * l2 + x * l1 * l2 + c).ravel()


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
def load_toml_to_struct(filepath: str) -> Config:
    """Load simulation configuration from a TOML file into a Config object."""
    data = toml.load(filepath)
    return Config(**data)

def load_nc_toml_to_struct(filepath: str) -> NumericalConfig:
    """Load numerical configuration from TOML into a NumericalConfig object."""
    data = toml.load(filepath)
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
