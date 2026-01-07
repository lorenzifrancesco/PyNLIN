from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy import polyval
from scipy.constants import speed_of_light as c0

from pynlin.utils import (
    BaseModel,
    ConfigDict,
    EXTRA_IGNORE_CONFIG,
    dispersion_to_beta2,
    oi_law,
    oi_polynomial_expansion,
    _toml_load,
)


def _extract_fiber_section(data):
    """Return the fiber subsection if present, otherwise the raw mapping."""
    if not isinstance(data, Mapping):
        return data
    fiber_data = data.get("fiber", data)
    if isinstance(fiber_data, Mapping):
        fiber_data = dict(fiber_data)
        wdm = data.get("wdm")
        if isinstance(wdm, Mapping):
            for key in ("center_frequency", "n_modes"):
                if key not in fiber_data and key in wdm:
                    fiber_data[key] = wdm[key]
    return fiber_data


def _load_fiber_csv(path: Path, center_frequency: Optional[float] = None):
    """
    Load wavelength-dependent fiber data from CSV.

    Expected headers (matching input/fiber_data/smf28.csv):
    frequency_Hz, Aeff_um2, alpha_dB_per_km, beta2_s2_per_m, beta_1_per_m, beta1_s_per_m
    """
    data = np.genfromtxt(path, delimiter=",", names=True)
    freq = data["frequency_Hz"]
    wavelengths = c0 / freq
    order = np.argsort(wavelengths)  # ascending wavelength for interpolation
    wavelengths = wavelengths[order]
    aeff = data["Aeff_um2"][order] * 1e-12  # um^2 -> m^2
    alpha = data["alpha_dB_per_km"][order] * 1e-3  # dB/km -> dB/m
    beta2 = data["beta2_s2_per_m"][order] if "beta2_s2_per_m" in data.dtype.names else None
    beta = data["beta_1_per_m"][order] if "beta_1_per_m" in data.dtype.names else None
    beta1 = data["beta1_s_per_m"][order] if "beta1_s_per_m" in data.dtype.names else None

    center_wl = c0 / center_frequency if center_frequency else None

    def _interp(arr):
        if arr is None:
            return None
        if center_wl is not None:
            return float(np.interp(center_wl, wavelengths, arr))
        return float(np.mean(arr))

    beta2_center = _interp(beta2)
    beta_center = _interp(beta)
    beta1_center = _interp(beta1)
    aeff_center = _interp(aeff)
    alpha_center = _interp(alpha)
    return {
        "frequencies": freq[order],
        "wavelengths": wavelengths,
        "aeff_profile": aeff,
        "alpha_profile": alpha,
        "beta_profile": beta,
        "beta1_profile": beta1,
        "beta2_profile": beta2,
        "aeff_center": aeff_center,
        "alpha_center": alpha_center,
        "beta_center": beta_center,
        "beta1_center": beta1_center,
        "beta2_center": beta2_center,
    }


class FiberConfig(BaseModel):
    effective_area: float
    fiber_length: float
    path_to_csv: Optional[str] = None
    losses: Optional[Sequence[float]] = None
    raman_coefficient: float = 7e-14
    gamma: float = 1.3e-3

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"

    @classmethod
    def from_toml(cls, filepath):
        data = _extract_fiber_section(_toml_load(Path(filepath)))
        return cls(**data)


class SMFiberConfig(FiberConfig):
    beta2: Optional[float] = None
    dispersion: Optional[float] = None
    center_frequency: Optional[float] = None
    effective_area_wavelengths: Optional[Sequence[float]] = None
    effective_area_values: Optional[Sequence[float]] = None
    attenuation_wavelengths: Optional[Sequence[float]] = None
    attenuation_values: Optional[Sequence[float]] = None

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"

    def resolved_beta2(self, fallback: float) -> float:
        if self.beta2 is not None:
            return self.beta2
        if self.dispersion is not None and self.center_frequency is not None:
            wavelength = 3e8 / self.center_frequency
            return dispersion_to_beta2(self.dispersion, wavelength)
        return fallback


class SMFiberNarrowbandConfig(SMFiberConfig):
    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


class MMFiberConfig(FiberConfig):
    n_modes: int = 4

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"

    @classmethod
    def from_toml(cls, filepath):
        data = _extract_fiber_section(_toml_load(Path(filepath)))
        return cls(**data)


class Fiber:
    """Object collecting parameters of a Single Mode optical fiber."""

    def __init__(
        self,
        fiber_type,
        losses=None,
        raman_coefficient=7e-14,
        effective_area=80e-12,
        gamma=1.3 * 1e-3,
        n_modes=1,
        length=100e3
    ):
        self.fiber_type = fiber_type
        self.effective_area = effective_area
        self.raman_coefficient = raman_coefficient
        self.gamma = gamma
        self.n_modes = n_modes
        self.length = length
        if losses is not None:
            try:
                self.losses = list(losses)
            except:
                self.losses = [losses]
        else:
            # coefficients of the [0, 1, 2]-th order coefficients of a quadratic fit in
            # powers of wavelength (in m). Result in units of dB/m
            self.losses = np.array([2.26786883e-06 * 1e18, -
                                    7.12461042e-03 * 1e9, 5.78789219e00]) * 1e-3
        self.raman_efficiency = self.raman_coefficient / self.effective_area

    def loss_profile(self, wavelengths):
        """Get the fiber losses (in dB/m) at the specified wavelengths (in
        meters)."""
        return polyval(self.losses, wavelengths)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class SMFiber(Fiber):
    """Single Mode fiber with optional frequency-dependent Aeff/attenuation."""

    _DEFAULT_BETA2 = 20 * 1e-24 / 1e3

    def __init__(
        self,
        losses=None,
        raman_coefficient=7e-14,
        effective_area=80e-12,
        beta=None,
        beta1=None,
        beta2=_DEFAULT_BETA2,
        gamma=1.3 * 1e-3,
        length=100e3,
        effective_area_profile: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        attenuation_profile: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        beta_profile: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        beta1_profile: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        beta2_profile: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    ):
        super().__init__("SM",
                         losses=losses,
                         raman_coefficient=raman_coefficient,
                         effective_area=effective_area,
                         gamma=gamma,
                         length=length)
        self.effective_area = effective_area
        self.raman_coefficient = raman_coefficient
        self.gamma = gamma
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self._effective_area_profile = self._build_profile(effective_area_profile)
        self._attenuation_profile = self._build_profile(attenuation_profile)
        self._beta_profile = self._build_profile(beta_profile)
        self._beta1_profile = self._build_profile(beta1_profile)
        self._beta2_profile = self._build_profile(beta2_profile)
        self._freq_profile = None

        self.raman_efficiency = self.raman_coefficient / self.effective_area

    @staticmethod
    def _build_profile(profile):
        if profile is None:
            return None
        wavelengths, values = profile
        return (np.array(wavelengths), np.array(values))
    def _interp_profile(self, profile, wavelength: float) -> Optional[float]:
        if profile is None:
            return None
        wl, values = profile
        return float(np.interp(wavelength, wl, values))

    def effective_area_at(self, wavelength: float) -> float:
        """Return effective area interpolated on wavelength if profile provided."""
        if self._effective_area_profile is None:
            return self.effective_area
        wl, values = self._effective_area_profile
        return float(np.interp(wavelength, wl, values))

    def attenuation_at(self, wavelength: float) -> float:
        """Return attenuation interpolated on wavelength if profile provided."""
        if self._attenuation_profile is None:
            return self.loss_profile(wavelength)
        wl, values = self._attenuation_profile
        return float(np.interp(wavelength, wl, values))

    def loss_profile(self, wavelengths):
        """Override to use attenuation profile when provided."""
        wl = np.asarray(wavelengths)
        if self._attenuation_profile is not None:
            x, y = self._attenuation_profile
            return np.interp(wl, x, y)
        return super().loss_profile(wl)

    def beta_at(self, wavelength: float) -> Optional[float]:
        return self._interp_profile(self._beta_profile, wavelength)

    def beta1_at(self, wavelength: float) -> Optional[float]:
        return self._interp_profile(self._beta1_profile, wavelength)

    def beta2_at(self, wavelength: float) -> Optional[float]:
        if self._beta2_profile is not None:
            return self._interp_profile(self._beta2_profile, wavelength)
        return self.beta2

    def beta_profile(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (wavelengths, beta) if loaded."""
        return self._beta_profile

    def beta1_profile(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (wavelengths, beta1) if loaded."""
        return self._beta1_profile

    def beta2_profile(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (wavelengths, beta2) if loaded."""
        return self._beta2_profile
    
    @property
    def frequency_profile(self) -> Optional[np.ndarray]:
        """Return the frequency grid corresponding to the loaded profiles, if any."""
        return self._freq_profile

    @classmethod
    def from_config(cls, config: SMFiberConfig) -> "SMFiber":
        if config.path_to_csv:
            fiber_data = _load_fiber_csv(Path(config.path_to_csv), config.center_frequency)
            beta2 = fiber_data["beta2_center"] if fiber_data["beta2_center"] is not None else config.resolved_beta2(cls._DEFAULT_BETA2)
            eff_area = fiber_data["aeff_center"] if fiber_data["aeff_center"] is not None else config.effective_area
            eff_profile = (fiber_data["wavelengths"], fiber_data["aeff_profile"])
            att_profile = (fiber_data["wavelengths"], fiber_data["alpha_profile"])
            beta_profile = (fiber_data["wavelengths"], fiber_data["beta_profile"]) if fiber_data["beta_profile"] is not None else None
            beta1_profile = (fiber_data["wavelengths"], fiber_data["beta1_profile"]) if fiber_data["beta1_profile"] is not None else None
            beta2_profile = (fiber_data["wavelengths"], fiber_data["beta2_profile"]) if fiber_data["beta2_profile"] is not None else None
            beta_center = None  # do not expose scalar beta in SMFiber
            beta1_center = None
            freq_profile = fiber_data["frequencies"]
        else:
            beta2 = config.resolved_beta2(cls._DEFAULT_BETA2)
            eff_area = config.effective_area
            eff_profile = (
                (config.effective_area_wavelengths, config.effective_area_values)
                if config.effective_area_wavelengths and config.effective_area_values
                else None
            )
            att_profile = (
                (config.attenuation_wavelengths, config.attenuation_values)
                if config.attenuation_wavelengths and config.attenuation_values
                else None
            )
            beta_profile = beta1_profile = beta2_profile = None
            beta_center = beta1_center = None
            freq_profile = None
        fiber = cls(
            losses=config.losses,
            raman_coefficient=config.raman_coefficient,
            effective_area=eff_area,
            beta=beta_center,
            beta1=beta1_center,
            beta2=beta2,
            gamma=config.gamma,
            length=config.fiber_length,
            effective_area_profile=eff_profile,
            attenuation_profile=att_profile,
            beta_profile=beta_profile,
            beta1_profile=beta1_profile,
            beta2_profile=beta2_profile,
        )
        if freq_profile is not None:
            fiber._freq_profile = np.array(freq_profile)
        return fiber

    @classmethod
    def from_toml(cls, filepath: Path | str) -> "SMFiber":
        return cls.from_config(SMFiberConfig.from_toml(filepath))

    def loss_profile(self, wavelengths):
        """Get the fiber losses (in dB/m) at the specified wavelengths (in
        meters)."""
        return polyval(self.losses, wavelengths)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class SMFiberNarrowband(SMFiber):
    """Narrowband SM fiber matching the legacy constant-parameter behavior."""

    @classmethod
    def from_config(cls, config: SMFiberNarrowbandConfig) -> "SMFiberNarrowband":
        beta2 = config.resolved_beta2(cls._DEFAULT_BETA2)
        return cls(
            losses=config.losses,
            raman_coefficient=config.raman_coefficient,
            effective_area=config.effective_area,
            beta2=beta2,
            gamma=config.gamma,
            length=config.fiber_length,
            effective_area_profile=(
                (config.effective_area_wavelengths, config.effective_area_values)
                if config.effective_area_wavelengths and config.effective_area_values
                else None
            ),
            attenuation_profile=(
                (config.attenuation_wavelengths, config.attenuation_values)
                if config.attenuation_wavelengths and config.attenuation_values
                else None
            ),
        )

    @classmethod
    def from_toml(cls, filepath: Path | str) -> "SMFiberNarrowband":
        return cls.from_config(SMFiberNarrowbandConfig.from_toml(filepath))


"""
OI fit storage and evaluation in a way which is compatible with torch
"""
@dataclass
class OICoefficients:
    values: list[torch.Tensor]

    def __init__(self, modes: int, input_values: np.ndarray):
        self.values = [torch.from_numpy(v[:modes, :modes]) for v in input_values]
        # self.num_modes, dim=2
        # ).float()

    def evaluate_oi_tensor(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """Evaluate the overlap integral between the pump and signal modes.+

        Parameters
        ----------
        coefficients : OICoefficients
            The overlap integral coefficients. The `values` attribute should be a list of length 6
              of torch.Tensor elements, each of size (num_modes, num_modes).

        wavelengths : torch.Tensor (batch_dim, num_frequencies (pumps + signals))
          Tensor containing all the wavelengths involved in the system.
        """
        return oi_polynomial_expansion(wavelengths, self.values)


@dataclass
class GroupDelay:
    """
    Dispersion data storage and evaluation for the walkoff and collision evaluation
    
    Third order fit data for the beta1 
    (and beta2) functions of frequency and modes
    
    Data format:
        values: (num_modes, 3)
    """
    values: list[np.array]

    def __init__(self, modes: int, beta1_values: np.ndarray):
        self.modes = modes
        self.values = beta1_values

    def evaluate_beta1(self, mode: int, frequency: float) -> torch.Tensor:
        return polyval(self.values[mode, :], frequency)

    def evaluate_beta2(self, mode: int, frequency: float) -> torch.Tensor:
        derivative_parameters = [self.values[mode, 0]/np.pi, self.values[mode, 1]/(2*np.pi)]
        return polyval(derivative_parameters, frequency)

class MMFiber(Fiber):
    def __init__(
        self,
        losses=None,
        raman_coefficient=7e-14,
        effective_area=80e-12,
        group_delay=None,
        gamma=1.3 * 1e-3,
        length=100e3,
        n_modes=4,
        overlap_integrals=None,
        mode_names=None,
    ):
        """
        Params
        ======
        overlap_integrals     : 6 quadratic fit parameters for each mode family pair
        overlap_integrals_avg : 1 oi for each mode family pair
        Strategy: if overlap_integrals is none, go to default case to overlap_integrals_avg
        both of them can be used with polyval (of course using the average is inefficient)

        Attributes
        =======
        self.overlap_integrals : (6, modes, modes) used only in the Numpy solver: can also contain also the average if needed!
        """
        super().__init__("MM",
                         losses=losses,
                         raman_coefficient=raman_coefficient,
                         effective_area=effective_area,
                         gamma=gamma,
                         n_modes=n_modes,
                         length=length)
        self.raman_efficiency = self.raman_coefficient / self.effective_area # FIXME 01

        # Structure of the overlap integrals
        # overlap_integrals[i, j] = [a1, b1, a2, b2, x, c]
        # i, j mode indexes,
        # all the quadratic fit parameters are used in oi_polynomial_expansion

        self.overlap_integrals = overlap_integrals
        self.torch_oi = OICoefficients(self.n_modes, overlap_integrals) if overlap_integrals is not None else None
        self.group_delay = GroupDelay(self.n_modes, group_delay) if group_delay is not None else None
        self.mode_names = mode_names

    @classmethod
    def from_config(cls, config: MMFiberConfig, **kwargs) -> "MMFiber":
        return cls(
            losses=config.losses,
            raman_coefficient=config.raman_coefficient,
            effective_area=config.effective_area,
            gamma=config.gamma,
            length=config.fiber_length,
            n_modes=config.n_modes,
            **kwargs,
        )

    @classmethod
    def from_toml(cls, filepath: Path | str, **kwargs) -> "MMFiber":
        return cls.from_config(MMFiberConfig.from_toml(filepath), **kwargs)
    
    def evaluate_oi(self, i, j, wavelength_i, wavelength_j):
        # original data were in um
        return oi_law(wavelength_i, wavelength_j, self.overlap_integrals[:, i, j])
      
def get_oi_matrix(self, modes, wavelengths):
  M = len(modes)
  W = len(wavelengths)
  mat = np.zeros((M*W, M*W))
  mat[:, :]
  for n in range(M):
    for m in range(M):
      for wn in range(W):
        for wm in range(W):
          mat[n+(wn*M), m+(wm*M)] = self.evaluate_oi(n, m, wavelengths[wn], wavelengths[wm])
  return mat


def fiber_summary(fiber: Fiber) -> str:
    """Return a human-friendly summary string for a Fiber/SMFiber/MMFiber."""
    def _format_profile_samples(wavelengths, values) -> str:
        wl = np.asarray(wavelengths, dtype=float)
        vals = np.asarray(values, dtype=float)
        if wl.size == 0 or vals.size == 0:
            return "[]"
        n = min(wl.size, vals.size)
        if n <= 4:
            idx = list(range(n))
        else:
            idx = [0, 1, n - 2, n - 1]
        pairs = ", ".join(f"({wl[i]:.3e},{vals[i]:.3e})" for i in idx)
        return f"[{pairs}]"

    lines = [
        f"Fiber type   : {fiber.fiber_type}",
        f"Length       : {fiber.length:.3e} m",
        f"Effective A  : {getattr(fiber, 'effective_area', float('nan')):.3e} m^2",
        f"Gamma        : {getattr(fiber, 'gamma', float('nan')):.3e} 1/W/m",
        f"Raman coeff. : {getattr(fiber, 'raman_coefficient', float('nan')):.3e}",
        f"Raman eff.   : {getattr(fiber, 'raman_efficiency', float('nan')):.3e}",
    ]
    # Loss coefficients (if present) as short tuple
    losses = getattr(fiber, "losses", None)
    if losses is not None and not isinstance(fiber, SMFiber):
        try:
            lines.append(f"Loss coeffs  : {tuple(losses)}")
        except Exception:
            lines.append("Loss coeffs  : <unavailable>")

    if isinstance(fiber, SMFiberNarrowband):
        lines.append(f"Beta2        : {fiber.beta2:.3e} s^2/m")
        if getattr(fiber, "_effective_area_profile", None) is not None:
            wl, vals = fiber._effective_area_profile
            lines.append(f"Aeff profile : {len(wl)} points [{wl[0]:.3e},{wl[-1]:.3e}] m")
        if getattr(fiber, "_attenuation_profile", None) is not None:
            wl, vals = fiber._attenuation_profile
            samples = _format_profile_samples(wl, vals)
            lines.append(f"Atten profile: {len(wl)} points [{wl[0]:.3e},{wl[-1]:.3e}] m {samples}")
    elif isinstance(fiber, SMFiber):
        # Profiles carry beta/beta1/beta2; no single-point betas shown here
        if getattr(fiber, "_effective_area_profile", None) is not None:
            wl, vals = fiber._effective_area_profile
            lines.append(f"Aeff profile : {len(wl)} points [{wl[0]:.3e},{wl[-1]:.3e}] m")
        if getattr(fiber, "_attenuation_profile", None) is not None:
            wl, vals = fiber._attenuation_profile
            samples = _format_profile_samples(wl, vals)
            lines.append(f"Atten profile: {len(wl)} points [{wl[0]:.3e},{wl[-1]:.3e}] m {samples}")
        if getattr(fiber, "_beta_profile", None) is not None:
            wl, vals = fiber._beta_profile
            lines.append(f"Beta profile : {len(wl)} points [{wl[0]:.3e},{wl[-1]:.3e}] m")
        if getattr(fiber, "_beta1_profile", None) is not None:
            wl, vals = fiber._beta1_profile
            samples = _format_profile_samples(wl, vals)
            lines.append(f"Beta1 profile: {len(wl)} points [{wl[0]:.3e},{wl[-1]:.3e}] m {samples}")
        if getattr(fiber, "_beta2_profile", None) is not None:
            wl, vals = fiber._beta2_profile
            samples = _format_profile_samples(wl, vals)
            lines.append(f"Beta2 profile: {len(wl)} points [{wl[0]:.3e},{wl[-1]:.3e}] m {samples}")
        if getattr(fiber, "_freq_profile", None) is not None:
            fp = fiber._freq_profile
            lines.append(f"Freq grid    : {len(fp)} points [{fp.min():.3e},{fp.max():.3e}] Hz")

    if isinstance(fiber, MMFiber):
        lines.extend([
            f"Modes        : {fiber.n_modes}",
            f"Mode names   : {fiber.mode_names if fiber.mode_names is not None else 'N/A'}",
            f"Group delay  : {'loaded' if fiber.group_delay is not None else 'None'}",
            f"Overlap ints : {'loaded' if fiber.overlap_integrals is not None else 'None'}",
        ])

    return "\n".join(lines)


def log_fiber(fiber: Fiber, logger=None, level: str = "trace"):
    """Log or print a fiber summary; defaults to loguru's trace level."""
    summary = fiber_summary(fiber)
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


if __name__ == "__main__":
    # Simple smoke test: load nested-fiber TOML and print a short summary
    cfg_path = Path("input/uwb_struct.toml")
    smf = SMFiber.from_toml(cfg_path)
    print(f"Loaded fiber from {cfg_path}:")
    print(fiber_summary(smf))
     
    def loss_profile(self, wavelengths):
        """Get the fiber losses (in dB/m) at the specified wavelengths (in
        meters)."""
        return polyval(self.losses, wavelengths)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()
