from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys
from typing import Mapping, Tuple

import numpy as np
import scipy.integrate

from pynlin.utils import PulseShape, _toml_load

# Helper to select values from multiple mappings (pulse section or root)
def _pick(key, *sections):
    for sec in sections:
        if isinstance(sec, Mapping) and key in sec:
            return sec[key]
    return None


@dataclass
class PulseConfig:
    baud_rate: float
    pulse_shape: int
    num_symbols: float | None = None
    samples_per_symbol: float | None = None
    rolloff: float | None = None

    @property
    def type(self) -> "PulseType":
        return PulseType.from_value(self.pulse_shape)

    @property
    def shape(self) -> PulseShape:
        return PulseShape(self.pulse_shape)

    @classmethod
    def from_mapping(cls, data: Mapping) -> "PulseConfig":
        root = data
        pulse = root.get("pulse", {}) if isinstance(root.get("pulse"), Mapping) else {}
        baud = _pick("baud_rate", root, pulse)
        shape = _pick("pulse_shape", root, pulse)
        if baud is None or shape is None:
            raise ValueError("PulseConfig requires baud_rate and pulse_shape in the TOML.")
        num_symbols = _pick("num_symbols", root, pulse)
        samples_per_symbol = _pick("samples_per_symbol", root, pulse)
        rolloff = _pick("rolloff", root, pulse)
        return cls(
            baud_rate=float(baud),
            pulse_shape=int(shape),
            num_symbols=float(num_symbols) if num_symbols is not None else None,
            samples_per_symbol=float(samples_per_symbol) if samples_per_symbol is not None else None,
            rolloff=float(rolloff) if rolloff is not None else None,
        )

    @classmethod
    def from_toml(cls, filepath: Path | str) -> "PulseConfig":
        return cls.from_mapping(_toml_load(Path(filepath)))


class PulseType(Enum):
    GAUSSIAN = "gaussian"
    NYQUIST = "nyquist"
    RAISED_COSINE = "raised_cosine"
    ROOT_RAISED_COSINE = "root_raised_cosine"

    @classmethod
    def from_value(cls, value) -> "PulseType":
        if isinstance(value, cls):
            return value
        if isinstance(value, PulseShape):
            return PULSE_SHAPE_TO_TYPE[value]
        if isinstance(value, int):
            if value in PULSE_INT_TO_TYPE:
                return PULSE_INT_TO_TYPE[value]
            try:
                return PULSE_SHAPE_TO_TYPE[PulseShape(value)]
            except Exception:
                pass
        if isinstance(value, str):
            key = value.lower()
            for member in cls:
                if member.value == key or member.name.lower() == key:
                    return member
        raise ValueError(f"Unknown pulse type: {value}")

# map legacy numeric/enum PulseShape to PulseType
PULSE_SHAPE_TO_TYPE = {
    PulseShape.GAU: PulseType.GAUSSIAN,
    PulseShape.NYQ: PulseType.NYQUIST,
    PulseShape.RAISED_COSINE: PulseType.RAISED_COSINE,
    PulseShape.ROOT_RAISED_COSINE: PulseType.ROOT_RAISED_COSINE,
}
PULSE_INT_TO_TYPE = {
    0: PulseType.GAUSSIAN,
    1: PulseType.NYQUIST,
    2: PulseType.RAISED_COSINE,
    3: PulseType.ROOT_RAISED_COSINE,
}

# from pynlin.fiber import Fiber
# from pynlin.wdm import WDM
# from pynlin.collisions import get_gvd

class Pulse(ABC):
    def __init__(
        self,
        baud_rate: float = 10e9,
        num_symbols: float = 1e3,
        samples_per_symbol: float = 2**5,
    ):
        self.baud_rate = baud_rate
        self.num_symbols = num_symbols
        self.samples_per_symbol = samples_per_symbol
        self.T0 = 1 / self.baud_rate

    @abstractmethod
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the pulse shape and the time axis."""
        pass


class RaisedCosinePulse(Pulse):
    def __init__(
        self,
        baud_rate: float = 10e9,
        num_symbols: float = 1e3,
        samples_per_symbol: float = 2**5,
        rolloff: float = 0.1,
    ):
        super().__init__(baud_rate, num_symbols, samples_per_symbol)
        self.rolloff = rolloff
        self._generate()

    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.g, self.t

    def _generate(self):
        dt = self.T0 / self.samples_per_symbol
        Ndt = self.samples_per_symbol * self.num_symbols
        t = np.arange(-Ndt / 2, Ndt / 2) * dt
        df = 1 / (max(t) - min(t))
        dw = 2 * np.pi * df
        f = np.arange(-Ndt / 2, Ndt / 2) * df

        rof = self.rolloff
        R = rof
        rate = 1 / self.T0
        freq = f
        wind1 = np.zeros_like(f)
        wind1[np.abs(freq) <= rate * (1 + R) / 2] = 1
        wind2 = np.zeros_like(f)
        wind2[np.abs(freq) >= rate * (1 - R) / 2] = 1
        wind = wind1 * wind2
        # wind1[np.abs(freq) >= R * (1 - R) / 2] = 1
        gf = (1 - wind) + wind * 0.5 * (
            1 + np.cos(np.pi / R / rate * (np.abs(freq) - rate * (1 - R) / 2))
        )
        gf[np.abs(freq) > rate * (1 + R) / 2] = 0

        gt = np.real(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(gf))))

        energy = scipy.integrate.trapezoid(np.abs(gt) ** 2, t)
        gt = gt / np.sqrt(energy)

        self.t = t
        self.g = gt

class RootRaisedCosinePulse(Pulse):
    def __init__(
        self,
        baud_rate: float = 10e9,
        num_symbols: float = 1e3,
        samples_per_symbol: float = 2**5,
        rolloff: float = 0.1,
    ):
        super().__init__(baud_rate, num_symbols, samples_per_symbol)
        self.rolloff = rolloff
        self._generate()

    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.g, self.t

    def _generate(self):
        dt = self.T0 / self.samples_per_symbol
        Ndt = self.samples_per_symbol * self.num_symbols
        t = np.arange(-Ndt / 2, Ndt / 2) * dt

        if self.rolloff <= 0:
            gt = np.sinc(t / self.T0) / np.sqrt(self.T0)
        else:
            df = 1 / (max(t) - min(t))
            f = np.arange(-Ndt / 2, Ndt / 2) * df

            R = self.rolloff
            rate = 1 / self.T0
            freq = f
            wind1 = np.zeros_like(f)
            wind1[np.abs(freq) <= rate * (1 + R) / 2] = 1
            wind2 = np.zeros_like(f)
            wind2[np.abs(freq) >= rate * (1 - R) / 2] = 1
            wind = wind1 * wind2
            gf = (1 - wind) + wind * 0.5 * (
                1 + np.cos(np.pi / R / rate * (np.abs(freq) - rate * (1 - R) / 2))
            )
            gf[np.abs(freq) > rate * (1 + R) / 2] = 0
            gf = np.sqrt(gf)

            gt = np.real(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(gf))))

        energy = scipy.integrate.trapezoid(np.abs(gt) ** 2, t)
        gt = gt / np.sqrt(energy)

        self.t = t
        self.g = gt

class NyquistPulse(Pulse):
    def __init__(
        self,
        baud_rate: float = 10e9,
        num_symbols: float = 1e3,
        samples_per_symbol: float = 2**5,
        rolloff: float = 0.1,
    ):
        super().__init__(baud_rate, num_symbols, samples_per_symbol)
        self.rolloff = rolloff
        self._generate()

    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.g, self.t

    def _generate(self):
        dt = self.T0 / self.samples_per_symbol
        Ndt = self.samples_per_symbol * self.num_symbols
        t = np.arange(-Ndt / 2, Ndt / 2) * dt

        gt = np.sinc(t/self.T0)/np.sqrt(self.T0)

        # Correct analytical normalization (finiteness of interval make it imprecise)
        energy = scipy.integrate.trapezoid(np.abs(gt) ** 2, t)
        gt = gt / np.sqrt(energy)

        self.t = t
        self.g = gt

class GaussianPulse(Pulse):
    def __init__(
        self,
        baud_rate: float = 10e9,
        num_symbols: float = 1e3,
        samples_per_symbol: float = 2**5,
    ):
        super().__init__(baud_rate, num_symbols, samples_per_symbol)
        self._generate()

    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.g, self.t

    def _generate(self):
        dt = self.T0 / self.samples_per_symbol
        Ndt = self.samples_per_symbol * self.num_symbols
        t = np.arange(-Ndt / 2, Ndt / 2) * dt

        # this underflows pretty much systematically. We suppress warnings
        with np.errstate(under='ignore'):
            gt = np.exp(-t**2/(2*(self.T0**2))) / np.sqrt((np.sqrt(np.pi) * self.T0))
        
            # Correct analytical normalization (finiteness of interval make it imprecise)
            energy = scipy.integrate.trapezoid(np.abs(gt) ** 2, t)
            gt = gt / np.sqrt(energy)
        self.t = t
        self.g = gt


# map PulseType to concrete pulse classes
PULSE_TYPE_TO_CLASS = {
    PulseType.GAUSSIAN: GaussianPulse,
    PulseType.NYQUIST: NyquistPulse,
    PulseType.RAISED_COSINE: RaisedCosinePulse,
    PulseType.ROOT_RAISED_COSINE: RootRaisedCosinePulse,
}


def pulse_from_config(cfg: PulseConfig, **overrides) -> Pulse:
    """Instantiate the correct Pulse subclass based on PulseConfig/PulseType."""
    ptype = cfg.type
    pulse_cls = PULSE_TYPE_TO_CLASS.get(ptype)
    if pulse_cls is None:
        raise ValueError(f"No pulse class registered for {ptype}")

    kwargs = {
        "baud_rate": cfg.baud_rate,
        "num_symbols": cfg.num_symbols if cfg.num_symbols is not None else 1e3,
        "samples_per_symbol": cfg.samples_per_symbol if cfg.samples_per_symbol is not None else 2**5,
    }
    for key, val in overrides.items():
        if val is not None:
            kwargs[key] = val

    if ptype in (PulseType.RAISED_COSINE, PulseType.ROOT_RAISED_COSINE) and "rolloff" not in kwargs:
        kwargs["rolloff"] = cfg.rolloff if cfg.rolloff is not None else 0.1

    return pulse_cls(**kwargs)


def pulse_from_toml(filepath: Path | str, **overrides) -> Pulse:
    """Load pulse section from TOML and instantiate the matching Pulse subclass."""
    cfg = PulseConfig.from_toml(filepath)
    return pulse_from_config(cfg, **overrides)
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # quick load test from TOML (structured or flat)
    toml_paths = sys.argv[1:] if len(sys.argv) > 1 else ["input/smf_struct.toml", "input/smf.toml"]
    for path in toml_paths:
        try:
            cfg = PulseConfig.from_toml(path)
            pulse_obj = pulse_from_config(cfg)
            print(f"[{path}] -> {pulse_obj.__class__.__name__}, baud={pulse_obj.baud_rate:.3e}, type={cfg.type.value}")
        except Exception as exc:
            print(f"[{path}] load failed: {exc}")

    # legacy plotting demo
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=5e2, samples_per_symbol=2**5)
    g, t = pulse.data()
    vari = 100
    mid = len(t)//2
    plt.plot(t[mid-vari:mid+vari], np.abs(g[mid-vari:mid+vari]), label="Gaussian")
    pulse = NyquistPulse(baud_rate=10e9, num_symbols=5e2, samples_per_symbol=2**5)
    g, t = pulse.data()
    plt.plot(t[mid-vari:mid+vari], np.abs(g[mid-vari:mid+vari]), label="Nyquist")
    pulse = RaisedCosinePulse(baud_rate=10e9, num_symbols=5e2, samples_per_symbol=2**5, rolloff=1)
    g, t = pulse.data()
    plt.plot(t[mid-vari:mid+vari], np.abs(g[mid-vari:mid+vari]), label="Raised Cosine")
    plt.legend()
    plt.savefig("media/debug/pulse_shapes.png", dpi=300)
    print("Pulse shapes plot saved to media/debug/pulse_shapes.png")
