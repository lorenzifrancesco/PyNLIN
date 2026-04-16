#!/usr/bin/env python3
"""Compute and plot the cycle-averaged Wigner function of a modulated pulse train.

The Wigner-Ville distribution is defined as

    W_x(t, f) = \int_{-\infty}^{\infty} x(t + \tau / 2) x^*(t - \tau / 2) e^{-j 2\pi f \tau} d\tau.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.log_init import init_logging
from loguru import logger as lg
from pynlin.constellations import QAM
from pynlin.pulses import pulse_from_config
from pynlin.system import System
from pynlin.utils import BaseModel, ConfigDict, _toml_load


class WignerSystemConfig(BaseModel):
    config: str = "input/uwb_struct.toml"
    out: str = "media/wigner/wigner_cycle_average.pdf"
    pulse_out: str = "media/wigner/pulse_shape.pdf"
    qam_order: int = 16
    train_symbols: int = 512
    pulse_shape: str | int | None = None
    rolloff: Optional[float] = None
    pulse_symbols: Optional[int] = None
    samples_per_symbol: Optional[int] = None
    baud_rate: Optional[float] = 10e9
    pulse_width_ratio: float = 1.0
    seed: int = 1
    max_lag_symbols: int = 8
    phase_stride: int = 1
    symbol_stride: int = 1
    freq_bins: int = 512
    db: bool = False

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


def load_wigner_config(path: Optional[str | Path] = None) -> WignerSystemConfig:
    cfg_path = Path(path) if path is not None else Path("input/wigner_system.toml")
    if not cfg_path.exists():
        lg.warning("Wigner config file not found at {}; using defaults.", cfg_path)
        return WignerSystemConfig()
    data = _toml_load(cfg_path)
    if isinstance(data, dict):
        data = data.get("wigner_system", data)
    cfg = WignerSystemConfig(**(data or {}))
    lg.info("Loaded Wigner config from {}.", cfg_path)
    return cfg


def _parse_pulse_shape(value: str | int | None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    key = str(value).strip().lower()
    mapping = {
        "0": 0,
        "gaussian": 0,
        "gau": 0,
        "1": 1,
        "nyquist": 1,
        "nyq": 1,
        "2": 2,
        "raised_cosine": 2,
        "raised-cosine": 2,
        "rc": 2,
        "3": 3,
        "root_raised_cosine": 3,
        "root-raised-cosine": 3,
        "rrc": 3,
    }
    if key not in mapping:
        raise ValueError(f"Unknown pulse_shape override: {value}")
    return mapping[key]


def _build_pulse(system: System, args: WignerSystemConfig | argparse.Namespace):
    if system.pulse_config is None:
        if args.pulse_shape is not None or args.rolloff is not None:
            raise ValueError("pulse_shape/rolloff overrides require a system TOML with a [pulse] section.")
        pulse = system.pulse
        if args.baud_rate is not None:
            pulse.baud_rate = float(args.baud_rate)
        if args.samples_per_symbol is not None:
            pulse.samples_per_symbol = int(args.samples_per_symbol)
        if args.pulse_symbols is not None:
            pulse.num_symbols = int(args.pulse_symbols)
        return pulse

    pulse_cfg = system.pulse_config
    pulse_shape = _parse_pulse_shape(getattr(args, "pulse_shape", None))
    if pulse_shape is not None:
        pulse_cfg = replace(pulse_cfg, pulse_shape=pulse_shape)
    if getattr(args, "rolloff", None) is not None:
        pulse_cfg = replace(pulse_cfg, rolloff=float(args.rolloff))

    overrides = {}
    if args.baud_rate is not None:
        overrides["baud_rate"] = float(args.baud_rate)
    if args.samples_per_symbol is not None:
        overrides["samples_per_symbol"] = int(args.samples_per_symbol)
    if args.pulse_symbols is not None:
        overrides["num_symbols"] = int(args.pulse_symbols)
    return pulse_from_config(pulse_cfg, **overrides)


def _generate_symbols(constellation: np.ndarray, n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, constellation.size, size=n_symbols)
    return constellation[idx]


def _pulse_train(
    symbols: np.ndarray,
    pulse_samples: np.ndarray,
    samples_per_symbol: int,
) -> np.ndarray:
    upsampled = np.zeros(len(symbols) * samples_per_symbol, dtype=complex)
    upsampled[::samples_per_symbol] = symbols
    train = scipy.signal.convolve(upsampled, pulse_samples, mode="full")
    trim = len(pulse_samples) // 2
    if train.size > 2 * trim:
        train = train[trim:-trim]
    return train


def _scale_pulse_width(
    pulse_samples: np.ndarray,
    pulse_time: np.ndarray,
    width_ratio: float,
) -> np.ndarray:
    if width_ratio <= 0.0:
        raise ValueError("pulse_width_ratio must be positive.")
    if np.isclose(width_ratio, 1.0):
        return np.asarray(pulse_samples, dtype=complex)

    target_time = np.asarray(pulse_time, dtype=float)
    source_time = target_time / float(width_ratio)
    real = np.interp(source_time, target_time, np.real(pulse_samples), left=0.0, right=0.0)
    imag = np.interp(source_time, target_time, np.imag(pulse_samples), left=0.0, right=0.0)
    scaled = (real + 1j * imag) / np.sqrt(width_ratio)

    energy = np.trapezoid(np.abs(scaled) ** 2, target_time)
    if energy > 0.0:
        scaled = scaled / np.sqrt(energy)
    return scaled


def _plot_pulse_shape(
    pulse_time: np.ndarray,
    pulse_samples: np.ndarray,
    baud_rate: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots()
    time_symbols = pulse_time * baud_rate
    ax.plot(time_symbols, np.real(pulse_samples), lw=0.9, label="Re[g(t)]")
    if np.max(np.abs(np.imag(pulse_samples))) > 1e-12:
        ax.plot(time_symbols, np.imag(pulse_samples), lw=0.9, ls="--", label="Im[g(t)]")
    ax.set_xlabel(r"Time / $T_s$")
    ax.set_ylabel("Amplitude")
    ax.set_title("Pulse shape used for the Wigner estimate")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    lg.info("Saved pulse plot to {}", out_path)


def _cycle_averaged_wigner(
    waveform: np.ndarray,
    fs: float,
    samples_per_symbol: int,
    max_lag_symbols: int,
    phase_stride: int,
    symbol_stride: int,
    freq_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples_per_symbol = int(samples_per_symbol)
    max_lag_symbols = max(1, int(max_lag_symbols))
    phase_stride = max(1, int(phase_stride))
    symbol_stride = max(1, int(symbol_stride))
    lag_samples = max_lag_symbols * samples_per_symbol
    nfft = max(int(freq_bins), 2 * lag_samples + 1)
    nfft = 1 << (nfft - 1).bit_length()

    phase_indices = np.arange(0, samples_per_symbol, phase_stride, dtype=int)
    if phase_indices.size == 0:
        raise ValueError("phase_stride is too large for the symbol period.")

    total_symbols = waveform.size // samples_per_symbol
    first_symbol = max_lag_symbols
    last_symbol = total_symbols - max_lag_symbols - 1
    if last_symbol < first_symbol:
        raise ValueError("Not enough symbols for the requested Wigner lag window.")

    dt = 1.0 / fs
    plane = np.zeros((phase_indices.size, nfft), dtype=float)
    counts = np.zeros(phase_indices.size, dtype=int)

    for p_idx, phase in enumerate(phase_indices):
        accum = np.zeros(nfft, dtype=float)
        used = 0
        for symbol_idx in range(first_symbol, last_symbol + 1, symbol_stride):
            center = symbol_idx * samples_per_symbol + phase
            if center - lag_samples < 0 or center + lag_samples >= waveform.size:
                continue
            forward = waveform[center : center + lag_samples + 1]
            backward = waveform[center : center - lag_samples - 1 : -1]
            if forward.size != lag_samples + 1 or backward.size != lag_samples + 1:
                continue
            lag_corr = forward * np.conj(backward)
            lag_corr = np.concatenate([np.conj(lag_corr[1:][::-1]), lag_corr])
            spec = np.fft.fftshift(np.fft.fft(lag_corr, n=nfft)) * dt
            accum += np.real(spec)
            used += 1
        if used == 0:
            raise ValueError(f"No valid Wigner centers for phase index {phase}.")
        plane[p_idx] = accum / used
        counts[p_idx] = used

    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=dt))
    phase_axis = phase_indices.astype(float) / float(samples_per_symbol)
    return phase_axis, freq, plane


def _plot_wigner(
    phase_axis: np.ndarray,
    freq_axis_hz: np.ndarray,
    plane: np.ndarray,
    out_path: Path,
    title: str,
    db: bool,
) -> None:
    freq_ghz = freq_axis_hz / 1e9
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    if db:
        data = 10.0 * np.log10(np.maximum(np.abs(plane), 1e-30))
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[phase_axis[0], phase_axis[-1], freq_ghz[0], freq_ghz[-1]],
            cmap="magma",
        )
        cbar_label = "Wigner magnitude [dB/arb.]"
    else:
        finite = np.isfinite(plane)
        if np.any(finite):
            vmax = float(np.percentile(np.abs(plane[finite]), 99.0))
        else:
            vmax = 1.0
        if vmax <= 0.0:
            vmax = 1.0
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        im = ax.imshow(
            plane,
            origin="lower",
            aspect="auto",
            extent=[phase_axis[0], phase_axis[-1], freq_ghz[0], freq_ghz[-1]],
            cmap="RdBu_r",
            norm=norm,
        )
        cbar_label = "Wigner [arb.]"
    ax.set_xlabel(r"Local time within symbol $t/T_s$")
    ax.set_ylabel("Frequency [GHz]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    lg.info("Saved Wigner plot to {}", out_path)


def main() -> None:
    init_logging()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--wigner-config", type=str, default=None, help="Path to Wigner config TOML.")
    pre_args, _ = pre.parse_known_args()
    cfg = load_wigner_config(pre_args.wigner_config)

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--config", type=str, help="System config TOML (overrides Wigner config).")
    ap.add_argument("--out", type=str)
    ap.add_argument("--pulse-out", type=str)
    ap.add_argument("--qam-order", type=int)
    ap.add_argument("--train-symbols", type=int)
    ap.add_argument("--pulse-shape", type=str, help="Override pulse family: gaussian, nyquist, raised_cosine, rrc.")
    ap.add_argument("--rolloff", type=float, help="Override rolloff for RC/RRC pulses.")
    ap.add_argument("--pulse-symbols", type=int, help="Override pulse filter length in symbols.")
    ap.add_argument("--samples-per-symbol", type=int)
    ap.add_argument("--baud-rate", type=float)
    ap.add_argument("--pulse-width-ratio", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-lag-symbols", type=int)
    ap.add_argument("--phase-stride", type=int)
    ap.add_argument("--symbol-stride", type=int)
    ap.add_argument("--freq-bins", type=int)
    ap.add_argument("--db", action="store_true")
    ap.add_argument("--linear", dest="db", action="store_false")
    ap.set_defaults(**cfg.model_dump())
    args = ap.parse_args()

    system = System.from_toml(args.config)
    pulse = _build_pulse(system, args)
    lg.info(
        "Pulse: type={} baud_rate={:.3e} samples_per_symbol={} num_symbols={}",
        pulse.__class__.__name__,
        pulse.baud_rate,
        pulse.samples_per_symbol,
        pulse.num_symbols,
    )

    g_base, t_base = pulse.data()
    pulse_samples = _scale_pulse_width(g_base, t_base, float(args.pulse_width_ratio))
    _plot_pulse_shape(t_base, pulse_samples, pulse.baud_rate, Path(args.pulse_out))

    samples_per_symbol = int(pulse.samples_per_symbol)
    dt = 1.0 / (pulse.baud_rate * samples_per_symbol)
    fs = 1.0 / dt
    rng = np.random.default_rng(args.seed)

    qam = QAM(int(args.qam_order))
    symbols = qam.symbols()
    seq = _generate_symbols(symbols, int(args.train_symbols), rng)
    waveform = _pulse_train(seq, pulse_samples, samples_per_symbol)

    phase_axis, freq_axis, plane = _cycle_averaged_wigner(
        waveform,
        fs,
        samples_per_symbol,
        args.max_lag_symbols,
        args.phase_stride,
        args.symbol_stride,
        args.freq_bins,
    )
    _plot_wigner(
        phase_axis,
        freq_axis,
        plane,
        Path(args.out),
        f"Cycle-averaged Wigner function ({int(args.qam_order)}-QAM)",
        bool(args.db),
    )


if __name__ == "__main__":
    main()
