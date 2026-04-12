#!/usr/bin/env python3
"""Compute and plot PSD for arbitrary pulse shapes and uniform QAM constellations."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

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


class PsdSystemConfig(BaseModel):
    config: str = "input/uwb_struct.toml"
    out: str = "media/psd/psd_qam.pdf"
    pulse_out: str = "media/psd/pulse_shape.pdf"
    qam_orders: str | list[int] = "16"
    train_symbols: int = 10000
    pulse_shape: str | int | None = None
    rolloff: Optional[float] = None
    pulse_symbols: Optional[int] = None
    samples_per_symbol: Optional[int] = None
    baud_rate: Optional[float] = 10e9
    pulse_width_ratios: str | list[float] = "1.0"
    seed: int = 1
    method: str = "welch"
    window: str = "hann"
    nperseg: int = 4096
    db: bool = True
    bispectrum: bool = False
    bispectrum_max_ghz: float = 50.0
    bispectrum_bins: int = 64
    bispectrum_out: str = "media/psd/bispectrum.png"
    include_gaussian: bool = False
    fourth_order_proxy: bool = False
    fourth_out: str = "media/psd/fourth_order_proxy.pdf"
    factorization: bool = False
    factorization_out: str = "media/psd/psd_factorization.pdf"

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


def load_psd_config(path: Optional[str | Path] = None) -> PsdSystemConfig:
    cfg_path = Path(path) if path is not None else Path("input/psd_system.toml")
    if not cfg_path.exists():
        lg.warning("PSD config file not found at {}; using defaults.", cfg_path)
        return PsdSystemConfig()
    data = _toml_load(cfg_path)
    if isinstance(data, dict):
        data = data.get("psd_system", data)
    cfg = PsdSystemConfig(**(data or {}))
    lg.info("Loaded PSD config from {}.", cfg_path)
    return cfg


def _parse_int_list(value: str | list[int] | tuple[int, ...] | np.ndarray | int | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    parts = str(value).replace(",", " ").split()
    return [int(p) for p in parts if p.strip()]


def _parse_float_list(value: str | list[float] | tuple[float, ...] | np.ndarray | float | int | None) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    if isinstance(value, (float, int, np.floating, np.integer)):
        return [float(value)]
    parts = str(value).replace(",", " ").split()
    return [float(p) for p in parts if p.strip()]


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


def _build_pulse(system: System, args: PsdSystemConfig | argparse.Namespace):
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


def _generate_gaussian_symbols(n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    real = rng.normal(size=n_symbols)
    imag = rng.normal(size=n_symbols)
    return (real + 1j * imag) / np.sqrt(2.0)


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
        raise ValueError("pulse_width_ratios must be positive.")
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


def _delta_train(symbols: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    train = np.zeros(len(symbols) * samples_per_symbol, dtype=complex)
    train[::samples_per_symbol] = symbols
    return train


def _compute_psd(
    waveform: np.ndarray,
    fs: float,
    method: str,
    window: str,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "welch":
        freq, psd = scipy.signal.welch(
            waveform,
            fs=fs,
            nperseg=min(nperseg, waveform.size),
            window=window,
            return_onesided=False,
            detrend=False,
            scaling="density",
        )
    else:
        freq, psd = scipy.signal.periodogram(
            waveform,
            fs=fs,
            window=window,
            return_onesided=False,
            detrend=False,
            scaling="density",
        )
    freq = np.fft.fftshift(freq)
    psd = np.fft.fftshift(psd)
    return freq, psd


def _periodogram_fft(
    waveform: np.ndarray,
    fs: float,
    nfft: Optional[int] = None,
    scale_length: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(waveform)
    if nfft is None:
        nfft = arr.size
    if scale_length is None:
        scale_length = nfft
    spec = np.fft.fft(arr, n=nfft)
    freq = np.fft.fftfreq(nfft, d=1.0 / fs)
    psd = (np.abs(spec) ** 2) / (fs * scale_length)
    return np.fft.fftshift(freq), np.fft.fftshift(psd)


def _pulse_transfer(
    pulse_samples: np.ndarray,
    fs: float,
    nfft: int,
) -> tuple[np.ndarray, np.ndarray]:
    freq = np.fft.fftfreq(nfft, d=1.0 / fs)
    transfer = np.fft.fft(pulse_samples, n=nfft)
    return np.fft.fftshift(freq), np.fft.fftshift(np.abs(transfer) ** 2)


def _plot_pulse_shape(
    pulse_time: np.ndarray,
    pulse_map: list[tuple[float, np.ndarray]],
    baud_rate: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots()
    time_symbols = pulse_time * baud_rate
    for width_ratio, pulse_samples in pulse_map:
        ax.plot(time_symbols, np.real(pulse_samples), lw=0.9, label=fr"Re[g(t)], $T_p/T_s={width_ratio:g}$")
        if np.max(np.abs(np.imag(pulse_samples))) > 1e-12:
            ax.plot(time_symbols, np.imag(pulse_samples), lw=0.9, ls="--", label=fr"Im[g(t)], $T_p/T_s={width_ratio:g}$")
    ax.set_xlabel(r"Time / $T_s$")
    ax.set_ylabel("Amplitude")
    ax.set_title("Pulse shapes at fixed symbol period")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    lg.info("Saved pulse plot to {}", out_path)


def _plot_factorization(
    symbols: np.ndarray,
    pulse_map: list[tuple[float, np.ndarray]],
    samples_per_symbol: int,
    fs: float,
    out_path: Path,
    tag: str,
) -> None:
    impulses = _delta_train(symbols, samples_per_symbol)
    nfft = impulses.size + max(len(pulse_samples) for _, pulse_samples in pulse_map) - 1

    freq, psd_delta = _periodogram_fft(impulses, fs, nfft=nfft, scale_length=nfft)
    freq_ghz = freq / 1e9
    eps = 1e-30

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.0), sharex=True)

    axes[0].plot(freq_ghz, 10.0 * np.log10(np.maximum(psd_delta, eps)), lw=0.9)
    axes[0].set_ylabel(r"$S_\delta(f)$ [dB]")
    axes[0].set_title(f"Delta-train spectrum ({tag})")
    axes[0].grid(True, which="both", ls=":", lw=0.5, alpha=0.4)

    axes[1].set_ylabel(r"$|G(f)|^2$ [dB]")
    axes[1].set_title("Pulse transfer magnitude")
    axes[1].grid(True, which="both", ls=":", lw=0.5, alpha=0.4)

    axes[2].set_xlabel("Frequency [GHz]")
    axes[2].set_ylabel("PSD [dB/Hz]")
    axes[2].set_title("Output PSD and factorized prediction")
    axes[2].grid(True, which="both", ls=":", lw=0.5, alpha=0.4)

    for width_ratio, pulse_samples in pulse_map:
        waveform = scipy.signal.convolve(impulses, pulse_samples, mode="full")
        _, psd_total = _periodogram_fft(waveform, fs, nfft=nfft, scale_length=nfft)
        _, transfer = _pulse_transfer(pulse_samples, fs, nfft=nfft)
        predicted = psd_delta * transfer

        rel_err = np.zeros_like(psd_total)
        mask = predicted > np.max(predicted) * 1e-12
        rel_err[mask] = np.abs(psd_total[mask] - predicted[mask]) / predicted[mask]

        axes[1].plot(freq_ghz, 10.0 * np.log10(np.maximum(transfer, eps)), lw=0.9, label=fr"$T_p/T_s={width_ratio:g}$")
        axes[2].plot(
            freq_ghz,
            10.0 * np.log10(np.maximum(psd_total, eps)),
            lw=0.9,
            label=fr"actual, $T_p/T_s={width_ratio:g}$",
        )
        axes[2].plot(
            freq_ghz,
            10.0 * np.log10(np.maximum(predicted, eps)),
            lw=0.9,
            ls="--",
            label=fr"product, $T_p/T_s={width_ratio:g}$",
        )
        lg.info(
            "Factorization check {} width_ratio={}: max relative error {:.3e}.",
            tag,
            width_ratio,
            float(np.max(rel_err)),
        )

    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    lg.info("Saved PSD factorization plot to {}.", out_path)


def _fourth_order_proxy(
    waveform: np.ndarray,
    fs: float,
    method: str,
    window: str,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray]:
    intensity = np.abs(waveform) ** 2
    intensity = intensity - np.mean(intensity)
    return _compute_psd(intensity, fs, method, window, nperseg)


def _estimate_bispectrum(
    waveform: np.ndarray,
    fs: float,
    nperseg: int,
    window: str,
    max_freq_hz: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = min(int(nperseg), waveform.size)
    if nperseg < 8:
        raise ValueError("nperseg too small for bispectrum.")
    win = scipy.signal.get_window(window, nperseg)
    hop = nperseg
    n_segments = waveform.size // hop
    if n_segments < 1:
        raise ValueError("Not enough data for bispectrum.")
    fft_len = nperseg
    freqs = np.fft.fftfreq(fft_len, d=1.0 / fs)
    max_freq_hz = min(max_freq_hz, np.max(freqs))
    valid = np.where((freqs >= 0.0) & (freqs <= max_freq_hz))[0]
    if valid.size < 2:
        raise ValueError("max_freq_hz too low for bispectrum grid.")
    if n_bins > valid.size:
        n_bins = valid.size
    grid_idx = np.linspace(0, valid.size - 1, n_bins, dtype=int)
    bins = valid[grid_idx]
    f_axis = freqs[bins]
    max_pos_idx = int(np.max(valid))
    B = np.zeros((n_bins, n_bins), dtype=complex)
    denom12 = np.zeros((n_bins, n_bins), dtype=float)
    denom3 = np.zeros((n_bins, n_bins), dtype=float)

    for seg in range(n_segments):
        start = seg * hop
        segment = waveform[start : start + nperseg]
        if segment.size < nperseg:
            break
        segment = segment * win
        X = np.fft.fft(segment, n=fft_len)
        for i, k1 in enumerate(bins):
            X1 = X[k1]
            for j, k2 in enumerate(bins):
                k3 = k1 + k2
                if k3 >= X.size or k3 > max_pos_idx:
                    continue
                X2 = X[k2]
                X3 = X[k3]
                B[i, j] += X1 * X2 * np.conj(X3)
                denom12[i, j] += np.abs(X1 * X2) ** 2
                denom3[i, j] += np.abs(X3) ** 2

    if n_segments > 0:
        B = B / n_segments
        denom12 = denom12 / n_segments
        denom3 = denom3 / n_segments
    bicoherence = np.zeros_like(denom12)
    mask = (denom12 > 0.0) & (denom3 > 0.0)
    bicoherence[mask] = (np.abs(B[mask]) ** 2) / (denom12[mask] * denom3[mask])
    return f_axis, B, bicoherence


def main() -> None:
    init_logging()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--psd-config", type=str, default=None, help="Path to PSD config TOML.")
    pre_args, _ = pre.parse_known_args()
    cfg = load_psd_config(pre_args.psd_config)

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--config", type=str, help="System config TOML (overrides psd config).")
    ap.add_argument("--out", type=str)
    ap.add_argument("--pulse-out", type=str)
    ap.add_argument("--qam-orders", type=str)
    ap.add_argument("--train-symbols", type=int)
    ap.add_argument("--pulse-shape", type=str, help="Override pulse family: gaussian, nyquist, raised_cosine, rrc.")
    ap.add_argument("--rolloff", type=float, help="Override rolloff for RC/RRC pulses.")
    ap.add_argument("--pulse-symbols", type=int, help="Override pulse filter length in symbols.")
    ap.add_argument("--samples-per-symbol", type=int)
    ap.add_argument("--baud-rate", type=float)
    ap.add_argument("--pulse-width-ratios", type=str, help="Pulse width ratios relative to the symbol period.")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--method", choices=("welch", "periodogram"))
    ap.add_argument("--window", type=str)
    ap.add_argument("--nperseg", type=int)
    ap.add_argument("--db", action="store_true")
    ap.add_argument("--linear", dest="db", action="store_false")
    ap.add_argument("--bispectrum", action="store_true")
    ap.add_argument("--bispectrum-max-ghz", type=float)
    ap.add_argument("--bispectrum-bins", type=int)
    ap.add_argument("--bispectrum-out", type=str)
    ap.add_argument("--include-gaussian", action="store_true")
    ap.add_argument("--fourth-order-proxy", action="store_true")
    ap.add_argument("--fourth-out", type=str)
    ap.add_argument("--factorization", action="store_true")
    ap.add_argument("--factorization-out", type=str)
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
    pulse_width_ratios = _parse_float_list(args.pulse_width_ratios)
    if not pulse_width_ratios:
        pulse_width_ratios = [1.0]
    pulse_map = [
        (width_ratio, _scale_pulse_width(g_base, t_base, width_ratio))
        for width_ratio in pulse_width_ratios
    ]
    _plot_pulse_shape(t_base, pulse_map, pulse.baud_rate, Path(args.pulse_out))
    samples_per_symbol = int(pulse.samples_per_symbol)
    dt = 1.0 / (pulse.baud_rate * samples_per_symbol)
    fs = 1.0 / dt
    rng = np.random.default_rng(args.seed)
    qam_orders = _parse_int_list(args.qam_orders)
    if not qam_orders:
        raise ValueError("qam-orders must contain at least one integer value.")

    fig, ax = plt.subplots()
    psd_xlim_ghz = None
    for m in qam_orders:
        qam = QAM(m)
        symbols = qam.symbols()
        seq = _generate_symbols(symbols, args.train_symbols, rng)
        for width_ratio, pulse_samples in pulse_map:
            waveform = _pulse_train(seq, pulse_samples, samples_per_symbol)
            freq, psd = _compute_psd(waveform, fs, args.method, args.window, args.nperseg)
            freq_ghz = freq / 1e9
            label = f"{m}-QAM, Tp/Ts={width_ratio:g}"
            if args.db:
                psd_plot = 10.0 * np.log10(np.maximum(psd, 1e-30))
                ax.plot(freq_ghz, psd_plot, label=label, lw=0.8)
            else:
                ax.plot(freq_ghz, psd, label=label, lw=0.8)
            if psd_xlim_ghz is None:
                mask = psd > np.max(psd) * 1e-6
                if np.any(mask):
                    min_f = float(np.min(freq_ghz[mask]))
                    max_f = float(np.max(freq_ghz[mask]))
                    span = max_f - min_f
                    pad = 0.1 * span if span > 0 else 0.0
                    psd_xlim_ghz = (min_f - pad, max_f + pad)

    if args.include_gaussian:
        seq = _generate_gaussian_symbols(args.train_symbols, rng)
        for width_ratio, pulse_samples in pulse_map:
            waveform = _pulse_train(seq, pulse_samples, samples_per_symbol)
            freq, psd = _compute_psd(waveform, fs, args.method, args.window, args.nperseg)
            freq_ghz = freq / 1e9
            label = f"Gaussian, Tp/Ts={width_ratio:g}"
            if args.db:
                psd_plot = 10.0 * np.log10(np.maximum(psd, 1e-30))
                ax.plot(freq_ghz, psd_plot, label=label, lw=0.8)
            else:
                ax.plot(freq_ghz, psd, label=label, lw=0.8)

    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("PSD [dB/Hz]" if args.db else "PSD [1/Hz]")
    ax.set_title("Pulse-shaped QAM PSD")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    if psd_xlim_ghz is not None:
        ax.set_xlim(psd_xlim_ghz)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    lg.info("Saved PSD plot to {}", out_path)

    if args.factorization:
        m_sel = qam_orders[0]
        qam = QAM(m_sel)
        symbols = qam.symbols()
        seq = _generate_symbols(symbols, args.train_symbols, rng)
        _plot_factorization(
            seq,
            pulse_map,
            samples_per_symbol,
            fs,
            Path(args.factorization_out),
            f"{m_sel}-QAM",
        )

    if args.fourth_order_proxy:
        fig4, ax4 = plt.subplots()
        for m in qam_orders:
            qam = QAM(m)
            symbols = qam.symbols()
            seq = _generate_symbols(symbols, args.train_symbols, rng)
            for width_ratio, pulse_samples in pulse_map:
                waveform = _pulse_train(seq, pulse_samples, samples_per_symbol)
                freq4, psd4 = _fourth_order_proxy(waveform, fs, args.method, args.window, args.nperseg)
                freq4_ghz = freq4 / 1e9
                label = f"{m}-QAM, Tp/Ts={width_ratio:g}"
                if args.db:
                    psd4_plot = 10.0 * np.log10(np.maximum(psd4, 1e-30))
                    ax4.plot(freq4_ghz, psd4_plot, label=label, lw=0.8)
                else:
                    ax4.plot(freq4_ghz, psd4, label=label, lw=0.8)
        if args.include_gaussian:
            seq = _generate_gaussian_symbols(args.train_symbols, rng)
            for width_ratio, pulse_samples in pulse_map:
                waveform = _pulse_train(seq, pulse_samples, samples_per_symbol)
                freq4, psd4 = _fourth_order_proxy(waveform, fs, args.method, args.window, args.nperseg)
                freq4_ghz = freq4 / 1e9
                label = f"Gaussian, Tp/Ts={width_ratio:g}"
                if args.db:
                    psd4_plot = 10.0 * np.log10(np.maximum(psd4, 1e-30))
                    ax4.plot(freq4_ghz, psd4_plot, label=label, lw=0.8)
                else:
                    ax4.plot(freq4_ghz, psd4, label=label, lw=0.8)
        ax4.set_xlabel("Frequency [GHz]")
        ax4.set_ylabel("PSD of $|x|^2$ [dB/Hz]" if args.db else "PSD of $|x|^2$ [1/Hz]")
        ax4.set_title(r"Fourth-order proxy PSD ($|x|^2$ fluctuations)")
        ax4.legend(fontsize=8)
        ax4.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
        if psd_xlim_ghz is not None:
            ax4.set_xlim(psd_xlim_ghz)
        fig4.tight_layout()
        out4 = Path(args.fourth_out)
        out4.parent.mkdir(parents=True, exist_ok=True)
        fig4.savefig(out4, dpi=200, bbox_inches="tight")
        lg.info("Saved fourth-order proxy PSD to {}", out4)

    if args.bispectrum:
        def _plot_bicoherence(tag: str, waveform: np.ndarray, out_path: Path) -> None:
            f_axis, _, bicoherence = _estimate_bispectrum(
                waveform,
                fs,
                args.nperseg,
                args.window,
                args.bispectrum_max_ghz * 1e9,
                args.bispectrum_bins,
            )
            f_ghz = f_axis / 1e9
            fig_bi, ax_bi = plt.subplots()
            im = ax_bi.imshow(
                bicoherence,
                origin="lower",
                extent=[f_ghz[0], f_ghz[-1], f_ghz[0], f_ghz[-1]],
                aspect="equal",
                cmap="magma",
            )
            ax_bi.set_xlabel(r"$f_1$ [GHz]")
            ax_bi.set_ylabel(r"$f_2$ [GHz]")
            ax_bi.set_title(f"Bicoherence ({tag})")
            fig_bi.colorbar(im, ax=ax_bi, label="Bicoherence (normalized)")
            fig_bi.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig_bi.savefig(out_path, dpi=200, bbox_inches="tight")
            lg.info("Saved bispectrum plot to {}", out_path)

        out_bi = Path(args.bispectrum_out)
        m_sel = qam_orders[0]
        lg.info("Computing bispectrum for {}-QAM.", m_sel)
        qam = QAM(m_sel)
        symbols = qam.symbols()
        seq = _generate_symbols(symbols, args.train_symbols, rng)
        waveform = _pulse_train(seq, pulse_map[0][1], samples_per_symbol)
        _plot_bicoherence(f"{m_sel}-QAM, Tp/Ts={pulse_map[0][0]:g}", waveform, out_bi)

        if args.include_gaussian:
            lg.info("Computing bispectrum for Gaussian modulation.")
            seq = _generate_gaussian_symbols(args.train_symbols, rng)
            waveform = _pulse_train(seq, pulse_map[0][1], samples_per_symbol)
            out_gauss = out_bi.with_name(out_bi.stem + "_gaussian" + out_bi.suffix)
            _plot_bicoherence(f"Gaussian, Tp/Ts={pulse_map[0][0]:g}", waveform, out_gauss)


if __name__ == "__main__":
    main()
