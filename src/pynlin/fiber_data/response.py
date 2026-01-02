#!/usr/bin/env python3
"""
response.py (rewritten)

Implements Raman gain *scaling* following Rottwitt et al. (2003), Eq. (39),
using only effective-area data Aeff(f) (Gaussian overlap approximation, Eq. (38)).

Key points:
- We compute a reference Raman gain spectrum g_ref(Δν; λp_ref) from an impulse-response model
  (same Hollenbeck–Cantrell modal sum and FFT logic as your original script).
- We scale it to a new pump wavelength λp_new using:
    g(Δν, λp_new) = g(Δν, λp_ref) * (λs_new / λs_ref) * (Aps_ref / Aps_new)
  where λs is the signal wavelength corresponding to the same Raman shift Δν:
    νs = νp - Δν,  λs = c/νs
  and the pump–signal overlap effective area is approximated by:
    Aps(Δν, λp) ≈ (Aeff(λp) + Aeff(λs)) / 2

- Aeff is loaded from input/fiber_data/smf28.csv (frequency + Aeff columns).
- Output:
  - Plots (optional) 
  - CSV with columns: dnu_Hz, g_ref_cm_per_W, g_scaled_cm_per_W, scale_factor

Notes on units:
- Your original "total_gain_spectrum * 1e2" plot label suggests cm/W.
  This rewrite keeps that convention for the reference and scaled spectra.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.constants import epsilon_0, c as C_SI

# Keep compatibility with your environment: original script imported c0 from pynlin.utils
try:
    from pynlin.utils import c0 as C0  # type: ignore
except Exception:
    C0 = C_SI

# -------------------------------------------------------------------------
# Hollenbeck–Cantrell-style parameters (as in your original response.py)
# -------------------------------------------------------------------------
# positions are in cm^-1 in your file, then multiplied by 1e2 => 1/m
positions = (
    np.array(
        [
            56.25,
            100.00,
            231.25,
            362.50,
            463.00,
            497.00,
            611.50,
            691.67,
            793.67,
            835.50,
            930.00,
            1080.00,
            1215.00,
        ]
    )
    * 1e2
)

intensities = np.array(
    [1.00, 11.40, 36.67, 67.67, 74.00, 4.50, 6.80, 4.60, 4.20, 4.50, 2.70, 3.10, 3.00]
)

g_fwhm = (
    np.array(
        [
            52.10,
            110.42,
            175.00,
            162.50,
            135.33,
            24.50,
            41.50,
            155.00,
            59.50,
            64.30,
            150.00,
            91.00,
            160.00,
        ]
    )
    * 1e2
)

l_fwhm = (
    np.array(
        [
            17.37,
            38.81,
            58.33,
            54.17,
            45.11,
            8.17,
            13.83,
            51.67,
            19.83,
            21.43,
            50.00,
            30.33,
            53.33,
        ]
    )
    * 1e2
)

# Vibrational angular frequencies (rad/s)
omega_v = 2 * np.pi * C0 * positions

# Lorentzian / Gaussian parameters (rad/s)
gamma_L = np.pi * C0 * l_fwhm
Gamma_G = np.pi * C0 * g_fwhm

# Amplitudes
amplitudes = intensities * omega_v
num_components = len(positions)


# -------------------------------------------------------------------------
# Helpers: IO and interpolation for Aeff(f)
# -------------------------------------------------------------------------
def _pick_column(names: list[str], candidates: list[str]) -> str:
    low = {n.lower(): n for n in names}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    raise KeyError(f"Could not find any of columns {candidates} in CSV header {names}")


def load_Aeff_of_f(
    csv_path: Path,
    *,
    f_col_candidates: list[str] = ["frequency_Hz", "frequency", "f_Hz", "f"],
    A_col_candidates: list[str] = ["Aeff_um2", "Aeff", "A_eff_um2", "Aeff_m2", "A_eff_m2"],
) :
    """
    Load Aeff table and return callable Aeff(f_Hz) -> Aeff (m^2).

    Assumptions:
    - frequency in Hz
    - Aeff in um^2 OR m^2 (auto-detected by column name heuristic)
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    names = list(data.dtype.names or [])
    if not names:
        raise ValueError(f"No header detected in {csv_path}; expected a header row with column names.")

    f_col = _pick_column(names, f_col_candidates)
    A_col = _pick_column(names, A_col_candidates)

    f = np.asarray(data[f_col], dtype=float)
    A = np.asarray(data[A_col], dtype=float)

    idx = np.argsort(f)
    f, A = f[idx], A[idx]

    # Convert Aeff to m^2 if it is likely in um^2
    # Heuristic: common convention in your project uses *_um2.
    if "um2" in A_col.lower():
        A_m2 = A * 1e-12
    else:
        # If it's already m^2, keep it; if it's um^2 without being labeled, user should rename column.
        A_m2 = A

    f_min, f_max = float(f[0]), float(f[-1])

    def Aeff_of_f(freq_Hz: np.ndarray | float) -> np.ndarray:
        freq = np.asarray(freq_Hz, dtype=float)
        if np.any(freq < f_min) or np.any(freq > f_max):
            raise ValueError(
                f"Requested frequency outside Aeff table range: "
                f"[{f_min:.3e}, {f_max:.3e}] Hz; got [{freq.min():.3e}, {freq.max():.3e}] Hz"
            )
        return np.interp(freq, f, A_m2)

    return Aeff_of_f, (f_min, f_max), (f_col, A_col)


# -------------------------------------------------------------------------
# Raman impulse response and gain spectrum (kept compatible with original script)
# -------------------------------------------------------------------------
def impulse_response(fs: float, num_samples: int, normalize: bool = False):
    """
    Compute Raman impulse response h(t) sampled at sampling frequency fs with num_samples.

    This retains your original modal-sum structure:
      exp(-gamma t) * exp(-(Gamma^2 t^2)/4) * sin(omega t)
    """
    dt = 1.0 / fs
    t = np.arange(num_samples) * dt

    modes = (
        np.reshape(amplitudes / omega_v, (num_components, 1))
        * np.exp(-np.outer(gamma_L, t))
        * np.exp(-np.outer(Gamma_G**2, t**2 / 4.0))
        * np.sin(np.outer(omega_v, t))
    )
    response = np.sum(modes, axis=0)

    if normalize:
        mx = np.max(np.abs(response))
        if mx > 0:
            response = response / mx

    return response, t


def gain_spectrum(fs: float, num_samples: int):
    """
    Backward-compatible helper returning a one-sided Raman gain spectrum.
    Only used by legacy imports in Raman solvers.
    """
    response, _ = impulse_response(fs=fs, num_samples=num_samples, normalize=False)
    spectrum = np.fft.fft(response)
    # Keep the positive frequencies, mirror of usage in torch solvers
    return -np.imag(spectrum[: math.ceil((num_samples + 1) / 2)])


def reference_gain_spectrum_cm_per_W(
    *,
    pump_wavelength_m: float,
    fs: float,
    duration_s: float,
    fR: float = 0.245,
    n2_m2_per_W: float = 2.6e-20,
):
    """
    Compute a *reference* Raman gain spectrum vs Raman shift Δν >= 0.

    This mirrors your original "__main__" logic:
      gamma = n2 * (2π/λ) * (2πν/c)??  (your script used n2*lambda2nu(λ)*2π/c0)
      total_gain_spectrum = 2*gamma*Im( FFT(total_response)/fs )

    We keep the same structure to remain numerically compatible with what you were plotting.
    """
    # sampling
    dt = 1.0 / fs
    num_samples = int(np.ceil(duration_s / dt))
    if num_samples < 16:
        raise ValueError("duration/fs too small; increase duration or fs.")

    # Raman response
    response, t = impulse_response(fs=fs, num_samples=num_samples, normalize=False)

    # total response in original script is fR * response (already “Raman part”)
    total_response = fR * response

    # frequency axis (Raman shift) for one-sided plot: Δν = k * fs/N (>=0)
    f = np.arange(num_samples) * fs / num_samples  # Hz

    # spectrum: original script divided by fs
    H = np.fft.fft(total_response) / fs

    # original script:
    #   gamma = n2 * lambda2nu(wavelength) * 2*pi / c0
    # where lambda2nu(wavelength) = c/λ = ν_p
    nu_p = C0 / pump_wavelength_m
    gamma_eff = n2_m2_per_W * nu_p * 2.0 * np.pi / C0

    g = 2.0 * gamma_eff * np.imag(H)  # “gain spectrum” (SI-ish internal)
    g_cm_per_W = -g * 1e2            # keep your sign convention + cm/W plotting

    return f, g_cm_per_W


# -------------------------------------------------------------------------
# Rottwitt Eq.(39) scaling using Aeff table (Gaussian overlap approximation Eq.(38))
# -------------------------------------------------------------------------
def scale_eq39(
    *,
    dnu_hz: np.ndarray,
    g_ref: np.ndarray,
    lam_p_ref_m: float,
    lam_p_new_m: float,
    Aeff_of_f,
):
    """
    Implements Rottwitt et al. (2003) Eq. (39) with Eq. (38) approximation:
      g(Δν, λp) = g(Δν, λp_ref) * (λs/λs_ref) * (Aps_ref/Aps_new)
    where:
      νs = νp - Δν (Stokes),
      λ = c/ν,
      Aps ≈ (Aeff(λp)+Aeff(λs))/2.

    Returns:
      g_scaled, scale_factor
    """
    dnu_hz = np.asarray(dnu_hz, dtype=float)
    g_ref = np.asarray(g_ref, dtype=float)

    nu_p_ref = C0 / lam_p_ref_m
    nu_p_new = C0 / lam_p_new_m

    nu_s_ref = nu_p_ref - dnu_hz
    nu_s_new = nu_p_new - dnu_hz

    # Valid only for Stokes shifts where νs>0
    valid = (nu_s_ref > 0.0) & (nu_s_new > 0.0)

    # λs/λs_ref = (c/νs_new)/(c/νs_ref) = νs_ref/νs_new
    lam_ratio = np.ones_like(dnu_hz)
    lam_ratio[valid] = nu_s_ref[valid] / nu_s_new[valid]

    # Aps terms (Eq. 38)
    Aeff_p_ref = Aeff_of_f(nu_p_ref)
    Aeff_p_new = Aeff_of_f(nu_p_new)

    Aps_ref = np.full_like(dnu_hz, np.nan, dtype=float)
    Aps_new = np.full_like(dnu_hz, np.nan, dtype=float)

    Aps_ref[valid] = 0.5 * (Aeff_p_ref + Aeff_of_f(nu_s_ref[valid]))
    Aps_new[valid] = 0.5 * (Aeff_p_new + Aeff_of_f(nu_s_new[valid]))

    scale = np.full_like(dnu_hz, np.nan, dtype=float)
    scale[valid] = lam_ratio[valid] * (Aps_ref[valid] / Aps_new[valid])

    g_scaled = np.full_like(g_ref, np.nan, dtype=float)
    g_scaled[valid] = g_ref[valid] * scale[valid]

    return g_scaled, scale, valid


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aeff-csv", type=str, default="input/fiber_data/smf28.csv",
                    help="CSV file with frequency and Aeff columns (OESCLU band).")
    ap.add_argument("--pump", type=float, default=1550e-9,
                    help="Target pump wavelength [m] to scale to.")
    ap.add_argument("--pump-ref", type=float, default=1455e-9,
                    help="Reference pump wavelength [m] for g_ref(Δν).")
    ap.add_argument("--fs", type=float, default=1e15,
                    help="Sampling frequency [Hz] for time-domain impulse response.")
    ap.add_argument("--duration", type=float, default=1e-11,
                    help="Duration [s] for impulse response window.")
    ap.add_argument("--fmax-thz", type=float, default=40.0,
                    help="Max Raman shift [THz] to include in output/plots.")
    ap.add_argument("--out-csv", type=str, default="output/raman_gain_scaled_eq39.csv",
                    help="Output CSV path.")
    ap.add_argument("--no-plot", action="store_true", help="Disable plots.")
    args = ap.parse_args()

    aeff_csv = Path(args.aeff_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    Aeff_of_f, (f_min, f_max), (f_col, A_col) = load_Aeff_of_f(aeff_csv)

    lam_p_ref = float(args.pump_ref)
    lam_p_new = float(args.pump)
    nu_p_ref = C0 / lam_p_ref
    nu_p_new = C0 / lam_p_new

    # Reference gain spectrum on Δν axis (Hz)
    dnu_hz_full, g_ref_full = reference_gain_spectrum_cm_per_W(
        pump_wavelength_m=lam_p_ref,
        fs=float(args.fs),
        duration_s=float(args.duration),
        fR=0.245,
        n2_m2_per_W=2.6e-20,
    )

    # Restrict to Δν in [0, fmax]
    fmax = float(args.fmax_thz) * 1e12
    m = (dnu_hz_full >= 0.0) & (dnu_hz_full <= fmax)
    dnu_hz = dnu_hz_full[m]
    g_ref = g_ref_full[m]

    # Apply Eq.(39) scaling
    g_scaled, scale, valid = scale_eq39(
        dnu_hz=dnu_hz,
        g_ref=g_ref,
        lam_p_ref_m=lam_p_ref,
        lam_p_new_m=lam_p_new,
        Aeff_of_f=Aeff_of_f,
    )

    # Safety: ensure pump/signal frequencies are within Aeff table
    # (the interpolator already enforces bounds; we provide context)
    # νp must be inside table
    if not (f_min <= nu_p_ref <= f_max):
        raise ValueError(f"Reference pump frequency {nu_p_ref:.3e} Hz outside Aeff table [{f_min:.3e},{f_max:.3e}] Hz")
    if not (f_min <= nu_p_new <= f_max):
        raise ValueError(f"Target pump frequency {nu_p_new:.3e} Hz outside Aeff table [{f_min:.3e},{f_max:.3e}] Hz")

    # Write CSV
    header = ",".join([
        "dnu_Hz",
        "dnu_THz",
        "g_ref_cm_per_W",
        "g_scaled_cm_per_W",
        "scale_factor_eq39",
        "valid",
    ])
    data = np.column_stack([
        dnu_hz,
        dnu_hz * 1e-12,
        g_ref,
        g_scaled,
        scale,
        valid.astype(int),
    ])
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    print(f"Wrote: {out_csv}")
    print(f"Aeff table: {aeff_csv} (using columns {f_col}, {A_col})")
    print(f"Pump ref: {lam_p_ref*1e9:.3f} nm; Pump new: {lam_p_new*1e9:.3f} nm")
    print(f"Aeff table frequency span: [{f_min*1e-12:.2f}, {f_max*1e-12:.2f}] THz")

    if args.no_plot:
        return

    # Plots
    plt.figure(figsize=(7, 4))
    plt.plot(dnu_hz * 1e-12, g_ref, label=f"ref {lam_p_ref*1e9:.1f} nm")
    plt.plot(dnu_hz[valid] * 1e-12, g_scaled[valid], label=f"scaled {lam_p_new*1e9:.1f} nm (Eq.39)")
    plt.xlabel("Raman shift Delta nu [THz]")
    plt.ylabel("Raman gain spectrum [cm/W]")
    plt.xlim(0, args.fmax_thz)
    plt.minorticks_on()
    plt.grid(which="both", alpha=0.3)
    plt.legend()

    plt.figure(figsize=(7, 4))
    plt.plot(dnu_hz[valid] * 1e-12, scale[valid])
    plt.xlabel("Raman shift Delta nu [THz]")
    plt.ylabel("Scaling factor (Eq.39)")
    plt.xlim(0, args.fmax_thz)
    plt.minorticks_on()
    plt.grid(which="both", alpha=0.3)

    plt.savefig(out_csv.with_suffix(".pdf"), bbox_inches='tight', pad_inches=0.01)
    print(f"Plots saved as {out_csv.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
