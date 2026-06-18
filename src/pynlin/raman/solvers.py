from typing import Tuple
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize
try:
    import seaborn as sns
except Exception:  # plotting-only dependency
    sns = None
from numpy import polyval
from scipy.constants import Boltzmann as kB
from scipy.constants import Planck as h_planck
from scipy.constants import lambda2nu, nu2lambda

import pynlin.utils
from pynlin.fiber import Fiber, MMFiber, SMFiberNarrowband
from pynlin.pulses import Pulse
from pynlin.fiber_data.response import gain_spectrum
from pynlin.utils import (
    alpha_to_linear,
    dBm2watt,
    BaseModel,
    ConfigDict,
    oi_law,
    _toml_load,
    watt2dBm,
    wavelength_to_frequency,
)
from pynlin.wdm import IrregularWDM, PumpSpec
from pynlin.log_init import init_logging
from loguru import logger as lg

init_logging()


class TimeOut(Exception):
    """Raise a timeout exception to stop the shooting algorithm."""
    pass


class RamanAmplifierConfig(BaseModel):
    """Configuration for Raman amplifier pumps/targets."""

    pumps: list[PumpSpec] | None = None
    target_raman_gain: float = 0.0

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"

    @classmethod
    def from_mapping(cls, data):
        data = data or {}
        pumps_data = data.get("pumps") if isinstance(data, dict) else None
        pumps = None
        if isinstance(pumps_data, list):
            pumps = [PumpSpec(**p) if not isinstance(p, PumpSpec) else p for p in pumps_data]
        target = data.get("target") if isinstance(data, dict) else {}
        raman_gain = target.get("raman_gain") if isinstance(target, dict) else data.get("raman_gain", 0.0)
        return cls(pumps=pumps, target_raman_gain=float(raman_gain))

    @classmethod
    def from_toml(cls, filepath):
        raw = _toml_load(filepath)
        amp = raw.get("amplification") if isinstance(raw, dict) else {}
        return cls.from_mapping(amp)


class SMRamanAmplifierConfig(RamanAmplifierConfig):
    """Single-mode Raman amplifier config."""

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


class MMRamanAmplifierConfig(RamanAmplifierConfig):
    """Multimode Raman amplifier config."""

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


_GAIN_SPECTRUM_CACHE: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
_GAIN_MATRIX_CACHE: dict[tuple[bytes, int, float], np.ndarray] = {}


class RamanAmplifier:
    @staticmethod
    def _log_first_nonfinite(name: str, arr: np.ndarray) -> bool:
        """Log the first NaN/inf entry; return True if any were found."""
        bad = ~np.isfinite(arr)
        if not np.any(bad):
            return False
        idx = tuple(np.argwhere(bad)[0])
        lg.error(f"{name} has non-finite value at {idx}: {arr[idx]}")
        return True

    def __init__(
        self,
        fiber: Fiber,
        response_bandwidth: float = 40e12,
        viz: bool = False
    ):
        self.fiber = fiber
        self.response_bandwidth = response_bandwidth

        super().__init__()

    @staticmethod
    def _gain_resolution(frequencies: np.ndarray, fallback: float = 50e9) -> float:
        """Resolution to sample Raman gain; derived from grid spacing if available."""
        freqs = np.sort(np.unique(np.asarray(frequencies, dtype=float)))
        diffs = np.diff(freqs)
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(np.min(diffs))
        return float(fallback)

    def _build_gain_spectrum(self, max_shift_hz: float, resolution_hz: float):
        """Precompute reference Raman gain spectrum up to max_shift_hz and cache it."""
        max_shift_hz = float(max_shift_hz)
        resolution_hz = max(float(resolution_hz), 1e9)
        for (cached_max, cached_res), (shifts, gains) in _GAIN_SPECTRUM_CACHE.items():
            if cached_max >= max_shift_hz and cached_res <= resolution_hz:
                return shifts, gains

        num_samples = int(np.ceil(2 * max_shift_hz / resolution_hz)) + 1
        num_samples = max(num_samples, 1024)
        fs = resolution_hz * num_samples
        g_ref = gain_spectrum(fs=fs, num_samples=num_samples)
        shifts = np.arange(len(g_ref)) * fs / num_samples
        peak = np.max(np.abs(g_ref))
        gains = g_ref / peak if peak else g_ref
        _GAIN_SPECTRUM_CACHE[(max_shift_hz, resolution_hz)] = (shifts, gains)
        lg.debug(
            f"Cached Raman gain spectrum up to {max_shift_hz*1e-12:.2f} THz "
            f"(Δf≈{resolution_hz*1e-9:.1f} GHz, samples={num_samples})"
        )
        return shifts, gains

    def _interpolate_gain(self, delta_freqs, resolution_hz: float):
        """Interpolate Raman gain for frequency shifts delta_freqs (array)."""
        abs_shifts = np.abs(delta_freqs)
        max_shift = np.max(abs_shifts)
        shifts, gains_ref = self._build_gain_spectrum(max_shift, resolution_hz)
        gains = np.interp(abs_shifts, shifts, gains_ref, left=0.0, right=0.0)
        gains[delta_freqs < 0] *= -1
        return gains

    def propagate(self):
        pass

    def solve(
        self,
        signal_power,
        signal_wavelength,
        pump_power,
        pump_wavelength,
        z,
        pump_direction=1,
        use_power_at_fiber_start=False,
        check_photon_count=False,
        reference_bandwidth=0.1,
        temperature=300,
        odeint_kwargs=None,
        ase=False,
        solver: str = "ivp",
        pump_power_floor: float = 0.0,
    ):

        # raman_coefficient = self.fiber.raman_coefficient
        # effective_core_area = self.fiber.effective_area

        num_signals = signal_power.shape[0]
        num_pumps = pump_power.shape[0]
        total_signals = num_pumps + num_signals
        num_ase = num_signals if ase else 0

        signal_direction = np.ones_like(signal_power)
        if num_pumps == 0:
            pump_direction = np.array([])
        elif np.isscalar(pump_direction):
            pump_direction = np.sign(pump_direction) * np.ones_like(pump_power)
        else:
            pump_direction = np.atleast_1d(pump_direction)

        # organization of gain matrix
        self.direction = np.concatenate((pump_direction, signal_direction))

        # check if we need a shooting algorithm
        shooting = np.any(self.direction < 0)

        if pump_power_floor and pump_power.size:
            pump_power = np.maximum(pump_power, pump_power_floor)

        self.pump_power = pump_power
        self.signal_power = signal_power
        self.pump_wavelengths = pump_wavelength
        self.signal_wavelengths = signal_wavelength

        wavelengths = np.concatenate((self.pump_wavelengths, self.signal_wavelengths))
        frequencies = lambda2nu(wavelengths)

        # input_power = np.concatenate((self.pump_power, self.signal_power))
        input_power = np.concatenate((self.pump_power, self.signal_power))
        if ase:
            input_power = np.concatenate((input_power, np.zeros(num_ase)))
        lg.debug(
            f"[SM amp] inputs: pump {pump_power.shape}, signal {signal_power.shape}, "
            f"power range [{input_power.min():.3e},{input_power.max():.3e}], z={len(z)}"
        )
        lg.info(f"[SM amp] initial power vector (W): {input_power}")

        losses_linear = self.get_linear_losses(wavelengths)

        gain_matrix = self.compute_gain_matrix(frequencies)
        lg.debug(f"[SM amp] gain matrix shape: {gain_matrix.shape}")
        lg.debug(
            f"[SM amp] gain stats: min={np.nanmin(gain_matrix):.3e}, max={np.nanmax(gain_matrix):.3e}"
        )

        # Compute the frequency shifts for each wave
        frequency_shifts = np.zeros((total_signals, total_signals))
        for i in range(total_signals):
            frequency_shifts[i, :] = frequencies - frequencies[i]

        if ase:
            h_planck = 6.626e-34
            kB = 1.380649e-23
            Hinv = np.exp(h_planck * np.abs(frequency_shifts) / (kB * temperature)) - 1
            np.fill_diagonal(Hinv, 1e56)  # random fill the diagonal
            eta_plus = 1 + 1 / Hinv
            np.fill_diagonal(eta_plus, 0)
            gain_matrix_ase = eta_plus * gain_matrix

        kw = dict(rtol=1e-3, atol=1e-9, mxstep=10000)
        if isinstance(odeint_kwargs, dict):
            kw.update(odeint_kwargs)
        # User can override max_step via odeint_kwargs; otherwise derive a conservative default.
        kw_max_step = kw.pop("max_step", None)
        default_max_step = (
            min(1e100, (z[-1] - z[0]) / 5000) if len(z) > 1 else np.inf
        )
        max_step = default_max_step if kw_max_step is None else float(kw_max_step)
        max_step = 0.1
        lg.warning(f"[SM amp] using max_step={max_step:.3e} m for solver")
        if ase:
            sol = scipy.integrate.odeint(
                RamanAmplifier.raman_ode_with_ase,
                input_power,
                z,
                args=(losses_linear, gain_matrix, gain_matrix_ase, np.hstack((self.direction, np.ones(
                    num_ase))), temperature, reference_bandwidth, num_pumps, num_signals, frequencies),
                **kw,
            )
            pump_solution = sol[:, :num_pumps]
            signal_solution = sol[:, num_pumps:num_pumps + num_signals]
            ase_solution = sol[:, -num_ase:] if num_ase else np.empty((len(z), 0))
        else:
            if solver == "log_ivp":
                # Experimental: integrate log-power to reduce dynamic range issues.
                log_floor = 1e-30
                P0 = np.maximum(input_power, log_floor)
                logP0 = np.log(P0)

                def _ode_log(z_val, logP):
                    P_lin = np.exp(logP)
                    dlogPdz = (-losses_linear + gain_matrix @ P_lin)
                    if not np.all(np.isfinite(dlogPdz)):
                        raise FloatingPointError("Non-finite dlogPdz in log_ivp")
                    return dlogPdz * self.direction

                lg.debug(
                    f"[SM amp] log_ivp: max_step={max_step:.3e}, "
                    f"rtol={kw.get('rtol')}, atol={kw.get('atol')}, "
                    f"logP0 range [{logP0.min():.3e},{logP0.max():.3e}]"
                )
                ivp = scipy.integrate.solve_ivp(
                    _ode_log,
                    (z[0], z[-1]),
                    logP0,
                    method="Radau",
                    t_eval=z,
                    rtol=kw.get("rtol", 1e-6),
                    atol=kw.get("atol", 1e-6),
                    max_step=max_step,
                )
                if not ivp.success:
                    lg.error(f"[SM amp] log_ivp solver failed: {ivp.message}")
                else:
                    lg.debug(f"[SM amp] log_ivp steps: {ivp.nfev} f evals, {ivp.t.size} outputs")
                sol_lin = np.exp(ivp.y.T)
                # Guard against underflow/overflow after exponentiation
                sol_lin = np.where(np.isfinite(sol_lin), sol_lin, np.nan)
                sol = sol_lin
            elif solver in ("ivp", "bdf", "BDF", "radau", "Radau"):
                # Use stiff solver for better robustness (Radau), integrating log-power.
                eval_count = {"n": 0}

                log_every = int(kw.pop("log_every", 1000))
                floor = max(pump_power_floor, 1e-30)
                logP0 = np.log(np.maximum(input_power, floor))

                def _ode(z_val, logP):
                    P = np.exp(logP)
                    eval_count["n"] += 1
                    gP = gain_matrix @ P
                    if log_every and eval_count["n"] % log_every == 0:
                        lg.debug(
                            f"[SM amp][Radau] eval {eval_count['n']} t={z_val:.3e}: "
                            f"max|P|={np.max(np.abs(P)):.3e}, max|gP|={np.max(np.abs(gP)):.3e}"
                        )
                    dlogPdz = (-losses_linear + gP)
                    return dlogPdz * self.direction

                lg.debug(
                    f"[SM amp] Radau (logP) solve: max_step={max_step:.3e}, "
                    f"rtol={kw.get('rtol')}, atol={kw.get('atol')}, "
                    f"logP0 range [{logP0.min():.3e},{logP0.max():.3e}]"
                )
                ivp = None
                max_step_try = max_step
                for attempt in range(2):
                    try:
                        ivp = scipy.integrate.solve_ivp(
                            _ode,
                            (z[0], z[-1]),
                            logP0,
                            method="Radau",
                            t_eval=z,
                            rtol=kw.get("rtol", 1e-6),
                            atol=kw.get("atol", 1e-9),
                            max_step=max_step_try,
                        )
                    except Exception as exc:
                        lg.warning(f"[SM amp] Radau attempt {attempt+1} failed (max_step={max_step_try:.3e}): {exc}")
                        ivp = None
                    if ivp is not None and ivp.success:
                        break
                    max_step_try *= 0.1
                if ivp is None or not ivp.success:
                    msg = ivp.message if ivp is not None else "no result"
                    lg.error(f"[SM amp] Radau solver failed after retries: {msg}")
                    sol = np.full((len(z), len(input_power)), np.nan)
                    sol_log = np.log(sol + np.finfo(float).eps)
                else:
                    lg.debug(f"[SM amp] Radau steps: {ivp.nfev} f evals, {ivp.t.size} outputs (max_step_used={max_step_try:.3e})")
                    sol_log = ivp.y.T
                    sol = np.exp(sol_log)
                    sol = np.where(np.isfinite(sol), sol, np.nan)
            else:
                sol = scipy.integrate.odeint(
                    RamanAmplifier.raman_ode,
                    input_power,
                    z,
                    args=(losses_linear, gain_matrix, np.hstack((self.direction,))),
                    **kw,
                )
                lg.debug(f"[SM amp] odeint solve done; sol shape {sol.shape}")
            pump_solution = sol[:, :num_pumps] if num_pumps else np.zeros((len(z), 0))
            signal_solution = sol[:, num_pumps:num_pumps + num_signals]
            ase_solution = np.empty((len(z), 0))

        bad_pump = self._log_first_nonfinite("[SM amp] pump_solution", pump_solution) if pump_solution.size else False
        bad_sig = self._log_first_nonfinite("[SM amp] signal_solution", signal_solution)
        if bad_pump or bad_sig:
            lg.error("[SM amp] non-finite values in solution; check gain matrix and step size/tolerances")
        else:
            pump_range = (
                f"[{pump_solution.min():.3e},{pump_solution.max():.3e}]" if pump_solution.size else "[]"
            )
            lg.debug(
                f"[SM amp] done solve: pump_sol={pump_solution.shape}, sig_sol={signal_solution.shape}; "
                f"power ranges pump{pump_range} "
                f"signal[{signal_solution.min():.3e},{signal_solution.max():.3e}]"
            )

        if check_photon_count:
            photon_count = np.sum(sol / frequencies, axis=1)
            return pump_solution, signal_solution, photon_count
        else:
            return pump_solution, signal_solution, ase_solution

    def solve_ase_with_fixed_powers(
        self,
        pump_solution: np.ndarray,
        signal_solution: np.ndarray,
        pump_wavelength: np.ndarray,
        signal_wavelength: np.ndarray,
        z: np.ndarray,
        ase_initial: np.ndarray | None = None,
        temperature: float = 300,
        reference_bandwidth: float = 0.1,
        odeint_kwargs=None,
        solver: str = "radau",
        ase_direction=1.0,
        ase_decimation: int = 1,
    ) -> np.ndarray:
        """Compute ASE evolution along fixed pump/signal profiles.

        pump_solution/signal_solution must be shaped (len(z), n). reference_bandwidth
        follows the same units used by solve(..., ase=True). ase_decimation keeps
        every Nth ASE channel, sums the signal powers per group, and scales the
        ASE bandwidth by N.
        """
        lg.info(f"[SM amp] solve_ase_with_fixed_powers: z len={len(z)}, "
                f"pump_sol={pump_solution.shape}, sig_sol={signal_solution.shape}, "
                f"ase_decimation={ase_decimation}, solver={solver}"
        )
        z = np.asarray(z, dtype=float)
        if z.ndim != 1 or z.size < 2:
            raise ValueError("z must be a 1D array with at least two points.")
        if np.any(np.diff(z) <= 0):
            raise ValueError("z must be strictly increasing for interpolation.")

        def _coerce_profile(profile: np.ndarray, name: str) -> np.ndarray:
            arr = np.asarray(profile, dtype=float)
            if arr.size == 0:
                return np.zeros((len(z), 0))
            if arr.ndim == 1:
                if arr.shape[0] != len(z):
                    raise ValueError(f"{name} length must match z length.")
                return arr.reshape((len(z), 1))
            if arr.ndim != 2:
                raise ValueError(f"{name} must be 1D or 2D.")
            if arr.shape[0] != len(z):
                raise ValueError(f"{name} first dimension must match z length.")
            return arr

        pump_solution = _coerce_profile(pump_solution, "pump_solution")
        signal_solution_full = _coerce_profile(signal_solution, "signal_solution")

        num_pumps = pump_solution.shape[1]
        num_signals_full = signal_solution_full.shape[1]
        if num_signals_full == 0:
            raise ValueError("signal_solution must have at least one signal.")

        ase_decimation = int(ase_decimation)
        if ase_decimation < 1:
            raise ValueError("ase_decimation must be >= 1.")
        ase_indices = np.arange(0, num_signals_full, ase_decimation, dtype=int)
        if ase_indices.size == 0:
            raise ValueError("ase_decimation removed all ASE channels.")
        group_slices = [
            slice(idx, min(idx + ase_decimation, num_signals_full)) for idx in ase_indices
        ]
        num_ase = len(group_slices)

        pump_wavelength = np.asarray(pump_wavelength, dtype=float)
        signal_wavelength_full = np.asarray(signal_wavelength, dtype=float)
        if num_pumps != pump_wavelength.size:
            raise ValueError("pump_wavelength size must match pump_solution columns.")
        if num_signals_full != signal_wavelength_full.size:
            raise ValueError("signal_wavelength size must match signal_solution columns.")

        signal_solution = np.stack(
            [signal_solution_full[:, slc].sum(axis=1) for slc in group_slices], axis=1
        )
        signal_wavelength = signal_wavelength_full[ase_indices]
        num_signals = signal_solution.shape[1]

        wavelengths = (
            np.concatenate((pump_wavelength, signal_wavelength))
            if num_pumps
            else signal_wavelength
        )
        frequencies = lambda2nu(wavelengths)
        gain_matrix = self.compute_gain_matrix(frequencies)

        frequency_shifts = frequencies[None, :] - frequencies[:, None]
        Hinv = np.exp(h_planck * np.abs(frequency_shifts) / (kB * temperature)) - 1
        np.fill_diagonal(Hinv, 1e56)
        eta_plus = 1 + 1 / Hinv
        np.fill_diagonal(eta_plus, 0)
        gain_matrix_ase = eta_plus * gain_matrix

        losses_s = self.get_linear_losses(signal_wavelength)
        signal_freqs = lambda2nu(signal_wavelength)
        ase_bandwidth = float(reference_bandwidth) * ase_decimation

        if ase_initial is None:
            ase_initial = np.zeros((num_ase,), dtype=float)
        else:
            ase_initial = np.asarray(ase_initial, dtype=float).reshape(-1)
            if ase_initial.size == num_signals_full:
                ase_initial = np.array(
                    [np.sum(ase_initial[slc]) for slc in group_slices], dtype=float
                )
            elif ase_initial.size != num_ase:
                raise ValueError("ase_initial must match number of ASE channels.")

        if np.isscalar(ase_direction):
            ase_direction = np.sign(ase_direction) * np.ones((num_ase,))
        else:
            ase_direction = np.asarray(ase_direction, dtype=float).reshape(-1)
            if ase_direction.size == num_signals_full:
                ase_direction = np.array(
                    [np.sign(np.sum(ase_direction[slc])) for slc in group_slices], dtype=float
                )
            elif ase_direction.size != num_ase:
                raise ValueError("ase_direction must match number of ASE channels.")

        pump_interp = None
        if num_pumps:
            pump_interp = scipy.interpolate.interp1d(
                z,
                pump_solution,
                axis=0,
                bounds_error=False,
                fill_value=(pump_solution[0], pump_solution[-1]),
            )
        signal_interp = scipy.interpolate.interp1d(
            z,
            signal_solution,
            axis=0,
            bounds_error=False,
            fill_value=(signal_solution[0], signal_solution[-1]),
        )

        def _ode_ivp(z_val, ase_power):
            pump_vec = pump_interp(z_val) if num_pumps else np.zeros((0,))
            signal_vec = signal_interp(z_val)
            P = np.concatenate((pump_vec, signal_vec))

            gain_factor = gain_matrix @ P
            gain_factor_ase = gain_matrix_ase @ P
            gain_sig = gain_factor[-num_signals:]
            gain_factor_ase = gain_factor_ase[-num_signals:]

            dASEdz = (-losses_s + gain_sig) * ase_power
            dASEdz += (
                gain_factor_ase
                * 2
                * h_planck
                * signal_freqs
                * ase_bandwidth
            )
            return dASEdz * ase_direction

        def _ode_odeint(ase_power, z_val):
            return _ode_ivp(z_val, ase_power)

        kw = dict(rtol=1e-3, atol=1e-9)
        if isinstance(odeint_kwargs, dict):
            kw.update(odeint_kwargs)

        if solver.lower() == "radau":
            if "max_step" not in kw:
                kw["max_step"] = (z[-1] - z[0]) / 3500 if len(z) > 1 else np.inf
            sol = scipy.integrate.solve_ivp(
                _ode_ivp,
                (z[0], z[-1]),
                ase_initial,
                method="Radau",
                t_eval=z,
                **kw,
            )
            if not sol.success:
                lg.error(f"[SM amp] Radau (ASE-only) failed: {sol.message}")
            return sol.y.T
        else:
            kw.pop("max_step", None)
            sol = scipy.integrate.odeint(_ode_odeint, ase_initial, z, **kw)
            return sol

    def compute_gain_matrix(self, frequencies):
        """Generate the matrix of Raman gains between each pair of frequencies."""
        frequencies = np.asarray(frequencies, dtype=float)
        resolution = self._gain_resolution(frequencies)
        # Include a version tag so older cached matrices (built with the wrong sign convention)
        # are not reused across process runs.
        cache_key = (frequencies.tobytes(), id(self.fiber), float(resolution), "raman_sign_v2")
        if cache_key in _GAIN_MATRIX_CACHE:
            return _GAIN_MATRIX_CACHE[cache_key]

        num_frequencies = len(frequencies)
        # Raman transfer depends on the shift (ν_j - ν_i): power flows from higher ν_j to lower ν_i.
        # The previous sign convention (ν_i - ν_j) inverted this flow, causing unphysical gain blow-up.
        frequency_shifts = frequencies[None, :] - frequencies[:, None]

        gains = self._interpolate_gain(frequency_shifts, resolution)

        wavelengths = nu2lambda(frequencies)
        if hasattr(self.fiber, "effective_area_at"):
            areas = np.array([self.fiber.effective_area_at(wl) for wl in wavelengths], dtype=float)
        else:
            areas = np.full_like(frequencies, getattr(self.fiber, "effective_area", 1.0), dtype=float)
        # use the average Aeff of the interacting pair to approximate overlap
        area_matrix = 0.5 * (areas[:, None] + areas[None, :])
        min_area = np.min(areas[areas > 0]) if np.any(areas > 0) else 1e-18
        area_matrix = np.where(area_matrix > 0, area_matrix, min_area)

        gains = gains * (self.fiber.raman_coefficient / area_matrix)
        np.fill_diagonal(gains, 0)

        freqs = np.expand_dims(frequencies, axis=-1)
        # Use the physical ν_i / ν_j factor (no clipping) so Stokes/anti-Stokes pairs
        # exchange power consistently and do not create artificial gain.
        freq_scaling = freqs * (1.0 / freqs.T)
        gain_matrix = freq_scaling * gains
        if self._log_first_nonfinite("[SM amp] gain_matrix", gain_matrix):
            raise ValueError("Non-finite values in Raman gain matrix")
        _GAIN_MATRIX_CACHE[cache_key] = gain_matrix
        lg.debug(
            f"Cached Raman gain matrix for {num_frequencies} frequencies "
            f"(Δf≈{resolution*1e-9:.1f} GHz, max shift {np.max(np.abs(frequency_shifts))*1e-12:.2f} THz)."
        )
        return gain_matrix

    @staticmethod
    def raman_ode(P, z, losses, gain_matrix, direction):
        """System of equations describing a Raman amplifier. No ASE. No Rayleigh back-reflection."""
        try:
            dPdz = (
                -losses[:, np.newaxis] + np.matmul(gain_matrix, P[:, np.newaxis])
            ) * P[:, np.newaxis]
        except ValueError as exc:
            lg.error(f"[SM amp] raman_ode shape mismatch: losses {losses.shape}, gain {gain_matrix.shape}, P {P.shape}, dir {direction.shape}")
            raise
        return direction * np.squeeze(dPdz)

    @staticmethod
    def raman_ode_with_ase(P, 
                           z, 
                           losses, 
                           gain_matrix, 
                           gain_matrix_ase, 
                           direction, 
                           temperature, 
                           ref_bandwidth, 
                           num_pumps, 
                           num_signals, 
                           frequencies):
        """System of equations describing a Raman amplifier. With ASE. No Rayleigh back-reflection. No ASE-to-pump power transfer"""
        h_planck = 6.626e-34

        num_ase = num_signals
        num_power = (num_signals + num_pumps)

        P_ = P[:num_power]
        P_ase = P[-num_ase:]
        losses_ase = losses[-num_ase:]

        gain_factor = np.matmul(gain_matrix, P_[:, np.newaxis])

        #      | Pumps | Signals |
        # Pumps     A        B
        # ------
        # ASE      C        D
        #
        try:

            dPowerdz = (-losses[:, np.newaxis] + gain_factor) * P_[:, np.newaxis]

            gain_factor_ase = np.matmul(gain_matrix_ase, P_[:, np.newaxis])
            gain_factor_ase = gain_factor_ase[-num_ase:]

            dASEdz = -losses[-num_ase:, np.newaxis] * P_ase[:, np.newaxis]
            dASEdz += gain_factor[-num_ase:] * P_ase[:, np.newaxis]
            dASEdz += (
                gain_factor_ase * 2 * h_planck *
                frequencies[-num_ase:, np.newaxis] * ref_bandwidth
            )

        except ValueError:
            breakpoint()
        dPdz = np.vstack((dPowerdz, dASEdz))
        return direction * np.squeeze(dPdz)

    @staticmethod
    def extended_raman_ode(P, z, losses, gain_matrix, direction):
        """System of equations describing a Raman amplifier. No ASE. No Rayleigh back-reflection."""

        # breakpoint()
        dPdz = np.ones_like(P)
        for z_ind in range(P.shape[1]):
            spectrum = P[:, z_ind]
            dPdz_z = (
                -losses[:, np.newaxis] + np.matmul(gain_matrix, spectrum[:, np.newaxis])
            ) * spectrum[:, np.newaxis]
            dPdz[:, z_ind] = direction * np.squeeze(dPdz_z)
        return dPdz

    def get_linear_losses(self, wavelengths):
        """Compute the linear loss coefficient for the given wavelengths."""
        losses = self.fiber.loss_profile(wavelengths)
        losses_linear = alpha_to_linear(losses)
        return losses_linear

    # --- Boundary-value solve (signals forward, pumps backward) ---
    def solve_bvp_signals_forward_pumps_backward(
        self,
        signal_power: np.ndarray,
        signal_wavelength: np.ndarray,
        pump_power: np.ndarray,
        pump_wavelength: np.ndarray,
        z: np.ndarray,
        tol: float = 1e-6,
        max_nodes: int = 100000,
        power_cap: float = 1e3,
    ):
        """
        Solve Raman power profiles as a boundary-value problem:
        signals are specified at z=0 (forward) and pumps at z=L (backward).

        This uses scipy.integrate.solve_bvp and assumes pumps propagate backward
        (direction = -1) while signals propagate forward (direction = +1).
        """
        import scipy.integrate

        z_arr = np.asarray(z, dtype=float).reshape(-1)
        if z_arr.size < 2:
            raise ValueError("z must contain at least two points for BVP solve")
        z0, zL = float(z_arr[0]), float(z_arr[-1])
        span = zL - z0
        if span <= 0:
            raise ValueError("z must be strictly increasing for BVP solve")

        sigP = np.asarray(signal_power, dtype=float).reshape(-1)
        sigW = np.asarray(signal_wavelength, dtype=float).reshape(-1)
        pumP = np.asarray(pump_power, dtype=float).reshape(-1)
        pumW = np.asarray(pump_wavelength, dtype=float).reshape(-1)

        Ns = sigP.size
        Np = pumP.size
        if sigW.size != Ns:
            raise ValueError("signal_power and signal_wavelength size mismatch")
        if pumW.size != Np:
            raise ValueError("pump_power and pump_wavelength size mismatch")

        pump_dir = -np.ones(Np, dtype=float)
        sig_dir = np.ones(Ns, dtype=float)
        direction = np.concatenate([pump_dir, sig_dir], axis=0)

        wavelengths = np.concatenate([pumW, sigW], axis=0)
        freqs = lambda2nu(wavelengths)
        losses = self.get_linear_losses(wavelengths)
        gain_matrix = self.compute_gain_matrix(freqs)

        lg.info(
            f"[SM amp] BVP solve_bvp: pumps={Np} (backward), signals={Ns} (forward), "
            f"span={span:.3e} m, tol={tol:.1e}, power_cap={power_cap:.1e} W"
        )
        # Scale the domain to [0,1] and powers to O(1) to avoid singular Jacobians
        s_grid = (z_arr - z0) / span
        power_scale = max(
            np.max(sigP) if Ns else 0.0,
            np.max(pumP) if Np else 0.0,
            1.0,
        )
        tol_scaled = tol / power_scale
        if tol_scaled > 1e2:
            lg.warning(
                f"[SM amp] BVP tolerance is very loose after scaling (tol_scaled={tol_scaled:.1e}); "
                "solution may be under-constrained."
            )
        lg.debug(
            f"[SM amp] BVP scaling: power_scale={power_scale:.3e} W, "
            f"tol_scaled={tol_scaled:.2e}, span={span:.3e} m"
        )

        # Loss-only initial guess that satisfies boundaries (physical units), then scale
        z_rel = z_arr - z0
        pump_guess = pumP[:, None] * np.exp(-losses[:Np, None] * (zL - z_arr)[None, :]) if Np else np.zeros((0, z_arr.size))
        sig_guess = sigP[:, None] * np.exp(-losses[Np:, None] * z_rel[None, :]) if Ns else np.zeros((0, z_arr.size))
        y_init = np.vstack([pump_guess, sig_guess])
        y_init_scaled = y_init / power_scale
        pump_bc_scaled = pumP / power_scale if Np else np.array([])
        sig_bc_scaled = sigP / power_scale if Ns else np.array([])

        def ode(s_local, Y_scaled):
            Y_phys = np.clip(Y_scaled * power_scale, 0.0, power_cap)
            gmY = gain_matrix @ Y_phys
            gmY = np.clip(gmY, -power_cap, power_cap)
            rhs_phys = (-losses[:, None] + gmY) * Y_phys
            return direction[:, None] * (rhs_phys * (span / power_scale))

        def bc(ya_scaled, yb_scaled):
            res = np.zeros_like(ya_scaled)
            if Np:
                res[:Np] = yb_scaled[:Np] - pump_bc_scaled
            if Ns:
                res[Np:] = ya_scaled[Np:] - sig_bc_scaled
            return res

        sol = scipy.integrate.solve_bvp(
            ode,
            bc,
            s_grid,
            y_init_scaled,
            tol=tol_scaled,
            max_nodes=max_nodes,
        )
        if not sol.success:
            raise RuntimeError(
                f"solve_bvp failed: {sol.message} "
                f"[span={span:.3e} m, power_scale={power_scale:.3e} W, tol_scaled={tol_scaled:.3e}]"
            )

        # Evaluate solution on the original grid and rescale to physical units
        try:
            y_scaled = sol.sol(s_grid)
        except Exception:
            y_scaled = sol.y
        y = y_scaled * power_scale  # shape (N, M)
        pump_solution = y[:Np, :].T if Np else np.zeros((z_arr.size, 0))
        signal_solution = y[Np:, :].T if Ns else np.zeros((z_arr.size, 0))
        ase_solution = np.empty((z_arr.size, 0))
        return pump_solution, signal_solution, ase_solution

    def solve_shooting(self, pump_power, signal_power, z, solver="scipy"):
        """Shooting algorithm presented in [1].

        References
        ----------
        [1] Hai Ming Jiang and Kang Xie, "Efficient and robust shooting algorithm
          for numerical design of bidirectionally pumped Raman fiber amplifiers,"
          J. Opt. Soc. Am. B 29, 8-14 (2012)
        """

        if solver == "scipy":
            return self.solve_shooting_scipy(pump_power, signal_power, z)
        elif solver == "jiang":
            return self.solve_shooting_jiang(pump_power, signal_power, z)

    def solve_shooting_scipy(
        self,
        pump_power,
        signal_power,
        z,
        pump_wavelength=None,
        signal_wavelength=None,
        pump_power_floor: float = 0.0,
        guess_scale: float = 1e-3,
    ):
        """Shooting using scipy.solve_bvp to meet pump boundary conditions."""
        num_signals = len(signal_power)
        num_pumps = len(pump_power)
        alpha = 0.01

        # 1): construct the vector of initial conditions

        # 1.a) Order the frequencies from largest to smallest

        def reverse_sort(a, b):
            sort_idx = np.argsort(a)
            a = a[sort_idx]
            b = b[sort_idx]
            return a[::-1], b[::-1]

        # ensure internal state mirrors inputs
        self.pump_power = pump_power
        self.signal_power = signal_power
        if pump_wavelength is not None:
            self.pump_wavelengths = pump_wavelength
        elif not hasattr(self, "pump_wavelengths") or self.pump_wavelengths is None:
            raise ValueError("pump_wavelength must be provided for shooting solver.")
        if signal_wavelength is not None:
            self.signal_wavelengths = signal_wavelength
        elif not hasattr(self, "signal_wavelengths") or self.signal_wavelengths is None:
            raise ValueError("signal_wavelength must be provided for shooting solver.")
        self.direction = np.ones(num_pumps + num_signals)

        pump_frequencies = lambda2nu(self.pump_wavelengths)
        sorted_pump_frequencies, sorted_pump_powers = reverse_sort(
            pump_frequencies, pump_power
        )

        pump_gain_matrix = self.compute_gain_matrix(sorted_pump_frequencies)
        sorted_pump_losses = self.get_linear_losses(nu2lambda(sorted_pump_frequencies))
        signal_losses = self.get_linear_losses(self.signal_wavelengths)

        def build_initial_conditions(scale: float):
            direction_tmp = np.ones_like(sorted_pump_powers)
            guess_pumps = np.maximum(sorted_pump_powers * scale, pump_power_floor)
            sol_guess = scipy.integrate.odeint(
                RamanAmplifier.raman_ode,
                guess_pumps,
                z,
                args=(sorted_pump_losses, pump_gain_matrix, direction_tmp),
            )
            pump_solution_initial_cond = sol_guess[::-1, :num_pumps]
            signal_solution_initial_cond = self.signal_power * np.exp(
                -z[:, np.newaxis] * signal_losses
            )
            return np.hstack((pump_solution_initial_cond, signal_solution_initial_cond)).T

        wavelengths = np.concatenate(
            (nu2lambda(sorted_pump_frequencies), self.signal_wavelengths)
        )
        frequencies = lambda2nu(wavelengths)
        losses_linear = self.get_linear_losses(wavelengths)
        gain_matrix = self.compute_gain_matrix(frequencies)
        target_power_spectrum = np.concatenate((sorted_pump_powers, self.signal_power))

        def boundary_residuals(ya, yb):
            residuals = np.zeros_like(ya)
            fwd_idx = self.direction > 0
            bwd_idx = self.direction < 0
            residuals[fwd_idx] = ya[fwd_idx]
            residuals[bwd_idx] = yb[bwd_idx]
            return target_power_spectrum - residuals

        def ode(z, P):
            return RamanAmplifier.extended_raman_ode(
                P, z, losses_linear, gain_matrix, self.direction
            )

        scales = [guess_scale, guess_scale * 0.1, guess_scale * 0.01]
        last_err = None
        for sc in scales:
            try:
                init = build_initial_conditions(sc)
                result = scipy.integrate.solve_bvp(
                    ode, boundary_residuals, z, init, verbose=1
                )
                if result.success:
                    lg.info(f"[SM amp] Shooting converged with initial scale {sc}")
                    return result.y.transpose()
                last_err = result.message
                lg.warning(f"[SM amp] Shooting failed (scale {sc}): {result.message}")
            except Exception as exc:
                last_err = str(exc)
                lg.warning(f"[SM amp] Shooting exception (scale {sc}): {exc}")
        raise RuntimeError(f"Shooting did not converge: {last_err}")

    def solve_shooting_jiang(self, pump_power, signal_power, z):
        """Shooting algorithm presented in [1].

        References
        ----------
        [1] Hai Ming Jiang and Kang Xie, "Efficient and robust shooting algorithm
          for numerical design of bidirectionally pumped Raman fiber amplifiers,"
          J. Opt. Soc. Am. B 29, 8-14 (2012)
        """
        num_pumps = len(pump_power)
        alpha = 0.01

        # 1): construct the vector of initial conditions

        # 1.a) Order the frequencies from largest to smallest

        def reverse_sort(a, b):
            sort_idx = np.argsort(a)
            a = a[sort_idx]
            b = b[sort_idx]
            return a[::-1], b[::-1]

        pump_frequencies = lambda2nu(self.pump_wavelengths)
        sorted_pump_frequencies, sorted_pump_powers = reverse_sort(
            pump_frequencies, pump_power
        )

        gain_matrix = self.compute_gain_matrix(sorted_pump_frequencies)
        sorted_pump_losses = self.get_linear_losses(nu2lambda(sorted_pump_frequencies))

        # 1.b) Propagate the pumps adding them one by one.
        x0 = np.zeros_like(sorted_pump_powers)
        input_power_tmp = np.zeros_like(sorted_pump_powers)
        direction_tmp = np.ones_like(sorted_pump_powers)

        for i, Pp in enumerate(sorted_pump_powers):

            input_power_tmp[i] = sorted_pump_powers[i]

            sol = scipy.integrate.odeint(
                RamanAmplifier.raman_ode,
                input_power_tmp,
                z,
                args=(sorted_pump_losses, gain_matrix, direction_tmp),
            )

            pump_solution = sol[:, :num_pumps]
            signal_solution = sol[:, num_pumps:]

            P_current_pump_z0 = pump_solution[-1, i]
            x0[i] = P_current_pump_z0

            # plt.figure()
            # plt.plot(z[::-1], pynlin.utils.watt2dBm(pump_solution))
            # plt.title(f"Phase 1, iteration {i+1}: find initial guesses")

        # 1.c) Determine the scaling vector S
        S = np.ones_like(x0) / 1e3
        x0 = x0 * S

        # Step 2) Iterate to correct the initial guesses

        wavelengths = np.concatenate(
            (nu2lambda(sorted_pump_frequencies), self.signal_wavelengths)
        )
        frequencies = lambda2nu(wavelengths)
        losses_linear = self.get_linear_losses(wavelengths)
        gain_matrix = self.compute_gain_matrix(frequencies)

        def solve_system(x):
            """Get the solution of the system for `x` input pump powers."""
            input_power = np.concatenate((x, self.signal_power))
            sol = scipy.integrate.odeint(
                RamanAmplifier.raman_ode,
                input_power,
                z,
                args=(losses_linear, gain_matrix, self.direction),
            )
            return sol

        def get_output_pump_powers(x):
            """Get the power of the pumps at the end of the fiber for the current guess `x`."""
            sol = solve_system(x)
            return sol[-1, :num_pumps]

        def compute_error(x):
            P_out = get_output_pump_powers(x)
            D = P_out - pump_power
            return D

        def pump_mse(x):
            err = compute_error(x)
            return np.mean(err**2)

        def gradient(x, h=1e-3):
            grad = np.zeros((x.shape[0],))

            for i in range(num_pumps):
                x_p = np.copy(x)
                x_m = np.copy(x)
                x_p[i] = x_p[i] + h
                x_m[i] = x_m[i] - h
                mse_p = pump_mse(x_p)
                mse_m = pump_mse(x_m)
                grad[i] = (mse_p - mse_m) / (2 * h)
            return grad

        def compute_jacobian(x0, h=1e-3):
            """Compute the Jacobian matrix."""
            J = np.zeros((x0.shape[0], x0.shape[0]))
            D = compute_error(x0)

            for i in range(num_pumps):
                x_p = np.copy(x0)
                x_m = np.copy(x0)
                x_p[i] = x_p[i] + h
                x_m[i] = x_m[i] - h
                D_p = compute_error(x_p)
                D_m = compute_error(x_m)

                J[:, i] = (D_p - D_m) / (2 * h)

            return J

        # Iterate on the initial guesses
        num_iter = 0

        Ds = []

        while num_iter < 1000:
            D = compute_error(x0)

            lg.debug(f"Iteration {num_iter}")
            lg.debug(f"\tx0={pynlin.utils.watt2dBm(x0)} dBm")
            lg.debug(f"\tError={D * 1e3} mW")
            Ds.append(D)

            sol = solve_system(x0)
            pump_solution = sol[:, :num_pumps]
            signal_solution = sol[:, num_pumps:]

            # fig, ax = plt.subplots(ncols=2)
            # ax[0].plot(z, pynlin.utils.watt2dBm(pump_solution), color="red")
            # ax[0].plot(z, pynlin.utils.watt2dBm(signal_solution), color="black")
            # ax[0].set_title(f"Iteration {num_iter}")
            # ax[1].semilogy(np.abs(np.stack(Ds)), marker="x")
            # plt.show()

            eta = 0.0001
            try:

                G = gradient(x0)
                # J = compute_jacobian(x0)

                # delta_P = -np.dot(np.linalg.inv(J), D)
                # cap the maximum change to 10 mW
                # signs = np.sign(delta_P)
                # abs = np.abs(delta_P)
                # delta_P_capped = np.minimum(abs * 1e3, 100)
                # delta_P = signs * delta_P_capped * 1e-3

                #     # breakpoint()

                x0 = x0 - eta * G
                lg.debug(f"\tx0_new={pynlin.utils.watt2dBm(x0)} dBm")
                num_iter += 1
            except np.linalg.LinAlgError:
                break

        return solve_system(x0)


class MMFRamanAmplifier(RamanAmplifier):
    # def __init__(self, bandwidth=40e12):
    #     super().__init__()

    def solve(
        self,
        signal_power,
        signal_wavelength,
        pump_power,
        pump_wavelength,
        z,
        fiber,
        counterpumping=False,
        ase=False,
        reference_bandwidth=0.1 * 1e-9,
        temperature=300,
        shooting=None,
        initial_guesses=None,
        direction=None,
        odeint_kwargs=None
    ):
        lg.debug(f"[MM amp] start solve: signals={signal_power.shape}, pumps={pump_power.shape}, z={len(z)}")
        """Solve the multi-mode Raman amplifier equations [1].

        Params
        ------
        signal_power: np.ndarray
          The input signal power in a 2d ndarray of shape (wavelengths, modes)
          expressed in Watt.
        signal_wavelength: np.ndarray
          The input signal wavelengths in a 2d ndarray of shape (wavelengths,)
          expressed in meters.
        pump_power: np.ndarray
          The input pump power in a 2d ndarray of shape (wavelengths, modes)
          expressed in Watt.
        pump_wavelength: np.ndarray
          The input pump wavelengths in a 1d ndarray of shape (wavelengths,)
          expressed in meters.
        z: np.ndarray
          The z-axis along which to integrate the equations, expressed in
          meters.
        fiber: pyraman.Fiber
          Fiber object defining the amplifier
        counterpumping: bool, optional
          If True, `pump_power` is considered to be the pump power
          at the start of the fiber, i.e. at z = 0. The signs
          of the equations for the pump power evolution are set to -1,
          so the losses actually amplify the pump power during
          propagation in the z: 0->L direction. Optional, by default False.
        reference_bandwidth: float, optional
          The reference optical bandwidth (nm) for ASE measurement.
          Optional, by default 0.1 nm.
        temperature: float, optional
          The optical fiber temperature (K). Optional, by default 300 K.
        shooting : bool, optional
          Use a shooting method to solve the counterpumping case,
          by default None.

        References
        ----------
        .. [1] Ryf, Roland, Rene Essiambre, Johannes von Hoyningen-Huene,
            and Peter Winzer. 2012. “Analysis of Mode-Dependent Gain in
            Raman Amplified Few-Mode Fiber.” In Optical Fiber Communication
            Conference, Los Angeles, California: OSA, OW1D.2.
            https://www.osapublishing.org/abstract.cfm?uri=OFC-2012-OW1D.2
            (June 5, 2019).
        """
        num_signals = signal_power.shape[0]
        num_pumps = pump_power.shape[0]

        total_wavelengths = num_signals + num_pumps
        num_modes = fiber.n_modes
        total_signals = total_wavelengths
        pump_power_ = pump_power.reshape((num_modes * num_pumps))
        signal_power_ = signal_power.reshape((num_modes * num_signals))

        # Structure of the waves: modes and frequencies
        # - waves: | (pumps)
        #          | freq1                        | freqn                      |
        # array:   | LP01                LPxx     | LP01                LPxx   |

        wavelengths = np.concatenate((pump_wavelength, signal_wavelength))
        input_power = np.concatenate((pump_power_, signal_power_))
        frequencies = wavelength_to_frequency(wavelengths)
        resolution = self._gain_resolution(frequencies)

        loss_coeffs = fiber.losses
        losses_ = polyval(loss_coeffs, wavelengths)
        losses_linear = alpha_to_linear(losses_)
        losses_linear = np.repeat(losses_linear, fiber.n_modes)

        frequency_shifts = np.zeros((total_wavelengths, total_wavelengths))
        for i in range(total_wavelengths):
            frequency_shifts[i, :] = frequencies - frequencies[i]

        gains = self._interpolate_gain(frequency_shifts, resolution)
        gains *= fiber.raman_coefficient

        # Must be multiplied by overlap integrals
        # Force diagonal to be 0
        np.fill_diagonal(gains, 0)
        # gains = np.triu(gains) + np.triu(gains, 1).T

        # compute the frequency scaling factor
        freqs = np.expand_dims(frequencies, axis=-1)
        freq_scaling = np.maximum(1, freqs * (1 / freqs.T))

        mode_list = np.array(range(fiber.n_modes))
        # change the order of creation
        oi = fiber.get_oi_matrix(mode_list, wavelengths)
        
        gain_matrix = freq_scaling * gains
        gain_matrix = gain_matrix.repeat(fiber.n_modes, axis=0).repeat(
            fiber.n_modes, axis=1
        )
        gains_mmf = gain_matrix * oi
    
        np.save("np_gains.npy", frequencies)
        if not ase:
            if direction is None:
                direction = np.ones((total_wavelengths * fiber.n_modes,))

            if counterpumping or shooting:
                direction[: num_pumps * fiber.n_modes] = -1

            if not shooting:
                kw = dict(rtol=1e-6, atol=1e-12, mxstep=10000)
                if isinstance(odeint_kwargs, dict):
                    kw.update(odeint_kwargs)
                sol = scipy.integrate.odeint(
                    MMFRamanAmplifier.raman_ode,
                    input_power,
                    z,
                    args=(losses_linear, gains_mmf, direction),
                    **kw,
                )
                sol = sol.reshape((len(z), total_signals, fiber.n_modes))
                pump_solution = sol[:, :num_pumps, :]
                signal_solution = sol[:, num_pumps:, :]
                lg.debug(f"[MM amp] done solve (no ASE, no shooting): pump_sol={pump_solution.shape}, sig_sol={signal_solution.shape}")
            else:
                # SHOOTING METHOD
                pump_losses = losses_linear[: num_pumps * fiber.n_modes]

                if initial_guesses is None:
                    initial_guesses = pump_power_ * np.exp(-z[-1] * pump_losses) / 10

                callback_info = {
                    "max_error": float("inf"),
                    "iter": 0,
                    "threshold": 0.01,
                    "params": None,
                }

                def callback(x):
                    max_error = callback_info["max_error"]
                    threshold = callback_info["threshold"]
                    lg.debug(f"Max error: {max_error} mW")
                    if max_error < threshold:
                        # print(f"Shooting error < {threshold}, stopping optimization")
                        callback_info["params"] = np.copy(x)
                        raise TimeOut("Stopping optimization")

                def optim_fun(x0):

                    x0_ = 10 ** (x0 / 10)
                    input_power = np.concatenate((x0_, signal_power_))
                    kw = dict(rtol=1e-3, atol=1e-6, mxstep=10000)
                    if isinstance(odeint_kwargs, dict):
                        kw.update(odeint_kwargs)
                    sol = scipy.integrate.odeint(
                        MMFRamanAmplifier.raman_ode,
                        input_power,
                        z,
                        args=(losses_linear, gains_mmf, direction),
                        **kw,
                    )
                    sol = sol.reshape((len(z), total_signals, fiber.n_modes))
                    pump_solution = sol[-1, :num_pumps, :].flatten()
                    # return the MSE between desired solution and obtained values
                    cost = np.sqrt(
                        np.mean((pump_solution * 1e3 - pump_power_ * 1e3) ** 2)
                    )
                    max_error = np.max(np.abs(pump_solution - pump_power_)) * 1e3
                    callback_info["max_error"] = max_error
                    # print(f"RMSE: {cost:.3f} mW\tMax. Error {max_error:.2f} mW")
                    return cost

                # bounds = [(0, None) for _ in range(num_pumps * fiber.n_modes)]

                try:
                    result = scipy.optimize.minimize(
                        # optim_fun, initial_guesses, method="L-BFGS-B", bounds=bounds
                        optim_fun,
                        10 * np.log10(initial_guesses),
                        method="L-BFGS-B",
                        options={"maxfun": 1e10, "maxiter": 1e10, "ftol": 1e-15},
                        # bounds=bounds,
                        callback=callback,
                    )

                    x0 = result.x
                except TimeOut:
                    x0 = callback_info["params"]

                x0 = 10 ** (x0 / 10)

                # Result of the optimization process: pump power at z=0

                # Propagate
                input_power = np.concatenate((x0, signal_power_))

                sol = scipy.integrate.odeint(
                    MMFRamanAmplifier.raman_ode,
                    input_power,
                    z,
                    args=(losses_linear, gains_mmf, direction),
                )
                sol = sol.reshape((len(z), total_signals, fiber.n_modes))

                pump_solution = sol[:, :num_pumps, :]
                signal_solution = sol[:, num_pumps:, :]
            lg.debug(f"[MM amp] done solve (no ASE, shooting): pump_sol={pump_solution.shape}, sig_sol={signal_solution.shape}")
            return pump_solution, signal_solution, np.array([])
        else:
            direction = np.ones(((total_wavelengths + num_signals) * fiber.n_modes,))

            # Compute the phonon occupancy factor
            Hinv = np.exp(h_planck * np.abs(frequency_shifts) / (kB * temperature)) - 1
            Hinv = np.where(Hinv == 0, -1, Hinv)
            eta = 1 + 1 / Hinv
            np.fill_diagonal(eta, 0)
            # print(eta)
            eta = np.repeat(np.repeat(eta, fiber.n_modes, axis=0), fiber.n_modes, axis=1)

            # Compute the new Raman gain matrix
            gain_matrix_ase = eta * gains_mmf

            # Convert reference bandwidth in hertz using the
            # central signal wavelength as reference
            central_wavelength = (signal_wavelength.max() + signal_wavelength.min()) / 2
            w_a = central_wavelength - reference_bandwidth / 2
            w_b = central_wavelength + reference_bandwidth / 2
            f_a = lambda2nu(w_a)
            f_b = lambda2nu(w_b)
            reference_bandwidth_hz = np.abs(f_a - f_b)

            if counterpumping or shooting:
                direction[: num_pumps * fiber.n_modes] = -1

            # Initial conditions, ase power must be 0 at z=0
            input_power_ase = np.zeros((input_power.size + num_signals * fiber.n_modes,))
            input_power_ase[: input_power.size] = input_power

            signal_frequencies = wavelength_to_frequency(signal_wavelength)
            ase_frequencies = np.repeat(signal_frequencies, fiber.n_modes)

            kw = dict(rtol=1e-3, atol=1e-6, mxstep=10000)
            if isinstance(odeint_kwargs, dict):
                kw.update(odeint_kwargs)
            sol = scipy.integrate.odeint(
                MMFRamanAmplifier.raman_ode_with_ase,
                input_power_ase,
                z,
                args=(
                    losses_linear,
                    gains_mmf,
                    gain_matrix_ase,
                    ase_frequencies,
                    reference_bandwidth_hz,
                    direction,
                    num_signals,
                    num_pumps,
                    fiber.n_modes,
                ),
                **kw,
            )

            sol = sol.reshape((len(z), total_signals + num_signals, fiber.n_modes))
            power_solution = sol[:, :total_signals, :]
            pump_solution = power_solution[:, :num_pumps, :]
            signal_solution = power_solution[:, num_pumps:, :]
            ase_solution = sol[:, -num_signals:, :]

            lg.debug(f"[MM amp] done solve (with ASE): pump_sol={pump_solution.shape}, sig_sol={signal_solution.shape}")
            return pump_solution, signal_solution, ase_solution

    def solve_shooting():
        pass

    @staticmethod
    def raman_ode(P, z, losses, gain_matrix, direction):
        """Integration step of the multimode Raman system."""
        dPdz = (-losses[:, np.newaxis] + np.matmul(gain_matrix,
                                                   P[:, np.newaxis])) * P[:, np.newaxis]

        return np.squeeze(dPdz) * direction

    def raman_ode_with_ase(
        P,
        z,
        losses,
        gain_matrix,
        gain_matrix_ase,
        frequencies,
        ref_bandwidth,
        direction,
        num_signals,
        num_pumps,
        num_modes,
    ):
        """Integration step of the multimode Raman system with ASE."""
        num_ase = num_signals * num_modes
        num_power = (num_signals + num_pumps) * num_modes

        P_ = P[:num_power]
        P_ase = P[-num_ase:]
        losses_ase = losses[-num_ase:]

        gain_factor = np.matmul(gain_matrix, P_[:, np.newaxis])

        dPowerdz = (-losses[:, np.newaxis] + gain_factor) * P_[:, np.newaxis]

        gain_factor_ase = np.matmul(gain_matrix_ase, P_[:, np.newaxis])
        gain_factor_ase = gain_factor_ase[-num_ase:]

        dASEdz = -losses_ase[:, np.newaxis] * P_ase[:, np.newaxis]
        dASEdz += gain_factor[-num_ase:] * P_ase[:, np.newaxis]
        dASEdz += (
            gain_factor_ase * 2 * h_planck * frequencies[:, np.newaxis] * ref_bandwidth
        )

        dPdz = np.vstack((dPowerdz, dASEdz))
        return np.squeeze(dPdz) * direction

# Named amplifiers mirroring fiber naming (wideband by default)
class SMRamanAmplifier(RamanAmplifier):
    """Single-mode Raman amplifier (wideband)."""
    pass


class SMRamanAmplifierNarrowband(SMRamanAmplifier):
    """Single-mode narrowband Raman amplifier (placeholder for distinct behavior)."""
    pass


class MMRamanAmplifier(MMFRamanAmplifier):
    """Multimode Raman amplifier."""
    pass

# Wideband SMF Raman amplifier aware of irregular WDM launch powers
class SMWidebandRamanAmplifier(SMRamanAmplifier):
    """Single-mode Raman amplifier that honors per-band launch powers on irregular WDM grids."""

    @staticmethod
    def _band_launch_powers(system, default_dbm: float) -> np.ndarray:
        wdm = system.wdm
        n_ch = getattr(wdm, "num_channels", None)
        powers_dbm = np.full(n_ch, default_dbm)
        if isinstance(wdm, IrregularWDM):
            for name, slc in getattr(wdm, "_band_slices", {}).items():
                spec = wdm.band_specs.get(name)
                if spec and spec.launch_power_dbm is not None:
                    powers_dbm[slc] = spec.launch_power_dbm
        # power_scale = getattr(wdm, "power_scale", 1.0)
        # if power_scale not in (None, 1.0):
        #     lg.info(f"[SM amp] Detected WDM power_scale={power_scale:.3f}; undoing per-channel boost.")
        #     powers_dbm = powers_dbm - 10 * np.log10(power_scale)
        return powers_dbm

    def solve_from_system(self, system, z: np.ndarray, disable_pumps: bool = False, **kwargs):
        fiber = system.fiber
        wdm = system.wdm
        pumps = [] if disable_pumps else (system.pump_specs or [])

        freqs = wdm.frequency_grid()
        wavelengths = nu2lambda(freqs)

        launch_dbm_default = system.launch_power if system.launch_power is not None else -5.0
        signal_power_dbm = self._band_launch_powers(system, launch_dbm_default)
        signal_power_w = dBm2watt(signal_power_dbm)

        pump_wavelengths = np.array([p.wavelength for p in pumps]) if pumps else np.array([])
        pump_powers_w = dBm2watt(np.array([p.power_dbm for p in pumps])) if pumps else np.array([])
        pump_dirs = np.array([p.direction for p in pumps]) if pumps else np.array([])

        return self.solve(
            signal_power=signal_power_w,
            signal_wavelength=wavelengths,
            pump_power=pump_powers_w,
            pump_wavelength=pump_wavelengths,
            z=z,
            pump_direction=pump_dirs,
            use_power_at_fiber_start=True,
            check_photon_count=False,
            **kwargs,
        )

    def solve_signals_with_fixed_pumps(
        self,
        signal_power: np.ndarray,
        signal_wavelength: np.ndarray,
        pump_power_profile: np.ndarray,
        pump_wavelength: np.ndarray,
        z: np.ndarray,
        odeint_kwargs=None,
        solver: str = "radau",
    ):
        """Solve only signals with pumps fixed to a provided z-profile."""
        num_pumps = pump_power_profile.shape[1] if pump_power_profile.size else 0
        num_signals = signal_power.shape[0]
        wavelengths = np.concatenate((pump_wavelength, signal_wavelength)) if num_pumps else signal_wavelength
        frequencies = lambda2nu(wavelengths)
        gain_matrix = self.compute_gain_matrix(frequencies)
        losses_s = self.get_linear_losses(signal_wavelength)

        if num_pumps:
            Gsp = gain_matrix[num_pumps:, :num_pumps]
            Gss = gain_matrix[num_pumps:, num_pumps:]
        else:
            Gsp = np.zeros((num_signals, 0))
            Gss = gain_matrix

        kw = dict(rtol=1e-3, atol=1e-6, max_step=(z[-1]-z[0]) / 3500 if len(z) > 1 else None)
        if isinstance(odeint_kwargs, dict):
            kw.update(odeint_kwargs)

        if num_pumps:
            pump_interp = [
                lambda zz, col=i: float(np.interp(zz, z, pump_power_profile[:, col])) for i in range(num_pumps)
            ]
        else:
            pump_interp = []

        def _ode(z_val, S):
            pump_vec = np.array([fn(z_val) for fn in pump_interp]) if num_pumps else np.zeros((0,))
            rhs = -losses_s + (Gss @ S)
            if num_pumps:
                rhs += Gsp @ pump_vec
            return rhs * S

        if solver.lower() == "radau":
            sol = scipy.integrate.solve_ivp(
                _ode,
                (z[0], z[-1]),
                signal_power,
                method="Radau",
                t_eval=z,
                **kw,
            )
            if not sol.success:
                lg.error(f"[SM amp] Radau (signals-only) failed: {sol.message}")
            return sol.y.T
        else:
            sol = scipy.integrate.odeint(_ode, signal_power, z, **kw)
            return sol

    def solve_two_step_from_system(
        self,
        system,
        z: np.ndarray,
        decimation_factor: int = 100,
        disable_pumps: bool = False,
        pump_power_floor: float = 0.0,
        use_shooting: bool = False,
        shooting_guess_scale: float = 1e-3,
        conserve_power_on_decimation: bool = True,
        pump_input_scale: float = 1e-3,
        **kwargs,
    ):
        """Two-step solve: (1) decimated signals + pumps, (2) full signals with fixed pumps."""
        fiber = system.fiber
        wdm = system.wdm
        pumps = [] if disable_pumps else (system.pump_specs or [])

        # Optional signal decimation via WDM helper (no per-channel power rescale)
        wdm_dec = wdm.decimate(decimation_factor, rescale_power=False) if decimation_factor > 1 else wdm
        freqs = wdm_dec.frequency_grid()
        wavelengths = nu2lambda(freqs)

        launch_dbm_default = system.launch_power if system.launch_power is not None else -5.0
        # Use a lightweight system-like holder so _band_launch_powers uses the decimated grid
        dec_sys = type("TmpSystem", (), {"wdm": wdm_dec})
        signal_power_dbm = self._band_launch_powers(dec_sys, launch_dbm_default)
        signal_power_w = dBm2watt(signal_power_dbm)
        if decimation_factor > 1 and conserve_power_on_decimation and freqs.size:
            power_scale = len(wdm.frequency_grid()) / len(freqs)
            signal_power_w *= power_scale
            lg.info(f"[SM amp] Two-step: decimating signals by {decimation_factor} (kept {len(freqs)}); power scale {power_scale:.2f}")
        elif decimation_factor > 1:
            lg.info(f"[SM amp] Two-step: decimating signals by {decimation_factor} (kept {len(freqs)}) without power scaling")

        pump_wavelengths = np.array([p.wavelength for p in pumps]) if pumps else np.array([])
        pump_powers_target_w = dBm2watt(np.array([p.power_dbm for p in pumps])) if pumps else np.array([])
        pump_powers_w = pump_powers_target_w * pump_input_scale if pumps else np.array([])
        if pumps:
            lg.info(f"[SM amp] Using pump input scale {pump_input_scale:.1e}; targets (W) min/max {pump_powers_target_w.min():.3e}/{pump_powers_target_w.max():.3e}")
        pump_dirs = np.array([p.direction for p in pumps]) if pumps else np.array([])

        # Step 1: decimate signals but keep total power
        if decimation_factor > 1:
            sig_idx = np.arange(0, len(wavelengths), decimation_factor, dtype=int)
            scale = len(wavelengths) / len(sig_idx) if conserve_power_on_decimation else 1.0
            sig_power_dec = signal_power_w[sig_idx] * scale
            sig_wl_dec = wavelengths[sig_idx]
            lg.info(f"[SM amp] Two-step: decimating signals by {decimation_factor} (kept {len(sig_idx)})")
            if not conserve_power_on_decimation:
                lg.info("[SM amp] Decimation without power scaling (per-channel power unchanged)")
        else:
            sig_idx = np.arange(len(wavelengths))
            sig_power_dec = signal_power_w
            sig_wl_dec = wavelengths

        # Step 1: decimated forward solve (no shooting)
        pump_solution_step1, signal_solution_step1, _ = self.solve(
            signal_power=sig_power_dec,
            signal_wavelength=sig_wl_dec,
            pump_power=pump_powers_w,
            pump_wavelength=pump_wavelengths,
            z=z,
            pump_direction=pump_dirs,
            use_power_at_fiber_start=True,
            check_photon_count=False,
            pump_power_floor=pump_power_floor,
            **kwargs,
        )
        try:
            from pynlin.raman.plot_optimization import plot_profiles
            sig_step1_plot = signal_solution_step1[:, :, None] if signal_solution_step1.ndim == 2 else signal_solution_step1
            pump_step1_plot = pump_solution_step1[:, :, None] if pump_solution_step1.ndim == 2 else pump_solution_step1
            pump_powers_step1 = pump_powers_w[:, None] if pump_powers_w.size else np.zeros((0, 1))
            plot_profiles(
                signal_wavelengths=sig_wl_dec,
                signal_solution=sig_step1_plot,
                ase_solution=None,
                pump_wavelengths=pump_wavelengths,
                pump_solution=pump_step1_plot,
                pump_powers=pump_powers_step1,
                cf=system,
                wallpaper_mode=False,
            )
            lg.info("Saved decimated-step profiles via plot_profiles.")
        except Exception as e:
            lg.warning(f"Decimated-step plotting failed: {e}")

        # Step 2: solve full signals with pumps fixed to step1 profile
        signal_solution_full = self.solve_signals_with_fixed_pumps(
            signal_power=signal_power_w,
            signal_wavelength=wavelengths,
            pump_power_profile=pump_solution_step1 if pump_solution_step1.size else np.zeros((len(z), 0)),
            pump_wavelength=pump_wavelengths,
            z=z,
            odeint_kwargs=kwargs.get("odeint_kwargs"),
        )

        return pump_solution_step1, signal_solution_full, signal_solution_step1, sig_wl_dec

# Backward compatibility
RamanAmplifier = SMRamanAmplifier
MMFRamanAmplifier = MMRamanAmplifier
SMRamanAmplifierWideband = SMWidebandRamanAmplifier


# ---------------------------------------
#  MAIN
# ---------------------------------------


def main():
    import sys
    from pathlib import Path
    from pynlin.system import System
    from pynlin.utils import nu2lambda

    init_logging()
    level = os.getenv("LOGURU_LEVEL", "TRACE")
    lg.remove()
    lg.add(sys.stderr, level=level)

    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input/studies.toml")
    profile_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/uwb_power_profiles.npy")
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/uwb_ase_power_profile.npy")

    if not profile_path.exists():
        lg.error(f"[SM amp] profile file not found: {profile_path}")
        return

    system = System.from_toml(cfg_path)
    amp = SMWidebandRamanAmplifier(system.fiber)

    data = np.load(profile_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    if not isinstance(data, dict):
        lg.error(f"[SM amp] expected dict-like payload in {profile_path}, got {type(data)}")
        return

    def _get(keys):
        for key in keys:
            if key in data:
                return data[key]
        return None

    z = _get(["z"])
    if z is None:
        lg.error(f"[SM amp] missing z array in {profile_path}")
        return
    z = np.asarray(z, dtype=float)

    signal_solution = _get(["signal_sol", "signal_solution"])
    if signal_solution is None:
        lg.error(f"[SM amp] missing signal_sol in {profile_path}")
        return
    signal_solution = np.asarray(signal_solution, dtype=float)
    if signal_solution.ndim != 2 or signal_solution.shape[0] != len(z):
        lg.error(f"[SM amp] signal_sol shape mismatch: z={z.shape}, signal_sol={signal_solution.shape}")
        return

    pump_solution = _get(["pump_sol", "pump_solution"])
    if pump_solution is None:
        pump_solution = np.zeros((len(z), 0), dtype=float)
    else:
        pump_solution = np.asarray(pump_solution, dtype=float)
        if pump_solution.ndim == 1:
            if pump_solution.shape[0] != len(z):
                lg.error(f"[SM amp] pump_sol length mismatch: z={z.shape}, pump_sol={pump_solution.shape}")
                return
        elif pump_solution.ndim == 2:
            if pump_solution.shape[0] != len(z):
                lg.error(f"[SM amp] pump_sol shape mismatch: z={z.shape}, pump_sol={pump_solution.shape}")
                return
        else:
            lg.error(f"[SM amp] unsupported pump_sol shape: {pump_solution.shape}")
            return

    signal_wavelengths = _get(["signal_wavelengths", "signal_wavelength"])
    if signal_wavelengths is None:
        freqs = system.wdm.frequency_grid()
        if freqs.size != signal_solution.shape[1]:
            lg.error(
                "[SM amp] missing signal_wavelengths and WDM grid size does not match signal_sol."
            )
            return
        signal_wavelengths = nu2lambda(freqs)
    signal_wavelengths = np.asarray(signal_wavelengths, dtype=float)
    if signal_wavelengths.size != signal_solution.shape[1]:
        lg.error(
            f"[SM amp] signal_wavelengths size mismatch: wl={signal_wavelengths.size}, signals={signal_solution.shape[1]}"
        )
        return

    pump_wavelengths = _get(["pump_wavelengths", "pump_wavelength"])
    if pump_wavelengths is None and pump_solution.size:
        pumps = system.pump_specs or []
        pump_wavelengths = np.array([p.wavelength for p in pumps], dtype=float)
    if pump_wavelengths is None:
        pump_wavelengths = np.zeros((0,), dtype=float)
    pump_wavelengths = np.asarray(pump_wavelengths, dtype=float)

    num_pumps = 0 if pump_solution.size == 0 else (1 if pump_solution.ndim == 1 else pump_solution.shape[1])
    if num_pumps == 0:
        pump_wavelengths = np.zeros((0,), dtype=float)
    elif pump_wavelengths.size != num_pumps:
        lg.error(
            f"[SM amp] pump_wavelengths size mismatch: wl={pump_wavelengths.size}, pumps={num_pumps}"
        )
        return

    ase_initial = _get(["ase_sol", "ase_solution"])
    if ase_initial is not None:
        ase_initial = np.asarray(ase_initial, dtype=float)
        if ase_initial.ndim == 2 and ase_initial.shape[0] == len(z):
            ase_initial = ase_initial[0]
        elif ase_initial.ndim == 2 and ase_initial.shape[1] == len(z):
            ase_initial = ase_initial[:, 0]
        elif ase_initial.ndim != 1:
            lg.warning("[SM amp] ase_sol shape not usable for ase_initial; ignoring.")
            ase_initial = None
        if ase_initial is not None and ase_initial.size != signal_solution.shape[1]:
            lg.warning(
                "[SM amp] ase_initial length does not match number of signals; ignoring."
            )
            ase_initial = None

    reference_bandwidth = _get(["reference_bandwidth", "ref_bandwidth"])
    if reference_bandwidth is None:
        reference_bandwidth = system.baud_rate if system.baud_rate is not None else 0.1
    reference_bandwidth = float(reference_bandwidth)

    temperature = _get(["temperature"])
    if temperature is None:
        temperature = 300.0
    temperature = float(temperature)

    ase_decimation = _get(["ase_decimation", "ase_decimation_factor", "ase_stride"])
    if ase_decimation is None:
        ase_decimation = 1
    ase_decimation = int(ase_decimation)
    if ase_decimation > 1:
        lg.info(
            f"[SM amp] ASE decimation: every {ase_decimation} channels, "
            f"bandwidth scaled to {reference_bandwidth * ase_decimation:.3e}"
        )

    ase_solution = amp.solve_ase_with_fixed_powers(
        pump_solution=pump_solution,
        signal_solution=signal_solution,
        pump_wavelength=pump_wavelengths,
        signal_wavelength=signal_wavelengths,
        z=z,
        ase_initial=ase_initial,
        temperature=temperature,
        reference_bandwidth=reference_bandwidth,
        ase_decimation=ase_decimation,
    )

    lg.info(f"[SM amp] ASE-only solve done: ase_sol={ase_solution.shape}")
    if ase_solution.size:
        lg.info(
            f"[SM amp] ASE range [{ase_solution.min():.3e},{ase_solution.max():.3e}] W"
        )
        decimated_indices = np.arange(0, signal_wavelengths.size, ase_decimation, dtype=int)
        ase_out = ase_solution[-1]
        if isinstance(system.wdm, IrregularWDM):
            for name, slc in system.wdm._band_slices.items():
                band_mask = (decimated_indices >= slc.start) & (decimated_indices < slc.stop)
                if not np.any(band_mask):
                    lg.info(f"[SM amp] ASE avg band {name}: n/a (no decimated channels)")
                    continue
                band_mean_w = float(np.mean(ase_out[band_mask]))
                band_mean_dbm = float(watt2dBm(band_mean_w))
                lg.info(
                    f"[SM amp] ASE avg band {name}: {band_mean_w:.3e} W ({band_mean_dbm:.2f} dBm)"
                )
        else:
            band_mean_w = float(np.mean(ase_out))
            band_mean_dbm = float(watt2dBm(band_mean_w))
            lg.info(
                f"[SM amp] ASE avg (all): {band_mean_w:.3e} W ({band_mean_dbm:.2f} dBm)"
            )
    if output_path is not None:
        payload = dict(data)
        payload["ase_sol_fixed"] = ase_solution
        payload["ase_reference_bandwidth"] = reference_bandwidth
        payload["ase_reference_bandwidth_effective"] = reference_bandwidth * ase_decimation
        payload["ase_temperature"] = temperature
        payload["ase_decimation"] = ase_decimation
        ase_indices = np.arange(0, signal_wavelengths.size, ase_decimation, dtype=int)
        payload["ase_signal_indices"] = ase_indices
        payload["ase_signal_wavelengths"] = signal_wavelengths[ase_indices]
        np.save(output_path, payload)
        lg.info(f"[SM amp] saved ASE payload to {output_path}")

    try:
        from pynlin.raman.plot_optimization import plot_profiles

        if ase_decimation > 1:
            group_slices = [
                slice(idx, min(idx + ase_decimation, signal_solution.shape[1]))
                for idx in range(0, signal_solution.shape[1], ase_decimation)
            ]
            signal_solution_plot = np.stack(
                [signal_solution[:, slc].sum(axis=1) for slc in group_slices], axis=1
            )
            signal_wavelengths_plot = signal_wavelengths[::ase_decimation]
        else:
            signal_solution_plot = signal_solution
            signal_wavelengths_plot = signal_wavelengths

        if signal_solution_plot.ndim == 2:
            signal_solution_plot = signal_solution_plot[:, :, None]
        ase_solution_plot = ase_solution[:, :, None] if ase_solution.ndim == 2 else ase_solution

        pump_wavelengths_plot = pump_wavelengths
        pump_solution_plot = pump_solution
        mode_count = getattr(system, "n_modes", getattr(system.fiber, "n_modes", 1))
        if pump_solution_plot is None or np.size(pump_solution_plot) == 0:
            pump_wavelengths_plot = np.array([np.mean(signal_wavelengths_plot)])
            pump_solution_plot = np.zeros((len(z), 1, mode_count), dtype=float)
        else:
            pump_solution_plot = np.asarray(pump_solution_plot, dtype=float)
            if pump_solution_plot.ndim == 2:
                pump_solution_plot = pump_solution_plot[:, :, None]
            elif pump_solution_plot.ndim == 1:
                pump_solution_plot = pump_solution_plot.reshape((len(z), 1, 1))

        pump_powers = _get(
            ["pump_powers", "pump_power", "pump_power_w", "pump_powers_w"]
        )
        if pump_powers is None:
            pump_powers = np.zeros((pump_solution_plot.shape[1], 1), dtype=float)
        else:
            pump_powers = np.asarray(pump_powers, dtype=float)
            if pump_powers.ndim == 1:
                pump_powers = pump_powers[:, None]

        plot_profiles(
            signal_wavelengths=signal_wavelengths_plot,
            signal_solution=signal_solution_plot,
            ase_solution=ase_solution_plot,
            pump_wavelengths=pump_wavelengths_plot,
            pump_solution=pump_solution_plot,
            pump_powers=pump_powers,
            cf=system,
            wallpaper_mode=False,
            use_active_naming=False,
            plot_title=f"ASE decimation: {ase_decimation}x",
        )
        lg.info("[SM amp] saved ASE plot via plot_profiles.")
    except Exception as exc:
        lg.warning(f"[SM amp] ASE plotting skipped: {exc}")


def main_bak():
    import sys
    from pathlib import Path
    from pynlin.raman.plot_optimization import plot_profiles
    from pynlin.log_init import init_logging
    from loguru import logger as lg
    init_logging()
    # Ensure console logging is enabled even when env is unset
    level = os.getenv("LOGURU_LEVEL", "TRACE")
    lg.remove()
    lg.add(sys.stderr, level=level)
    # Simple smoke test: load system TOML (default smf_struct) and run single-mode amplification
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input/studies.toml")
    try:
        from pynlin.system import System
        from pynlin.utils import dBm2watt, nu2lambda
    except Exception as e:
        lg.error(f"Could not import System: {e}")
        sys.exit(1)

    system = System.from_toml(cfg_path)
    fiber = system.fiber
    wdm = system.wdm
    pumps = system.pump_specs or []

    try:
        out_fig = system.plot_launch_spectrum()
        lg.info(f"Saved launch spectrum plot to {out_fig}")
    except Exception as e:
        lg.warning(f"Launch spectrum plot skipped: {e}")

    freqs = wdm.frequency_grid()
    lg.info(f"Frequency grid: {freqs.shape[0]} points from {freqs.min()/1e12:.2f} THz to {freqs.max()/1e12:.2f} THz")

    # Diagnostic: compare wideband vs narrowband gain matrices on a reduced grid
    try:
        import matplotlib.pyplot as plt
        sample_size = min(200, len(freqs))
        idx = np.linspace(0, len(freqs) - 1, sample_size, dtype=int)
        freq_sample = freqs[idx]

        wide_amp = RamanAmplifier(fiber)
        gm_wide = wide_amp.compute_gain_matrix(freq_sample)

        nb_fiber = SMFiberNarrowband(
            losses=getattr(fiber, "losses", None),
            raman_coefficient=fiber.raman_coefficient,
            effective_area=getattr(fiber, "effective_area", 80e-12),
            beta2=getattr(fiber, "beta2", None),
            gamma=getattr(fiber, "gamma", None),
            length=fiber.length,
        )
        nb_amp = RamanAmplifier(nb_fiber)
        gm_narrow = nb_amp.compute_gain_matrix(freq_sample)

        def plot_matrix(mat, title, fname):
            plt.figure(figsize=(5, 4))
            plt.imshow(mat, origin="lower", aspect="auto")
            plt.colorbar(label="Gain (1/W·m)")
            plt.title(title)
            plt.tight_layout()
            out_path = Path("media/debug") / fname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=200)
            plt.close()
            lg.info(f"Saved gain matrix plot: {out_path}")

        plot_matrix(gm_wide, "Wideband gain matrix", "gain_matrix_wide.png")
        plot_matrix(gm_narrow, "Narrowband gain matrix", "gain_matrix_narrow.png")
        plot_matrix(gm_wide - gm_narrow, "Gain matrix diff (wide - narrow)", "gain_matrix_diff.png")

        # Reference case from the active studies TOML
        try:
            smf_ref = System.from_toml(Path("input/studies.toml"))
            ref_freqs = smf_ref.wdm.frequency_grid()
            ref_idx = np.linspace(0, len(ref_freqs) - 1, min(200, len(ref_freqs)), dtype=int)
            ref_freq_sample = ref_freqs[ref_idx]
            ref_amp = RamanAmplifier(smf_ref.fiber)
            gm_ref = ref_amp.compute_gain_matrix(ref_freq_sample)
            plot_matrix(gm_ref, "Narrowband gain (smf_struct)", "gain_matrix_smf_struct.png")
        except Exception as e:
            lg.warning(f"Reference smf_struct gain plot skipped: {e}")
    except Exception as e:
        lg.warning(f"Gain matrix diagnostic skipped: {e}")

    # Boundary-value solve only (signals forward, pumps backward)
    z = np.linspace(0, fiber.length, 400)
    amp = SMWidebandRamanAmplifier(fiber)
    launch_dbm_default = system.launch_power if system.launch_power is not None else -5.0
    # Decimate signals (20x) via WDM helper; no per-channel power rescale
    signal_power_dbm = amp._band_launch_powers(system, launch_dbm_default)
    signal_power_w = dBm2watt(signal_power_dbm)
    sig_wl_full = nu2lambda(freqs)
    wdm_dec = wdm.decimate(2, rescale_power=True)
    freqs_dec = wdm_dec.frequency_grid()
    sig_wl_dec = nu2lambda(freqs_dec)
    sig_power_dec = signal_power_w[: len(sig_wl_dec)]
    power_scale = len(sig_wl_full) / len(sig_wl_dec) if len(sig_wl_dec) else 1.0
    # if decim > 1 and power_scale != 1.0:
    #     sig_power_dec = sig_power_dec * power_scale
    #     lg.info(f"[BVP] Decimating signals by {decim} (kept {len(sig_wl_dec)} of {len(sig_wl_full)}); power scale {power_scale:.2f}")
    # else:
    #     lg.info(f"[BVP] Decimating signals by {decim} (kept {len(sig_wl_dec)} of {len(sig_wl_full)}) without power scaling")
    pump_wavelengths = np.array([p.wavelength for p in pumps]) if pumps else np.array([])
    pump_power_w = dBm2watt(np.array([p.power_dbm for p in pumps])) if pumps else np.array([])
    # guard against zero/negative powers before BVP
    pump_power_w = np.maximum(pump_power_w, 1e-12) if pump_power_w.size else pump_power_w
    sig_power_dec = np.maximum(sig_power_dec, 1e-12)
    pump_sol, sig_sol, _ = amp.solve_bvp_signals_forward_pumps_backward(
        signal_power=sig_power_dec,
        signal_wavelength=sig_wl_dec,
        pump_power=pump_power_w,
        pump_wavelength=pump_wavelengths,
        z=z,
        tol=10,
        power_cap=1e3,
    )
    sig_finite = np.isfinite(sig_sol)
    pump_finite = np.isfinite(pump_sol) if pump_sol.size else np.array([True])
    lg.info(
        f"[BVP] pump {pump_sol.shape}, signal {sig_sol.shape}; "
        f"signal finite {sig_finite.mean():.4f}, pump finite {pump_finite.mean():.4f}"
    )
    lg.info(
        f"[BVP] signal min/max dBm [{watt2dBm(np.nanmin(sig_sol)):.2f}, {watt2dBm(np.nanmax(sig_sol)):.2f}]"
    )
    if pump_sol.size:
        lg.info(
            f"[BVP] pump min/max dBm [{watt2dBm(np.nanmin(pump_sol)):.2f}, {watt2dBm(np.nanmax(pump_sol)):.2f}]"
        )

    try:
        pump_power_dbm = np.array([p.power_dbm for p in pumps]) if pumps else np.array([])
        pump_power_w = dBm2watt(pump_power_dbm) if pump_power_dbm.size else np.array([])

        # Build mock pump data if none were present so plots can proceed
        if pump_sol.size == 0 or pump_power_w.size == 0:
            mode_count = getattr(system, "n_modes", getattr(system.fiber, "n_modes", 1))
            pump_wavelengths = np.array([nu2lambda(np.median(freqs))])
            pump_solution = np.zeros((len(z), 1, mode_count))
            pump_powers = np.zeros((1, 1))
            lg.warning("No pumps in solve; using mock zero-power pump for plotting.")
        else:
            pump_wavelengths = np.array([p.wavelength for p in pumps])
            pump_solution = pump_sol[:, :, None] if pump_sol.ndim == 2 else pump_sol
            pump_powers = pump_power_w[:, None]

        signal_solution = sig_sol[:, :, None] if sig_sol.ndim == 2 else sig_sol
        signal_solution_dec = signal_solution

        plot_profiles(
            signal_wavelengths=sig_wl_dec,
            signal_solution=signal_solution,
            ase_solution=None,
            pump_wavelengths=pump_wavelengths,
            pump_solution=pump_solution,
            pump_powers=pump_powers,
            cf=system,
            wallpaper_mode=False,
            use_active_naming=False
        )
        lg.info("Saved smoke-test profiles via plot_profiles.")

        print("Saved BVP profiles via plot_profiles.")
    except Exception as e:
        lg.error(f"Plotting skipped due to error: {e}")


if __name__ == "__main__":
    main()
