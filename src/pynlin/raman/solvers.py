from typing import Tuple
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
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
from pynlin.fiber import Fiber, MMFiber
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

        losses_linear = self.get_linear_losses(wavelengths)

        gain_matrix = self.compute_gain_matrix(frequencies)
        lg.debug(f"[SM amp] gain matrix shape: {gain_matrix.shape}")

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

        kw = dict(rtol=1e-3, atol=1e-3, mxstep=10000)
        if isinstance(odeint_kwargs, dict):
            kw.update(odeint_kwargs)

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
            if solver in ("ivp", "bdf", "BDF", "radau", "Radau"):
                # Use stiff solver for better robustness (Radau)
                def _ode(z_val, P):
                    P = np.asarray(P)
                    dPdz = (-losses_linear + gain_matrix @ P) * P
                    return dPdz * self.direction

                max_step = min(
                    1e100,
                    (z[-1]-z[0]) / 2000 if len(z) > 1 else np.inf,
                )
                ivp = scipy.integrate.solve_ivp(
                    _ode,
                    (z[0], z[-1]),
                    input_power,
                    method="Radau",
                    t_eval=z,
                    rtol=kw.get("rtol", 1e-3),
                    atol=kw.get("atol", 1e-6),
                    max_step=max_step,
                )
                if not ivp.success:
                    lg.error(f"[SM amp] Radau solver failed: {ivp.message}")
                else:
                    lg.debug(f"[SM amp] Radau steps: {ivp.nfev} f evals, {ivp.t.size} outputs")
                sol = ivp.y.T
            else:
                sol = scipy.integrate.odeint(
                    RamanAmplifier.raman_ode,
                    input_power,
                    z,
                    args=(losses_linear, gain_matrix, np.hstack((self.direction,))),
                    **kw,
                )
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

    def compute_gain_matrix(self, frequencies):
        """Generate the matrix of Raman gains between each pair of frequencies."""
        frequencies = np.asarray(frequencies, dtype=float)
        resolution = self._gain_resolution(frequencies)
        cache_key = (frequencies.tobytes(), id(self.fiber), float(resolution))
        if cache_key in _GAIN_MATRIX_CACHE:
            return _GAIN_MATRIX_CACHE[cache_key]

        num_frequencies = len(frequencies)
        frequency_shifts = frequencies[:, None] - frequencies[None, :]

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
        freq_scaling = np.maximum(1.0, freqs * (1 / freqs.T))
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
        except ValueError:
            breakpoint()
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

    def solve_shooting_scipy(self, pump_power, signal_power, z):
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
        signal_losses = self.get_linear_losses(self.signal_wavelengths)

        direction_tmp = np.ones_like(sorted_pump_powers)

        sol = scipy.integrate.odeint(
            RamanAmplifier.raman_ode,
            sorted_pump_powers,
            z,
            args=(sorted_pump_losses, gain_matrix, direction_tmp),
        )

        pump_solution_initial_cond = sol[::-1, :num_pumps]

        # signal_solution_initial_cond = sorted_pump_powers * np.exp(
        #     -z[::-1, np.newaxis] * sorted_pump_losses
        # )

        signal_solution_initial_cond = self.signal_power * np.exp(
            -z[:, np.newaxis] * signal_losses
        )

        # plt.figure()
        # plt.plot(z, pynlin.utils.watt2dBm(signal_solution_initial_cond))
        # plt.plot(z, pynlin.utils.watt2dBm(pump_solution_initial_cond))
        # # plt.show()

        initial_conditions = np.hstack(
            (pump_solution_initial_cond, signal_solution_initial_cond)
        ).T

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

        result = scipy.integrate.solve_bvp(
            ode, boundary_residuals, z, initial_conditions, verbose=1
        )
        return result.y.transpose()

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

            print(f"Iteration {num_iter}")
            print(f"\tx0={pynlin.utils.watt2dBm(x0)} dBm")
            print(f"\tError={D * 1e3} mW")
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

                # print(f"\tdelta_P={delta_P * 1e3} mW")
                # x0 = x0 + alpha * delta_P
                x0 = x0 - eta * G
                print(f"\tx0_new={pynlin.utils.watt2dBm(x0)} dBm")
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
                    print(f"Max error: {max_error} mW")
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

# Backward compatibility
RamanAmplifier = SMRamanAmplifier
MMFRamanAmplifier = MMRamanAmplifier
SMRamanAmplifierWideband = SMWidebandRamanAmplifier


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from pynlin.raman.plot_optimization import plot_profiles
    from pynlin.log_init import init_logging
    from loguru import logger as lg
    init_logging()
    # Ensure console logging is enabled even when env is unset
    level = os.getenv("LOGURU_LEVEL", "DEBUG")
    lg.remove()
    lg.add(sys.stderr, level=level)
    # Simple smoke test: load system TOML (default smf_struct) and run single-mode amplification
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input/dummy_struct.toml")
    try:
        from pynlin.system import System
        from pynlin.utils import dBm2watt, nu2lambda
    except Exception as e:
        print(f"Could not import System: {e}")
        sys.exit(1)

    system = System.from_toml(cfg_path)
    fiber = system.fiber
    wdm = system.wdm
    pumps = system.pump_specs or []

    try:
        out_fig = system.plot_launch_spectrum()
        print(f"Saved launch spectrum plot to {out_fig}")
    except Exception as e:
        print(f"Launch spectrum plot skipped: {e}")

    freqs = wdm.frequency_grid()
    print(f"Frequency grid: {freqs.shape[0]} points from {freqs.min()/1e12:.2f} THz to {freqs.max()/1e12:.2f} THz")

    # Build per-band launch powers automatically
    z = np.linspace(0, fiber.length, 400)
    amp = SMWidebandRamanAmplifier(fiber)
    pump_sol, sig_sol, _ = amp.solve_from_system(
        system,
        z=z,
        disable_pumps=False,
        solver="ivp",
        odeint_kwargs={"rtol": 1e-2, "atol": 1e-4, "mxstep": 500},
        ase=False,
        pump_power_floor=1e-6,
    )
    print(f"Pump solution shape: {pump_sol.shape}, Signal solution shape: {sig_sol.shape}")
    sig_finite = np.isfinite(sig_sol)
    pump_finite = np.isfinite(pump_sol) if pump_sol.size else np.array([True])
    print(
        f"Signal finite fraction: {sig_finite.mean():.4f}, "
        f"min/max (dBm) [{watt2dBm(np.nanmin(sig_sol)):.2f}, {watt2dBm(np.nanmax(sig_sol)):.2f}]"
    )
    if pump_sol.size:
        print(
            f"Pump finite fraction: {pump_finite.mean():.4f}, "
            f"min/max (dBm) [{watt2dBm(np.nanmin(pump_sol)):.2f}, {watt2dBm(np.nanmax(pump_sol)):.2f}]"
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
            print("No pumps in solve; using mock zero-power pump for plotting.")
        else:
            pump_wavelengths = np.array([p.wavelength for p in pumps])
            pump_solution = pump_sol[:, :, None] if pump_sol.ndim == 2 else pump_sol
            pump_powers = pump_power_w[:, None]

        signal_solution = sig_sol[:, :, None] if sig_sol.ndim == 2 else sig_sol

        plot_profiles(
            signal_wavelengths=nu2lambda(freqs),
            signal_solution=signal_solution,
            ase_solution=None,
            pump_wavelengths=pump_wavelengths,
            pump_solution=pump_solution,
            pump_powers=pump_powers,
            cf=system,
            wallpaper_mode=False,
        )
        print("Saved smoke-test profiles via plot_profiles.")
    except Exception as e:
        print(f"Plotting skipped due to error: {e}")
