"""
solvers_homotopy_continuation.py

Self-contained implementation of the fast unidirectional power-profile estimator with
homotopy/continuation intended for very high Raman pump powers (robustness first).

Design intent
-------------
- Provide a dedicated SMF wideband amplifier class (SMFWidebandAmplifier) that can be
  called from the existing solvers.py __main__ without invasive refactors.
- Reuse existing gain/loss machinery by importing RamanAmplifier / SMWidebandRamanAmplifier
  from solvers.py at runtime (no circular import if you import this file only inside
  the solvers.py __main__ block).

Core algorithm
--------------
- Inner loop: Jiang-style unidirectional fixed-point iteration with pump rescaling to
  meet backward pump boundary conditions at z=L.
- Outer loop: homotopy continuation that scales pump targets from a small factor to 1,
  using the previous solution as initialization.

Notes
-----
- This implementation targets ASE=False. If ASE is required, keep using the existing
  ODE-based path in solvers.py.
- This implementation assumes z is (approximately) uniform. If not, we internally
  remesh to uniform spacing and interpolate back.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from loguru import logger as lg
from scipy.constants import lambda2nu, nu2lambda
from pynlin.raman.solvers import SMWidebandRamanAmplifier

# Local project imports: we keep them optional to make this file "self-sufficient" within the repo.
# The only hard dependency is that solvers.py exists and exposes RamanAmplifier + SMWidebandRamanAmplifier.
try:
    from pynlin.utils import dBm2watt, watt2dBm
    from pynlin.wdm import IrregularWDM
except Exception:  # pragma: no cover
    dBm2watt = None
    watt2dBm = None
    IrregularWDM = None


@dataclass
class JiangIterativeConfig:
    """
    Configuration for the Jiang-style unidirectional solver + homotopy.

    inner_iters:
        If set, force this many inner iterations. If None, choose automatically from ts_dB rule.
    delta_db_max:
        Initial (max) dB increment per inner iteration before normalization to match total ts_dB.
    recovery_db_per_100_iter:
        How many dB are "recovered" over 100 iterations in the schedule (paper rule-of-thumb).
    early_stop_rtol:
        If not None, enable early stopping when max relative change in signals is below this tolerance.
    pump_power_floor:
        Clamp pumps everywhere to at least this value (W) to avoid division-by-zero during rescaling.
    homotopy_steps:
        Number of outer continuation steps from pump_scale_start to 1.0.
    pump_scale_start:
        Starting scaling factor for pump powers in homotopy. Must be >0 and <=1.
    """

    inner_iters: Optional[int] = None
    delta_db_max: float = 0.2
    recovery_db_per_100_iter: float = 10.0
    early_stop_rtol: Optional[float] = 1e-4
    pump_power_floor: float = 1e-12

    iterative_steps: int = 10
    pump_scale_start: float = 1e-3

    # Safety / numerics
    max_exponent: float = 700.0  # exp(700) ~ 1e304 (avoid overflow)
    nan_guard: bool = True


def _is_uniform_grid(z: np.ndarray, rtol: float = 1e-9, atol: float = 0.0) -> bool:
    if z.size < 3:
        return True
    dz = np.diff(z)
    return np.allclose(dz, dz[0], rtol=rtol, atol=atol)


def _uniformize_grid(z: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return uniform grid with same endpoints and same length; also return dz."""
    if z.size < 2:
        return z.copy(), 0.0
    z0, z1 = float(z[0]), float(z[-1])
    M = int(z.size)
    z_u = np.linspace(z0, z1, M)
    dz = float(z_u[1] - z_u[0]) if M > 1 else 0.0
    return z_u, dz


def _cumtrapz_rows(P: np.ndarray, dz: float) -> np.ndarray:
    """
    Cumulative trapezoid integration along axis=1 for each row.

    P shape: (N, M)
    Returns I shape: (N, M) where I[:,0]=0 and I[:,m]=int_0^{z_m} P(xi) dxi.
    """
    if P.ndim != 2:
        raise ValueError(f"P must be 2D (N,M), got {P.shape}")
    N, M = P.shape
    I = np.zeros((N, M), dtype=float)
    if M <= 1:
        return I
    # Increment: I[m] = I[m-1] + 0.5*(P[m-1]+P[m])*dz
    I[:, 1:] = np.cumsum(0.5 * (P[:, :-1] + P[:, 1:]) * dz, axis=1)
    return I


def _array_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Return small, nan-aware stats for logging."""
    arr_np = np.asarray(arr)
    finite_mask = np.isfinite(arr_np)
    finite_vals = arr_np[finite_mask]
    stats: Dict[str, Any] = {
        "shape": arr_np.shape,
        "size": int(arr_np.size),
        "nan_count": int(np.isnan(arr_np).sum()),
        "finite": bool(np.all(finite_mask)),
    }
    if finite_vals.size:
        stats.update(
            {
                "min": float(np.min(finite_vals)),
                "max": float(np.max(finite_vals)),
                "mean": float(np.mean(finite_vals)),
            }
        )
    else:
        stats.update({"min": None, "max": None, "mean": None})
    return stats


def _log_profile_stats(label: str, arr: np.ndarray, level: str = "info") -> None:
    """Lightweight logger for array stats."""
    stats = _array_stats(arr)
    msg = (
        f"{label} shape={stats['shape']}, size={stats['size']}, "
        f"nan={stats['nan_count']}, finite={stats['finite']}"
    )
    if stats["min"] is not None:
        msg += f", min={stats['min']:.3e}, max={stats['max']:.3e}, mean={stats['mean']:.3e}"
    else:
        msg += ", min/max undefined (no finite entries)"
    log_fn = getattr(lg, level, lg.info)
    log_fn(msg)


def _to_dbm_safe(power_w: np.ndarray, floor: float = 1e-18) -> np.ndarray:
    """Convert power in W to dBm with a small floor to avoid -inf."""
    power = np.maximum(np.asarray(power_w, dtype=float), floor)
    if watt2dBm is not None:
        return watt2dBm(power)
    return 10.0 * np.log10(power / 1e-3)


def _log_power_stats(label: str, power_w: np.ndarray, level: str = "info", floor: float = 1e-18) -> None:
    """Log stats for power in both W and dBm to spot unit issues quickly."""
    _log_profile_stats(f"{label} [W]", power_w, level=level)
    # power_dbm = _to_dbm_safe(power_w, floor=floor)
    # stats_dbm = _array_stats(power_dbm)
    # msg = (
    #     f"{label} [dBm] shape={stats_dbm['shape']}, size={stats_dbm['size']}, "
    #     f"nan={stats_dbm['nan_count']}, finite={stats_dbm['finite']}"
    # )
    # if stats_dbm["min"] is not None:
    #     msg += (
    #         f", min={stats_dbm['min']:.2f}, max={stats_dbm['max']:.2f}, "
    #         f"mean={stats_dbm['mean']:.2f}"
    #     )
    # else:
    #     msg += ", min/max undefined (no finite entries)"
    # log_fn = getattr(lg, level, lg.info)
    # log_fn(msg)


def _plot_gain_matrix(G: np.ndarray, wl_all: np.ndarray, out_dir: Path) -> Optional[Path]:
    """Plot gain matrix heatmap to a file; return path or None if plotting fails."""
    if G.size == 0:
        lg.warning("[Jiang/Init] Skipping gain matrix plot (empty matrix)")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting optional
        lg.warning(f"Skipping gain matrix plot (matplotlib not available): {exc}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"gain_matrix_{ts}.png"

    wl_nm = wl_all * 1e9
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(G, origin="lower", aspect="auto")
    n_ticks = min(6, wl_nm.size)
    tick_idx = np.linspace(0, wl_nm.size - 1, n_ticks, dtype=int) if wl_nm.size else np.array([], dtype=int)
    if tick_idx.size:
        tick_labels = [f"{wl_nm[i]:.1f}" for i in tick_idx]
        ax.set_xticks(tick_idx)
        ax.set_yticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Channel index (lambda nm)")
    ax.set_ylabel("Channel index (lambda nm)")
    ax.set_title("Raman gain matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Gain coeff.")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    lg.info(f"[Jiang/Init] Saved gain matrix plot to {out_path}")
    return out_path


class SMFWidebandAmplifier:
    """
    Dedicated wideband SMF Raman amplifier using Jiang 2025-style unidirectional solver with homotopy.

    This class is intentionally thin: it delegates gain/loss computation to the existing
    RamanAmplifier implementation in solvers.py to avoid duplicating wideband Raman gain logic.
    """

    def __init__(self, fiber, response_bandwidth: float = 40e12):
        # Runtime import avoids circular imports when solvers.py imports this file in __main__.
        from pynlin.raman.solvers import SMWidebandRamanAmplifier  

        self._amp = SMWidebandRamanAmplifier(fiber, response_bandwidth=response_bandwidth)
        self.fiber = fiber
        
    @staticmethod
    def _band_launch_powers(system, default_dbm: float) -> np.ndarray:
        """Return per-channel launch powers in dBm, honoring band overrides."""
        if hasattr(system, "_initial_signal_powers_dbm"):
            powers_dbm = system._initial_signal_powers_dbm()
            lg.info(f"[Jiang/Init] Signal launch powers per channel (dBm): {powers_dbm}")
            return powers_dbm

        wdm = system.wdm
        n_ch = getattr(wdm, "num_channels", 0)
        powers_dbm = np.full(n_ch, default_dbm, dtype=float)
        if IrregularWDM is not None and isinstance(wdm, IrregularWDM):
            for name, slc in getattr(wdm, "_band_slices", {}).items():
                spec = wdm.band_specs.get(name)
                if spec and spec.launch_power_dbm is not None:
                    powers_dbm[slc] = spec.launch_power_dbm
        # power_scale = getattr(wdm, "power_scale", 1.0)
        # if power_scale not in (None, 1.0):
        #     lg.info(f"[Jiang/Init] Detected WDM power_scale={power_scale:.3f}; undoing per-channel boost.")
        #     powers_dbm = powers_dbm - 10 * np.log10(power_scale)
        lg.info(f"[Jiang/Init] Signal launch powers per channel (dBm): {powers_dbm}")
        return powers_dbm

    def solve_with_jiang(
        self,
        system,
        z: np.ndarray,
        cfg: Optional[JiangIterativeConfig] = None,
        disable_pumps: bool = False,
        **kwargs,
    ):
        """
        Entry point analogous to SMWidebandRamanAmplifier.solve_from_system(), but uses homotopy solver. kwargs are reserved for future compatibility; currently unused to keep behavior explicit.
        """
        if cfg is None:
            cfg = JiangIterativeConfig()

        wdm = system.wdm
        pumps = [] if disable_pumps else (system.pump_specs or [])

        freqs = wdm.frequency_grid()
        sig_wl = nu2lambda(freqs)

        launch_dbm_default = system.launch_power if system.launch_power is not None else -5.0
        sig_dbm = self._band_launch_powers(system, launch_dbm_default)
        sig_w = dBm2watt(sig_dbm)

        pump_wl = np.array([p.wavelength for p in pumps], dtype=float) if pumps else np.array([], dtype=float)
        pump_target_w = dBm2watt(np.array([p.power_dbm for p in pumps], dtype=float)) if pumps else np.array([], dtype=float)
        pump_dir = np.array([p.direction for p in pumps], dtype=float) if pumps else np.array([], dtype=float)

        return self.solve_iterative(
            signal_power=sig_w,
            signal_wavelength=sig_wl,
            pump_power_target=pump_target_w,
            pump_wavelength=pump_wl,
            z=z,
            pump_direction=pump_dir,
            cfg=cfg,
        )

    def solve_iterative(
        self,
        signal_power: np.ndarray,
        signal_wavelength: np.ndarray,
        pump_power_target: np.ndarray,
        pump_wavelength: np.ndarray,
        z: np.ndarray,
        pump_direction=1,
        cfg: Optional[JiangIterativeConfig] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve with outer homotopy on pump power targets and inner Jiang-style unidirectional iteration.

        Returns
        -------
        pump_solution: (M, Np)
        signal_solution: (M, Ns)
        ase_solution: empty (M, 0)
        """
        if cfg is None:
            cfg = JiangIterativeConfig()

        # Grid handling
        z_in = np.asarray(z, dtype=float)
        z_u, dz = (z_in, float(z_in[1] - z_in[0])) if _is_uniform_grid(z_in) else _uniformize_grid(z_in)
        if z_u.size < 2:
            raise ValueError("z must have at least 2 points.")

        sigP = np.asarray(signal_power, dtype=float).reshape(-1)
        sigW = np.asarray(signal_wavelength, dtype=float).reshape(-1)
        pumP_tgt = np.asarray(pump_power_target, dtype=float).reshape(-1)
        pumW = np.asarray(pump_wavelength, dtype=float).reshape(-1)

        Ns = sigP.size
        Np = pumP_tgt.size
        N = Ns + Np
        if sigW.size != Ns:
            raise ValueError("signal_power and signal_wavelength size mismatch.")
        if pumW.size != Np:
            raise ValueError("pump_power_target and pump_wavelength size mismatch.")

        # Directions
        sig_dir = np.ones(Ns, dtype=float)
        if Np == 0:
            pump_dir = np.array([], dtype=float)
        elif np.isscalar(pump_direction):
            pump_dir = np.sign(float(pump_direction)) * np.ones(Np, dtype=float)
        else:
            pump_dir = np.sign(np.asarray(pump_direction, dtype=float).reshape(-1))
            if pump_dir.size != Np:
                raise ValueError("pump_direction length mismatch with pumps.")
        direction = np.concatenate([pump_dir, sig_dir], axis=0)  # shape (N,)
        lg.debug(f"[Jiang/Init] channel counts: signals={Ns}, pumps={Np}, total={N}, co-prop: {int(np.sum(direction>0))}, counter-prop: {int(np.sum(direction<0))}")

        # Assemble wavelength grid
        wl_all = np.concatenate([pumW, sigW], axis=0)
        freqs_all = lambda2nu(wl_all)

        # Gain / loss from existing implementation (wideband & Aeff-aware)
        losses = self._amp.get_linear_losses(wl_all)  # shape (N,)
        G = self._amp.compute_gain_matrix(freqs_all)  # shape (N,N)
        # np.fill_diagonal(G, 0.0) # TODO superfluous

        lg.info("[Jiang/Init] Starting homotopy solve")
        lg.debug(f"[Jiang/Init] cfg={cfg}")
        lg.debug(
            f"[Jiang/Init] grid points={z_in.size}, z0={z_in[0]:.3e}, zL={z_in[-1]:.3e}, uniform={_is_uniform_grid(z_in)}, dz={dz:.3e}"
        )
        span_m = float(z_in[-1] - z_in[0])
        lg.info(f"[Jiang/Init] span length={span_m:.3e} m ({span_m*1e-3:.3f} km), dz={dz:.3e} m")
        _log_profile_stats("[Jiang/Init] signal launch W", sigP, level="debug")
        _log_profile_stats("[Jiang/Init] pump target W", pumP_tgt, level="debug")
        _log_profile_stats("[Jiang/Init] pump direction", pump_dir if Np else np.array([]), level="debug")
        _log_profile_stats("[Jiang/Init] wavelength grid (m)", wl_all, level="debug")
        _log_profile_stats("[Jiang/Init] losses (1/m)", losses, level="debug")
        if losses.size:
            losses_db_per_m = losses * (10.0 / np.log(10.0))
            _log_profile_stats("[Jiang/Init] losses (dB/m)", losses_db_per_m, level="debug")
            _log_profile_stats("[Jiang/Init] losses (dB/km)", losses_db_per_m * 1e3, level="debug")
        _log_profile_stats("[Jiang/Init] gain matrix entries", G, level="debug")
        if Np:
            n_forward = int(np.sum(pump_dir > 0))
            n_backward = int(np.sum(pump_dir < 0))
            lg.info(f"[Jiang/Init] pumps forward={n_forward}, backward={n_backward} (sign on direction only)")
        _plot_gain_matrix(G, wl_all, out_dir=Path("logs") / "homotopy_debug")

        # Index helpers
        pump_idx = np.arange(0, Np, dtype=int)
        sig_idx = np.arange(Np, N, dtype=int)

        # Outer iterative schedule for pump targets
        if cfg.iterative_steps < 1:
            raise ValueError("cfg.iterative_steps must be >=1")
        if not (0.0 < cfg.pump_scale_start <= 1.0):
            raise ValueError("cfg.pump_scale_start must be in (0,1].")

        lam = np.geomspace(cfg.pump_scale_start, 1.0, cfg.iterative_steps) if Np else np.array([1.0])
        lg.debug(f"[Jiang/Init] iteration lambda schedule={lam}")
        P_prev_init = None

        pump_sol_u = np.zeros((z_u.size, Np), dtype=float)
        sig_sol_u = np.zeros((z_u.size, Ns), dtype=float)

        for s, lmb in enumerate(lam):
            pumP = pumP_tgt * float(lmb)
            lg.info(f" step {s+1}/{lam.size}: pump scale lambda={lmb:.3e}")
            _log_profile_stats(f" step {s+1} pump target W", pumP, level="debug")

            P_u = self._solve_jiang_unidirectional_inner(
                sigP=sigP,
                pumP=pumP,
                losses=losses,
                G=G,
                z=z_u,
                dz=dz,
                direction=direction,
                pump_idx=pump_idx,
                sig_idx=sig_idx,
                cfg=cfg,
                P_init=P_prev_init,
            )

            # Keep for warm-start next outer step
            P_prev_init = P_u
            _log_power_stats(f" step {s+1} combined profile", P_u, level="info")
            if Np:
                _log_power_stats(f" step {s+1} pumps", P_u[pump_idx, :], level="debug")
            _log_power_stats(f" step {s+1} signals", P_u[sig_idx, :], level="debug")

        if Np:
            pump_sol_u = P_u[pump_idx, :].T
        sig_sol_u = P_u[sig_idx, :].T

        # If we remeshed, interpolate back to original z
        if z_u is not z_in:
            pump_sol = np.zeros((z_in.size, Np), dtype=float)
            sig_sol = np.zeros((z_in.size, Ns), dtype=float)
            for i in range(Np):
                pump_sol[:, i] = np.interp(z_in, z_u, pump_sol_u[:, i])
            for i in range(Ns):
                sig_sol[:, i] = np.interp(z_in, z_u, sig_sol_u[:, i])
        else:
            pump_sol = pump_sol_u
            sig_sol = sig_sol_u

        ase_sol = np.empty((z_in.size, 0), dtype=float)
        _log_power_stats("[Jiang/Result] pump_sol", pump_sol, level="info")
        _log_power_stats("[Jiang/Result] sig_sol", sig_sol, level="info")
        return pump_sol, sig_sol, ase_sol

    def _solve_jiang_unidirectional_inner(
        self,
        sigP: np.ndarray,
        pumP: np.ndarray,
        losses: np.ndarray,
        G: np.ndarray,
        z: np.ndarray,
        dz: float,
        direction: np.ndarray,
        pump_idx: np.ndarray,
        sig_idx: np.ndarray,
        cfg: JiangIterativeConfig,
        P_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Inner Jiang unidirectional iteration with pump boundary enforcement by rescaling.

        Returns P of shape (N, M) where M=len(z).
        """
        M = z.size
        Np = pump_idx.size
        Ns = sig_idx.size
        N = Np + Ns
        span_m = float(z[-1] - z[0])
        lg.info(f" grid: M={M}, dz={dz:.3e} m, span={span_m:.3e} m ({span_m*1e-3:.3f} km)")

        # Build boundary vector at z=0 (signals fixed; pumps unknown but will be iteratively determined)
        # Use initial guess for pumps at z=0 derived from backward loss-only anchored at z=L.
        Pin0 = np.zeros(N, dtype=float)
        Pin0[sig_idx] = sigP
        if Np:
            L = float(z[-1] - z[0])
            # physical backward: P(z) = P(L) * exp(-loss*(L - z)) => at z=0: P0 = P(L)*exp(-loss*L)
            Pin0[pump_idx] = np.maximum(pumP * np.exp(-losses[pump_idx] * L), cfg.pump_power_floor)
        _log_power_stats(" Pin0 (signals+init pumps)", Pin0, level="info")

        # Initialize full profiles
        if P_init is not None and P_init.shape == (N, M):
            P = np.maximum(np.array(P_init, dtype=float, copy=True), cfg.pump_power_floor)
            # Ensure signals at z=0 are exactly correct
            P[sig_idx, 0] = sigP
            lg.debug(" Using warm-start P_init")
        else:
            P = np.zeros((N, M), dtype=float)
            # Signals: forward loss-only
            P[sig_idx, :] = sigP[:, None] * np.exp(-losses[sig_idx, None] * (z[None, :] - z[0]))
            # Pumps: backward loss-only anchored at z=L
            if Np:
                L = float(z[-1] - z[0])
                P[pump_idx, :] = pumP[:, None] * np.exp(-losses[pump_idx, None] * (L - (z[None, :] - z[0])))
                P[pump_idx, :] = np.maximum(P[pump_idx, :], cfg.pump_power_floor)
            lg.debug(" Initialized P with loss-only profiles")
        _log_power_stats(" P initial", P, level="info")

        # --- Jiang "ts_dB" initial down-scaling to avoid divergence when pumps dominate ---
        if Np:
            sum_sig0 = float(np.sum(sigP))
            sum_pumpL = float(np.sum(P[pump_idx, -1]))
            if sum_pumpL > 0 and sum_sig0 > 0 and sum_pumpL > sum_sig0:
                scale0 = sum_sig0 / sum_pumpL
                scale0 = float(np.clip(scale0, 1e-12, 1.0))
                P[pump_idx, :] *= scale0
                ts_db = -10.0 * np.log10(scale0)
            else:
                ts_db = 0.0
        else:
            ts_db = 0.0
        lg.info(f" Initial total pump scaling ts_dB={ts_db:.3f} dB")
        
        # Iteration count
        if cfg.inner_iters is None:
            liter = max(1, int(np.ceil(100.0 * ts_db / float(cfg.recovery_db_per_100_iter)))) if ts_db > 0 else 1
        else:
            liter = int(cfg.inner_iters)

        # Delta schedule normalized to total ts_db
        if ts_db > 0 and liter > 1:
            raw = np.linspace(cfg.delta_db_max, 0.0, liter, dtype=float)
            sraw = float(np.sum(raw))
            delta_db = raw * (ts_db / sraw) if sraw > 0 else np.zeros_like(raw)
        else:
            delta_db = np.zeros(liter, dtype=float)

        cum_db = np.cumsum(delta_db)
        if Np:
            pump_ref0 = np.array(P[pump_idx, -1], copy=True)
        else:
            pump_ref0 = np.array([], dtype=float)

        lg.info(
            f" ts_dB={ts_db:.3f}, liter={liter}, dz={dz:.3e}, M={M}, N={N}, pump_ref0(L) sum={float(np.sum(pump_ref0)) if pump_ref0.size else 0.0:.3e}"
        )
        if delta_db.size:
            _log_profile_stats(" delta_db schedule per iter", delta_db, level="debug")
            _log_profile_stats(" cumulative delta_db", cum_db, level="debug")
        if pump_ref0.size:
            _log_power_stats(" pump_ref0 at z=L", pump_ref0, level="info")

        # Fixed-point iterations
        last_sig = None
        for k in range(liter):
            lg.info(f" iteration {k+1}/{liter}")
            # Integrals of current powers
            I = _cumtrapz_rows(P, dz)  # (N,M)
            S = G @ I  # (N,M)
            _log_profile_stats(f" iter {k+1} integral I (W*m)", I, level="debug")
            _log_profile_stats(f" iter {k+1} source S (dimensionless)", S, level="debug")

            # Effective "input" at z=0 for this iteration
            Pin_k = np.array(P[:, 0], copy=False)

            # Exponent: direction * (-loss*z + S)
            z_rel = (z - z[0])[None, :]  # (1,M)
            expo_raw = (-losses[:, None] * direction[:, None] * z_rel) + (direction[:, None] * S)
            _log_profile_stats(f" iter {k+1} exponent raw", expo_raw, level="debug")

            # Clamp exponent to avoid overflow (still allows huge gain but prevents NaNs)
            expo = np.clip(expo_raw, -cfg.max_exponent, cfg.max_exponent)
            _log_profile_stats(f" iter {k+1} exponent clipped", expo, level="debug")

            P_tilde = Pin_k[:, None] * np.exp(expo)
            _log_power_stats(f" iter {k+1} P_tilde pre-boundary", P_tilde, level="info")
            _log_power_stats(f" iter {k+1} P_tilde z=0", P_tilde[:, 0], level="debug")
            _log_power_stats(f" iter {k+1} P_tilde z=L", P_tilde[:, -1], level="debug")

            if cfg.nan_guard:
                if not np.all(np.isfinite(P_tilde)):
                    bad = np.argwhere(~np.isfinite(P_tilde))[0]
                    raise FloatingPointError(f"Non-finite in Jiang inner iteration at k={k}, idx={tuple(bad)}")

            # Enforce signal boundary exactly
            P_tilde[sig_idx, 0] = sigP

            # Pump boundary enforcement by rescaling at z=L towards reference
            if Np:
                # reference at this iteration (moves toward pumP via ts_db schedule)
                # If ts_db==0, pump_ref equals current target at L (after initialization)
                ref = pump_ref0 * (10.0 ** (cum_db[k] / 10.0)) if ts_db > 0 else np.array(P_tilde[pump_idx, -1], copy=True)

                # If homotopy pumP is specified, we want the final boundary to match pumP.
                # Make ref never exceed the desired target (numerical safety for early k):
                ref = np.minimum(ref, pumP)

                out = P_tilde[pump_idx, -1]
                out = np.maximum(out, cfg.pump_power_floor)
                scale_vec = ref / out
                P_tilde[pump_idx, :] *= scale_vec[:, None]
                P_tilde[pump_idx, :] = np.maximum(P_tilde[pump_idx, :], cfg.pump_power_floor)
                _log_power_stats(f" iter {k+1} pump ref at z=L", ref, level="info")
                _log_power_stats(f" iter {k+1} pump out pre-scale z=L", out, level="info")
                _log_profile_stats(f" iter {k+1} pump scale_vec", scale_vec, level="info")
                _log_power_stats(f" iter {k+1} pumps post-scale z=0", P_tilde[pump_idx, 0], level="debug")
                _log_power_stats(f" iter {k+1} pumps post-scale z=L", P_tilde[pump_idx, -1], level="info")

            # Early stop (signals)
            if cfg.early_stop_rtol is not None and last_sig is not None:
                cur_sig = P_tilde[sig_idx, :]
                denom = np.maximum(np.abs(last_sig), 1e-30)
                rchg = float(np.max(np.abs(cur_sig - last_sig) / denom))
                lg.info(f" iter {k+1} max rel change signals={rchg:.2e}")
                if rchg < float(cfg.early_stop_rtol) and (k > max(5, int(0.3 * liter))):
                    lg.debug(f" early stop at k={k} (max rel change signals={rchg:.2e})")
                    P = P_tilde
                    break
                last_sig = cur_sig.copy()
            else:
                last_sig = P_tilde[sig_idx, :].copy()

            P = P_tilde

        # Final boundary sanity: set pump boundary exactly to pumP (z=L) by one last rescale
        if Np:
            outL = np.maximum(P[pump_idx, -1], cfg.pump_power_floor)
            scale_final = pumP / outL
            P[pump_idx, :] *= scale_final[:, None]
            P[pump_idx, :] = np.maximum(P[pump_idx, :], cfg.pump_power_floor)
            _log_profile_stats(" final pump scale factors", scale_final, level="info")
            _log_power_stats(" final pumps z=L", P[pump_idx, -1], level="info")
            _log_power_stats(" final pumps z=0", P[pump_idx, 0], level="debug")

        # Ensure exact signal boundary
        P[sig_idx, 0] = sigP
        _log_power_stats(" Final P", P, level="info")
        return P


if __name__ == "__main__":  # pragma: no cover
    import os
    import sys

    import matplotlib

    # Use a non-interactive backend so the plot can be saved headlessly.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pynlin.system import System

    level = os.getenv("LOGURU_LEVEL", "INFO")
    lg.remove()
    lg.add(sys.stderr, level=level)

    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input/dummy_struct.toml")
    system = System.from_toml(cfg_path)
    try:
        out_fig = system.plot_launch_spectrum()
        lg.info(f"Saved launch spectrum plot to {out_fig}")
    except Exception as e:
        lg.warning(f"Launch spectrum plot skipped: {e}")
    # decimate the signals 
    system.wdm = system.wdm.decimate(factor=1, rescale_power=True) # FIXME: only when decimating without rescaling we do get reasonable results (al netto dei segnali che vanno a zero)
    fiber = system.fiber

    z = np.linspace(0.0, float(fiber.length), 400)

    amp = SMFWidebandAmplifier(fiber)
    cfg = JiangIterativeConfig(
        iterative_steps=100,
        pump_scale_start=1e-6,
        inner_iters=None,
        early_stop_rtol=1e-4,
        pump_power_floor=1e-12,
    )
    pump_sol, sig_sol, _ = amp.solve_with_jiang(system, z=z, cfg=cfg, disable_pumps=False)

    try:
        from pynlin.raman.plot_optimization import plot_profiles
    except Exception as exc:
        lg.error(f"Plotting skipped: plot_profiles not available ({exc})")
    else:
        pump_power_dbm = np.array([p.power_dbm for p in (system.pump_specs or [])], dtype=float)
        pump_power_w = dBm2watt(pump_power_dbm) if pump_power_dbm.size else np.zeros((0,))
        pump_wl = np.array([p.wavelength for p in (system.pump_specs or [])], dtype=float)
        sig_wl = nu2lambda(system.wdm.frequency_grid())

        pump_solution_plot = pump_sol[:, :, None] if pump_sol.ndim == 2 else pump_sol
        signal_solution_plot = sig_sol[:, :, None] if sig_sol.ndim == 2 else sig_sol
        pump_powers = pump_power_w[:, None] if pump_power_w.size else np.zeros((0, 1))

        plot_profiles(
            signal_wavelengths=sig_wl,
            signal_solution=signal_solution_plot,
            ase_solution=None,
            pump_wavelengths=pump_wl,
            pump_solution=pump_solution_plot,
            pump_powers=pump_powers,
            cf=system,
            wallpaper_mode=False,
            use_active_naming=False
        )
