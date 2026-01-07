"""Compute and save Raman power profiles for a given system using the Jiang solver.

This script targets the dummy_struct configuration and saves the resulting power
profiles to a .npy file for downstream noise calculations. No optimization is
performed; pumps and signals come directly from the system definition.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger as lg
from scipy.constants import nu2lambda

from pynlin.raman.solvers_jiang import SMFWidebandAmplifier, JiangIterativeConfig
from pynlin.system import System
from pynlin.utils import dBm2watt


def simulate_power_profiles(
    cfg_path: Path | str = Path("input/dummy_struct.toml"),
    output_path: Path | str = Path("results/dummy_power_profiles.npy"),
    z_points: int = 400,
    decimation_factor: int = 1,
    jiang_cfg: Optional[JiangIterativeConfig] = None,
) -> Path:
    """Run Jiang solver on the provided system and persist power profiles."""
    system = System.from_toml(Path(cfg_path))
    if decimation_factor > 1:
        system.wdm = system.wdm.decimate(decimation_factor, rescale_power=True)
        lg.info(f"Decimated WDM by {decimation_factor}; channels kept: {system.wdm.num_channels}")

    fiber = system.fiber
    z = np.linspace(0.0, float(fiber.length), int(z_points))

    amp = SMFWidebandAmplifier(fiber)
    cfg = jiang_cfg or JiangIterativeConfig(
        iterative_steps=80,
        pump_scale_start=1e-6,
        inner_iters=None,
        early_stop_rtol=1e-4,
        pump_power_floor=1e-12,
    )
    pump_sol, sig_sol, ase_sol = amp.solve_with_jiang(system, z=z, cfg=cfg, disable_pumps=False)

    pump_power_dbm = np.array([p.power_dbm for p in (system.pump_specs or [])], dtype=float)
    pump_power_w = dBm2watt(pump_power_dbm) if pump_power_dbm.size else np.zeros((0,))
    pump_wl = np.array([p.wavelength for p in (system.pump_specs or [])], dtype=float)
    sig_wl = nu2lambda(system.wdm.frequency_grid())

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pump_sol": pump_sol,
        "signal_sol": sig_sol,
        "ase_sol": ase_sol,
        "pump_wavelengths": pump_wl,
        "pump_powers": pump_power_w,
        "signal_wavelengths": sig_wl,
        "z": z,
    }
    np.save(out_path, payload)
    lg.info(f"Saved power profiles to {out_path}")
    return out_path


def _init_logging() -> None:
    level = os.getenv("LOGURU_LEVEL", "INFO")
    lg.remove()
    lg.add(sys.stderr, level=level)


if __name__ == "__main__":
    _init_logging()
    simulate_power_profiles(cfg_path="input/uwb_struct.toml", output_path="results/uwb_power_profiles.npy")
