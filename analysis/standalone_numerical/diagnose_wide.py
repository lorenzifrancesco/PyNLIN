"""Check convergence for q=2.0 with larger windows."""

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401
from loguru import logger as lg
from analysis.log_init import init_logging
from pynlin.methods.td.xhkm_sums import compute_xhkm_sums
from pynlin.methods.td.xhkm_mc import estimate_xhkm_sums_mc
from pynlin.methods.td.xpm_kernel import compute_xpm_kernel_fft
from pynlin.pulses import pulse_from_config
from pynlin.system import System

system = System.from_toml("input/studies.toml")
pulse = pulse_from_config(system.pulse_config, num_symbols=128, samples_per_symbol=16)
T = 1.0 / float(pulse.baud_rate)
norm = T ** 2
length = float(system.fiber_length)
beta2_si = float(system.numerics.gvd)
beta2_mc = beta2_si / (T ** 2)
spacing = 2.0
dgd = abs(beta2_si) * 2.0 * np.pi * spacing * float(pulse.baud_rate)
llw = length * float(pulse.baud_rate) * dgd
lg.info(f"q={spacing} LLW={llw:.2f}")

mc = estimate_xhkm_sums_mc(
    beta2=beta2_mc, alpha=0.0, length=length,
    channel_spacing_over_baud=spacing, n_samples=500000,
    seed=1234, system=system,
)
lg.info(f"MC N1={mc.n1:.6e} N2={mc.n2:.6e}")

# Try with auto_refine to let the code handle z-resolution automatically
for h_r, m_r, label in [
    (5, 9, "base"),
    (7, 11, "medium"),
]:
    h = np.arange(-h_r, h_r + 1)
    r = np.arange(-h_r, h_r + 1)
    m = np.arange(-m_r, m_r + 1)
    z0 = np.linspace(0.0, length, 9)
    table = compute_xpm_kernel_fft(
        pulse, z0, h, r, m,
        dgd=dgd, gvda=beta2_si, gvdb=beta2_si,
        auto_refine=True, discretization_action="silent",
        min_pts_per_collision=3.0,
    )
    d = compute_xhkm_sums(table.X, table.h_values, table.r_values, table.m_values)
    actual_nz = table.metadata["n_z"]
    r1 = d.n1 * norm / mc.n1
    r2 = d.n2 * norm / mc.n2
    n_entries = h.size * r.size * m.size
    lg.info(f"  {label}: h_r={h_r} m_r={m_r} nz_auto={actual_nz} "
            f"entries={n_entries}  N1/MC={r1:.4f}  N2/MC={r2:.4f}")

init_logging()
