from __future__ import annotations

import numpy as np

from pynlin.constellation_stats import gaussian_mu0, qam_mu0
from pynlin.methods.td import _td_modulation_components
from pynlin.system import System

_MODULATION_MU0 = {
    "gaussian": gaussian_mu0,
    "qpsk": lambda: qam_mu0(4),
    "16qam": lambda: qam_mu0(16),
    "64qam": lambda: qam_mu0(64),
}

_KNOWN_MODULATIONS = frozenset(_MODULATION_MU0)


def compute_chi1_chi2(
    system: System,
    collision_coeffs: np.ndarray,
    launch_powers_w: np.ndarray | None = None,
    use_kappa: bool = True,
    use_x_mode: bool = True,
    exclude_self_channel: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (chi1, chi2, prefactor) from TD collision coefficients."""
    const_pref, sum_a, sum_b = _td_modulation_components(
        system,
        collision_coeffs,
        launch_powers_w,
        use_kappa=use_kappa,
        use_x_mode=use_x_mode,
        exclude_self_channel=exclude_self_channel,
    )
    return sum_b, sum_a, const_pref


def nlin_from_chi(
    chi1: np.ndarray,
    chi2: np.ndarray,
    prefactor: np.ndarray,
    mu0: float,
) -> np.ndarray:
    """Return launch-referenced NLIN = prefactor * (mu0 * chi2 + chi1)."""
    return prefactor * (mu0 * chi2 + chi1)


def resolve_mu0(modulation: str) -> float:
    """Return the fourth-order cumulant mu0 for a named modulation."""
    if modulation not in _MODULATION_MU0:
        raise ValueError(
            f"Unknown modulation {modulation!r}. "
            f"Known: {sorted(_KNOWN_MODULATIONS)}."
        )
    return float(_MODULATION_MU0[modulation]())
