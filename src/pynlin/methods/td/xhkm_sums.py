from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class XhkmSums:
    """Prefactor-free Dar-style sums contracted from ``X[h,r,m]``.

    ``n1`` and ``n2`` correspond to the collision-sum parts of the Dar
    ``chi1`` and ``chi2`` terms. Physical prefactors are intentionally not
    included here.

    3PC classification follows the FWM degeneracy rule: only ``h=0``
    (same pump symbol, 3PCa) or ``k=m`` (same signal/probe symbol, 3PCb)
    create a genuine 3PC.  The ``h=k`` and ``h=m`` cases (formerly named
    ``n_3pc_other``) are non-degenerate FWM entries that happen to share
    a symbol index by chance; they belong to the 4PC sector.
    """

    n1: float
    n2: float
    n_2pc: float
    n_3pc_total: float
    n_3pca: float
    n_3pcb: float
    # DEPRECATED: kept for back-compat with old .npz files; always 0.
    n_3pc_other: float
    n_3pc_k_eq_m: float
    n_4pc: float
    n_k_neq_m: float

    @property
    def n2_over_n1(self) -> float:
        if self.n1 <= 0.0:
            return float("nan")
        return self.n2 / self.n1


def _single_zero_index(values: np.ndarray, name: str) -> int:
    matches = np.flatnonzero(values == 0)
    if matches.size != 1:
        raise ValueError(f"{name} must contain exactly one zero")
    return int(matches[0])


def compute_xhkm_sums(
    X: np.ndarray,
    h_values: np.ndarray,
    r_values: np.ndarray,
    m_values: np.ndarray,
) -> XhkmSums:
    """Contract ``X[h,r,m] = X_{h,m+r,m}`` into prefactor-free sums."""
    X = np.asarray(X)
    h_values = np.asarray(h_values, dtype=int).reshape(-1)
    r_values = np.asarray(r_values, dtype=int).reshape(-1)
    m_values = np.asarray(m_values, dtype=int).reshape(-1)

    if X.ndim != 3:
        raise ValueError(f"Expected a rank-3 X tensor, got shape {X.shape}")
    ih0 = _single_zero_index(h_values, "h_values")
    ir0 = _single_zero_index(r_values, "r_values")
    expected_shape = (h_values.size, r_values.size, m_values.size)
    if X.shape != expected_shape:
        raise ValueError(f"X shape {X.shape} does not match index shape {expected_shape}")

    abs2 = np.abs(X) ** 2
    n1 = float(np.sum(abs2))
    n2 = float(np.sum(abs2[:, ir0, :]))
    n_2pc = float(np.sum(abs2[ih0, ir0, :]))
    n_3pc_k_eq_m = n2 - n_2pc
    n_k_neq_m = n1 - n2

    H = h_values[:, None, None]
    R = r_values[None, :, None]
    M = m_values[None, None, :]
    K = M + R
    shape = abs2.shape
    mask_2pc = np.broadcast_to((H == 0) & (R == 0), shape)
    mask_3pca = np.broadcast_to((H == 0) & (R != 0), shape)
    mask_3pcb = np.broadcast_to((H != 0) & (R == 0), shape)
    # h=k / h=m cases (formerly 3PC other) are non-degenerate FWM —
    # they belong to the 4PC sector, not to 3PC.
    mask_3pc = mask_3pca | mask_3pcb
    mask_4pc = ~(mask_2pc | mask_3pc)

    n_3pca = float(np.sum(abs2[mask_3pca]))
    n_3pcb = float(np.sum(abs2[mask_3pcb]))
    n_3pc_other = 0.0  # DEPRECATED — always 0
    n_3pc_total = float(np.sum(abs2[mask_3pc]))
    n_4pc = float(np.sum(abs2[mask_4pc]))

    return XhkmSums(
        n1=n1,
        n2=n2,
        n_2pc=n_2pc,
        n_3pc_total=n_3pc_total,
        n_3pca=n_3pca,
        n_3pcb=n_3pcb,
        n_3pc_other=n_3pc_other,
        n_3pc_k_eq_m=float(n_3pc_k_eq_m),
        n_4pc=n_4pc,
        n_k_neq_m=float(n_k_neq_m),
    )


__all__ = ["XhkmSums", "compute_xhkm_sums"]
