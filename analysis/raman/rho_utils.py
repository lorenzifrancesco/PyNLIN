from __future__ import annotations

import numpy as np
from loguru import logger as lg


def _fmt_c(val: complex) -> str:
    return f"{val.real:.3e}{val.imag:+.3e}j"


def rho_undepleted(
    mu: np.ndarray,
    A: complex,
    alpha_p: float,
    length: float,
    *,
    log_points: bool = False,
    dps: int = 80,
) -> np.ndarray:
    """Compute rho from the closed-form undepleted-pump expression."""
    try:
        import mpmath as mp
    except ImportError as exc:
        raise ImportError(
            "mpmath is required for complex incomplete Gamma evaluation. "
            "Install it or provide real-valued inputs."
        ) from exc
    mp.mp.dps = dps
    lg.debug("mpmath precision set to {dps} digits.", dps=mp.mp.dps)
    mu_arr = np.asarray(mu, dtype=complex)
    A_mp = mp.mpc(A)
    exp_factor = mp.e ** (2.0 * alpha_p * length)
    lg.debug(
        "rho inputs: alpha_p={alpha_p:.3e}, L={length:.3e}, A={A}, exp_factor={exp_factor}",
        alpha_p=alpha_p,
        length=length,
        A=_fmt_c(complex(A)),
        exp_factor=str(exp_factor),
    )
    out = np.empty(mu_arr.shape, dtype=float)
    fail_count = 0
    for idx, mu_val in np.ndenumerate(mu_arr):
        mu_mp = mp.mpc(mu_val)
        if log_points:
            lg.debug(
                "gammainc call idx={} mu={} A={} Aexp={}",
                idx,
                _fmt_c(complex(mu_val)),
                _fmt_c(complex(A_mp)),
                _fmt_c(complex(A_mp * exp_factor)),
            )
        try:
            exp_mu = mp.e ** (-2.0 * alpha_p * length * mu_mp)
            term = mp.expint(1.0 + mu_mp, A_mp) - exp_mu * mp.expint(1.0 + mu_mp, A_mp * exp_factor)
            rho_val = mp.e ** (A_mp) / (2.0 * alpha_p) * abs(term) ** 2
            out_val = float(mp.re(rho_val))
            out[idx] = out_val
            if log_points and not np.isfinite(out_val):
                lg.debug(
                    "non-finite rho at idx={} mu={} rho={}",
                    idx,
                    _fmt_c(complex(mu_val)),
                    _fmt_c(complex(rho_val)),
                )
        except Exception:
            fail_count += 1
            if log_points:
                lg.debug("gammainc failed at idx={} mu={}", idx, _fmt_c(complex(mu_val)))
            out[idx] = np.nan
    if fail_count:
        lg.warning("gammainc failed for {} of {} points; values set to NaN.", fail_count, out.size)
    return out


def rho_attenuation(alpha: float, delta_beta: np.ndarray, length: float) -> np.ndarray:
    """Attenuation-only efficiency with full dispersion-based delta_beta."""
    alpha = float(alpha)
    delta_beta = np.asarray(delta_beta, dtype=float)
    denom = 2.0 * alpha - 1j * delta_beta
    numer = 1.0 - np.exp(-(2.0 * alpha - 1j * delta_beta) * length)
    return np.abs(numer / denom) ** 2


def effective_length(alpha: float, length: float) -> float:
    """Return the effective interaction length for power attenuation."""
    alpha = float(alpha)
    length = float(length)
    if alpha <= 0.0:
        return length
    return (1.0 - np.exp(-2.0 * alpha * length)) / (2.0 * alpha)
