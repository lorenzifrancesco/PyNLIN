"""Compatibility namespace for the former :mod:`pynlin.nlin` package.

New code should import from :mod:`pynlin.methods`, but these re-exports keep
existing scripts and notebooks working after the method-package refactor.
"""

from pynlin.methods.td.time_integrals import (
    X0mm_space_integral,
    compute_all_collisions_time_integrals,
    m_th_time_integral,
    m_th_time_integral_general,
)

__all__ = [
    "m_th_time_integral",
    "m_th_time_integral_general",
    "X0mm_space_integral",
    "compute_all_collisions_time_integrals",
]
