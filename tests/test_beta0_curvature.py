"""beta0 must carry the fiber's curvature, not just its value and slope.

The FWM phase mismatch u0 is a *second* difference of beta0 over channel
separations of a few tens of GHz.  Interpolating a tabulated beta0 destroys
that: a piecewise-linear interpolant has zero curvature inside every table
cell, so every tuple whose legs fall in one cell is reported as perfectly
phase matched (u0 = 0).  These tests pin the reconstruction.
"""

import numpy as np
import pytest

from pynlin.methods.td.fullband_mc import (
    _beta0_abs_from_fiber,
    _beta0_from_curvature,
    decimated_frequency_grid,
)
from pynlin.methods.td.fast_nlin import fwm_tuple_variables
from pynlin.system import System

TWO_PI = 2.0 * np.pi


def _analytic_grid(n=401, df=25e9, f0=193.4e12, beta2=-21.0e-27, beta3=0.12e-39):
    """A grid with exactly known beta1/beta2 (cubic beta)."""
    freqs = f0 + df * np.arange(n)
    dw = TWO_PI * (freqs - f0)
    beta1 = 1.0e-9 + beta2 * dw + 0.5 * beta3 * dw**2
    beta2_grid = beta2 + beta3 * dw
    return freqs, beta1, beta2_grid


def test_reconstruction_reproduces_beta2():
    freqs, beta1, beta2 = _analytic_grid()
    beta0 = _beta0_from_curvature(freqs, beta1, beta2)

    w = TWO_PI * freqs
    h = float(np.diff(w).mean())
    curvature = (beta0[2:] - 2.0 * beta0[1:-1] + beta0[:-2]) / h**2

    assert np.allclose(curvature, beta2[1:-1], rtol=2e-6)


def test_reconstruction_preserves_slope():
    freqs, beta1, beta2 = _analytic_grid()
    beta0 = _beta0_from_curvature(freqs, beta1, beta2)

    slope = np.gradient(beta0, TWO_PI * freqs)
    assert np.allclose(slope, beta1, rtol=1e-6)


def test_u0_matches_the_closed_form():
    """u0 = L (w_a - w_c)(w_c - w_b) beta_2(wbar), exact for cubic beta."""
    freqs, beta1, beta2 = _analytic_grid()
    beta0 = _beta0_from_curvature(freqs, beta1, beta2)

    L = 100e3
    t = freqs.size // 2
    w = TWO_PI * freqs

    def u0_of(a, b, c):
        rel = lambda k: beta0[k] - beta0[t] - beta1[t] * (w[k] - w[t])
        return (rel(a) + rel(b) - rel(c)) * L

    for i, j in ((1, 1), (1, 2), (3, 4), (10, 20), (50, 60)):
        a, c, b = t + i + j, t + j, t - i
        assert a + b - c == t
        beta2_bar = np.interp(0.5 * (freqs[a] + freqs[b]), freqs, beta2)
        closed = L * (w[a] - w[c]) * (w[c] - w[b]) * beta2_bar
        assert u0_of(a, b, c) == pytest.approx(closed, rel=1e-6)


def test_compact_tuples_are_not_reported_as_phase_matched():
    """The regression itself: nearest-neighbour tuples must not get u0 = 0."""
    freqs, beta1, beta2 = _analytic_grid()
    beta0 = _beta0_from_curvature(freqs, beta1, beta2)

    v = fwm_tuple_variables(freqs, beta0, beta1, beta2, 24.5e9, 100e3, freqs.size // 2)
    assert v.u0.size > 0
    assert np.min(np.abs(v.u0)) > 1.0


def test_real_system_grid_carries_curvature():
    system = System.from_toml("input/studies.toml")
    _, freqs = decimated_frequency_grid(system, 8)
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)

    beta0 = _beta0_abs_from_fiber(system, freqs, beta1)

    order = np.argsort(freqs)
    w = TWO_PI * freqs[order]
    h = float(np.diff(w).mean())
    curvature = (beta0[order][2:] - 2.0 * beta0[order][1:-1] + beta0[order][:-2]) / h**2

    ratio = curvature / beta2[order][1:-1]
    assert np.median(np.abs(ratio - 1.0)) < 0.05
