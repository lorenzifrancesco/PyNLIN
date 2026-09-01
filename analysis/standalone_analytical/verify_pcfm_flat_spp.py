"""Verification of ``pcfm_flat_spp_phase_diagram.md``.

Checks, in order:
  1. the ladder functions: the closed form (6) for Psi1, the identity
     (X Psi1')' = Khat, the asymptotic constant C_inf = pi (gamma - 1),
     and PCFM1's I_{0,1}(L; lambda) = (L/2) Psi1(lambda L);
  2. the corner second-difference formula (8) against brute 2D quadrature
     on rectangles drawn from all four regions;
  3. the region laws (19)-(22) derived from the corner formula, including
     the exact 2/GM^2 law and the sheet logarithm;
  4. the mapping (17) against ``fwm_tuple_variables`` on a synthetic
     constant-beta2 grid, and masked QMC ground truth vs the
     A(d) x corner-formula overlay;
  5. the trilinear rung (25) against brute 3D quadrature;
  6. the phase diagram derived from the corner formula alone
     (exports/pcfm_phase_diagram.png).

    .venv/bin/python analysis/standalone_analytical/verify_pcfm_flat_spp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import CubicSpline
from scipy.special import sici

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pynlin.methods.td.fast_nlin import (  # noqa: E402
    fwm_tuple_variables,
    kernel_abs2,
    qmc_tuple_ground_truth,
    support_acceptance,
)

EULER_GAMMA = 0.5772156649015328606
C_INF = np.pi * (EULER_GAMMA - 1.0)

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


# ============================================================
# Ladder functions
# ============================================================

def psi0(x):
    """Psi0(X) = int_0^X Khat = 2 Si(X) - 2 (1 - cos X)/X, odd, Psi0(0)=0."""
    x = np.asarray(x, dtype=float)
    si, _ = sici(np.abs(x))
    out = 2.0 * si - 2.0 * np.where(np.abs(x) > 1e-8,
                                    (1.0 - np.cos(x)) / np.where(x == 0, 1, np.abs(x)),
                                    np.abs(x) / 2.0)
    return np.sign(x) * out


ASYM_SWITCH = 200.0


def psi1_asym(x):
    """Large-|X| expansion (14) extended two orders by parts:
    pi ln|X| + C_inf + 2/X + 2 cos X/X^3 + 10 sin X/X^4 - 52 cos X/X^5."""
    ax = np.abs(x)
    return (np.sign(x) * (np.pi * np.log(ax) + C_INF) + 2.0 / x
            + 2.0 * np.cos(x) / x**3 + 10.0 * np.sin(x) / x**4
            - 52.0 * np.cos(x) / x**5)


def _psi1_quad_scalar(x):
    """Reference Psi1: quadrature of Psi0(s)/s up to the switch, extended
    asymptotic beyond (direct quadrature over thousands of radians hits
    oscillatory roundoff)."""
    if x == 0.0:
        return 0.0
    if abs(x) > ASYM_SWITCH:
        return float(psi1_asym(np.array([x]))[0])
    val, _ = quad(lambda s: psi0(s) / s if s > 0 else 1.0, 0.0, abs(x),
                  limit=800, epsabs=1e-13, epsrel=1e-12)
    return np.sign(x) * val

# Dense cumulative table for |X| <= ASYM_SWITCH (vectorized midrange path).
_s = np.linspace(0.0, ASYM_SWITCH, 400_001)
_g = np.empty_like(_s)
_g[0] = 1.0  # lim Psi0(s)/s
_g[1:] = psi0(_s[1:]) / _s[1:]
_tab = cumulative_trapezoid(_g, _s, initial=0.0)
_psi1_spline = CubicSpline(_s, _tab)


def psi1(x):
    """Odd Psi1 for any real argument (spline midrange, asymptotic tail)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mid = np.abs(x) <= ASYM_SWITCH
    out[mid] = np.sign(x[mid]) * _psi1_spline(np.abs(x[mid]))
    out[~mid] = psi1_asym(x[~mid])
    return out


def corner_E(theta, p1_lims, p2_lims):
    """<Khat(theta p1 p2)> over [a,b] x [c,d]: Delta^2 Psi1 / Delta^2 X, eq. (8)."""
    a, b = p1_lims
    c, d = p2_lims
    num = (psi1(np.array([theta * b * d])) - psi1(np.array([theta * b * c]))
           - psi1(np.array([theta * a * d])) + psi1(np.array([theta * a * c])))[0]
    return num / (theta * (b - a) * (d - c))


def corner_E_quad(theta, p1_lims, p2_lims):
    """Corner formula with quadrature-accurate Psi1 (for tiny second
    differences, where the tabulated path's interpolation error matters)."""
    a, b = p1_lims
    c, d = p2_lims
    num = (_psi1_quad_scalar(theta * b * d) - _psi1_quad_scalar(theta * b * c)
           - _psi1_quad_scalar(theta * a * d) + _psi1_quad_scalar(theta * a * c))
    return num / (theta * (b - a) * (d - c))


def brute_E(theta, p1_lims, p2_lims, n=None):
    """Tensor Gauss-Legendre average of Khat(theta p1 p2) over the rectangle."""
    a, b = p1_lims
    c, d = p2_lims
    span = abs(theta) * max(abs(a), abs(b)) * (d - c) + abs(theta) * max(abs(c), abs(d)) * (b - a)
    if n is None:
        n = int(min(3000, max(400, 8 * span / np.pi)))
    x1, w1 = np.polynomial.legendre.leggauss(n)
    p1 = 0.5 * (b - a) * x1 + 0.5 * (a + b)
    p2 = 0.5 * (d - c) * x1 + 0.5 * (c + d)
    u = theta * p1[:, None] * p2[None, :]
    vals = kernel_abs2(u)
    return 0.25 * np.einsum("i,j,ij->", w1, w1, vals)


# ============================================================
# 1. Ladder identities
# ============================================================
print("[1] ladder identities")

xs = np.array([0.3, 2.0, 7.0, 20.0, 120.0])
si_x, _ = sici(xs)
j_x = np.array([quad(lambda t: sici(t)[0] / t if t > 0 else 1.0, 0, x,
                     limit=800, epsabs=1e-13)[0] for x in xs])
closed = 2.0 * (j_x - si_x + (1.0 - np.cos(xs)) / xs)
direct = np.array([_psi1_quad_scalar(x) for x in xs])
err = np.max(np.abs(closed - direct) / np.abs(direct))
check("Psi1 closed form (6) == quadrature of (5)", err < 1e-9, f"max rel {err:.1e}")

# (X Psi1')' = Khat via Psi0: X Psi1' = Psi0 by construction; Psi0' = Khat.
h = 1e-5
for x0 in (0.7, 5.0, 33.0):
    d_psi0 = (psi0(x0 + h) - psi0(x0 - h)) / (2 * h)
    if not abs(d_psi0 - kernel_abs2(np.array([x0]))[0]) < 1e-8:
        check("(X Psi1')' = Khat (numeric)", False, f"x0={x0}")
        break
else:
    check("(X Psi1')' = Khat (numeric)", True)

c_est = _psi1_quad_scalar(500.0) - np.pi * np.log(500.0) - 2.0 / 500.0
check("C_inf = pi (gamma - 1)", abs(c_est - C_INF) < 1e-6,
      f"{c_est:.8f} vs {C_INF:.8f}")

spl_err = max(abs(psi1(np.array([x]))[0] - _psi1_quad_scalar(x))
              for x in np.random.default_rng(0).uniform(0.05, 195, 40))
check("midrange spline of Psi1 vs quadrature", spl_err < 1e-7, f"max abs {spl_err:.1e}")

# PCFM1 eq. (19): I_{0,1}(L; lam) = int_0^L u^-1 (L-u) Si(lam u) du = (L/2) Psi1(lam L)
for L, lam in [(1.0, 3.0), (2.5, 40.0), (10.0, 0.2)]:
    i01, _ = quad(lambda u: (L - u) * sici(lam * u)[0] / u if u > 0 else (L * lam),
                  0, L, limit=800, epsabs=1e-13)
    ref = 0.5 * L * psi1(np.array([lam * L]))[0]
    if not abs(i01 - ref) < 1e-8 * max(1, abs(ref)):
        check("PCFM1 I_{0,1} = (L/2) Psi1(lam L)", False, f"L={L} lam={lam}")
        break
else:
    check("PCFM1 I_{0,1} = (L/2) Psi1(lam L)", True)

# SCI special case (11): symmetric box -> Psi1(x)/x
xsci = 4.7
e_sym = corner_E(1.0, (-xsci**0.5, xsci**0.5), (-xsci**0.5, xsci**0.5))
check("SCI special case: E = Psi1(x)/x", abs(e_sym - psi1(np.array([xsci]))[0] / xsci) < 1e-9)

# ============================================================
# 2. Corner formula vs brute quadrature
# ============================================================
print("[2] corner formula (8) vs 2D quadrature")

rng = np.random.default_rng(1)
cases = [
    # (theta, p1_lims, p2_lims, label)
    (0.02, (1.0, 3.0), (2.0, 5.0), "plateau"),
    (0.5, (5.0, 11.0), (7.0, 13.0), "moderate"),
    (6.0, (18.0, 25.0), (31.0, 37.0), "gapped far"),
    (3.0, (4.0, 10.0), (-3.0, 3.0), "axis straddle"),
    (3.0, (-2.0, 4.0), (-3.0, 5.0), "double straddle"),
    (0.05, (100.0, 106.0), (200.0, 206.0), "gapped coherent"),
    (-1.7, (2.0, 8.0), (-4.0, 2.5), "negative theta"),
    (8.0, (0.0, 6.28), (0.0, 6.28), "corner at origin"),
]
worst = 0.0
worst_fast = 0.0
for theta, l1, l2, lab in cases:
    ec = corner_E_quad(theta, l1, l2)
    eb = brute_E(theta, l1, l2)
    rel = abs(ec - eb) / max(abs(eb), 1e-300)
    worst = max(worst, rel)
    worst_fast = max(worst_fast, abs(corner_E(theta, l1, l2) - eb) / max(abs(eb), 1e-300))
    print(f"      {lab:16s} E_corner={ec: .6e}  E_brute={eb: .6e}  rel={rel:.1e}")
check("corner formula matches quadrature on all rectangles", worst < 1e-6,
      f"worst rel {worst:.1e}; fast tabulated path worst rel {worst_fast:.1e}")

# ============================================================
# 3. Region laws
# ============================================================
print("[3] region laws from the corner formula")

# Region 1: plateau -> 1.
e1 = corner_E(1e-4, (-np.pi + 2 * np.pi * 1.0, np.pi + 2 * np.pi), (-np.pi + 2 * np.pi, np.pi + 2 * np.pi))
check("region 1: E -> 1", abs(e1 - 1.0) < 1e-3, f"E={e1:.6f}")

# Region 3 exact law (20): E = 2 / (theta^2 (Phi_a^2-pi^2)(Phi_b^2-pi^2)).
ok3 = True
for theta, na, nb in [(5.0, 4, 7), (40.0, 3, 9), (2.0, 12, 20)]:
    Pa, Pb = 2 * np.pi * na, 2 * np.pi * nb
    e = corner_E(theta, (Pa - np.pi, Pa + np.pi), (Pb - np.pi, Pb + np.pi))
    law = 2.0 / (theta**2 * (Pa**2 - np.pi**2) * (Pb**2 - np.pi**2))
    rel = abs(e - law) / law
    print(f"      R3 theta={theta:5.1f} (n_a,n_b)=({na},{nb}) E={e:.4e} 2/GM^2={law:.4e} rel={rel:.1e}")
    ok3 &= rel < 0.05
check("region 3: exact 2/GM^2 law (20)", ok3)

# Region 2 sheet law (22): straddling rectangle.
ok2 = True
for theta, na in [(2.0, 6), (10.0, 12), (50.0, 25)]:
    Pa = 2 * np.pi * na
    l1 = (Pa - np.pi, Pa + np.pi)
    l2 = (-np.pi, np.pi)
    e = corner_E(theta, l1, l2)
    law = 2 * np.pi * np.log(l1[1] / l1[0]) / (abs(theta) * (l1[1] - l1[0]) * (l2[1] - l2[0]))
    rel = abs(e - law) / law
    print(f"      R2 theta={theta:5.1f} n_a={na}  E={e:.4e}  sheet law={law:.4e}  rel={rel:.1e}")
    ok2 &= rel < 0.05
check("region 2: sheet law (22) with logarithm", ok2)

# Degenerate corner: rectangle touching both axes stays bounded by 1.
e_deg = corner_E(1000.0, (0.0, 2 * np.pi), (0.0, 2 * np.pi))
check("degenerate double-zero rectangle: finite, 0 < E < 1", 0.0 < e_deg < 1.0,
      f"E={e_deg:.4e}")

# Region 4: coherent fringes -> Khat(u0), including a null.
ok4 = True
for u0_target in (3 * np.pi, 4 * np.pi):  # side lobe and exact null
    Pa = Pb = 200.0 * 2 * np.pi
    theta = -u0_target / (Pa * Pb)
    e = corner_E_quad(theta, (Pa - np.pi, Pa + np.pi), (Pb - np.pi, Pb + np.pi))
    kh = kernel_abs2(np.array([u0_target]))[0]
    print(f"      R4 u0={u0_target/np.pi:.0f}pi  E={e:.4e}  Khat(u0)={kh:.4e}")
    ok4 &= abs(e - kh) < 5e-3 * max(kh, 1e-2)
check("region 4: E -> Khat(u0) with nulls", ok4)

# Log-log slopes along rays: sheet ray -> -1, gapped ray -> -2.
thetas = np.logspace(0.5, 3.5, 13)
na, nb = 8, 8
Pa, Pb = 2 * np.pi * na, 2 * np.pi * nb
e_sheet = np.array([corner_E(t, (Pa - np.pi, Pa + np.pi), (-np.pi, np.pi)) for t in thetas])
e_gap = np.array([corner_E(t, (Pa - np.pi, Pa + np.pi), (Pb - np.pi, Pb + np.pi)) for t in thetas])
s_sheet = np.polyfit(np.log(thetas), np.log(e_sheet), 1)[0]
s_gap = np.polyfit(np.log(thetas), np.log(e_gap), 1)[0]
check("ray slopes: sheet -1, gapped -2",
      abs(s_sheet + 1.0) < 0.02 and abs(s_gap + 2.0) < 0.02,
      f"slopes {s_sheet:.3f}, {s_gap:.3f}")

# ============================================================
# 4. Mapping to fwm_tuple_variables + masked overlay
# ============================================================
print("[4] mapping (17) and masked overlay")

B = 25e9
L = 100e3
beta2_const = -21e-27  # s^2/m
n_ch = 41
spacing = 1.02 * B
f0 = 193.4e12
freqs = f0 + spacing * (np.arange(n_ch) - n_ch // 2)
omega = 2 * np.pi * freqs
om_t = omega[n_ch // 2]
beta0_abs = 0.5 * beta2_const * (omega - om_t) ** 2
beta1 = beta2_const * (omega - om_t)
beta2 = np.full(n_ch, beta2_const)

var = fwm_tuple_variables(freqs, beta0_abs, beta1, beta2, B, L, n_ch // 2)
theta_phys = beta2_const * B**2 * L
Phi_a = (omega[var.a] - om_t) / B
Phi_b = (omega[var.b] - om_t) / B
d0 = np.abs(var.d) < 1e-9
u0_pred = -theta_phys * Phi_a * Phi_b
nu_pred = theta_phys * Phi_a
err_u0 = np.max(np.abs(var.u0[d0] - u0_pred[d0]) / np.maximum(np.abs(var.u0[d0]), 1e-12))
err_nu = np.max(np.abs(var.nu_a[d0] - nu_pred[d0]) / np.maximum(np.abs(var.nu_a[d0]), 1e-12))
check("u0 = -theta Phi_a Phi_b and nu_a = theta Phi_a (d=0 tuples)",
      err_u0 < 1e-10 and err_nu < 1e-10, f"rel {err_u0:.1e}, {err_nu:.1e}")

# Masked QMC (exact bilinear phase: include_quadratic with equal q's) vs
# A(d) x corner formula, on tuples across regimes.
print("      masked QMC vs A(d) x E_corner (overlay quality, not an identity):")
q_val = 0.5 * beta2_const * B**2 * L
sel = []
for want in ["gap-far", "gap-near", "sheet-adjacent"]:
    if want == "gap-far":
        idx = np.argmin(np.abs(np.abs(Phi_a) - 2 * np.pi * 8) + np.abs(np.abs(Phi_b) - 2 * np.pi * 12) + 1e6 * (~d0))
    elif want == "gap-near":
        idx = np.argmin(np.abs(np.abs(Phi_a) - 2 * np.pi * 2) + np.abs(np.abs(Phi_b) - 2 * np.pi * 3) + 1e6 * (~d0))
    else:
        idx = np.argmin(np.abs(np.abs(Phi_a) - 2 * np.pi * 1) + np.abs(np.abs(Phi_b) - 2 * np.pi * 15) + 1e6 * (~d0))
    sel.append((want, int(idx)))
ok_overlay = True
for lab, i in sel:
    mc, se = qmc_tuple_ground_truth(
        u0=float(var.u0[i]),
        nu=(float(var.nu_a[i]), float(var.nu_b[i]), float(var.nu_c[i])),
        q=(q_val, q_val, q_val, q_val),
        d=float(var.d[i]), n_points=1 << 15, n_replicates=4,
    )
    ec = corner_E(-theta_phys, (Phi_a[i] - np.pi, Phi_a[i] + np.pi),
                  (Phi_b[i] - np.pi, Phi_b[i] + np.pi))
    overlay = float(support_acceptance(np.array([var.d[i]]))[0]) * ec
    ratio = mc / overlay if overlay > 0 else np.inf
    print(f"      {lab:15s} masked QMC={mc:.4e} (se {se:.1e})  A(d)*E_corner={overlay:.4e}  ratio={ratio:.3f}")
    ok_overlay &= 0.5 < ratio < 2.0
check("overlay A(d) x E_corner within 2x of masked QMC", ok_overlay)

# ============================================================
# 5. Trilinear rung (25)
# ============================================================
print("[5] trilinear corner formula (25)")

_s2 = np.linspace(0.0, ASYM_SWITCH, 400_001)
_g2 = np.empty_like(_s2)
_g2[0] = 1.0
_g2[1:] = psi1(_s2[1:]) / _s2[1:]
_tab2 = cumulative_trapezoid(_g2, _s2, initial=0.0)
_psi2_spline = CubicSpline(_s2, _tab2)


def psi2(x):
    x = np.asarray(x, dtype=float)
    if np.any(np.abs(x) > ASYM_SWITCH):
        raise ValueError("psi2 table range exceeded")
    return np.sign(x) * _psi2_spline(np.abs(x))


def corner_E3(theta3, l1, l2, l3):
    signs = []
    vals = []
    for i, p1 in enumerate(l1):
        for j, p2 in enumerate(l2):
            for k, p3 in enumerate(l3):
                signs.append((-1) ** (i + j + k))
                vals.append(theta3 * p1 * p2 * p3)
    num = sum(s * psi2(np.array([v]))[0] for s, v in zip(signs, vals))
    den = theta3 * (l1[1] - l1[0]) * (l2[1] - l2[0]) * (l3[1] - l3[0])
    # signs: + for (upper,upper,upper) parity -> (-1)^(i+j+k) with i=0 lower
    return -num / den  # (-1)^3 orientation: third difference sign


def brute_E3(theta3, l1, l2, l3, n=160):
    x, w = np.polynomial.legendre.leggauss(n)

    def nodes(lo, hi):
        return 0.5 * (hi - lo) * x + 0.5 * (lo + hi)

    p1, p2, p3 = nodes(*l1), nodes(*l2), nodes(*l3)
    u = theta3 * p1[:, None, None] * p2[None, :, None] * p3[None, None, :]
    return 0.125 * np.einsum("i,j,k,ijk->", w, w, w, kernel_abs2(u))


ok5 = True
for theta3, l1, l2, l3 in [
    (0.01, (1.0, 3.0), (2.0, 4.0), (0.5, 2.5)),
    (0.3, (2.0, 5.0), (-1.5, 1.5), (1.0, 4.0)),
    (0.05, (3.0, 9.0), (4.0, 10.0), (-2.0, 2.0)),
]:
    e3c = corner_E3(theta3, l1, l2, l3)
    e3b = brute_E3(theta3, l1, l2, l3)
    rel = abs(e3c - e3b) / max(abs(e3b), 1e-300)
    print(f"      theta3={theta3}  E_corner3={e3c:.6e}  E_brute3={e3b:.6e}  rel={rel:.1e}")
    ok5 &= rel < 1e-4
check("trilinear Delta^3 Psi2 formula matches 3D quadrature", ok5)

# ============================================================
# 6. Phase diagram from the corner formula
# ============================================================
print("[6] phase diagram figure")

def corner_E_symmetric(x_grad, u0_abs):
    """Dense-map section: symmetric split Phi_a = Phi_b = Phi.

    Given (x_grad, |u0|), invert (17): Phi = sqrt6 |u0|/x, |theta| =
    x^2/(6 |u0|); the corner formula (8) on the box (Phi +- pi)^2 is fully
    vectorized through psi1.
    """
    phi = np.sqrt(6.0) * u0_abs / x_grad
    theta = x_grad**2 / (6.0 * u0_abs)
    x1 = theta * (phi + np.pi) ** 2
    x2 = theta * (phi + np.pi) * (phi - np.pi)
    x3 = theta * (phi - np.pi) ** 2
    second_diff = (psi1(x1) - 2.0 * psi1(x2) + psi1(x3)) / (theta * 4.0 * np.pi**2)
    # Far all-positive corners (Phi > pi and min corner beyond the
    # asymptotic switch): the corner sum cancels catastrophically when
    # Delta^2 X << eps * Psi1, but for this split X1 X3 = X2^2 exactly, so
    # the log part of Delta^2 Psi1 vanishes identically and
    # Delta^2 (2/X) = 8 pi^2 theta / X2^2 exactly, leaving only the small
    # oscillatory terms to difference:
    #   E = 2/X2^2 + Delta^2(osc)/(4 pi^2 theta),
    # the exact region-3 law (20) plus the fringe residue.

    def osc(x):
        return (2.0 * np.cos(x) / x**3 + 10.0 * np.sin(x) / x**4
                - 52.0 * np.cos(x) / x**5)

    far = (phi > np.pi) & (x3 > ASYM_SWITCH)
    with np.errstate(divide="ignore", invalid="ignore"):
        stable = (2.0 / x2**2
                  + (osc(x1) - 2.0 * osc(x2) + osc(x3)) / (4.0 * np.pi**2 * theta))
    # Tiny corner spread outside the far branch: the mixed-derivative
    # limit (19) is the exact asymptote, E -> Khat(u0) + O(spread^2).
    tiny = (~far) & (np.abs(x1 - x3) < 0.3)
    out = np.where(far, stable, second_diff)
    return np.where(tiny, kernel_abs2(theta * phi**2), out)


# Panel (a): dense symmetric-split map.
x_ax = np.logspace(-3, 4, 460)
u_ax = np.logspace(-2, 6, 460)
XX, UU = np.meshgrid(x_ax, u_ax)
EE = corner_E_symmetric(XX, UU)

# Panel (b): the physically reachable ray points of a WDM grid.
r_grid = 1.02
n_idx = np.unique(np.round(np.logspace(0, 2.6, 24)).astype(int))
pairs = [(na, nb) for na in n_idx for nb in n_idx]
pairs += [(na, -nb) for na in n_idx for nb in n_idx if na != nb]
thetas_fig = np.logspace(-6, 2, 33)
xg, u0a, ee = [], [], []
for na, nb in pairs:
    Pa, Pb = 2 * np.pi * r_grid * na, 2 * np.pi * r_grid * nb
    for t in thetas_fig:
        e = corner_E(-t, (Pa - np.pi, Pa + np.pi), (Pb - np.pi, Pb + np.pi))
        xg.append(t * np.sqrt(Pa**2 + Pb**2 + (Pa + Pb) ** 2))
        u0a.append(abs(t * Pa * Pb))
        ee.append(max(e, 1e-16))
xg, u0a, ee = map(np.asarray, (xg, u0a, ee))

fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.2), sharey=True)


def draw_boundaries(ax, xspan):
    xx = np.logspace(np.log10(xspan[0]), np.log10(xspan[1]), 200)
    ax.plot(xx, np.pi * xx / np.sqrt(3), ls="--", color="crimson", lw=1.6,
            label=r"$|u_0| = \pi x_\nabla/\sqrt3$")
    ax.plot(xx, np.pi * np.sqrt(3) * xx, ls=":", color="crimson", lw=1.6,
            label=r"$|u_0| = W$")
    ax.axvline(1.0, color="crimson", lw=1.2, ls="-.", label=r"$x_\nabla = 1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x_\nabla$ [rad]")


pcm = axes[0].pcolormesh(x_ax, u_ax, np.log10(np.maximum(EE, 1e-16)),
                         cmap="viridis", vmin=-10, vmax=0, shading="auto",
                         rasterized=True)
draw_boundaries(axes[0], (x_ax[0], x_ax[-1]))
axes[0].set_xlim(x_ax[0], x_ax[-1])
axes[0].set_ylim(u_ax[0], u_ax[-1])
axes[0].set_ylabel(r"$|u_0|$ [rad]")
axes[0].set_title("(a) symmetric-split section, corner formula")
axes[0].legend(loc="lower right", framealpha=0.35)

sc = axes[1].scatter(xg, u0a, c=np.log10(ee), s=7, cmap="viridis", vmin=-10,
                     vmax=0, rasterized=True)
draw_boundaries(axes[1], (xg.min(), xg.max()))
axes[1].set_xlim(xg.min(), xg.max())
axes[1].set_title("(b) WDM-grid rays, $\\theta$ swept")

cb = fig.colorbar(pcm, ax=axes, ticks=[-10, -8, -6, -4, -2, 0], pad=0.015)
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
cb.set_label(r"$\log_{10}\ \langle\hat K\rangle_{\rm box}$")
out = Path(__file__).resolve().parent / "exports" / "pcfm_phase_diagram.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=170, bbox_inches="tight")
print(f"      wrote {out}")

# ============================================================
print()
n_fail = sum(1 for _, ok in results if not ok)
print(f"{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
