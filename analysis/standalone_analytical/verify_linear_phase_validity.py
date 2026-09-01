"""Verification of the claims in section 10.5 of ``lorenzi_fast_method.md``.

Section 10.5 asks when the locally linear phase model of section 4 stops being
adequate.  The answer rests on the observation that, for a global dispersion
model truncated at ``beta3``, the accumulated phase is an exact product of
three factors, each affine in the in-band offsets:

    u / L = (w_a - w_c) (w_c - w_b) beta_2(wbar),      wbar = (w_a + w_b)/2

with every frequency carrying its in-band offset.  Identities are checked
symbolically, distributional constants by exact integration, and sizes by
Monte-Carlo against the exact phase.

    python analysis/standalone_analytical/verify_linear_phase_validity.py
"""

import numpy as np
import sympy as sp


results = []


def check(name, ok):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


# ============================================================
# 1. Symbolic identities
# ============================================================

xa, xb, xc, xd, d, B, L = sp.symbols('x_a x_b x_c x_d d B L', real=True)
b0, b1, b2, b3 = sp.symbols('beta0 beta1 beta2 beta3', real=True)
wa, wb, wc, wt = sp.symbols('omega_a omega_b omega_c omega_t', real=True)

beta = lambda w: b0 + b1 * w + b2 * w**2 / 2 + b3 * w**3 / 6
beta_1 = lambda w: sp.expand(b1 + b2 * w + b3 * w**2 / 2)
beta_2 = lambda w: b2 + b3 * w
q_of = lambda w: beta_2(w) * B**2 * L / 2

print("\n1. Exact forms of the accumulated phase")

wl = wa + wb - wc                       # landing centre, d = 0
exact = sp.expand(L * (beta(wa + B * xa) + beta(wb + B * xb)
                       - beta(wc + B * xc) - beta(wl + B * (xa + xb - xc))))

# --- the three-factor form -------------------------------------------------
P = (wa + B * xa) - (wc + B * xc)
Q = (wc + B * xc) - (wb + B * xb)
G = beta_2(((wa + B * xa) + (wb + B * xb)) / 2)

check("(check 1) u = L (w_a-w_c)(w_c-w_b) beta_2(wbar), exactly",
      sp.simplify(sp.expand(exact - L * P * Q * G)) == 0)
check("        all three factors are affine in (x_a, x_b, x_c)",
      all(sp.Poly(sp.expand(f), xa, xb, xc).total_degree() == 1
          for f in (P, Q, G)))

# --- the additive form terminates at third order ---------------------------
exact4 = sp.expand(L * (beta(wa + B * xa) + beta(wb + B * xb)
                        - beta(wc + B * xc) - beta(wt + B * xd)))
const = sp.expand(L * (beta(wa) + beta(wb) - beta(wc) - beta(wt)))
lin = sp.expand(L * B * (beta_1(wa) * xa + beta_1(wb) * xb
                         - beta_1(wc) * xc - beta_1(wt) * xd))
quad = sp.expand(q_of(wa) * xa**2 + q_of(wb) * xb**2
                 - q_of(wc) * xc**2 - q_of(wt) * xd**2)
cubic = sp.expand(b3 * B**3 * L / 6 * (xa**3 + xb**3 - xc**3 - xd**3))

check("(check 2) exact = const + linear + quadratic + cubic, no remainder",
      sp.simplify(exact4 - (const + lin + quad + cubic)) == 0)
check("        the expansion terminates (no quartic or higher)",
      sp.Poly(exact4, xa, xb, xc, xd).total_degree() == 3)

# --- collapse of the four-term quadratic form ------------------------------
qsym = sp.Symbol('q', real=True)
four = qsym * xa**2 + qsym * xb**2 - qsym * xc**2 - qsym * (xa + xb - xc)**2
check("(check 3) equal local GVDs: the four q_j terms collapse to "
      "-2q (x_a-x_c)(x_b-x_c)",
      sp.simplify(sp.expand(four + 2 * qsym * (xa - xc) * (xb - xc))) == 0)

# --- the multiplicative form -----------------------------------------------
zero = {xa: 0, xb: 0, xc: 0}
P0, Q0, G0 = P.subs(zero), Q.subs(zero), G.subs(zero)
eP, eQ, eG = (sp.simplify((P - P0) / P0), sp.simplify((Q - Q0) / Q0),
              sp.simplify((G - G0) / G0))

check("(check 4) u = u_0 (1+eps_P)(1+eps_Q)(1+eps_G)",
      sp.simplify(sp.expand(exact - L * P0 * Q0 * G0
                            * (1 + eP) * (1 + eQ) * (1 + eG))) == 0)
check("        the linear model is the first-order truncation of that product",
      sp.simplify(sp.expand(
          sp.expand(exact - L * P0 * Q0 * G0 * (1 + eP + eQ + eG))
          - L * P0 * Q0 * G0 * (eP * eQ + eP * eG + eQ * eG
                                + eP * eQ * eG))) == 0)

print(f"\n    eps_P = {eP}")
print(f"    eps_Q = {eQ}")
print(f"    eps_G = {sp.simplify(eG)}")


# ============================================================
# 2. Exact distributional constants
# ============================================================

print("\n2. Constants of the relative-error law "
      "(rectangular-Nyquist model, x ~ U(-pi, pi))")

t = sp.Symbol('t')
uni = sp.Rational(1, 2) / sp.pi          # density of U(-pi, pi)
ya, yb, yc = sp.symbols('y_a y_b y_c', real=True)
prod = (ya - yc) * (yc - yb)


def expectation(expr):
    out = expr
    for y in (ya, yb, yc):
        out = sp.integrate(sp.expand(out) * uni, (y, -sp.pi, sp.pi))
    return sp.simplify(out)


mean = expectation(prod)
second = expectation(prod**2)
variance = sp.simplify(second - mean**2)

print(f"    E[(x_a-x_c)(x_c-x_b)]   = {mean}")
print(f"    E[((x_a-x_c)(x_c-x_b))^2] = {second}")

check("(check 5) mean = -pi^2/3   (a systematic bias, not just spread)",
      sp.simplify(mean + sp.pi**2 / 3) == 0)
check("(check 6) second moment = 8 pi^4 / 15",
      sp.simplify(second - 8 * sp.pi**4 / 15) == 0)
check("(check 7) variance = pi^4 (19/45)",
      sp.simplify(variance - sp.pi**4 * sp.Rational(19, 45)) == 0)

# The prefactors of 1/(r^2 |dn_ac dn_cb|):  divide by 4 pi^2.
c_bias = sp.simplify(sp.Abs(mean) / (4 * sp.pi**2))
c_std = sp.simplify(sp.sqrt(variance) / (4 * sp.pi**2))
c_rms = sp.simplify(sp.sqrt(second) / (4 * sp.pi**2))

print(f"    bias prefactor = {c_bias} = {float(c_bias):.4f}")
print(f"    std  prefactor = {c_std} = {float(c_std):.4f}")
print(f"    rms  prefactor = {c_rms} = {float(c_rms):.4f}")

check("(check 8) bias prefactor = 1/12",
      sp.simplify(c_bias - sp.Rational(1, 12)) == 0)
check("(check 9) std prefactor = sqrt(19/45)/4",
      sp.simplify(c_std - sp.sqrt(sp.Rational(19, 45)) / 4) == 0)
check("(check 10) rms prefactor = sqrt(1/30)",
      sp.simplify(c_rms - sp.sqrt(sp.Rational(1, 30))) == 0)


# ============================================================
# 3. Monte-Carlo against the exact phase
# ============================================================

TWOPI, THZ = 2 * np.pi, 1e12
B_V, L_V, DF = 24.5e9, 100e3, 25e9                 # the note's active case
R = DF / B_V
BETA2, BETA3 = -21.1687712327e-27, 0.1285064367e-39
W_REF = TWOPI * 193.4 * THZ
F0 = 193.4

nbeta = lambda w: BETA2 * (w - W_REF)**2 / 2 + BETA3 * (w - W_REF)**3 / 6
nbeta_1 = lambda w: BETA2 * (w - W_REF) + BETA3 * (w - W_REF)**2 / 2
nbeta_2 = lambda w: BETA2 + BETA3 * (w - W_REF)
W_ZDW = W_REF - BETA2 / BETA3

rng = np.random.default_rng(0)
N = 400_000
x = rng.uniform(-np.pi, np.pi, (N, 3))
XA, XB, XC = x.T
XD = XA + XB - XC


def phases(na, nb, nc):
    w = [TWOPI * (F0 * THZ + n * DF) for n in (na, nb, nc)]
    wa_, wb_, wc_ = w
    wl_ = wa_ + wb_ - wc_

    u0 = L_V * (nbeta(wa_) + nbeta(wb_) - nbeta(wc_) - nbeta(wl_))
    ex = L_V * (nbeta(wa_ + B_V * XA) + nbeta(wb_ + B_V * XB)
                - nbeta(wc_ + B_V * XC) - nbeta(wl_ + B_V * XD))
    li = u0 + L_V * B_V * ((nbeta_1(wa_) - nbeta_1(wl_)) * XA
                           + (nbeta_1(wb_) - nbeta_1(wl_)) * XB
                           - (nbeta_1(wc_) - nbeta_1(wl_)) * XC)
    return u0, ex, li, nbeta_2((wa_ + wb_) / 2)


print(f"\n3. Monte-Carlo, {L_V/1e3:g} km, {B_V/1e9:g} GBd, "
      f"grid {DF/1e9:g} GHz (r = {R:.4f}); ZDW at {W_ZDW/TWOPI/THZ:.3f} THz")
print(f"    {'n_a':>6}{'n_b':>7}{'n_c':>6} {'|u0| rad':>13} {'std|du|':>9} "
      f"{'measured':>10} {'predicted':>10} {'ratio':>6}")

worst = 0.0
for na, nb, nc in [(1, -1, 0), (2, -2, 1), (3, 1, 2), (10, -10, 5),
                   (50, -150, 0), (400, -1200, 0)]:
    u0, ex, li, _ = phases(na, nb, nc)
    dn_ac, dn_cb = na - nc, nc - nb

    measured = float(np.std(ex - li) / abs(u0))
    predicted = float(c_std) / (R**2 * abs(dn_ac * dn_cb))
    worst = max(worst, abs(measured / predicted - 1.0))

    print(f"    {na:>6}{nb:>7}{nc:>6} {abs(u0):13.1f} "
          f"{np.std(ex-li):9.3f} {measured:10.3e} {predicted:10.3e} "
          f"{measured/predicted:6.2f}")

check("(check 11) the law holds within 30% over four decades of |dn dn|",
      worst < 0.30)

u0, ex, li, _ = phases(10, -10, 5)
dn = (10 - 5) * (5 + 10)
bias_pred = -1.0 / (12 * R**2 * dn)
check("(check 12) the predicted systematic bias is reproduced",
      abs(float(np.mean(ex - li) / u0) / bias_pred - 1.0) < 0.05)

# --- self-protection: the dominant error is proportional to beta_2(wbar) ---
print("\n4. Self-protection near the ZDW")
n_zdw = int(round((W_ZDW / TWOPI - F0 * THZ) / DF))

cases = [("C-band pump mean", 50, -150, 0),
         ("pump mean at the ZDW", n_zdw + 100, n_zdw - 100, n_zdw + 40),
         ("all legs at the ZDW", n_zdw + 2, n_zdw - 2, n_zdw + 1)]

sizes = {}
for label, na, nb, nc in cases:
    u0, ex, li, g = phases(na, nb, nc)
    sizes[label] = float(np.std(ex - li))
    print(f"    {label:22s} beta_2(wbar) = {g*1e27:+8.3f} ps^2/km   "
          f"std|du| = {sizes[label]:8.4f} rad   |u0| = {abs(u0):12.1f}")

check("(check 13) the absolute error collapses as beta_2(wbar) -> 0",
      sizes["all legs at the ZDW"] < 0.05 * sizes["C-band pump mean"])
check("(check 14) it is below one radian wherever u_0 can reach the sheet",
      max(sizes["pump mean at the ZDW"], sizes["all legs at the ZDW"]) < 1.0)

# --- the sheet is unreachable in the C band --------------------------------
u0_min, _, _, _ = phases(1, -1, 0)
print(f"\n    minimum |u_0| on the grid, C band, |dn_ac| = |dn_cb| = 1: "
      f"{abs(u0_min):.1f} rad")
check("(check 15) C-band tuples cannot reach the coherent/sheet regime",
      abs(u0_min) > 10.0)


# ============================================================
print("\n" + "=" * 64)
failed = [name for name, ok in results if not ok]
if failed:
    print(f"{len(failed)} FAILED:")
    for name in failed:
        print(f"   {name}")
    raise SystemExit(1)

print(f"all {len(results)} checks passed")
