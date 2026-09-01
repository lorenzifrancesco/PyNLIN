"""Verification of ``docs/source/xpm_in_channel_curvature.md``.

Checks, in order:
  1. Proposition 1 -- the mismatch of an XPM quadruplet is exactly bilinear
     for a cubic beta, with beta2 evaluated at the mean of the two
     annihilated photons; and the residual of freezing that mean;
  2. Proposition 2 -- u = y (nu - 2 q s) against the directly evaluated
     Delta_beta L;
  3. Proposition 3 / eq. (5.3) -- the closed-form total pair efficiency
     against a direct Monte Carlo and against the q = 0 reduction of
     ``lorenzi_fast_method.md`` sec. 12;
  4. eq. (7.1) -- the closed-form 2PC sector against the library's direct
     CRN sector estimator, its plateau law and its nu -> infinity constants
     (``npc_sector_asymptotics.md`` sec. 4);
  5. the comparison figure
     (docs/source/_static/lorenzi-fast/xpm_curvature_2pc.png).

    .venv/bin/python analysis/standalone_analytical/verify_xpm_curvature_closed_form.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, simpson
from scipy.special import sici

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pynlin.methods.td.xhkm_mc import estimate_xhkm_sectors_direct_mc  # noqa: E402

TP = 2.0 * np.pi
FIG = REPO_ROOT / "docs" / "source" / "_static" / "lorenzi-fast" / "xpm_curvature_2pc.png"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


# ============================================================
# Closed forms
# ============================================================
def J(kappa, Y):
    """int_{-Y}^{Y} (Y - |y|) Khat(kappa y) dy, eq. (5.2)."""
    z = kappa * Y
    if abs(z) < 1e-9:
        return Y**2 - kappa**2 * Y**4 / 72.0
    si, ci = sici(abs(z))
    cin = np.euler_gamma + np.log(abs(z)) - ci
    return 4.0 / kappa**2 * (abs(z) * si - (1.0 - np.cos(z)) - cin)


def efficiency_total(nu, lam):
    """Total pair efficiency N T^2/L^2, eq. (5.3).  lam = L/L_D, nu = L/L_W."""
    v, _ = quad(lambda s: J(nu - lam * s, TP - abs(s)), -TP, TP, limit=1000)
    return v / TP**3


def _antider_P(t):
    """int Lambda_hat dt = Si(t) + i Cin(t)."""
    si, ci = sici(np.abs(t))
    at = np.abs(t)
    cin = np.where(at > 1e-300, np.euler_gamma + np.log(at + 1e-300) - ci, 0.0)
    return np.sign(t) * si + 1j * cin


def _antider_Q(t):
    """int t Lambda_hat dt = -exp(i t) + i t."""
    return -np.exp(1j * t) + 1j * t


def _link_hat(d):
    d = np.asarray(d, dtype=float)
    out = np.ones_like(d, dtype=complex)
    nz = np.abs(d) > 1e-12
    out[nz] = (np.exp(1j * d[nz]) - 1.0) / (1j * d[nz])
    return out


def coherent_inner(x, nu, lam):
    """G(x) = E_{a,u}[ mask * Lambda_hat ] at shared outer frequency x."""
    x = np.asarray(x, dtype=float)
    ell = TP - np.abs(x)
    A = nu * x
    B = abs(lam) * np.abs(x)
    small = B * ell < 1e-8
    Bs = np.where(small, 1.0, B)
    tm, tp = A - Bs * ell, A + Bs * ell
    Pp, Pm, Pa = _antider_P(tp), _antider_P(tm), _antider_P(A)
    Qp, Qm, Qa = _antider_Q(tp), _antider_Q(tm), _antider_Q(A)
    inner = ell * (Pp - Pm) - (1.0 / Bs) * (
        A * (Pa - Pm) - (Qa - Qm) - A * (Pp - Pa) + (Qp - Qa)
    )
    return np.where(small, (ell / TP) ** 2 * _link_hat(A), inner / (4.0 * np.pi**2 * Bs))


def efficiency_2pc(nu, lam, n=40001):
    """2PC collision sector, eq. (7.1)."""
    x = np.linspace(1e-9, TP, n)
    return simpson(np.abs(coherent_inner(x, nu, lam)) ** 2, x=x) / np.pi


def C_total(r):
    """nu -> infinity constant of the total, npc_sector_asymptotics.md sec. 3."""
    tail = 0.0 if abs(r - 1.0) < 1e-12 else (r - 1) * np.log(r - 1)
    return r * ((r + 1) * np.log(r + 1) + tail - 2 * r * np.log(r))


def sector_mc(nu, q, n_samples=1_500_000, seed=3):
    lam = nu / (TP * q)
    s = estimate_xhkm_sectors_direct_mc(
        beta2=lam, alpha=0.0, length=1.0, channel_spacing_over_baud=q,
        n_samples=n_samples, seed=seed,
    )
    return s


# ============================================================
# 1. Proposition 1: exact bilinear mismatch through beta3
# ============================================================
print("\nProposition 1 -- bilinear mismatch, beta3 absorbed")
rng = np.random.default_rng(11)
beta2, beta3, df, B, L = -2.17e-26, 1.0e-40, 400e9, 100e9, 80e3
N = 200_000
xa, xb, xc = rng.uniform(-np.pi, np.pi, (3, N))
xt = xa + xb - xc
keep = np.abs(xt) < np.pi
xa, xb, xc, xt = xa[keep], xb[keep], xc[keep], xt[keep]
fa, fbb = B * xa / TP, df + B * xb / TP
fc, ftt = df + B * xc / TP, B * xt / TP


def beta_of(f):
    w = TP * f
    return 0.5 * beta2 * w**2 + beta3 * w**3 / 6.0


u_exact = (beta_of(fa) + beta_of(fbb) - beta_of(fc) - beta_of(ftt)) * L
rms = np.sqrt(np.mean(u_exact**2))
b2_bar = beta2 + beta3 * TP * (fc + ftt) / 2.0
u_bil = -4 * np.pi**2 * b2_bar * (fa - fc) * (fbb - fc) * L
check("Delta_beta = -4 pi^2 beta2(omega_bar) (fa-fc)(fb-fc)",
      np.max(np.abs(u_exact - u_bil)) / rms < 1e-12,
      f"max rel dev {np.max(np.abs(u_exact - u_bil)) / rms:.1e}")
b2_frozen = beta2 + beta3 * TP * df / 2.0
u_frozen = -4 * np.pi**2 * b2_frozen * (fa - fc) * (fbb - fc) * L
eps3 = np.pi * abs(beta3) * B / abs(beta2)
check("frozen beta2_eff residual is of order pi |beta3| B / |beta2|",
      np.max(np.abs(u_exact - u_frozen)) / rms < 3 * eps3,
      f"measured {np.max(np.abs(u_exact - u_frozen)) / rms:.2e} vs eps3 = {eps3:.2e}")

# ============================================================
# 2. Proposition 2: u = y (nu - 2 q s)
# ============================================================
print("\nProposition 2 -- factorized phase")
nu_p = TP * beta2 * df * B * L
q_p = 0.5 * beta2 * B**2 * L
u_fac = (xb - xc) * (nu_p - 2 * q_p * (xa - xc))
u_quad = (
    0.5 * beta2 * (TP * fa) ** 2 + 0.5 * beta2 * (TP * fbb) ** 2
    - 0.5 * beta2 * (TP * fc) ** 2 - 0.5 * beta2 * (TP * ftt) ** 2
) * L
check("u = y (nu - 2 q s)", np.max(np.abs(u_quad - u_fac)) < 1e-9 * max(1.0, rms),
      f"max abs dev {np.max(np.abs(u_quad - u_fac)):.1e} rad, rms|u| = "
      f"{np.sqrt(np.mean(u_quad**2)):.1f} rad")

# ============================================================
# 3. Total efficiency, eq. (5.3)
# ============================================================
print("\nEq. (5.3) -- total pair efficiency")


def H_doc(theta):
    if abs(theta) < 1e-8:
        return 2.0 / 3.0
    return 1.0 / (np.pi**2 * theta**2) - np.sin(TP * theta) / (2 * np.pi**3 * theta**3)


ok = True
for nu in (0.0, 0.5, 2.0, 7.0, 30.0, 120.0):
    ref, _ = quad(lambda t: 2 * (1 - t) * H_doc(nu * t), 0, 1, limit=400)
    ok &= abs(efficiency_total(nu, 0.0) - ref) < 1e-8
check("q = 0 limit reproduces lorenzi_fast_method.md sec. 12", ok, "8 digits")

rng = np.random.default_rng(5)
ok, worst = True, 0.0
for nu, lam in ((0.3, 0.3), (1.0, 0.5), (2.0, 2.0), (5.0, 1.0), (20.0, 10.0), (60.0, 50.0)):
    n = 2_000_000
    xa, xb, xc = rng.uniform(-np.pi, np.pi, (3, n))
    m = np.abs(xa + xb - xc) < np.pi
    u = (xb - xc)[m] * (nu - lam * (xa - xc)[m])
    du = np.where(np.abs(u) > 1e-12, u, 1.0)
    k = np.where(np.abs(u) > 1e-12, 4 * np.sin(du / 2) ** 2 / du**2, 1.0)
    mc = k.sum() / n
    se = np.sqrt((np.sum(k**2) / n - mc**2) / n)
    d = abs(mc - efficiency_total(nu, lam)) / se
    worst = max(worst, d)
    ok &= d < 4.0
check("eq. (5.3) vs direct Monte Carlo", ok, f"worst deviation {worst:.1f} sigma")

pl_ok = True
for lam, r in ((0.01, 1), (0.03, 1), (0.01, 4)):
    nu = TP * r * lam
    law = (2 / 3) * (1 - np.pi**2 * nu**2 / 30 - 2 * np.pi**4 * lam**2 / 315)
    pl_ok &= abs(law / efficiency_total(nu, lam) - 1) < 1e-3
check("plateau law (6.1)", pl_ok, "|error| < 0.1% for L/L_W < 0.2")

as_ok = True
for r in (2.0, 4.0):
    lam = 40.0
    nu = TP * r * lam
    as_ok &= abs(nu * efficiency_total(nu, lam) / C_total(r) - 1) < 5e-3
check("sheet law (6.2) with C(r)", as_ok, "|error| < 0.5% at nu > 10^3")

# ============================================================
# 4. 2PC sector, eq. (7.1)
# ============================================================
print("\nEq. (7.1) -- 2PC collision sector")
ok, worst = True, 0.0
rows = []
for q in (1.0, 2.0, 4.0):
    for nu in (0.0, 1.0, 10.0, 100.0, 300.0):
        lam = nu / (TP * q)
        s = sector_mc(nu, q)
        a, b = efficiency_total(nu, lam), efficiency_2pc(nu, lam)
        d1 = abs(s.n1 - a) / s.n1_stderr if s.n1_stderr > 0 else 0.0
        d2 = abs(s.n_2pc - b) / s.n_2pc_stderr if s.n_2pc_stderr > 0 else 0.0
        worst = max(worst, d1, d2)
        ok &= max(d1, d2) < 4.0
        rows.append((q, nu, a, s.n1, s.n1_stderr, b, s.n_2pc, s.n_2pc_stderr))
check("eqs. (5.3) and (7.1) vs estimate_xhkm_sectors_direct_mc", ok,
      f"worst deviation {worst:.1f} sigma over 15 (q, nu) points")

check("2PC plateau is 2/5", abs(efficiency_2pc(0.0, 0.0) - 0.4) < 1e-6)
pl_ok = True
for lam, r in ((0.01, 1), (0.02, 1), (0.005, 4)):
    nu = TP * r * lam
    law = (2 / 5) * (1 - np.pi**2 * nu**2 / 63 - 10 * np.pi**4 * lam**2 / 567)
    pl_ok &= abs(law / efficiency_2pc(nu, lam) - 1) < 1e-3
check("2PC plateau law (7.2)", pl_ok, "|error| < 0.1% for L/L_W < 0.15")

ref_C2 = {1.0: 0.8938, 2.0: 0.9172, 4.0: 0.9507}
as_ok = True
for q, c2 in ref_C2.items():
    lam = 1e4 / (TP * q)
    as_ok &= abs(1e4 * efficiency_2pc(1e4, lam, n=200001) / c2 - 1) < 1e-2
check("2PC constants match npc_sector_asymptotics.md sec. 4", as_ok,
      "within 1% at nu = 10^4")

# ============================================================
# 5. Figure
# ============================================================
print("\nFigure")
nus = np.logspace(-1.2, 3.2, 90)
qs = (1.0, 2.0, 4.0)
colors = {1.0: "#c0392b", 2.0: "#2980b9", 4.0: "#27ae60"}

tot = {q: np.array([efficiency_total(n, n / (TP * q)) for n in nus]) for q in qs}
pc2 = {q: np.array([efficiency_2pc(n, n / (TP * q)) for n in nus]) for q in qs}
tot_lin = np.array([efficiency_total(n, 0.0) for n in nus])
pc2_lin = np.array([efficiency_2pc(n, 0.0) for n in nus])

fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.1))

for q in qs:
    ax[0].loglog(nus, tot[q], color=colors[q], lw=1.6, label=f"total, $r={q:.0f}$")
    ax[0].loglog(nus, pc2[q], color=colors[q], lw=1.6, ls="--")
ax[0].axhline(2 / 3, color="0.6", lw=0.8)
ax[0].axhline(2 / 5, color="0.6", lw=0.8, ls="--")
ax[0].loglog(nus, C_total(4.0) / nus, color="0.4", lw=0.8, ls=":")
ax[0].set_xlabel(r"$L/L_W$")
ax[0].set_ylabel(r"$N\,T^2/L^2$")
ax[0].set_title("solid: total   dashed: 2PC")
ax[0].legend(fontsize=7, frameon=False)
ax[0].grid(alpha=0.25, which="both")

for q in qs:
    ax[1].semilogx(nus, nus * tot[q], color=colors[q], lw=1.6)
    ax[1].semilogx(nus, nus * pc2[q], color=colors[q], lw=1.6, ls="--")
    ax[1].axhline(C_total(q), color=colors[q], lw=0.7, ls=":")
    ax[1].axhline(ref_C2[q], color=colors[q], lw=0.7, ls=":")
ax[1].semilogx(nus, nus * tot_lin, color="k", lw=1.0, alpha=0.7,
               label=r"linear model, $L/L_D=0$")
ax[1].set_xlabel(r"$L/L_W$")
ax[1].set_ylabel(r"$(L/L_W)\,N\,T^2/L^2$")
ax[1].set_ylim(0, 1.6)
ax[1].set_title(r"compensated; dotted: $C_S(r)$")
ax[1].legend(fontsize=7, frameon=False, loc="lower right")
ax[1].grid(alpha=0.25, which="both")

for q in qs:
    ax[2].semilogx(nus, 100 * (tot[q] / tot_lin - 1), color=colors[q], lw=1.6,
                   label=f"total, $r={q:.0f}$")
    ax[2].semilogx(nus, 100 * (pc2[q] / pc2_lin - 1), color=colors[q], lw=1.6, ls="--")
ax[2].axhline(0, color="0.6", lw=0.8)
ax[2].set_xlabel(r"$L/L_W$")
ax[2].set_ylabel(r"departure from the linear model [%]")
ax[2].set_title("cost of setting $L/L_D=0$")
ax[2].legend(fontsize=7, frameon=False)
ax[2].grid(alpha=0.25, which="both")

fig.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=160)
print(f"  wrote {FIG.relative_to(REPO_ROOT)}")

print("\nSummary")
for name, ok in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
n_fail = sum(1 for _, ok in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
