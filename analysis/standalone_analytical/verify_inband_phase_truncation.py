"""Verification of every derivation and every number in
``docs/source/inband_phase_truncation.md``.

The note classifies closed-form GN models by how many of the two in-band
frequency coordinates of an XPM pair survive in the single-span phase.  This
script proves the symbolic identities with ``sympy``, evaluates the
distributional constants by exact integration, and reproduces the quadrature
tables with randomized Sobol sampling.

    python analysis/standalone_analytical/verify_inband_phase_truncation.py
"""

import numpy as np
import sympy as sp
from scipy.special import sici
from scipy.stats import qmc

C_LIGHT = 2.99792458e8

results = []


def check(name, ok):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


# ============================================================
# 1. Symbolic identities: the exact tuple phase
# ============================================================
print("\n1. Exact tuple phase (Sec. 2 of the note)")

b0, b1, b2, b3 = sp.symbols('beta0 beta1 beta2 beta3', real=True)
wa, wb, wc = sp.symbols('omega_a omega_b omega_c', real=True)
xa, xb, xc, Bs, L, Rs = sp.symbols('x_a x_b x_c B L R_s', real=True, positive=True)

beta2o = lambda w: b0 + b1 * w + b2 * w**2 / 2
beta3o = lambda w: b0 + b1 * w + b2 * w**2 / 2 + b3 * w**3 / 6

Wa, Wb, Wc = wa + Rs * xa, wb + Rs * xb, wc + Rs * xc
Wt = Wa + Wb - Wc                              # energy conservation

# (1a) beta0 and beta1 cancel identically for any energy-conserving tuple
gen0, gen1 = sp.symbols('g0 g1', real=True)
check("(1a) the beta0 and beta1 parts of Delta beta vanish identically",
      sp.simplify(sp.expand((gen0 + gen1 * Wa) + (gen0 + gen1 * Wb)
                            - (gen0 + gen1 * Wc) - (gen0 + gen1 * Wt))) == 0)

# (1b) uniform beta2: Delta beta = -beta2 (W_a - W_c)(W_b - W_c)
db2 = sp.expand(beta2o(Wa) + beta2o(Wb) - beta2o(Wc) - beta2o(Wt))
check("(1b) uniform beta2: Delta beta = -beta2 (W_a - W_c)(W_b - W_c)",
      sp.simplify(db2 + b2 * (Wa - Wc) * (Wb - Wc)) == 0)

# (1c) cubic beta: Delta beta = (W_a - W_c)(W_c - W_b) beta_2(mean(W_a,W_b))
db3 = sp.expand(beta3o(Wa) + beta3o(Wb) - beta3o(Wc) - beta3o(Wt))
fac = (Wa - Wc) * (Wc - Wb) * (b2 + b3 * (Wa + Wb) / 2)
check("(1c) cubic beta: Delta beta = (W_a-W_c)(W_c-W_b) beta_2((W_a+W_b)/2)",
      sp.simplify(sp.expand(db3 - fac)) == 0)
check("     each of the three factors is affine in (x_a, x_b, x_c)",
      all(sp.Poly(sp.expand(f), xa, xb, xc).total_degree() <= 1
          for f in ((Wa - Wc), (Wc - Wb), (b2 + b3 * (Wa + Wb) / 2))))

# (1d) the four-term quadratic form is the Gan Hessian on the centre slice
th_j, th_k, th_m = sp.symbols('vartheta_j vartheta_k vartheta_m', real=True)
four = th_j * xa**2 + th_k * xb**2 - th_m * xc**2 - 0          # x_t = 0 slice
four = four.subs(xc, xa + xb)
hess = ((th_j - th_m) * xa**2 + (th_k - th_m) * xb**2 - 2 * th_m * xa * xb)
check("(1d) Gan centre slice: the quadratic phase equals the Lorenzi Hessian",
      sp.simplify(sp.expand(th_j * xa**2 + th_k * xb**2
                            - th_m * (xa + xb)**2 - hess)) == 0)


# ============================================================
# 2. Symbolic identity: the XPM island reduction u = nu X g
# ============================================================
print("\n2. XPM island reduction (Sec. 3)")

xin, x1, x2, r = sp.symbols('x_in x_1 x_2 r', real=True)
df = r * Bs                                    # channel spacing
wt_, wi_ = sp.symbols('omega_t omega_i', real=True)

# CUT input at x_in, interferer unconjugated at x_2, conjugated at x_1
Wa_ = wt_ + Bs * xin
Wb_ = wi_ + Bs * x2
Wc_ = wi_ + Bs * x1
db = -b2 * (Wa_ - Wc_) * (Wb_ - Wc_)
db = db.subs(wi_, wt_ + 2 * sp.pi * df)

nu = 2 * sp.pi * b2 * Bs * df * L
X = x2 - x1
y = x1 - xin
target = nu * X * (1 + y / (2 * sp.pi * r))
check("(2a) u = nu X (1 + y/(2 pi r)) exactly, with X = x_2-x_1, y = x_1-x_in",
      sp.simplify(sp.expand(L * db - target)) == 0)

xout = xin + x2 - x1
check("(2b) y = x_1 - x_in = x_2 - x_out identically",
      sp.simplify((x2 - xout) - y) == 0)

# the same phase written as the four-term quadratic form used by
# ``qmc_xpm_ground_truth`` in src/pynlin/methods/td/fast_nlin.py
qq = b2 * Bs**2 * L / 2
four_term = qq * (xin**2 + x2**2 - x1**2 - xout**2)
check("(2c) u - nu X equals the four-term quadratic phase q(x_in^2+x_2^2"
      "-x_1^2-x_out^2)",
      sp.simplify(sp.expand(target - nu * X - four_term)) == 0)


# ============================================================
# 3. The three asymptotic constants, by exact integration
# ============================================================
print("\n3. Asymptotic constants (Sec. 4)")

t, rr = sp.symbols('t r', real=True, positive=True)
xo = sp.symbols('x_out', real=True)

Cex_cf = rr * ((rr + 1) * sp.log(rr + 1) + (rr - 1) * sp.log(rr - 1)
               - 2 * rr * sp.log(rr))
Cfr_cf = rr * sp.log((rr + sp.Rational(1, 2)) / (rr - sp.Rational(1, 2)))
Cxo_cf = rr * sp.log((rr + sp.Rational(1, 2) - xo / (2 * sp.pi))
                     / (rr - sp.Rational(1, 2) - xo / (2 * sp.pi)))

R_TEST = [sp.Rational(25, 24), sp.Rational(5, 4), sp.Rational(3, 2),
          sp.Rational(2), sp.Rational(7, 2)]


def num(expr, **subs):
    return sp.N(expr.subs(subs), 30)


# C_exact: y triangular on (-2 pi, 2 pi), i.e. t = y/(2 pi) with density 1 - |t|
ok = True
for rv in R_TEST:
    lhs = sp.integrate((1 - sp.Abs(t)) / (1 + t / rv), (t, -1, 1))
    ok &= abs(num(lhs) - num(Cex_cf, rr=rv)) < sp.Rational(1, 10**20)
check("(3a) C_exact = r[(r+1)ln(r+1)+(r-1)ln(r-1)-2r ln r]  (triangular y)", ok)

# C_froz: y uniform on (-pi, pi), i.e. t uniform on (-1/2, 1/2)
ok = True
for rv in R_TEST:
    lhs = sp.integrate(1 / (1 + t / rv), (t, -sp.Rational(1, 2), sp.Rational(1, 2)))
    ok &= abs(num(lhs) - num(Cfr_cf, rr=rv)) < sp.Rational(1, 10**20)
check("(3b) C_froz = r ln[(r+1/2)/(r-1/2)]  (uniform y)", ok)

# C_lin: point mass at y = 0
check("(3c) C_lin = 1  (y frozen at 0)",
      sp.simplify((1 / (1 + t / rr)).subs(t, 0)) == 1)

# band-resolved profile: y uniform on (-pi - x_out, pi - x_out)
ok = True
for rv in R_TEST:
    for xov in (-sp.pi * sp.Rational(9, 10), 0, sp.pi * sp.Rational(9, 10)):
        lhs = sp.integrate(1 / (1 + (t - xov / (2 * sp.pi)) / rv),
                           (t, -sp.Rational(1, 2), sp.Rational(1, 2)))
        ok &= abs(num(lhs) - num(Cxo_cf, rr=rv, xo=xov)) < sp.Rational(1, 10**20)
check("(3d) C(x_out) = r ln[(r+1/2-x_out/2pi)/(r-1/2-x_out/2pi)]", ok)

# the band average of C(x_out) is C_exact: x_out uniform and x_2 uniform and
# independent make y = x_2 - x_out triangular
ok = True
for rv in R_TEST:
    avg = sp.N(sp.Integral(Cxo_cf.subs(rr, rv), (xo, -sp.pi, sp.pi)) / (2 * sp.pi), 25)
    ok &= abs(avg - num(Cex_cf, rr=rv)) < sp.Rational(1, 10**15)
check("(3e) band average of C(x_out) equals C_exact", ok)


def C_lin(r):
    return np.ones_like(np.asarray(r, dtype=float))


def C_froz(r):
    r = np.asarray(r, dtype=float)
    return r * np.log((r + 0.5) / (r - 0.5))


def C_exact(r):
    r = np.asarray(r, dtype=float)
    return r * ((r + 1) * np.log(r + 1) + (r - 1) * np.log(r - 1)
                - 2 * r * np.log(r))


# large-r expansions
e = sp.symbols('e', positive=True)   # e = 1/r
ser_f = sp.series(Cfr_cf.subs(rr, 1 / e), e, 0, 5).removeO()
ser_e = sp.series(Cex_cf.subs(rr, 1 / e), e, 0, 5).removeO()
check("(3f) C_froz = 1 + 1/(12 r^2) + O(r^-4)",
      sp.simplify(ser_f - (1 + e**2 / 12 + e**4 / 80)) == 0)
check("(3g) C_exact = 1 + 1/(6 r^2) + O(r^-4)",
      sp.simplify(sp.expand(ser_e - (1 + e**2 / 6 + e**4 / 15))) == 0)

R_GAN = 100.0 / 96.0
print(f"\n  r = {R_GAN:.6f}:  C_lin = 1  C_froz = {C_froz(R_GAN):.5f} "
      f"({10*np.log10(C_froz(R_GAN)):+.3f} dB)  "
      f"C_exact = {C_exact(R_GAN):.5f} ({10*np.log10(C_exact(R_GAN)):+.3f} dB)")

print("\n  Table: separation of the three constants versus r")
print("  | r | C_froz [dB] | C_exact [dB] | C_exact/C_froz [dB] |")
print("  |---:|---:|---:|---:|")
for rv in (R_GAN, 1.25, 1.5, 2.0, 4.0, 8.0):
    a, b = 10 * np.log10(C_froz(rv)), 10 * np.log10(C_exact(rv))
    print(f"  | {rv:.3f} | {a:.3f} | {b:.3f} | {b-a:.3f} |")


# ============================================================
# 4. Strict-FWM relative-error moments, by exact integration
# ============================================================
print("\n4. Strict-FWM relative-error laws (Sec. 8)")

u_ = sp.symbols('u', real=True)
dn1, dn2 = sp.symbols('Delta_n_ac Delta_n_cb', real=True, positive=True)
eP = (xa - xc) / (2 * sp.pi * rr * dn1)
eQ = (xc - xb) / (2 * sp.pi * rr * dn2)
prod = sp.expand(eP * eQ)


def Eunif(expr):
    e = expr
    for v in (xa, xb, xc):
        e = sp.integrate(e, (v, -sp.pi, sp.pi)) / (2 * sp.pi)
    return sp.simplify(e)


m1 = Eunif(prod)
m2 = Eunif(prod**2)
var = sp.simplify(m2 - m1**2)
check("(4a) bias  E[eps_P eps_Q] = -1/(12 r^2 dn_ac dn_cb)",
      sp.simplify(m1 + 1 / (12 * rr**2 * dn1 * dn2)) == 0)
check("(4b) rms   sqrt(E[.^2]) = sqrt(1/30)/(r^2 dn_ac dn_cb)",
      sp.simplify(m2 - sp.Rational(1, 30) / (rr**4 * dn1**2 * dn2**2)) == 0)
check("(4c) std   sqrt(Var)    = sqrt(19/45)/(4 r^2 dn_ac dn_cb)",
      sp.simplify(var - sp.Rational(19, 45) / (16 * rr**4 * dn1**2 * dn2**2)) == 0)


# ============================================================
# 5. Multi-span phased-array identity
# ============================================================
print("\n5. Multi-span accumulation (Sec. 9)")

phi = sp.symbols('varphi', real=True)
for N in (2, 3, 4, 5):
    S = sp.Add(*[sp.exp(sp.I * n * phi) for n in range(N)])
    chi = sp.simplify(sp.expand(sp.re(S)**2 + sp.im(S)**2))
    dirichlet = sp.sin(N * phi / 2)**2 / sp.sin(phi / 2)**2
    cosine = N + 2 * sp.Add(*[(N - n) * sp.cos(n * phi) for n in range(1, N)])
    ok = (sp.simplify(sp.trigsimp(chi - cosine)) == 0
          and abs(complex(chi.subs(phi, sp.Rational(3, 7)))
                  - complex(dirichlet.subs(phi, sp.Rational(3, 7)))) < 1e-12)
    if not ok:
        break
check("(5a) chi_N = |sum e^{i n phi}|^2 = sin^2(N phi/2)/sin^2(phi/2)"
      " = N + 2 sum (N-n) cos(n phi)", ok)


# ============================================================
# 6. Masked-kernel quadrature
# ============================================================
print("\n6. Masked-kernel quadrature (Secs. 6 and 10)")


def kernel(u, a):
    """|int_0^1 A(s) e^{ius} ds|^2 normalized to its value at u = 0.

    A(s) = exp(-a s); a = alpha * L in nepers.  a = 0 is the flat profile.
    """
    u = np.asarray(u, dtype=float)
    if a == 0.0:
        out = np.ones_like(u)
        nz = u != 0.0
        out[nz] = 4.0 * np.sin(u[nz] / 2.0) ** 2 / u[nz] ** 2
        return out
    z = 1j * u - a
    lam = np.expm1(z) / z
    lam0 = -np.expm1(-a) / a
    return (np.abs(lam) / lam0) ** 2


def pair_efficiency(nu, r, alpha_l, models, m=16, seeds=4, seed0=12345):
    """E[k(u) 1{|x_out|<pi}] for each phase model, on common random numbers.

    Returns (mean, standard error over the independent scrambles).
    """
    acc = {k: [] for k in models}
    for s in range(seeds):
        pts = qmc.Sobol(3, scramble=True, seed=seed0 + s).random_base2(m)
        xin, x1, x2 = (2 * np.pi * (pts.T - 0.5))
        xout = xin + x2 - x1
        mask = np.abs(xout) < np.pi
        X = x2 - x1
        g = {'lin': np.ones_like(X),
             'froz': 1.0 + x2 / (2 * np.pi * r),
             'exact': 1.0 + (x2 - xout) / (2 * np.pi * r)}
        for k in models:
            acc[k].append(np.mean(kernel(nu * X * g[k], alpha_l) * mask))
    return ({k: float(np.mean(v)) for k, v in acc.items()},
            {k: float(np.std(v, ddof=1) / np.sqrt(seeds)) for k, v in acc.items()})


EULER = 0.5772156649015328606


def _Phi(s):
    """int_0^s k(t) dt = 2 Si(s) - 4 sin^2(s/2)/s, with k the flat kernel."""
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    nz = s != 0.0
    si, _ = sici(s[nz])
    out[nz] = 2.0 * si - 4.0 * np.sin(s[nz] / 2.0) ** 2 / s[nz]
    return out


def _Psi(s):
    """int_0^s |t| k(t) dt = sign(s) 2 Cin(|s|)."""
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    nz = s != 0.0
    a = np.abs(s[nz])
    _, ci = sici(a)
    out[nz] = np.sign(s[nz]) * 2.0 * (EULER + np.log(a) - ci)
    return out


_GL = {}


def _gl(n, a, b):
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    x, w = _GL[n]
    a = np.asarray(a, dtype=float)[..., None]
    b = np.asarray(b, dtype=float)[..., None]
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def F_reduced(nu, r, model, n=None):
    """Deterministic flat-profile pair efficiency, X integrated in closed form.

    'lin' and 'exact' use (x_in, y, X) coordinates, in which g depends on y
    only, so the X integral is [Phi(nu g X_hi) - Phi(nu g X_lo)]/(nu g).
    'froz' uses (x_2, x_in, X), in which g depends on x_2 only; x_in is
    integrated first, giving the weight (2 pi - |X|), and the X integral is a
    combination of Phi and Psi.
    """
    if n is None:
        n = 4000 if model == 'froz' else 600
    if model == 'froz':
        x2, w2 = _gl(n, -np.pi, np.pi)
        g = 1.0 + x2 / (2 * np.pi * r)
        ng = nu * g
        lo, hi = x2 - np.pi, x2 + np.pi
        J = (2 * np.pi * (_Phi(ng * hi) - _Phi(ng * lo)) / ng
             - (_Psi(ng * hi) - _Psi(ng * lo)) / ng**2)
        return float(np.sum(w2 * J) / (2 * np.pi) ** 3)
    xi, wi = _gl(n, -np.pi, np.pi)
    total = 0.0
    for lo_y, hi_y in ((-np.pi - xi, np.zeros_like(xi)),
                       (np.zeros_like(xi), np.pi - xi)):
        yy, wy = _gl(n, lo_y, hi_y)
        g = np.ones_like(yy) if model == 'lin' else 1.0 + yy / (2 * np.pi * r)
        xv = xi[:, None]
        lo = np.maximum(-np.pi - xv - yy, -np.pi - xv)
        hi = np.minimum(np.pi - xv - yy, np.pi - xv)
        ng = nu * g
        total += np.sum(wi[:, None] * wy * (_Phi(ng * hi) - _Phi(ng * lo)) / ng)
    return float(total / (2 * np.pi) ** 3)


print("\n  Convergence of |nu| F to the three constants at r = %.6f"
      " (flat span, X integrated exactly)" % R_GAN)
print("  | nu | lin | froz | exact |")
print("  |---:|---:|---:|---:|")
prev = None
for nu_v in (300.0, 1000.0, 3000.0, 10000.0, 30000.0):
    vals = {k: nu_v * F_reduced(nu_v, R_GAN, k) for k in ('lin', 'froz', 'exact')}
    print(f"  | {nu_v:.0f} | {vals['lin']:.5f} | {vals['froz']:.5f} "
          f"| {vals['exact']:.5f} |")
    prev = vals
print(f"  | limit | 1.00000 | {C_froz(R_GAN):.5f} | {C_exact(R_GAN):.5f} |")
check("(6a) |nu| F approaches the three closed forms within 0.1 % at nu = 3e4",
      abs(prev['lin'] - 1.0) < 1e-3
      and abs(prev['froz'] / C_froz(R_GAN) - 1) < 1e-3
      and abs(prev['exact'] / C_exact(R_GAN) - 1) < 1e-3)

stable = all(abs(F_reduced(3000.0, R_GAN, k, n=n_a)
                 / F_reduced(3000.0, R_GAN, k, n=2 * n_a) - 1) < 1e-6
             for k, n_a in (('lin', 600), ('exact', 600), ('froz', 2000)))
check("(6b) the reduced quadrature is converged in the node count", stable)

# the randomized Sobol estimator used for the Gan grid agrees at moderate nu
mean, se = pair_efficiency(18.4, R_GAN, 0.0, ('lin', 'froz', 'exact'), m=20)
check("(6c) Sobol and reduced quadrature agree at nu = 18.4 (Gan grid)",
      all(abs(mean[k] / F_reduced(18.4, R_GAN, k) - 1) < 3e-3
          for k in ('lin', 'froz', 'exact')))

# band-resolved profile at large nu
print("\n  Band-resolved adjacent-pair kernel at nu = 1000, r = %.6f" % R_GAN)
print("  | x_out/pi | masked kernel x nu | analytic C(x_out) | vs centre [dB] |")
print("  |---:|---:|---:|---:|")


def profile_at(xout_fixed, nu_v, r, m=20, seeds=4, seed0=777):
    """|nu| E[k] at fixed CUT output offset: x_in and x_2 free, x_1 determined."""
    vals = []
    for s in range(seeds):
        pts = qmc.Sobol(2, scramble=True, seed=seed0 + s).random_base2(m)
        xin, x2 = (2 * np.pi * (pts.T - 0.5))
        x1 = xin + x2 - xout_fixed
        mask = np.abs(x1) < np.pi
        X = x2 - x1
        g = 1.0 + (x2 - xout_fixed) / (2 * np.pi * r)
        # the x_1 marginal is the free direction absorbed by the delta in X
        vals.append(nu_v * np.mean(kernel(nu_v * X * g, 0.0) * mask))
    return float(np.mean(vals))


C0 = float(C_froz(R_GAN))
prof_ok = True
for xo_v in (-0.9, -0.5, 0.0, 0.5, 0.9):
    ana = R_GAN * np.log((R_GAN + 0.5 - xo_v / 2) / (R_GAN - 0.5 - xo_v / 2))
    num = profile_at(xo_v * np.pi, 1000.0, R_GAN)
    print(f"  | {xo_v:+.2f} | {num:.5f} | {ana:.5f} | "
          f"{10*np.log10(ana/C0):+.3f} |")
    prof_ok &= abs(num - ana) / ana < 0.03
print(f"  | band average | -- | {C_exact(R_GAN):.5f} | "
      f"{10*np.log10(C_exact(R_GAN)/C0):+.3f} |")
check("(6d) the band-resolved kernel matches C(x_out)", prof_ok)


# ============================================================
# 7. Gan O-band grid
# ============================================================
print("\n7. Gan O-band grid (Sec. 10)")

LAM0, SLOPE, CURV = 1302.3e-9, 0.087e3, -9.714e-5 * 1e12   # SI: s/m^3, s/m^4
B_SYM, DF, L_SPAN, N_CH = 96e9, 100e9, 80e3, 161
ALPHA_DB = 0.34


def D_of(lam):
    dl = lam - LAM0
    return SLOPE * dl + 0.5 * CURV * dl**2          # s/m^2  (== ps/nm/km * 1e-6)


def beta2_of(lam):
    return -D_of(lam) * lam**2 / (2 * np.pi * C_LIGHT)


def lam_of_f(f):
    return C_LIGHT / f


def pair_walkoff(lam):
    """nu = B L |beta1(omega_i) - beta1(omega_t)|, integrating beta2 across the pair."""
    ft = C_LIGHT / lam
    fi = ft + DF
    w = np.linspace(2 * np.pi * ft, 2 * np.pi * fi, 2001)
    b2 = beta2_of(lam_of_f(w / (2 * np.pi)))
    return B_SYM * L_SPAN * abs(np.trapezoid(b2, w))


LAMS = [1349.19e-9, 1337.15e-9, 1325.33e-9, 1313.71e-9, 1307.98e-9,
        1302.30e-9, 1291.08e-9, 1280.06e-9, 1258.56e-9]
def _alpha_np(a_db):
    return a_db / 10.0 * np.log(10.0) * (L_SPAN / 1e3)


alpha_l = _alpha_np(ALPHA_DB)
for a_db in (0.28, ALPHA_DB, 0.40):
    al = _alpha_np(a_db)
    print(f"  alpha = {a_db:.2f} dB/km:  alpha L = {al:.3f} Np,  L_eff = "
          f"{(1 - np.exp(-al)) / (al / L_SPAN) / 1e3:.2f} km")

print("\n  Adjacent pair, r = %.6f, relative to the linear phase" % R_GAN)
print("  | lambda [nm] | D [ps/(nm km)] | L/L_D | |nu| | froz [dB] | exact [dB]"
      " | residual [dB] | froz loss [dB] | exact loss [dB] | residual loss [dB] |")
print("  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
adjacent = {}
se_max = 0.0
for lam in LAMS:
    b2 = beta2_of(lam)
    d = abs(b2) * B_SYM**2 * L_SPAN
    nu_v = pair_walkoff(lam)
    row = [lam * 1e9, D_of(lam) * 1e6, d, nu_v]
    for al in (0.0, alpha_l):
        mean, se = pair_efficiency(nu_v, R_GAN, al, ('lin', 'froz', 'exact'), m=18)
        f_db = 10 * np.log10(mean['froz'] / mean['lin'])
        e_db = 10 * np.log10(mean['exact'] / mean['lin'])
        row += [f_db, e_db, e_db - f_db]
    adjacent[lam] = row
    se_max = max(se_max, max(se[k] / mean[k] for k in mean))
    print("  | " + " | ".join(f"{v:.3f}" if i else f"{v:.2f}"
                              for i, v in enumerate(row)) + " |")

print(f"  largest relative standard error in the table above: {se_max:.2e}"
      f"  ({10*np.log10(1+se_max):.4f} dB)")

print("\n  Sum over 80 interferer spacings at locally uniform beta2")
print("  | lambda [nm] | L/L_D | loss [dB/km] | froz [dB] | exact [dB] | residual [dB] |")
print("  |---:|---:|---:|---:|---:|---:|")
for lam in (1349.19e-9, 1325.33e-9, 1258.56e-9):
    nu1 = pair_walkoff(lam)
    d = abs(beta2_of(lam)) * B_SYM**2 * L_SPAN
    for al, adb in ((0.0, 0.0), (alpha_l, ALPHA_DB)):
        tot = {'lin': 0.0, 'froz': 0.0, 'exact': 0.0}
        for m_idx in range(1, 81):
            mean, _ = pair_efficiency(m_idx * nu1, m_idx * R_GAN, al,
                                      ('lin', 'froz', 'exact'), m=14, seeds=2)
            for k in tot:
                tot[k] += mean[k]
        f_db = 10 * np.log10(tot['froz'] / tot['lin'])
        e_db = 10 * np.log10(tot['exact'] / tot['lin'])
        print(f"  | {lam*1e9:.2f} | {d:.3f} | {adb:.2f} | {f_db:.3f} "
              f"| {e_db:.3f} | {e_db-f_db:.3f} |")

print("\n  Reference point r = 1 (adjacent Nyquist spacing), flat span")
for nu_v in (300.0, 3000.0, 30000.0):
    fl = F_reduced(nu_v, 1.0, 'lin')
    fe = F_reduced(nu_v, 1.0, 'exact')
    print(f"    nu = {nu_v:>7.0f}:  |nu| F_lin = {nu_v*fl:.4f}, "
          f"|nu| F_exact = {nu_v*fe:.4f}, linear deficit = "
          f"{100*(1 - fl/fe):.1f} %")
print(f"    C_exact(1) = 2 ln 2 = {2*np.log(2):.4f}, "
      f"limiting deficit = {100*(1 - 1/(2*np.log(2))):.1f} %")


# ============================================================
print("\n" + "=" * 62)
bad = [n for n, ok in results if not ok]
print(f"{len(results) - len(bad)}/{len(results)} checks passed")
for n in bad:
    print("  FAILED:", n)
