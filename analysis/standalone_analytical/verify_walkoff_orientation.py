"""Verification of the claims in section 10.4 of ``lorenzi_fast_method.md``.

Every assertion of that section is checked here, symbolically where the claim is
an identity and by Monte-Carlo where the claim is about a probability density.

    python analysis/standalone_analytical/verify_walkoff_orientation.py

The dispersion model is the global cubic one,

    beta(w) = beta0 + beta1 w + beta2 w^2/2 + beta3 w^3/6,

and the four waves obey  w_a + w_b = w_c + w_l  (energy conservation), with
w_l the landing frequency of the mixing product.  Coordinates are the common
mean and the two half-splittings

    wbar = (w_a + w_b)/2 = (w_c + w_l)/2,
    Dp   = (w_a - w_b)/2,
    Dm   = (w_c - w_l)/2,

written ``(S, p, q)`` in phase_matching_planes.md.
"""

import itertools

import numpy as np
import sympy as sp


PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok):
    results.append((name, bool(ok)))
    print(f"  [{PASS if ok else FAIL}] {name}")
    return ok


# ============================================================
# Symbolic part
# ============================================================

wbar, Dp, Dm = sp.symbols('wbar Delta_p Delta_m', real=True)
b1, b2, b3 = sp.symbols('beta1 beta2 beta3', real=True)
w_t = sp.symbols('omega_t', real=True)

beta = lambda w: b1 * w + b2 * w**2 / 2 + b3 * w**3 / 6
beta_1 = lambda w: sp.expand(b1 + b2 * w + b3 * w**2 / 2)     # group delay
beta_2 = lambda w: b2 + b3 * w                                # local GVD (cubic)

w_a, w_b = wbar + Dp, wbar - Dp
w_c, w_l = wbar + Dm, wbar - Dm

Dbeta = sp.expand(beta(w_a) + beta(w_b) - beta(w_c) - beta(w_l))

print("\nSymbolic identities")

check("(10.4.1) Delta-beta = (Dp^2 - Dm^2) * beta_2(wbar)",
      sp.simplify(Dbeta - (Dp**2 - Dm**2) * beta_2(wbar)) == 0)

# --- the walk-off vector is the gradient of Delta-beta ---------------------
# Free coordinates are the three legs; the landing frequency follows them.
xa, xb, xc = sp.symbols('x_a x_b x_c', real=True)
legs = {'a': w_a + xa, 'b': w_b + xb, 'c': w_c + xc}
landing = legs['a'] + legs['b'] - legs['c']

Dbeta_shifted = (beta(legs['a']) + beta(legs['b'])
                 - beta(legs['c']) - beta(landing))

gradient = [sp.expand(sp.diff(Dbeta_shifted, x).subs({xa: 0, xb: 0, xc: 0}))
            for x in (xa, xb, xc)]

walkoff = {k: sp.expand(beta_1(v) - beta_1(w_l)) for k, v in
           {'a': w_a, 'b': w_b, 'c': w_c}.items()}

check("(10.4.2) grad(Delta-beta) = (nu_a, nu_b, -nu_c), landing frame",
      all(sp.simplify(g - c) == 0 for g, c in
          zip(gradient, [walkoff['a'], walkoff['b'], -walkoff['c']])))

# --- midpoint law ----------------------------------------------------------
check("(10.4.3) nu_j = (w_j - w_l) * beta_2((w_j + w_l)/2)",
      all(sp.simplify(walkoff[k] - (w - w_l) * beta_2((w + w_l) / 2)) == 0
          for k, w in {'a': w_a, 'b': w_b, 'c': w_c}.items()))

# --- orientation on each component of the phase-matching locus -------------
surfaces = {
    "P1  (w_a = w_c)":  {Dm: Dp},
    "P2  (w_b = w_c)":  {Dm: -Dp},
    "Q   (beta_2(wbar) = 0)": {b2: -b3 * wbar},
}

print("\nWalk-off orientation on the phase-matching locus")
directions = {}
for label, sub in surfaces.items():
    nu = [sp.factor(sp.simplify(walkoff[k].subs(sub, simultaneous=True)))
          for k in 'abc']
    common = sp.gcd(sp.gcd(nu[0], nu[1]), nu[2])
    shape = [sp.simplify(n / common) for n in nu]

    # Normalise the sign so the first nonzero entry is positive; only the
    # direction is meaningful, the scale is the gradient magnitude x_grad.
    lead = next(s for s in shape if s != 0)
    if lead < 0:
        shape = [sp.simplify(-s) for s in shape]

    directions[label] = np.array([float(s) for s in shape])
    print(f"    {label:24s} (nu_a, nu_b, nu_c) ~ {shape}")

check("(10.4.4) on Q the two '+' legs share a group velocity",
      sp.simplify((beta_1(w_a) - beta_1(w_b)).subs(
          {b2: -b3 * wbar}, simultaneous=True).expand()) == 0)
check("(10.4.4) on Q leg c is frozen to the landing frequency",
      sp.simplify((beta_1(w_c) - beta_1(w_l)).subs(
          {b2: -b3 * wbar}, simultaneous=True).expand()) == 0)
check("        beta_1(w_a) - beta_1(w_b) = 2 Dp beta_2(wbar)",
      sp.simplify(beta_1(w_a) - beta_1(w_b) - 2 * Dp * beta_2(wbar)) == 0)

# --- the q_res plane and its beta3 tilt ------------------------------------
sum_landing = sp.simplify(walkoff['a'] + walkoff['b'] - walkoff['c'])
check("(10.4.5) landing frame: nu_a + nu_b - nu_c = beta3 (Dp^2 - Dm^2)",
      sp.simplify(sum_landing - b3 * (Dp**2 - Dm**2)) == 0)
check("        equivalently = beta3 * Delta-beta / beta_2(wbar)",
      sp.simplify(sum_landing - b3 * Dbeta / beta_2(wbar)) == 0)

walkoff_t = {k: sp.expand(beta_1(w) - beta_1(w_t))
             for k, w in {'a': w_a, 'b': w_b, 'c': w_c}.items()}
sum_target = sp.simplify(walkoff_t['a'] + walkoff_t['b'] - walkoff_t['c'])
check("(10.4.6) target frame adds exactly beta_1(w_l) - beta_1(w_t)",
      sp.simplify(sum_target - sum_landing - (beta_1(w_l) - beta_1(w_t))) == 0)
check("        at beta3 = 0 it reduces to beta2 * (carrier residual)",
      sp.simplify(sum_target.subs(b3, 0) - b2 * (w_l - w_t)) == 0)


# ============================================================
# Geometric / statistical part
# ============================================================

print("\nOrientation classes")

mask = np.array([1.0, 1.0, -1.0])


def signed_vector(nu):
    """The direction c = (nu_a, nu_b, -nu_c) of section 10.3."""
    return np.array([nu[0], nu[1], -nu[2]])


named = {
    "P1  (w, 0, w)": signed_vector(directions["P1  (w_a = w_c)"]),
    "P2  (0, w, w)": signed_vector(directions["P2  (w_b = w_c)"]),
    "Q   (w, w, 0)": signed_vector(directions["Q   (beta_2(wbar) = 0)"]),
    "10.3 equal   (w, w, w)": signed_vector([1.0, 1.0, 1.0]),
    "10.3 two-leg (w, w, 0)": signed_vector([1.0, 1.0, 0.0]),
    "10.3 one-leg (W, 0, 0)": signed_vector([1.0, 0.0, 0.0]),
}

for label, c in named.items():
    cosine = c @ mask / np.linalg.norm(c) / np.linalg.norm(mask)
    print(f"    {label:24s} c = {c}   cos(c, n) = {cosine:+.6f}")

check("(10.4.7) all three surfaces give |cos(c, n)| = sqrt(2/3)",
      all(abs(abs(named[k] @ mask / np.linalg.norm(named[k])
                  / np.linalg.norm(mask)) - np.sqrt(2 / 3)) < 1e-12
          for k in ("P1  (w, 0, w)", "P2  (0, w, w)", "Q   (w, w, 0)")))

# Signed permutations preserving the cube and the mask normal.
group = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product([1, -1], repeat=3):
        M = np.zeros((3, 3))
        for i, p in enumerate(perm):
            M[i, p] = signs[i]
        if np.allclose(M @ mask, mask) or np.allclose(M @ mask, -mask):
            group.append(M)

target = named["10.3 two-leg (w, w, 0)"]


def in_orbit(c):
    """Same direction class as the two-leg reference, up to sign and scale."""
    unit = c / np.linalg.norm(c)
    reference = target / np.linalg.norm(target)

    return any(np.allclose(M @ unit, reference)
               or np.allclose(M @ unit, -reference)
               for M in group)


check(f"(10.4.8) the symmetry group of (cube, mask) has 12 elements",
      len(group) == 12)
check("(10.4.8) P1, P2, Q lie in the orbit of the two-leg direction",
      all(in_orbit(named[k]) for k in
          ("P1  (w, 0, w)", "P2  (0, w, w)", "Q   (w, w, 0)")))
check("(10.4.8) the equal and one-leg directions do not",
      not in_orbit(named["10.3 equal   (w, w, w)"])
      and not in_orbit(named["10.3 one-leg (W, 0, 0)"]))

# Monte-Carlo: the masked densities must coincide, not merely the correlation.
print("\nMasked densities on D_0 (Monte-Carlo)")
rng = np.random.default_rng(0)
draws = 4_000_000
x = rng.uniform(-np.pi, np.pi, size=(draws, 3))
x = x[np.abs(x @ mask) <= np.pi]

print(f"    acceptance A(0) = {len(x) / draws:.4f}   (exact 2/3)")

edges = np.linspace(-1.0, 1.0, 61)
reference = None
deviations = {}

for label in ("Q   (w, w, 0)", "P1  (w, 0, w)", "P2  (0, w, w)",
              "10.3 equal   (w, w, w)", "10.3 one-leg (W, 0, 0)"):
    c = named[label]
    v = x @ c / (np.pi * np.linalg.norm(c))       # common normalisation
    density, _ = np.histogram(v, bins=edges, density=True)

    if reference is None:
        reference = density

    deviations[label] = float(np.max(np.abs(density - reference)))
    print(f"    {label:24s} max|rho - rho_Q| = {deviations[label]:.4f}")

noise = max(deviations["P1  (w, 0, w)"], deviations["P2  (0, w, w)"])
check("(10.4.8) P1, P2 reproduce the Q density to sampling noise",
      noise < 0.05)
check("(10.4.8) equal and one-leg are a different class",
      min(deviations["10.3 equal   (w, w, w)"],
          deviations["10.3 one-leg (W, 0, 0)"]) > 5 * noise)

# Second moments: the 1 : 2 : 3 law of (10.3.3), equal : two-leg : one-leg.
print("\nAccepted second moments at fixed x_grad (10.3.3)")
moments = {}
for label in ("10.3 equal   (w, w, w)", "10.3 two-leg (w, w, 0)",
              "10.3 one-leg (W, 0, 0)"):
    c = named[label]
    v = x @ c / np.linalg.norm(c)                 # fixed gradient scale
    moments[label] = float(np.mean(v**2) * len(x) / draws)

base = moments["10.3 equal   (w, w, w)"]
ratios = [moments[k] / base for k in
          ("10.3 equal   (w, w, w)", "10.3 two-leg (w, w, 0)",
           "10.3 one-leg (W, 0, 0)")]
print(f"    M2 ratios equal : two-leg : one-leg = "
      f"{ratios[0]:.3f} : {ratios[1]:.3f} : {ratios[2]:.3f}   (exact 1 : 2 : 3)")
check("(10.3.3) second-moment ratios are 1 : 2 : 3",
      abs(ratios[1] - 2) < 0.02 and abs(ratios[2] - 3) < 0.03)


# ============================================================
print("\n" + "=" * 62)
failed = [name for name, ok in results if not ok]
if failed:
    print(f"{len(failed)} FAILED:")
    for name in failed:
        print(f"   {name}")
    raise SystemExit(1)

print(f"all {len(results)} checks passed")
