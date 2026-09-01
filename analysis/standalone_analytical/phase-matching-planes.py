"""Phase-matching surfaces in the 3D FWM frequency-offset space.

Companion to ``the-domain-in-freq-space-*.py`` (which draw the *frequency*
matching support) and to ``analysis/fwm/fwm-efficiency/fwm_and_dispersion.ipynb``
(which draws the *phase* matching contours as 2D lines).

Convention, identical to the sibling scripts:

    omega_1 - omega_2 + omega_3 = omega_4          (energy conservation)
    nu_j = omega_j - omega_COI                     (offsets from the COI)
    nu_4 = nu_1 - nu_2 + nu_3                      (eliminated)

so (nu_1, nu_2, nu_3) is a complete coordinate system and every FWM tuple is
one point of it.  The phase mismatch of the tuple is

    Dbeta = beta(w1) + beta(w3) - beta(w2) - beta(w4).

With beta expanded to fourth order around a reference w_ref, this factorises
exactly (see phase_matching_planes.md):

    Dbeta = (nu1 - nu2)(nu2 - nu3) * [ b2(S) + beta4 * ((nu1-nu2)^2 + (nu2-nu3)^2) / 24 ]

    S  = (w1 + w3)/2 - w_ref = delta + (nu1 + nu3)/2
    b2(S) = beta2 + beta3 * S + beta4 * S^2 / 2      (local GVD at the pump mean)

Hence Dbeta = 0 is the union of

    P1: nu1 = nu2          (equivalently w1 = w2, w3 = w4)  -- degenerate/XPM
    P2: nu2 = nu3          (equivalently w3 = w2, w1 = w4)  -- degenerate/XPM
    Q : b2(S) + beta4 (...)/24 = 0                          -- the true FWM sheet

P1 and P2 are exact planes for any dispersion order.  Q is a plane when
beta4 = 0 (it degenerates to nu1 + nu3 = 2 nu_ZDF) and a quadric otherwise.

The notebook's 2D map is the slice nu_4 = 0 of this picture; its three
phase-matching lines are the traces of P1, P2 and Q on that slice.

Unlike the plotly siblings this script uses matplotlib, so it runs headless
and writes a static figure.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


TWOPI = 2.0 * np.pi
THZ = 1e12

OUT_MEDIA = Path("media/standalone_analytical")


# ============================================================
# Parameters
# ============================================================

# SMF-28 Taylor coefficients, same defaults as the dispersion notebook.
# The COI is placed in the E band so that the zero-dispersion sheet falls
# inside the plotted box.  With the notebook default COI = ref = 193.4 THz the
# sheet sits ~35 THz away and only the two XPM planes are visible -- which is
# exactly why the notebook's default map shows just the two axis lines.
F_COI_THZ = 220.0
F_REF_THZ = 193.4
BETA2 = -21.1687712327e-27   # s^2 / m
BETA3 = 0.1285064367e-39     # s^3 / m
BETA4 = -0.00028828828e-51   # s^4 / m

SPAN_THZ = 15.0              # half-width of the plotted offset box
CHANNEL_SPACING_THZ = 3.0    # Delta_f, only used to draw the channel grid
SYMBOL_RATE_THZ = 1.0        # R_s = B / 2pi, the per-channel cube side
CHANNEL_INDICES = range(-5, 6)


# ============================================================
# Dispersion helpers.  All internal frequencies are angular [rad/s].
# ============================================================

def local_beta2(s, beta2=BETA2, beta3=BETA3, beta4=BETA4):
    """GVD at angular offset ``s`` from the reference frequency."""
    return beta2 + beta3 * s + 0.5 * beta4 * s**2


def delta_beta(nu1, nu2, nu3, delta, beta2=BETA2, beta3=BETA3, beta4=BETA4):
    """Phase mismatch from the factorised closed form."""
    u = nu1 - nu2
    v = nu2 - nu3
    s = delta + 0.5 * (nu1 + nu3)

    return u * v * (
        local_beta2(s, beta2, beta3, beta4)
        + beta4 * (u**2 + v**2) / 24.0
    )


def delta_beta_direct(nu1, nu2, nu3, delta, beta2=BETA2, beta3=BETA3, beta4=BETA4):
    """Phase mismatch evaluated term by term, used as a cross-check."""
    def beta(x):
        return 0.5 * beta2 * x**2 + beta3 * x**3 / 6.0 + beta4 * x**4 / 24.0

    nu4 = nu1 - nu2 + nu3

    return (
        beta(delta + nu1)
        + beta(delta + nu3)
        - beta(delta + nu2)
        - beta(delta + nu4)
    )


def zero_dispersion_offset(delta, beta2=BETA2, beta3=BETA3, beta4=BETA4):
    """Angular offsets from the COI at which the local GVD vanishes.

    Returns the roots ``s`` of ``b2(s) = 0`` translated to COI-referred
    offsets ``s - delta``.
    """
    if beta4 == 0.0:
        if beta3 == 0.0:
            return np.array([])
        return np.array([-beta2 / beta3]) - delta

    roots = np.roots([0.5 * beta4, beta3, beta2])
    roots = np.sort(roots[np.isreal(roots)].real)

    return roots - delta


# ============================================================
# Geometry: clip a plane / a surface to the plotted box
# ============================================================

BOX_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(8)
    if i < j
    and bin(i ^ j).count("1") == 1
]


def box_corners(half):
    return np.array([
        [(-half, half)[(i >> 0) & 1],
         (-half, half)[(i >> 1) & 1],
         (-half, half)[(i >> 2) & 1]]
        for i in range(8)
    ])


def plane_box_polygon(normal, offset, half):
    """Convex polygon of ``normal . nu = offset`` inside the cube [-half, half]^3."""
    corners = box_corners(half)
    values = corners @ normal - offset

    points = []

    for i, j in BOX_EDGES:
        fi, fj = values[i], values[j]

        if fi * fj > 0.0:
            continue
        if abs(fi - fj) < 1e-30:
            continue

        t = fi / (fi - fj)

        if -1e-12 <= t <= 1.0 + 1e-12:
            points.append(corners[i] + t * (corners[j] - corners[i]))

    if len(points) < 3:
        return np.empty((0, 3))

    points = np.asarray(points)

    # Deduplicate.
    keep = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 1e-9 * half for q in keep):
            keep.append(p)
    points = np.asarray(keep)

    if len(points) < 3:
        return np.empty((0, 3))

    # Order the vertices by angle in the plane.
    n = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(n @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    centre = points.mean(axis=0)
    angles = np.arctan2((points - centre) @ e2, (points - centre) @ e1)

    return points[np.argsort(angles)]


def fwm_sheet(delta, half, resolution=81):
    """Sample the sheet ``b2(S) + beta4 (p^2 + q^2)/12 = 0``.

    The natural parametrisation is by the two splittings

        p = (nu1 - nu3)/2,    q = nu2 - (nu1 + nu3)/2,

    because the condition then fixes ``S = delta + (nu1 + nu3)/2`` alone:

        (beta4/2) S^2 + beta3 S + beta2 + beta4 (p^2 + q^2)/12 = 0.

    The branch continuous with the beta4 = 0 limit ``S = -beta2/beta3`` is
    selected.  Returns ``(nu1, nu2, nu3)`` arrays, NaN where no real root
    exists.
    """
    grid = np.linspace(-half, half, resolution)
    p, q = np.meshgrid(grid, grid, indexing="ij")

    constant = BETA2 + BETA4 * (p**2 + q**2) / 12.0

    if BETA4 == 0.0:
        s = -constant / BETA3
    else:
        discriminant = BETA3**2 - 2.0 * BETA4 * constant
        discriminant = np.where(discriminant >= 0.0, discriminant, np.nan)
        root = np.sqrt(discriminant)

        # (beta4/2) S^2 + beta3 S + constant = 0.  The branch that tends to
        # -constant/beta3 as beta4 -> 0 is the one with the minus sign in front
        # of the square root when beta3 > 0, written stably as below.
        s = -2.0 * constant / (BETA3 + root)

    offset = s - delta

    nu1 = offset + p
    nu3 = offset - p
    nu2 = offset + q

    return nu1, nu2, nu3


# ============================================================
# Channel grid
# ============================================================

def channel_cube_edges(n1, n2, n3, spacing, side):
    """Wireframe of the per-tuple support cube, in angular units."""
    centre = spacing * np.array([n1, n2, n3], dtype=float)
    corners = box_corners(0.5 * side) + centre

    segments = []
    for i, j in BOX_EDGES:
        segments.append((corners[i], corners[j]))

    return segments


# ============================================================
# Figure
# ============================================================

def main():
    delta = TWOPI * (F_COI_THZ - F_REF_THZ) * THZ
    half = TWOPI * SPAN_THZ * THZ
    spacing = TWOPI * CHANNEL_SPACING_THZ * THZ
    side = TWOPI * SYMBOL_RATE_THZ * THZ

    scale = 1.0 / (TWOPI * THZ)   # angular rad/s -> THz of offset

    # Cross-check the factorisation on random tuples.
    rng = np.random.default_rng(0)
    sample = rng.uniform(-half, half, size=(2000, 3))
    residual = np.max(np.abs(
        delta_beta(sample[:, 0], sample[:, 1], sample[:, 2], delta)
        - delta_beta_direct(sample[:, 0], sample[:, 1], sample[:, 2], delta)
    ))
    reference = np.max(np.abs(
        delta_beta_direct(sample[:, 0], sample[:, 1], sample[:, 2], delta)
    ))
    print(f"factorisation residual: {residual:.3e} / {reference:.3e} 1/m")

    nu_zdf = zero_dispersion_offset(delta)
    print("zero-dispersion offsets from COI [THz]:",
          np.array2string(nu_zdf * scale, precision=3))

    fig = plt.figure(figsize=(13.0, 6.0))

    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_2d = fig.add_subplot(1, 2, 2)

    # ---- 3D panel -------------------------------------------------

    # Degenerate planes nu1 = nu2 and nu2 = nu3.
    for normal, colour, label in (
        (np.array([1.0, -1.0, 0.0]), "#fc8d62", r"$\nu_1=\nu_2$  (XPM)"),
        (np.array([0.0, 1.0, -1.0]), "#8da0cb", r"$\nu_2=\nu_3$  (XPM)"),
    ):
        polygon = plane_box_polygon(normal, 0.0, half) * scale

        ax_3d.add_collection3d(Poly3DCollection(
            [polygon],
            facecolor=colour,
            edgecolor=colour,
            alpha=0.30,
            linewidth=1.2,
        ))
        ax_3d.plot([], [], color=colour, linewidth=6, alpha=0.5, label=label)

    # Tangent plane of the third surface at p = q = 0: nu1 + nu3 = 2 nu_ZDF,
    # where nu_ZDF is the true zero of the local GVD (beta4 included).  This is
    # the surface exactly when beta4 = 0.
    s_zdf_plane = nu_zdf[0]

    plane_polygon = plane_box_polygon(
        np.array([1.0, 0.0, 1.0]), 2.0 * s_zdf_plane, half
    ) * scale

    if len(plane_polygon):
        ax_3d.add_collection3d(Poly3DCollection(
            [plane_polygon], facecolor="#66c2a5", edgecolor="#1b7837",
            alpha=0.35, linewidth=1.5,
        ))
    ax_3d.plot([], [], color="#66c2a5", linewidth=6, alpha=0.6,
               label=r"$\nu_1+\nu_3=2\nu_{\rm ZDF}$  (tangent plane)")

    # The exact sheet, which bends away from that plane through beta4.
    nu1, nu2, nu3 = fwm_sheet(delta, half)

    inside = (
        (np.abs(nu1) <= half) & (np.abs(nu2) <= half) & (np.abs(nu3) <= half)
    )
    bend = np.nanmax(np.abs((nu1 + nu3) - 2.0 * s_zdf_plane)[inside]) * scale
    print(f"max in-box bend of the sheet off its tangent plane: {bend:.3f} THz")

    masked = [np.where(inside, component, np.nan) for component in (nu1, nu2, nu3)]

    ax_3d.plot_surface(
        masked[0] * scale, masked[1] * scale, masked[2] * scale,
        color="#1b7837", alpha=0.85, linewidth=0,
        antialiased=True, rstride=2, cstride=2, shade=True,
    )
    ax_3d.plot([], [], color="#1b7837", linewidth=6,
               label=r"exact sheet $\bar\beta_2(S)+\beta_4(p^2+q^2)/12=0$")

    # The notebook slice nu_4 = 0, i.e. nu1 - nu2 + nu3 = 0.
    slice_polygon = plane_box_polygon(
        np.array([1.0, -1.0, 1.0]), 0.0, half
    ) * scale

    ax_3d.add_collection3d(Poly3DCollection(
        [slice_polygon],
        facecolor="none",
        edgecolor="black",
        alpha=1.0,
        linewidth=1.8,
        linestyle="--",
    ))
    ax_3d.plot([], [], color="black", linewidth=1.8, linestyle="--",
               label=r"notebook slice $\nu_4=0$")

    # Channel grid, drawn only for the cubes the FWM sheet actually crosses.
    # A cube is crossed when the sheet function changes sign over its corners.
    def sheet_function(nu):
        u = nu[:, 0] - nu[:, 1]
        v = nu[:, 1] - nu[:, 2]
        s = delta + 0.5 * (nu[:, 0] + nu[:, 2])
        return local_beta2(s) + BETA4 * (u**2 + v**2) / 24.0

    drawn = 0
    for n1 in CHANNEL_INDICES:
        for n2 in CHANNEL_INDICES:
            for n3 in CHANNEL_INDICES:
                centre = spacing * np.array([n1, n2, n3], dtype=float)
                corners = box_corners(0.5 * side) + centre

                if np.max(np.abs(corners)) > half:
                    continue

                values = sheet_function(corners)
                if values.min() > 0.0 or values.max() < 0.0:
                    continue

                for a, b in channel_cube_edges(n1, n2, n3, spacing, side):
                    ax_3d.plot(
                        [a[0] * scale, b[0] * scale],
                        [a[1] * scale, b[1] * scale],
                        [a[2] * scale, b[2] * scale],
                        color="0.25", linewidth=0.6, alpha=0.7,
                    )
                drawn += 1

    print(f"channel cubes crossed by the FWM sheet: {drawn}")

    limit = half * scale
    ax_3d.set_xlim(-limit, limit)
    ax_3d.set_ylim(-limit, limit)
    ax_3d.set_zlim(-limit, limit)
    ax_3d.set_xlabel(r"$\nu_1/2\pi$ [THz]")
    ax_3d.set_ylabel(r"$\nu_2/2\pi$ [THz]")
    ax_3d.set_zlabel(r"$\nu_3/2\pi$ [THz]")
    ax_3d.set_box_aspect((1, 1, 1))
    ax_3d.view_init(elev=22, azim=-52)
    ax_3d.set_title(r"$\Delta\beta=0$ in the FWM offset space")
    ax_3d.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # ---- 2D panel: the notebook slice -----------------------------

    span_2d = TWOPI * SPAN_THZ * THZ
    grid = np.linspace(-span_2d, span_2d, 601)
    om1, om2 = np.meshgrid(grid, grid, indexing="xy")

    # On nu_4 = 0 the notebook coordinates are (Omega_1, Omega_2) with
    # nu1 = Omega_1, nu3 = Omega_2, nu2 = Omega_1 + Omega_2.
    mismatch = delta_beta_direct(om1, om1 + om2, om2, delta)

    scale_2d = scale
    extent = [-span_2d * scale_2d, span_2d * scale_2d] * 2

    image = ax_2d.imshow(
        np.sign(mismatch) * np.log10(1.0 + np.abs(mismatch)),
        extent=extent, origin="lower", cmap="RdBu_r", aspect="equal",
    )
    ax_2d.contour(
        om1 * scale_2d, om2 * scale_2d, mismatch,
        levels=[0.0], colors="k", linewidths=1.6,
    )

    fig.colorbar(image, ax=ax_2d, shrink=0.82,
                 label=r"$\mathrm{sgn}\,\Delta\beta\;\log_{10}(1+|\Delta\beta|)$")

    ax_2d.set_xlabel(r"$\Omega_1/2\pi$ [THz]")
    ax_2d.set_ylabel(r"$\Omega_2/2\pi$ [THz]")
    ax_2d.set_title(r"slice $\nu_4=0$: traces of the same surfaces")

    fig.tight_layout()

    OUT_MEDIA.mkdir(parents=True, exist_ok=True)
    out_path = OUT_MEDIA / "phase_matching_planes.pdf"
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
