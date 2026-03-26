from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from scipy.io import loadmat


# -----------------------------------------------------------------------------
# Load .mat file
# -----------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = Path.cwd()

candidate_paths = [
    script_dir / "input" / "aquila_15_LP_modes.mat",
    script_dir / "aquila_15_LP_modes.mat",
    project_root / "input" / "aquila_15_LP_modes.mat",
    project_root / "aquila_15_LP_modes.mat",
]

mat_file = next((p for p in candidate_paths if p.exists()), None)

if mat_file is None:
    searched = "\n".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        "Could not find aquila_15_LP_modes.mat. Searched in:\n" + searched
    )

print(f"Loading: {mat_file}")
data = loadmat(mat_file, struct_as_record=False, squeeze_me=True)

EMFpol = data["EMFpol"]
info = data.get("info", None)


# -----------------------------------------------------------------------------
# Output directory
# -----------------------------------------------------------------------------
out_dir = script_dir / "media" / "aquila_modes"
out_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Polar grid -> Cartesian grid
# -----------------------------------------------------------------------------
erre = np.asarray(EMFpol.erre)
phi = np.asarray(EMFpol.phi)
E = np.asarray(EMFpol.E)

R, PHI = np.meshgrid(erre, phi, indexing="xy")
X = R * np.cos(PHI)
Y = R * np.sin(PHI)

rmax = float(np.max(erre))

# Expected shape: (ncomp, nphi, nr, nmodes)
ncomp = E.shape[0]
nmodes = E.shape[3]

if ncomp < 3:
    raise ValueError(f"Expected at least 3 field components, got ncomp={ncomp}")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_mode_name(mode_index_zero_based):
    fallback = f"mode_{mode_index_zero_based + 1:03d}"

    if info is None or not hasattr(info, "tags"):
        return fallback

    try:
        tag = info.tags[mode_index_zero_based]
        if isinstance(tag, np.ndarray):
            if tag.shape == ():
                tag = tag.item()
            elif tag.size == 1:
                tag = tag.flat[0]
        tag = str(tag).strip()
        return tag if tag else fallback
    except Exception:
        return fallback


def sanitize_filename(name):
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "unnamed_mode"


# -----------------------------------------------------------------------------
# Build triangulation once
# -----------------------------------------------------------------------------
x_flat = X.ravel()
y_flat = Y.ravel()
triang = mtri.Triangulation(x_flat, y_flat)


# -----------------------------------------------------------------------------
# Loop over modes
# -----------------------------------------------------------------------------
for m in range(nmodes):
    # Components on the transverse plane
    Z0 = np.squeeze(np.real(E[0, :, :, m]))
    Z1 = np.squeeze(np.real(E[1, :, :, m]))
    Z2 = np.squeeze(np.real(E[2, :, :, m]))

    # Common zero-centered color scale across the 3 components
    all_vals = np.concatenate([Z0.ravel(), Z1.ravel(), Z2.ravel()])
    vabs = np.max(np.abs(all_vals))

    if np.isclose(vabs, 0.0):
        vabs = 1e-12

    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    mode_name = get_mode_name(m)
    filename = f"{m+1:03d}_" + sanitize_filename(mode_name) + ".png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    component_titles = ["component 1", "component 2", "component 3"]
    component_data = [Z0, Z1, Z2]

    pcm = None
    for ax, Z, comp_title in zip(axes, component_data, component_titles):
        pcm = ax.tripcolor(
            triang,
            Z.ravel(),
            shading="gouraud",
            cmap="coolwarm",
            norm=norm,
        )
        ax.set_aspect("equal")
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(comp_title)

    fig.suptitle(f"{m+1:03d} — {mode_name}")

    cbar = fig.colorbar(pcm, ax=axes, shrink=0.9)
    cbar.set_label("Re(E)")

    save_path = out_dir / filename
    fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")