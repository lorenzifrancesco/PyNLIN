#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
import string

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


MATFILE = "input/aquila_15_LP_modes.mat"

CSV_FULL_OUT = Path("output/overlap_aquila/oi_matrix_full.csv")
CSV_REDUCED_OUT = Path("output/overlap_aquila/oi_matrix_reduced.csv")

FIG_FULL_OUT = Path("media/overlap_aquila/oi_matrix_full.png")
FIG_REDUCED_OUT = Path("media/overlap_aquila/oi_matrix_reduced.png")

# Tolerance for deciding whether paired modes are "the same enough"
PAIR_RTOL = 10e-2
PAIR_ATOL = 1e-10


def load_data(matfile: str):
    data = loadmat(matfile, struct_as_record=False, squeeze_me=True)

    if "EMFpol" not in data:
        raise KeyError("Variable 'EMFpol' not found in the .mat file.")

    EMFpol = data["EMFpol"]

    r = np.asarray(EMFpol.erre, dtype=float).squeeze()
    phi = np.asarray(EMFpol.phi, dtype=float).squeeze()
    E = np.asarray(EMFpol.E)

    # Expected ordering: component, phi, r, mode
    if E.ndim != 4:
        raise ValueError(f"EMFpol.E must have 4 dimensions, found shape {E.shape}")

    ncomp, nphi, nr, nmodes = E.shape

    if len(phi) != nphi:
        raise ValueError(f"len(phi)={len(phi)} but E.shape[1]={nphi}")
    if len(r) != nr:
        raise ValueError(f"len(r)={len(r)} but E.shape[2]={nr}")

    neff = None
    if "neff" in data:
        neff = np.asarray(data["neff"]).squeeze()

    tags = None
    if "info" in data and hasattr(data["info"], "tags"):
        tags_raw = data["info"].tags
        if isinstance(tags_raw, np.ndarray):
            tags = [str(x) for x in tags_raw.flat]
        else:
            tags = [str(tags_raw)]

    return r, phi, E, neff, tags


def integrate_over_section(F: np.ndarray, r: np.ndarray, phi: np.ndarray) -> float:
    """
    Integrate a scalar field F(phi, r) over the fiber cross-section:
        ∫∫ F(phi, r) r dr dphi
    """
    if F.shape != (len(phi), len(r)):
        raise ValueError(f"Expected shape ({len(phi)}, {len(r)}), got {F.shape}")

    integrand = F * r[np.newaxis, :]
    return np.trapezoid(np.trapezoid(integrand, x=r, axis=1), x=phi, axis=0)


def mode_intensity(E: np.ndarray, mode_index: int) -> np.ndarray:
    """
    Build intensity profile I_a(phi, r) for one mode from the electric field:
        I_a = sum_c |E_c|^2

    E has shape (component, phi, r, mode).
    """
    Em = E[:, :, :, mode_index]
    I = np.sum(np.abs(Em) ** 2, axis=0)
    return np.real_if_close(I)


def compute_oi_matrix(E: np.ndarray, r: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute the matrix:
        OI_ab = ∫ I_a I_b dS / [ (∫ I_a dS)(∫ I_b dS) ]

    Since I_a is normalized by integrated intensity only, OI has units of 1/area.
    """
    nmodes = E.shape[3]

    intensities = []
    integrals = np.zeros(nmodes, dtype=float)

    for a in range(nmodes):
        Ia = mode_intensity(E, a)
        Sa = integrate_over_section(Ia, r, phi)

        if np.isclose(Sa, 0.0):
            raise ValueError(f"Mode {a+1} has zero integrated intensity.")

        intensities.append(Ia)
        integrals[a] = Sa

    OI = np.zeros((nmodes, nmodes), dtype=float)

    for a in range(nmodes):
        Ia = intensities[a]
        for b in range(nmodes):
            Ib = intensities[b]
            numerator = integrate_over_section(Ia * Ib, r, phi)
            denominator = integrals[a] * integrals[b]
            OI[a, b] = numerator / denominator

    return OI


def sanitize_mode_name(name: str) -> str:
    """
    Make mode names safe and compact for CSV headers and plot labels.

    Example:
        LP(0,1)   -> LP01
        LP(1, 2)a -> LP12a
    """
    s = str(name).strip()
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"[^A-Za-z0-9_+\-]", "", s)
    return s


def strip_trailing_letter_suffix(name: str) -> str:
    """
    For family grouping in the reduced representation, strip a single trailing
    lowercase letter if present.

    Example:
        LP11a -> LP11
        LP11b -> LP11
        LP01  -> LP01
    """
    return re.sub(r"[a-z]$", "", name)


def add_family_suffixes(names: list[str]) -> list[str]:
    """
    If reduced mode names repeat within the same family, rename them as
    family + a, family + b, ...

    Example:
        [LP01, LP11, LP11, LP21, LP21, LP21]
        -> [LP01, LP11a, LP11b, LP21a, LP21b, LP21c]

    Family is determined after removing a possible trailing lowercase suffix.
    """
    families = [strip_trailing_letter_suffix(name) for name in names]

    counts = {}
    for fam in families:
        counts[fam] = counts.get(fam, 0) + 1

    running = {}
    out = []

    for name, fam in zip(names, families):
        if counts[fam] == 1:
            out.append(fam)
            continue

        k = running.get(fam, 0)
        if k >= len(string.ascii_lowercase):
            raise ValueError(
                f"Too many repeated modes in family {fam!r}; "
                "only up to 26 suffixes are supported."
            )
        out.append(f"{fam}{string.ascii_lowercase[k]}")
        running[fam] = k + 1

    return out


def build_mode_names(tags, nmodes: int) -> list[str]:
    """
    Build readable mode names for CSV/plot labels.
    Prefer tags if available, otherwise use mode_01, mode_02, ...
    """
    if tags is not None and len(tags) == nmodes:
        return [sanitize_mode_name(str(t)) for t in tags]
    return [f"mode_{i+1:02d}" for i in range(nmodes)]


def check_pairwise_duplicate_structure(
    oi: np.ndarray,
    names: list[str],
    rtol: float = PAIR_RTOL,
    atol: float = PAIR_ATOL,
):
    """
    Check whether the modes come in pairs (0,1), (2,3), ...
    such that each pair has approximately the same OI row and column.
    """
    n = oi.shape[0]

    if n % 2 != 0:
        return False, list(range(n)), names

    reduced_indices = []
    reduced_names = []

    for i in range(0, n, 2):
        j = i + 1

        rows_close = np.allclose(oi[i, :], oi[j, :], rtol=rtol, atol=atol)
        cols_close = np.allclose(oi[:, i], oi[:, j], rtol=rtol, atol=atol)

        if not (rows_close and cols_close):
            max_row_diff = np.max(np.abs(oi[i, :] - oi[j, :]))
            max_col_diff = np.max(np.abs(oi[:, i] - oi[:, j]))
            print(
                f"Pair ({i+1}, {j+1}) is NOT approximately equal: "
                f"max row diff = {max_row_diff:.3e}, "
                f"max col diff = {max_col_diff:.3e}"
            )
            return False, list(range(n)), names

        reduced_indices.append(i)
        reduced_names.append(names[i])

        print(
            f"Pair ({i+1}, {j+1}) matched: "
            f"'{names[i]}' / '{names[j]}'"
        )

    return True, reduced_indices, reduced_names


def reduce_paired_matrix(oi: np.ndarray, indices: list[int]) -> np.ndarray:
    idx = np.array(indices, dtype=int)
    return oi[np.ix_(idx, idx)]


def save_csv(oi: np.ndarray, path: Path, mode_names: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(mode_names) != oi.shape[0]:
        raise ValueError(
            f"len(mode_names)={len(mode_names)} but matrix size is {oi.shape[0]}"
        )

    for name in mode_names:
        if "," in name:
            raise ValueError(f"Mode name still contains a comma: {name!r}")

    header = ",".join(["mode"] + mode_names)

    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i, name in enumerate(mode_names):
            row = ",".join([name] + [f"{oi[i, j]:.16e}" for j in range(oi.shape[1])])
            f.write(row + "\n")


def save_colormap(oi: np.ndarray, path: Path, mode_names: list[str], title: str):
    path.parent.mkdir(parents=True, exist_ok=True)

    n = oi.shape[0]

    fig, ax = plt.subplots(figsize=(8*0.6, 6*0.6))
    im = ax.imshow(oi, origin="lower", aspect="equal")
    plt.colorbar(im, ax=ax, label=r"OI [1/m$^2$]")

    # ax.set_xlabel("mode b")
    # ax.set_ylabel("mode a")
    ax.set_title(title)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(mode_names, rotation=90, fontsize=7)
    ax.set_yticklabels(mode_names, fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def main():
    r, phi, E, neff, tags = load_data(MATFILE)

    print(f"Loaded: {MATFILE}")
    print(f"E shape = {E.shape}")
    print(f"nr = {len(r)}, nphi = {len(phi)}, nmodes = {E.shape[3]}")

    oi_full = compute_oi_matrix(E, r, phi)
    full_names = build_mode_names(tags, oi_full.shape[0])

    save_csv(oi_full, CSV_FULL_OUT, full_names)
    save_colormap(
        oi_full,
        FIG_FULL_OUT,
        full_names,
        title="Intensity-overlap matrix (full)",
    )

    print(f"Saved full CSV to: {CSV_FULL_OUT}")
    print(f"Saved full figure to: {FIG_FULL_OUT}")

    is_paired, reduced_indices, reduced_names = check_pairwise_duplicate_structure(
        oi_full,
        full_names,
        rtol=PAIR_RTOL,
        atol=PAIR_ATOL,
    )

    if is_paired:
        oi_reduced = reduce_paired_matrix(oi_full, reduced_indices)
        reduced_names = add_family_suffixes(reduced_names)

        save_csv(oi_reduced, CSV_REDUCED_OUT, reduced_names)
        save_colormap(
            oi_reduced,
            FIG_REDUCED_OUT,
            reduced_names,
            title="Intensity-overlap matrix",
        )

        print(
            f"All mode pairs are approximately equal. "
            f"Saved reduced matrix of shape {oi_reduced.shape}."
        )
        print(f"Saved reduced CSV to: {CSV_REDUCED_OUT}")
        print(f"Saved reduced figure to: {FIG_REDUCED_OUT}")
        print(f"Reduced mode names: {reduced_names}")
    else:
        print("Pairwise reduction not applied.")


if __name__ == "__main__":
    main()