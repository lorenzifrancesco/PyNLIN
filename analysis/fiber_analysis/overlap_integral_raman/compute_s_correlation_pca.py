#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.io import loadmat

MATFILE = Path('input/aquila_15_LP_modes.mat')
OUTDIR = Path('output/s_correlation_pca')
FIGDIR = Path('media/s_correlation_pca')
MAX_MODES = 8
ZERO_GREY = (0.5, 0.5, 0.5, 1.0)


def load_modes(matfile: str, max_modes: int | None = None):
    data = loadmat(matfile, struct_as_record=False, squeeze_me=True)
    if 'EMFpol' not in data:
        raise KeyError("Variable 'EMFpol' not found in the .mat file.")
    emf = data['EMFpol']
    r = np.asarray(emf.erre, dtype=float).squeeze()
    phi = np.asarray(emf.phi, dtype=float).squeeze()
    E = np.asarray(emf.E)
    if E.ndim != 4:
        raise ValueError(f'Expected E with 4 dims (component, phi, r, mode), got {E.shape}')
    _, nphi, nr, nmodes_total = E.shape
    if len(phi) != nphi or len(r) != nr:
        raise ValueError('Inconsistent r/phi lengths with field array shape.')
    if max_modes is not None:
        nmodes = min(max_modes, nmodes_total)
        E = E[:, :, :, :nmodes]
    else:
        nmodes = nmodes_total
    return r, phi, E, nmodes_total

def trapezoid_weights(x: np.ndarray):
    if x.ndim != 1:
        raise ValueError('Expected a 1D coordinate array for trapezoid weights.')
    if x.size < 2:
        raise ValueError('Need at least two points to build trapezoid weights.')
    dx = np.diff(x)
    w = np.empty_like(x, dtype=float)
    w[0] = dx[0] / 2
    w[-1] = dx[-1] / 2
    if x.size > 2:
        w[1:-1] = (dx[:-1] + dx[1:]) / 2
    return w


def modal_overlap_fields(E: np.ndarray):
    """
    Return G[mu,nu,phi,r] = sum_c conj(E_c(mu)) * E_c(nu)
    """
    # E: (component, phi, r, mode)
    return np.einsum('cprm,cprn->mnpr', np.conj(E), E, optimize=True)


def build_pair_labels(nmodes: int):
    return [(mu, nu) for mu in range(nmodes) for nu in range(nmodes)]


def compute_correlation_matrices(G: np.ndarray, r: np.ndarray, phi: np.ndarray):
    w_r = trapezoid_weights(r) * r
    w_phi = trapezoid_weights(phi)
    weights = np.outer(w_phi, w_r).reshape(-1)

    nmodes = G.shape[0]
    npairs = nmodes * nmodes
    Gflat = G.reshape(npairs, -1)
    weighted = Gflat * np.sqrt(weights)[np.newaxis, :]

    K = weighted @ weighted.T
    Kc = weighted @ weighted.conj().T
    return K, Kc


def compute_correlation_matrix(E: np.ndarray, r: np.ndarray, phi: np.ndarray):
    """
    K[(mu,nu),(mu',nu')] = ∫∫ (E_mu^*·E_nu)(E_mu'^*·E_nu') dA
    This follows the formula as written in the slide, without adding an extra conjugation
    on the second factor.
    """
    G = modal_overlap_fields(E)
    nmodes = E.shape[3]
    pairs = build_pair_labels(nmodes)
    K, Kc = compute_correlation_matrices(G, r, phi)
    return K, Kc, pairs, G


def compute_svd(K: np.ndarray, hermitian_hint: bool = False):
    if hermitian_hint:
        evals, evecs = np.linalg.eigh(K)
        order = np.argsort(evals)[::-1]
        svals = np.maximum(evals[order], 0.0)
        vecs = evecs[:, order]
    else:
        U, svals, _ = np.linalg.svd(K)
        vecs = U
    return svals, vecs


def save_matrix_csv(M: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.iscomplexobj(M):
        with path.open('w', encoding='utf-8') as f:
            for i in range(M.shape[0]):
                row = []
                for j in range(M.shape[1]):
                    z = M[i, j]
                    row.append(f'{z.real:.16e}{z.imag:+.16e}j')
                f.write(','.join(row) + '\n')
    else:
        np.savetxt(path, M, delimiter=',')


def make_zero_centered_coolwarm():
    base = plt.get_cmap('coolwarm')
    samples = base(np.linspace(0.0, 1.0, 256))
    mid = samples.shape[0] // 2
    samples[mid - 1:mid + 1] = ZERO_GREY
    return colors.ListedColormap(samples, name='coolwarm_zero_grey')


def centered_diverging_norm(M: np.ndarray):
    vmax = float(np.max(np.abs(M)))
    if vmax == 0.0:
        vmax = 1.0
    return colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def plot_heatmap(M: np.ndarray, title: str, path: Path, cmap: str = 'viridis'):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    imshow_kwargs = dict(origin='lower', aspect='equal')
    if cmap == 'coolwarm':
        imshow_kwargs['cmap'] = make_zero_centered_coolwarm()
        imshow_kwargs['norm'] = centered_diverging_norm(M)
    else:
        imshow_kwargs['cmap'] = cmap
    im = ax.imshow(M, **imshow_kwargs)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def principal_component_matrix(vec: np.ndarray, nmodes: int):
    return vec.reshape((nmodes, nmodes))


def main():
    r, phi, E, nmodes_total = load_modes(str(MATFILE), max_modes=MAX_MODES)
    nmodes = E.shape[3]
    K, Kc, pairs, _ = compute_correlation_matrix(E, r, phi)
    npairs = len(pairs)

    svals, vecs = compute_svd(K, hermitian_hint=False)
    svals_c, vecs_c = compute_svd(Kc, hermitian_hint=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    save_matrix_csv(K, OUTDIR / 'correlation_matrix_as_written.csv')
    save_matrix_csv(Kc, OUTDIR / 'correlation_matrix_hermitian.csv')
    np.savetxt(OUTDIR / 'svd_singular_values_as_written.csv', svals, delimiter=',')
    np.savetxt(OUTDIR / 'eigenvalues_hermitian.csv', svals_c, delimiter=',')

    plot_heatmap(np.abs(K), '|K| from formula as written', FIGDIR / 'correlation_matrix_abs_as_written.png')
    plot_heatmap(np.real(K), 'Re(K) from formula as written', FIGDIR / 'correlation_matrix_real_as_written.png')
    plot_heatmap(np.abs(Kc), '|K| Hermitian version', FIGDIR / 'correlation_matrix_abs_hermitian.png')
    plot_heatmap(np.real(Kc), 'Re(K) Hermitian version', FIGDIR / 'correlation_matrix_real_hermitian.png')

    max_components = min(12, nmodes * nmodes)
    for k in range(max_components):
        M = principal_component_matrix(vecs[:, k], nmodes)
        plot_heatmap(np.real(M), f'PC {k+1} matrix, real part (formula as written)', FIGDIR / f'pc_{k+1:02d}_real_as_written.png', cmap='coolwarm')
        if np.max(np.abs(np.imag(M))) > 1e-12:
            plot_heatmap(np.imag(M), f'PC {k+1} matrix, imag part (formula as written)', FIGDIR / f'pc_{k+1:02d}_imag_as_written.png', cmap='coolwarm')
        plot_heatmap(np.abs(M), f'PC {k+1} matrix, abs (formula as written)', FIGDIR / f'pc_{k+1:02d}_abs_as_written.png')

        Mc = principal_component_matrix(vecs_c[:, k], nmodes)
        plot_heatmap(np.real(Mc), f'PC {k+1} matrix, real part (Hermitian)', FIGDIR / f'pc_{k+1:02d}_real_hermitian.png', cmap='coolwarm')
        if np.max(np.abs(np.imag(Mc))) > 1e-12:
            plot_heatmap(np.imag(Mc), f'PC {k+1} matrix, imag part (Hermitian)', FIGDIR / f'pc_{k+1:02d}_imag_hermitian.png', cmap='coolwarm')
        plot_heatmap(np.abs(Mc), f'PC {k+1} matrix, abs (Hermitian)', FIGDIR / f'pc_{k+1:02d}_abs_hermitian.png')

    with (OUTDIR / 'pair_ordering.txt').open('w', encoding='utf-8') as f:
        for idx, (mu, nu) in enumerate(pairs):
            f.write(f'{idx},{mu},{nu}\n')

    print(f'Loaded modes: nmodes={nmodes} (from total {nmodes_total}, capped at {MAX_MODES})')
    print(f'Computed pair space size: {npairs}')
    print('Saved outputs to:')
    print(f'  {OUTDIR}')
    print(f'  {FIGDIR}')
    print('Leading singular/eigen values:')
    for k in range(min(10, len(svals))):
        print(f'  as-written SVD[{k}] = {svals[k]:.8e}')
    for k in range(min(10, len(svals_c))):
        print(f'  hermitian eig[{k}] = {svals_c[k]:.8e}')


if __name__ == '__main__':
    main()
