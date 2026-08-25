"""Lorenzi Fast S5: full-band XPM + FWM spectrum, fast path vs MC reference.

Computes the fast prefactor-free spectrum (process-parallel), plus the
exhaustive-support MC reference at a subset of probe targets, and overlays
both per component.

Decimation semantics: ``--decimation`` thins the TARGET LIST ONLY. Tuples
and XPM pairs are always enumerated on the full channel grid, so every
computed value is the physical one for the real system, just sampled at
fewer target channels. (The earlier semantics decimated the interferer
population too, which changes the physics: measured against the full grid it
left XPM low by ~4x and FWM low by ~100x with severe grid-commensurability
spike artifacts. Interferer decimation is therefore no longer expressible
here.)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pynlin  # noqa: F401
from loguru import logger as lg

from analysis.log_init import init_logging
from pynlin.methods.td.fast_nlin import target_fast_sums
from pynlin.methods.td.fullband_mc import (
    _beta0_abs_from_fiber,
    compute_fullband_prefactor_free_mc,
    decimated_frequency_grid,
    estimate_zdw_frequency,
)
from pynlin.system import System

_G: dict = {}


def _init_worker(freqs, beta0_abs, beta1, beta2, baud_rate, length, n_refine):
    _G.update(
        freqs=freqs, beta0_abs=beta0_abs, beta1=beta1, beta2=beta2,
        baud_rate=baud_rate, length=length, n_refine=n_refine,
    )


def _work(target: int):
    res = target_fast_sums(
        _G["freqs"], _G["beta0_abs"], _G["beta1"], _G["beta2"],
        _G["baud_rate"], _G["length"], int(target), n_refine=_G["n_refine"],
    )
    out = (
        int(target), res.xpm, res.fwm, res.fwm_tuples,
        res.regime_counts, res.refined_mass_fraction,
    )
    # Near the ZDW a single target can transiently hold multi-GB of tuple
    # arrays; glibc's allocator does not always return freed arena memory to
    # the OS between targets processed by the same long-lived worker, so RSS
    # can ratchet upward across a run even though each target's Python
    # objects are collected. Force both a GC pass and an explicit trim so
    # worker RSS reflects live objects, not retained arena high-water marks.
    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    return out


def _band_local_positions(
    kept_indices: np.ndarray, band_slices: dict[str, slice]
) -> dict[str, np.ndarray]:
    """Map each named ITU band (defined on the full grid) to the decimated-
    grid positions that fall inside it."""
    out: dict[str, np.ndarray] = {}
    for name, sl in band_slices.items():
        lo = int(np.searchsorted(kept_indices, sl.start, side="left"))
        hi = int(np.searchsorted(kept_indices, sl.stop, side="left"))
        if hi > lo:
            out[name] = np.arange(lo, hi)
    return out


def select_mc_probes(
    *,
    freqs: np.ndarray,
    signal: np.ndarray,
    band_slices: dict[str, slice] | None,
    kept_indices: np.ndarray,
    zdw_freq: float | None,
    total_probes: int,
    min_per_band: int,
) -> np.ndarray:
    """Allocate MC reference targets against the real WDM structure instead
    of a blind uniform stride over the decimated index.

    Every band always gets its two edge channels (where the FWM support
    truncates sharply against the next guard gap) plus the channel nearest
    the ZDW if the ZDW falls inside that band. The remaining budget is split
    across bands proportional to band size (with a floor so small bands like
    C are not starved) and, within each band, spent on the positions where
    the fast total signal varies fastest -- band edges, ZDW ripples, and
    oscillatory structure are exactly where the analytic regimes are under
    the most stress, so that is where an MC reference is most informative.
    Bandless systems (single-band RegularWDM) fall back to one group over
    the whole grid, still gradient-weighted rather than uniform.
    """
    n = freqs.size
    log_signal = np.log(np.maximum(signal, 1e-300))
    grad = np.zeros(n)
    if n > 2:
        grad[1:-1] = 0.5 * np.abs(log_signal[2:] - log_signal[:-2])
    if n > 1:
        grad[0] = abs(log_signal[1] - log_signal[0])
        grad[-1] = abs(log_signal[-1] - log_signal[-2])

    groups = (
        _band_local_positions(kept_indices, band_slices)
        if band_slices
        else {"all": np.arange(n)}
    )

    forced: set[int] = set()
    for pos in groups.values():
        forced.add(int(pos[0]))
        forced.add(int(pos[-1]))
        if zdw_freq is not None and np.isfinite(zdw_freq):
            band_freqs = freqs[pos]
            if band_freqs.min() <= zdw_freq <= band_freqs.max():
                forced.add(int(pos[int(np.argmin(np.abs(band_freqs - zdw_freq)))]))

    sizes = {name: pos.size for name, pos in groups.items()}
    total_size = max(sum(sizes.values()), 1)
    remaining_budget = max(total_probes - len(forced), 0)
    per_band = {
        name: max(min_per_band, int(round(remaining_budget * sizes[name] / total_size)))
        for name in groups
    }

    selected = set(forced)
    for name, pos in groups.items():
        candidates = np.array([p for p in pos if int(p) not in selected], dtype=int)
        budget = per_band[name]
        if candidates.size == 0 or budget <= 0:
            continue
        order = np.argsort(grad[candidates])[::-1]
        pick = candidates[order[:budget]]
        selected.update(int(p) for p in pick)

    result = np.array(sorted(selected), dtype=int)
    if result.size > total_probes:
        extra = np.array([p for p in result if p not in forced], dtype=int)
        keep_n = max(total_probes - len(forced), 0)
        keep_extra = extra[np.argsort(grad[extra])[::-1][:keep_n]]
        result = np.array(sorted(forced | set(keep_extra.tolist())), dtype=int)
    return result


def _load_checkpoint(
    path: Path, n: int, n_refine: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a per-target checkpoint if it matches this run's configuration.

    Returns (xpm_fast, fwm_fast, refined_mass, done); NaN/False for targets
    not yet computed. With targets-only decimation the stored per-target
    values are physical full-grid quantities whatever the decimation, so the
    stored decimation is provenance only and a run at any decimation can
    reuse (and extend) the same checkpoint. A mismatch in n_refine or grid
    size is still treated as no checkpoint.
    """
    xpm_fast = np.full(n, np.nan)
    fwm_fast = np.full(n, np.nan)
    refined_mass = np.full(n, np.nan)
    done = np.zeros(n, dtype=bool)
    if not path.exists():
        return xpm_fast, fwm_fast, refined_mass, done
    ck = np.load(path)
    if (
        int(ck["n_refine"]) != int(n_refine)
        or int(ck["xpm_fast"].size) != n
    ):
        lg.warning(f"checkpoint at {path} does not match this run's config; ignoring")
        return xpm_fast, fwm_fast, refined_mass, done
    xpm_fast[:] = ck["xpm_fast"]
    fwm_fast[:] = ck["fwm_fast"]
    refined_mass[:] = ck["refined_mass"]
    done[:] = ck["done"]
    return xpm_fast, fwm_fast, refined_mass, done


def _interleaved_order(indices: np.ndarray, stride: int = 41) -> np.ndarray:
    """Reorder targets into a progressive, full-band-first sweep.

    Sorting by ``index % stride`` (stable, so each residue class keeps
    ascending order) groups together indices that are ``stride`` apart --
    each residue class alone already spans the whole grid at that spacing.
    Processing residue classes in order means the first ~1/stride of the
    run gives one roughly-evenly-spaced sample across the *entire* band,
    refining band-wide as more residues complete, instead of exhausting one
    contiguous (and potentially very expensive, as near the ZDW) region
    before anything else is visible. ``stride`` is prime-ish and unrelated
    to the ~4-channel FWM recurrence period found in this band, so it
    doesn't systematically over- or under-sample any real physical feature.
    """
    return indices[np.argsort(indices % stride, kind="stable")]


def _save_checkpoint(
    path: Path,
    xpm_fast: np.ndarray,
    fwm_fast: np.ndarray,
    refined_mass: np.ndarray,
    done: np.ndarray,
    decimation: int,
    n_refine: int,
) -> None:
    """Atomically write the checkpoint (write-then-rename survives a kill
    mid-write, which a direct np.savez to the final path would not)."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        xpm_fast=xpm_fast, fwm_fast=fwm_fast, refined_mass=refined_mass,
        done=done, decimation=decimation, n_refine=n_refine,
    )
    tmp.replace(path)


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--mc-total-probes", type=int, default=48)
    parser.add_argument("--mc-min-per-band", type=int, default=4)
    parser.add_argument("--mc-frequency-samples", type=int, default=1000)
    parser.add_argument("--mc-xpm-samples", type=int, default=200_000)
    parser.add_argument("--n-refine", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--no-mc", action="store_true",
        help="Skip the exhaustive-support MC reference (fast path only).",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Per-target checkpoint file; auto-derived from decimation if omitted. "
        "Resumes automatically if it exists and matches this run's config.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Save the checkpoint after this many newly completed targets.",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpointing (always recompute from scratch, no resume file written).",
    )
    args = parser.parse_args()

    system = System.from_toml(args.config)
    # Full grid always: decimation must not touch the interferer population.
    _, freqs = decimated_frequency_grid(system, 1)
    baud_rate = float(system.pulse.baud_rate)
    length = float(system.fiber_length)
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta0_abs = _beta0_abs_from_fiber(system, freqs, beta1)
    n = freqs.size
    targets = np.arange(0, n, max(int(args.decimation), 1), dtype=int)
    workers = args.workers if args.workers > 0 else max((os.cpu_count() or 2) - 1, 1)
    lg.info(
        f"{n} channels (full grid); decimation={args.decimation} thins targets only: "
        f"{targets.size} targets, {workers} workers"
    )

    L2 = length**2
    # Decimation-free name: any-decimation runs share/extend one checkpoint.
    checkpoint_path = args.checkpoint or (
        args.out_dir / f"s5_checkpoint_nrefine{args.n_refine}.npz"
    )
    if args.no_checkpoint:
        xpm_fast = np.full(n, np.nan)
        fwm_fast = np.full(n, np.nan)
        refined_mass = np.full(n, np.nan)
        done = np.zeros(n, dtype=bool)
    else:
        xpm_fast, fwm_fast, refined_mass, done = _load_checkpoint(
            checkpoint_path, n, args.n_refine
        )
        if np.any(done):
            lg.info(f"resuming from checkpoint: {int(np.sum(done))}/{n} targets already done")

    remaining = _interleaved_order(targets[~done[targets]])
    t0 = time.perf_counter()
    if remaining.size:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        completed_since_save = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(freqs, beta0_abs, beta1, beta2, baud_rate, length, args.n_refine),
        ) as pool:
            futures = {pool.submit(_work, int(t)): int(t) for t in remaining}
            for fut in as_completed(futures):
                tgt, xpm, fwm, _tuples, _regimes, rmass = fut.result()
                xpm_fast[tgt] = xpm * L2
                fwm_fast[tgt] = fwm * L2
                refined_mass[tgt] = rmass
                done[tgt] = True
                completed_since_save += 1
                if not args.no_checkpoint and completed_since_save >= args.checkpoint_every:
                    _save_checkpoint(
                        checkpoint_path, xpm_fast, fwm_fast, refined_mass, done,
                        args.decimation, args.n_refine,
                    )
                    lg.info(f"checkpoint saved: {int(np.sum(done))}/{n} done")
                    completed_since_save = 0
        if not args.no_checkpoint:
            _save_checkpoint(
                checkpoint_path, xpm_fast, fwm_fast, refined_mass, done,
                args.decimation, args.n_refine,
            )
    t_fast = time.perf_counter() - t0
    lg.info(
        f"fast path: {t_fast:.1f} s for {remaining.size} newly computed targets "
        f"({targets.size} targets on the {n}-channel grid)"
    )

    zdw = estimate_zdw_frequency(system)
    band_slices = getattr(system.wdm, "_band_slices", None)
    # Probe selection runs on the computed-target subgrid (full-band arrays
    # hold NaN at non-target positions); returned positions are then mapped
    # back to full-grid channel indices.
    signal = xpm_fast + fwm_fast
    probes_local = select_mc_probes(
        freqs=freqs[targets],
        signal=signal[targets],
        band_slices=band_slices,
        kept_indices=targets,
        zdw_freq=zdw,
        total_probes=args.mc_total_probes,
        min_per_band=args.mc_min_per_band,
    )
    probes = targets[probes_local]
    if band_slices:
        band_groups = _band_local_positions(targets, band_slices)
        counts = ", ".join(
            f"{name}={np.intersect1d(pos, probes_local).size}/{pos.size}"
            for name, pos in band_groups.items()
        )
        lg.info(f"probe allocation per band: {counts}")
    if args.no_mc:
        probes = np.array([], dtype=int)
        mc_xpm = np.array([], dtype=float)
        mc_fwm = np.array([], dtype=float)
        t_mc = 0.0
        lg.info("MC reference skipped (--no-mc)")
    else:
        t0 = time.perf_counter()
        # decimation=1: the MC reference must see the full interferer
        # population too; probe indices are full-grid channel indices.
        diag = compute_fullband_prefactor_free_mc(
            system,
            decimation=1,
            target_indices=probes,
            include_xpm=True,
            include_fwm=True,
            xpm_samples=args.mc_xpm_samples,
            fwm_frequency_samples=args.mc_frequency_samples,
            fwm_tuple_selection="exhaustive_support_mc",
            n_workers=workers,
        )
        t_mc = time.perf_counter() - t0
        mc_xpm = diag.xpm
        mc_fwm = diag.fwm
        lg.info(
            f"MC reference: {t_mc:.1f} s for {probes.size} probes "
            f"({t_mc / probes.size:.2f} s/target vs {t_fast / max(targets.size, 1):.2f} fast)"
        )
        xr = xpm_fast[probes] / np.maximum(mc_xpm, 1e-300)
        fr = fwm_fast[probes] / np.maximum(mc_fwm, 1e-300)
        lg.info(
            f"probe ratios fast/MC: XPM mean={np.mean(xr):.3f} "
            f"[{np.min(xr):.3f}, {np.max(xr):.3f}]; "
            f"FWM mean={np.mean(fr):.3f} [{np.min(fr):.3f}, {np.max(fr):.3f}]"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "s5_fullband.npz",
        freqs=freqs, xpm_fast=xpm_fast, fwm_fast=fwm_fast,
        refined_mass=refined_mass, probes=probes, targets=targets,
        xpm_mc=mc_xpm, fwm_mc=mc_fwm,
        t_fast=t_fast, t_mc=t_mc, decimation=args.decimation,
    )

    f_thz = freqs * 1e-12
    ts = targets
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(f_thz[ts], xpm_fast[ts], lw=1.2, color="tab:blue", label="XPM fast")
    ax.plot(f_thz[ts], fwm_fast[ts], lw=1.2, color="tab:red", label="FWM fast")
    ax.plot(f_thz[ts], xpm_fast[ts] + fwm_fast[ts], lw=1.0, color="black", alpha=0.6, label="total fast")
    if probes.size:
        ax.plot(f_thz[probes], mc_xpm, "o", ms=4, mfc="none", color="tab:blue", label="XPM MC")
        ax.plot(f_thz[probes], mc_fwm, "s", ms=4, mfc="none", color="tab:red", label="FWM MC")
    ax.set_yscale("log")
    ax.set_ylabel("prefactor-free sum [m$^2$]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax2 = axes[1]
    if probes.size:
        ax2.plot(f_thz[probes], xr, "o-", ms=3, lw=0.8, color="tab:blue", label="XPM fast/MC")
        ax2.plot(f_thz[probes], fr, "s-", ms=3, lw=0.8, color="tab:red", label="FWM fast/MC")
        ax2.legend(fontsize=8, frameon=False)
    ax2.axhline(1.0, color="black", lw=0.6)
    ax2.set_ylim(0.85, 1.15)
    ax2.set_xlabel("target frequency [THz]")
    ax2.set_ylabel("ratio")
    ax2.grid(True, alpha=0.25)
    if zdw is not None and np.isfinite(zdw):
        for a in axes:
            a.axvline(zdw * 1e-12, color="crimson", lw=0.8, ls=":", alpha=0.7)
    if band_slices:
        f_t_thz = f_thz[ts]
        for name, pos in _band_local_positions(ts, band_slices).items():
            edge_thz = 0.5 * (
                f_t_thz[pos[-1]] + (f_t_thz[pos[0] - 1] if pos[0] > 0 else f_t_thz[pos[0]])
            )
            for a in axes:
                a.axvline(edge_thz, color="gray", lw=0.5, ls="--", alpha=0.4)
    fig.suptitle(
        f"Lorenzi Fast full-band spectrum ({n} channels, "
        f"{targets.size} targets [decimation {args.decimation}, targets only])\n"
        f"fast {t_fast/max(targets.size,1):.2f} s/target vs "
        f"MC {t_mc/max(probes.size,1):.1f} s/target ({probes.size} probes)"
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "s5_fullband.png", dpi=200)
    plt.close(fig)
    lg.success(f"S5 saved to {args.out_dir}")


if __name__ == "__main__":
    main()
