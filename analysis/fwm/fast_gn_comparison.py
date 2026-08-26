"""Ease-of-computation exploration: this pipeline vs GN-style FWM evaluation.

Maps the two GN reference models of the theory doc (lorenzi_fast_method.md
section 5.3) into the pipeline's normalized per-tuple units and measures, per
probe target, the wall time and the accuracy of

* REF          -- ``target_fast_sums``: exhaustive tuples, linear phase,
                  exact-conditional-acceptance refinement (reference path).
* FAST         -- ``target_analytic_sums``: epsilon-tube + analytic branches
                  with the self-certified truncation bound (production path).
* GN-CFM analog -- center-family triplets only, receiver at the COI center
                  (the x_d = 0 section, so x_c = x_a + x_b + d and the phase
                  has two effective legs nu_a - nu_c, nu_b - nu_c),
                  circumscribed rectangle (mask window Pi dropped), linear
                  phase; evaluated with the closed-form/CF machinery.  This is
                  the Gan et al. closed-form model transplanted onto the
                  lossless single-span kernel.
* GN-NI analog  -- all families, exact per-island 2-D midpoint integral at
                  x_d = 0 with the full quadratic in-channel phase: a coarse
                  Riemann pass on every tuple plus a fine pass on the top
                  contributors -- the brute-force numerical-integral model's
                  mathematics on the same kernel.

All four produce dimensionless per-target strict-FWM efficiency sums, so the
ratios directly quantify the GN assumption stack (locally-white receiver,
rectangle domain, family collapse) and the timings quantify ease.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
from pynlin.methods.td.fast_analytic import target_analytic_sums
from pynlin.methods.td.fast_nlin import (
    FWMTupleVariables,
    fwm_tuple_variables,
    target_fast_sums,
)
from pynlin.methods.td.gn_analog import (
    center_family_indices,
    gn_cfm_tuple_values,
    gn_ni_from_variables,
)
from pynlin.methods.td.fullband_mc import (
    _beta0_abs_from_fiber,
    decimated_frequency_grid,
)
from pynlin.system import System

# Fixed method order and CVD-validated categorical palette (Okabe-Ito subset).
METHODS = ("REF", "FAST", "GN-CFM", "GN-NI")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
SCHEMA_VERSION = "fast_gn_comparison.v1"
CODE_OR_MODEL_VERSION = "pynlin-0.2.0/gn-analog-v1"


def gn_cfm_analog(
    variables: FWMTupleVariables, family_step_dimensionless: float
) -> tuple[int, float]:
    """Center-family GN-CFM analog sum (see pynlin.methods.td.gn_analog)."""
    sel = center_family_indices(variables, family_step_dimensionless)
    if sel.size == 0:
        return 0, 0.0
    return int(sel.size), float(np.sum(gn_cfm_tuple_values(variables, sel)))


def gn_ni_analog(
    variables: FWMTupleVariables,
    n_coarse: int,
    n_fine: int,
    n_refine: int,
    n_xd: int = 0,
) -> tuple[float, float, int]:
    """GN-NI analog sum (see pynlin.methods.td.gn_analog)."""
    return gn_ni_from_variables(
        variables, n_coarse=n_coarse, n_fine=n_fine, n_refine=n_refine, n_xd=n_xd
    )


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("media/lorenzi-fast"))
    parser.add_argument(
        "--targets", type=int, nargs="+", default=[0, 380, 1141, 1720, 2283],
        help="Full-grid probe targets (default: O edge, near-ZDW, mid-E, mid-C, U edge).",
    )
    parser.add_argument("--decimation", type=int, default=1,
                        help="Grid decimation (smoke tests only; physics needs 1).")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--ni-coarse", type=int, default=8)
    parser.add_argument("--ni-fine", type=int, default=32)
    parser.add_argument("--ni-refine", type=int, default=65536)
    parser.add_argument("--skip-ref", action="store_true",
                        help="Skip the reference pipeline (smoke tests).")
    parser.add_argument("--closure-xd", type=int, default=0,
                        help="If > 0, also run the closure test: the GN-NI "
                        "quadrature averaged over this many x_d receiver "
                        "sections. This closes to the masked 3-D local-"
                        "quadratic integral; comparison with linear REF is "
                        "also sensitive to local quadratic phase.")
    args = parser.parse_args()

    system = System.from_toml(args.config)
    full_grid_indices, freqs = decimated_frequency_grid(system, args.decimation)
    baud_rate = float(system.pulse.baud_rate)
    length = float(system.fiber_length)
    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta0_abs = _beta0_abs_from_fiber(system, freqs, beta1)
    spacing = float(np.median(np.diff(np.sort(freqs))))
    family_step = 2.0 * np.pi * spacing / baud_rate  # d units per triplet family

    requested_targets = np.asarray(args.targets, dtype=int)
    full_grid_size = int(np.asarray(system.wdm.frequency_grid()).size)
    if np.any((requested_targets < 0) | (requested_targets >= full_grid_size)):
        raise ValueError(f"targets must be full-grid indices in [0, {full_grid_size})")
    target_positions = np.array(
        [int(np.argmin(np.abs(full_grid_indices - target))) for target in requested_targets]
    )
    if np.unique(target_positions).size != target_positions.size:
        raise ValueError("multiple requested targets map to the same decimated channel")
    target_indices_in_full_grid = full_grid_indices[target_positions]
    if np.any(target_indices_in_full_grid != requested_targets):
        lg.warning(
            "decimation mapped requested full-grid targets {} to nearest retained targets {}",
            requested_targets.tolist(), target_indices_in_full_grid.tolist(),
        )

    rows = []
    for target_idx, t in zip(target_indices_in_full_grid, target_positions):
        target_idx = int(target_idx)
        t = int(t)
        t0 = time.perf_counter()
        variables = fwm_tuple_variables(
            freqs, beta0_abs, beta1, beta2, baud_rate, length, t
        )
        t_enum = time.perf_counter() - t0
        n_total = int(variables.u0.size)
        lg.info(
            f"target {target_idx} (work position {t}, f={freqs[t]*1e-12:.1f} THz): "
            f"{n_total} tuples "
            f"enumerated in {t_enum:.1f} s"
        )

        if args.skip_ref:
            ref_fwm, t_ref = np.nan, np.nan
        else:
            t0 = time.perf_counter()
            ref = target_fast_sums(freqs, beta0_abs, beta1, beta2, baud_rate, length, t)
            t_ref = time.perf_counter() - t0
            ref_fwm = ref.fwm
            lg.info(f"  REF    fwm={ref_fwm:.6e}  t={t_ref:.1f}s (incl. enumeration)")

        t0 = time.perf_counter()
        ana = target_analytic_sums(
            freqs, beta0_abs, beta1, beta2, baud_rate, length, t, epsilon=args.epsilon
        )
        t_fast = time.perf_counter() - t0
        lg.info(
            f"  FAST   fwm={ana.fwm:.6e}  kept={ana.fwm_tuples_kept}/{n_total} "
            f"cert/kept={ana.certificate/max(ana.fwm,1e-300):.1e}  t={t_fast:.1f}s"
        )

        t0 = time.perf_counter()
        n_cfm, cfm_fwm = gn_cfm_analog(variables, family_step)
        t_cfm = t_enum + time.perf_counter() - t0
        lg.info(
            f"  GN-CFM fwm={cfm_fwm:.6e}  triplets={n_cfm} "
            f"ratio={cfm_fwm/ref_fwm if ref_fwm else np.nan:.4f}  "
            f"t={t_cfm:.2f}s (incl. enumeration)"
        )

        t0 = time.perf_counter()
        ni_fwm, ni_coarse_sum, ni_refined = gn_ni_analog(
            variables, args.ni_coarse, args.ni_fine, args.ni_refine
        )
        t_ni = t_enum + time.perf_counter() - t0
        lg.info(
            f"  GN-NI  fwm={ni_fwm:.6e}  ratio={ni_fwm/ref_fwm if ref_fwm else np.nan:.4f} "
            f"coarse-only={ni_coarse_sum:.6e} refined={ni_refined}  "
            f"t={t_ni:.1f}s (incl. enumeration)"
        )

        cl_fwm = np.nan
        t_cl = np.nan
        if args.closure_xd > 0:
            t0 = time.perf_counter()
            cl_fwm, _, _ = gn_ni_analog(
                variables, args.ni_coarse, args.ni_fine, args.ni_refine,
                n_xd=args.closure_xd,
            )
            t_cl = t_enum + time.perf_counter() - t0
            lg.info(
                f"  CLOSURE(x_d avg, {args.closure_xd} sections) "
                f"fwm={cl_fwm:.6e}  closure/REF={cl_fwm/ref_fwm if ref_fwm else np.nan:.4f}  "
                f"slice/closure={ni_fwm/cl_fwm if cl_fwm else np.nan:.4f}  "
                f"t={t_cl:.1f}s (incl. enumeration)"
            )

        rows.append((
            target_idx, t, freqs[t], n_total, t_enum,
            ref_fwm, t_ref,
            ana.fwm, ana.certificate, ana.fwm_tuples_kept, t_fast,
            cfm_fwm, n_cfm, t_cfm,
            ni_fwm, ni_coarse_sum, ni_refined, t_ni,
            cl_fwm, t_cl,
        ))

    arr = np.array(rows, dtype=float)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    units = {
        "target_indices_in_full_grid": "1",
        "target_positions_in_work_grid": "1",
        "target_frequencies_hz": "Hz",
        "strict_fwm_tuple_counts": "1",
        "enumeration_wall_times_s": "s",
        "ref_efficiency_sums": "1",
        "ref_wall_times_s": "s",
        "fast_efficiency_sums": "1",
        "fast_certificate_bounds": "1",
        "fast_kept_tuple_counts": "1",
        "fast_wall_times_s": "s",
        "gn_cfm_efficiency_sums": "1",
        "gn_cfm_tuple_counts": "1",
        "gn_cfm_wall_times_s": "s",
        "gn_ni_efficiency_sums": "1",
        "gn_ni_coarse_efficiency_sums": "1",
        "gn_ni_refined_tuple_counts": "1",
        "gn_ni_wall_times_s": "s",
        "gn_ni_closure_efficiency_sums": "1",
        "gn_ni_closure_wall_times_s": "s",
        "target_grid_stride": "1",
        "interferer_grid_stride": "1",
        "symbol_rate_baud": "baud",
        "fiber_length_m": "m",
        "truncation_epsilon_dimensionless": "1",
        "gn_ni_coarse_grid_size": "1",
        "gn_ni_fine_grid_size": "1",
        "gn_ni_refine_tuple_limit": "1",
        "gn_ni_closure_receiver_section_count": "1",
    }
    scalar_fields = {
        "target_grid_stride", "interferer_grid_stride", "symbol_rate_baud",
        "fiber_length_m", "truncation_epsilon_dimensionless",
        "gn_ni_coarse_grid_size", "gn_ni_fine_grid_size",
        "gn_ni_refine_tuple_limit",
        "gn_ni_closure_receiver_section_count",
    }
    axes = {
        name: ([] if name in scalar_fields else ["target"]) for name in units
    }
    frequency_grid_hash = hashlib.sha256(
        np.ascontiguousarray(freqs, dtype="<f8").tobytes()
    ).hexdigest()
    np.savez(
        args.out_dir / "gn_comparison.npz",
        schema_version=SCHEMA_VERSION,
        units=json.dumps(units, sort_keys=True),
        axes=json.dumps(axes, sort_keys=True),
        phase_model=json.dumps({
            "REF": "linear",
            "FAST": "linear",
            "GN-CFM": "linear_xd_zero_unmasked_rectangle",
            "GN-NI": "local_quad_xd_zero",
        }, sort_keys=True),
        power_basis="single_pol",
        reference_plane="launch_ref",
        tuple_ordering="ordered_ab",
        frequency_grid_hash=frequency_grid_hash,
        symbol_rate_baud=baud_rate,
        fiber_length_m=length,
        gamma_model="not_applied_prefactor_free_efficiency",
        code_or_model_version=CODE_OR_MODEL_VERSION,
        target_grid_stride=int(args.decimation),
        interferer_grid_stride=int(args.decimation),
        target_indices_in_full_grid=arr[:, 0].astype(np.int64),
        target_positions_in_work_grid=arr[:, 1].astype(np.int64),
        target_frequencies_hz=arr[:, 2],
        strict_fwm_tuple_counts=arr[:, 3].astype(np.int64),
        enumeration_wall_times_s=arr[:, 4],
        ref_efficiency_sums=arr[:, 5],
        ref_wall_times_s=arr[:, 6],
        fast_efficiency_sums=arr[:, 7],
        fast_certificate_bounds=arr[:, 8],
        fast_kept_tuple_counts=arr[:, 9].astype(np.int64),
        fast_wall_times_s=arr[:, 10],
        gn_cfm_efficiency_sums=arr[:, 11],
        gn_cfm_tuple_counts=arr[:, 12].astype(np.int64),
        gn_cfm_wall_times_s=arr[:, 13],
        gn_ni_efficiency_sums=arr[:, 14],
        gn_ni_coarse_efficiency_sums=arr[:, 15],
        gn_ni_refined_tuple_counts=arr[:, 16].astype(np.int64),
        gn_ni_wall_times_s=arr[:, 17],
        gn_ni_closure_efficiency_sums=arr[:, 18],
        gn_ni_closure_wall_times_s=arr[:, 19],
        truncation_epsilon_dimensionless=args.epsilon,
        gn_ni_coarse_grid_size=args.ni_coarse,
        gn_ni_fine_grid_size=args.ni_fine,
        gn_ni_refine_tuple_limit=args.ni_refine,
        gn_ni_closure_receiver_section_count=args.closure_xd,
    )

    freq_thz = arr[:, 2] * 1e-12
    sums = np.stack([arr[:, 5], arr[:, 7], arr[:, 11], arr[:, 14]])
    times = np.stack([arr[:, 6], arr[:, 10], arr[:, 13], arr[:, 17]])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for i, (name, color) in enumerate(zip(METHODS, COLORS)):
        axes[0].semilogy(freq_thz, sums[i], "o-", ms=5, lw=1.5, color=color, label=name)
        if i > 0:
            axes[1].plot(freq_thz, sums[i] / sums[0], "o-", ms=5, lw=1.5, color=color,
                         label=name)
        axes[2].semilogy(freq_thz, times[i], "o-", ms=5, lw=1.5, color=color)
    axes[0].set_ylabel(r"strict-FWM efficiency sum $S_t$")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].axhline(1.0, color="gray", lw=0.8, ls=":")
    axes[1].set_ylabel("ratio to REF")
    axes[1].legend(fontsize=7, frameon=False)
    axes[2].set_ylabel("wall time per target [s]")
    for ax in axes:
        ax.set_xlabel("target frequency [THz]")
        ax.grid(True, alpha=0.25)
        ax.ticklabel_format(axis="x", useOffset=False)
    fig.suptitle("FWM contribution per target: this pipeline vs GN-style evaluation")
    fig.tight_layout()
    fig.savefig(args.out_dir / "gn_comparison.png", dpi=200)
    plt.close(fig)
    lg.success(f"GN comparison saved to {args.out_dir}")


if __name__ == "__main__":
    main()
