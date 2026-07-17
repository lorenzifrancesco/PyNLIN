from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.config import MCMethodConfig, PCFMMethodConfig, TDMethodConfig
from analysis.methods.io import _launch_referenced_nlin_to_output_power
from analysis.methods.models import _load_or_compute_fullband_mc, _load_or_compute_pcfm_I, _load_or_compute_pcfm_general
from analysis.runtime.cache import method_cache_tag, td_cache_path
from analysis.runtime.context import RunContext
from pynlin.constellation_stats import qam_mu0
from pynlin.methods.mc import compute_chi1_chi2, nlin_from_chi
from pynlin.methods.pcfm import PcfmConfig
from pynlin.methods.td.estimator import collision_coeffs_system_uwb, total_nlin_uwb
from pynlin.methods.td.fullband_mc import FullbandMCDiagnostic, gamma_grid


@dataclass(frozen=True)
class TDResult:
    collision_coeffs: np.ndarray
    launch_nlin_w: np.ndarray
    output_nlin_w: np.ndarray


@dataclass(frozen=True)
class PCFMResult:
    launch_nlin_w: np.ndarray
    output_nlin_w: np.ndarray
    eq18_xci_launch_w: np.ndarray | None = None
    eq18_xci_output_w: np.ndarray | None = None


@dataclass(frozen=True)
class MCResult:
    chi1: np.ndarray
    chi2: np.ndarray
    prefactor: np.ndarray
    nlin_16qam_output_w: np.ndarray


def run_td(context: RunContext, config: TDMethodConfig, *, cache_scope: str) -> TDResult:
    ccfs = collision_coeffs_system_uwb(
        context.system,
        ipulse=1,
        recompute=config.mode == "recompute",
        profile_path=context.profile_path,
        m_lo_truncation=config.m_lo_truncation,
        time_integral_backend=config.time_integral_backend,
    )
    extra_tag = method_cache_tag(
        cache_scope,
        context.cache_tag,
        f"mtrunc{config.m_lo_truncation}",
        f"tib{config.time_integral_backend}",
        "xci" if config.exclude_self_channel else "all",
    )
    launch_nlin = total_nlin_uwb(
        context.system,
        ccfs,
        use_kappa=config.use_kappa,
        use_x_mode=config.use_x_mode,
        launch_powers_w=context.launch_powers_w,
        exclude_self_channel=config.exclude_self_channel,
        cache_path=td_cache_path(
            context.profile_path,
            use_kappa=config.use_kappa,
            use_x_mode=config.use_x_mode,
            extra_tag=extra_tag,
        ),
        recompute=config.mode == "recompute",
    )
    output_nlin = _launch_referenced_nlin_to_output_power(
        launch_nlin,
        context.output_over_launch_ratio,
    )
    return TDResult(ccfs, launch_nlin, output_nlin)


def run_pcfm(context: RunContext, config: PCFMMethodConfig, *, cache_scope: str) -> PCFMResult:
    pcfm_cfg = PcfmConfig(
        degree=config.degree,
        include_mci=config.include_mci,
        use_numeric_sci=config.numeric_sci,
        use_numeric_xci=config.numeric_xci,
    )
    cache_tag = method_cache_tag(
        cache_scope,
        context.cache_tag,
        f"deg{config.degree}",
        f"nsci{int(config.numeric_sci)}",
        f"nxci{int(config.numeric_xci)}",
        f"mci{int(config.include_mci)}",
    )
    launch_nlin = _load_or_compute_pcfm_I(
        system=context.system,
        profile_path=context.profile_path,
        launch_powers_w=context.launch_powers_w,
        output_path=context.out_dir / f"pcfm_{cache_tag}.npy",
        cfg=pcfm_cfg,
        recompute=config.mode == "recompute",
    )
    output_nlin = _launch_referenced_nlin_to_output_power(
        launch_nlin,
        context.output_over_launch_ratio,
    )
    eq18_launch = None
    eq18_output = None
    if config.eq18_xci:
        eq18_launch = _load_or_compute_pcfm_general(
            system=context.system,
            launch_powers_w=context.launch_powers_w,
            output_path=context.out_dir / f"pcfm_{cache_tag}_xci_eq18.npy",
            profile_path=context.profile_path,
            degree=config.degree,
            xci_model="eq18",
            recompute=config.mode == "recompute",
        )
        eq18_output = _launch_referenced_nlin_to_output_power(
            eq18_launch,
            context.output_over_launch_ratio,
        )
    return PCFMResult(launch_nlin, output_nlin, eq18_launch, eq18_output)


def run_mc(context: RunContext, td: TDResult, td_config: TDMethodConfig, config: MCMethodConfig) -> MCResult:
    _ = config
    chi1, chi2, prefactor = compute_chi1_chi2(
        context.system,
        td.collision_coeffs,
        context.launch_powers_w,
        use_kappa=td_config.use_kappa,
        use_x_mode=td_config.use_x_mode,
        exclude_self_channel=td_config.exclude_self_channel,
    )
    nlin_16qam = nlin_from_chi(chi1, chi2, prefactor, qam_mu0(16))
    nlin_16qam_output = _launch_referenced_nlin_to_output_power(
        nlin_16qam,
        context.output_over_launch_ratio,
    )
    return MCResult(chi1, chi2, prefactor, nlin_16qam_output)


@dataclass(frozen=True)
class FullbandMCResult:
    diagnostic: FullbandMCDiagnostic
    nlin_output_w: np.ndarray


def _fullband_mc_estimate_nlin(
    system, diagnostic: FullbandMCDiagnostic, launch_powers_w: np.ndarray
) -> np.ndarray:
    """Rough NLIN estimate from prefactor-free fullband MC total.

    N_j ≈ (16/81) * gamma² * P_j * P_avg * S_j
    where S_j = XPM_j + FWM_j is the prefactor-free sum.
    """
    freqs = np.asarray(system.wdm.frequency_grid(), dtype=float)
    gamma = gamma_grid(system, freqs)
    mean_launch = float(np.mean(launch_powers_w))
    nlin = np.zeros_like(diagnostic.total, dtype=float)
    for i, grid_idx in enumerate(diagnostic.target_indices):
        P_j = float(launch_powers_w[grid_idx])
        kappa2 = float(gamma[int(grid_idx)]) ** 2 * (16.0 / 81.0)
        nlin[i] = kappa2 * P_j * mean_launch * float(diagnostic.total[i])
    return np.asarray(nlin, dtype=float)


def run_fullband_mc(
    context: RunContext, config: MCMethodConfig, *, cache_scope: str
) -> FullbandMCResult:
    cache_tag = method_cache_tag(
        cache_scope, context.cache_tag,
        f"dec{config.channel_decimation}",
        f"td{config.target_decimation}_to{config.target_offset}_tl{config.target_limit}",
        f"xpm{config.xpm_samples}_fwm{config.fwm_samples}_ff{config.fwm_frequency_samples}",
        f"sel{config.fwm_tuple_selection}",
        f"seed{config.seed}_mt{config.max_fwm_tuples_per_target}",
        f"w{config.workers}",
    )
    diagnostic = _load_or_compute_fullband_mc(
        context.system,
        output_path=context.out_dir / f"fullband_mc_{cache_tag}.npz",
        channel_decimation=config.channel_decimation,
        target_decimation=config.target_decimation,
        target_offset=config.target_offset,
        target_limit=config.target_limit,
        xpm_samples=config.xpm_samples,
        fwm_samples=config.fwm_samples,
        fwm_frequency_samples=config.fwm_frequency_samples,
        seed=config.seed,
        max_fwm_tuples_per_target=config.max_fwm_tuples_per_target,
        fwm_tuple_selection=config.fwm_tuple_selection,
        workers=config.workers,
        recompute=config.mode == "recompute",
    )
    launch_nlin = _fullband_mc_estimate_nlin(context.system, diagnostic, context.launch_powers_w)
    output_nlin = _launch_referenced_nlin_to_output_power(
        launch_nlin,
        context.output_over_launch_ratio[diagnostic.target_indices],
    )
    return FullbandMCResult(diagnostic, output_nlin)
