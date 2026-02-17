from pathlib import Path

import numpy as np
from loguru import logger as lg
from scipy.constants import c

from pynlin.nlin import nlin_estimator_uwb as nlin_uwb
from pynlin.nlin import pcfm_gn as pcfm
from pynlin.nlin.pcfm_gn import PcfmConfig
from pynlin.system import System


def _format_array_snippet(values: np.ndarray, max_items: int = 6) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return "[]"
    fmt = lambda x: f"{x:.3e}"
    if arr.size <= max_items:
        return "[" + ", ".join(fmt(v) for v in arr) + "]"
    half = max_items // 2
    head = ", ".join(fmt(v) for v in arr[:half])
    tail = ", ".join(fmt(v) for v in arr[-half:])
    return "[" + head + ", ..., " + tail + "]"


def _summarize_array(name: str, values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return {
            "name": name,
            "shape": str(arr.shape),
            "min": "n/a",
            "mean": "n/a",
            "max": "n/a",
            "sample": "[]",
        }
    return {
        "name": name,
        "shape": str(arr.shape),
        "min": f"{float(np.min(flat)):.3e}",
        "mean": f"{float(np.mean(flat)):.3e}",
        "max": f"{float(np.max(flat)):.3e}",
        "sample": _format_array_snippet(flat),
    }


def _format_param_table(title: str, meta_lines: list[str], rows: list[dict]) -> str:
    headers = ["name", "shape", "min", "mean", "max", "sample"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))
    line_parts = [
        title,
        *meta_lines,
        "  " + "  ".join(h.ljust(widths[h]) for h in headers),
        "  " + "  ".join("-" * widths[h] for h in headers),
    ]
    for row in rows:
        line_parts.append(
            "  " + "  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
        )
    return "\n".join(line_parts)


def _log_td_pcfm_parameters(
    system: System,
    launch_powers_w: np.ndarray,
    profile_path: Path | str,
    cfg: PcfmConfig | None = None,
) -> None:
    """Print parameter tables for TD vs PCFM dispersion usage."""
    cfg = cfg or PcfmConfig()
    freqs = system.wdm.frequency_grid()
    wavelengths = c / freqs
    bandwidth_hz = float(system.pulse.baud_rate)
    launch_psd = launch_powers_w / bandwidth_hz
    beta1_td, beta2_td = system.beta_grids(freqs=freqs)

    beta2_pcfm = pcfm._beta2_array(system, freqs)
    aeff_td = nlin_uwb._effective_area_array(system, freqs)
    aeff_pcfm = pcfm._aeff_array(system, freqs)
    n2 = 2.6e-20
    gamma_td = n2 * (2.0 * np.pi * freqs) / (aeff_td * c)
    gamma_pcfm = n2 * (2.0 * np.pi * freqs) / (aeff_pcfm * c)

    beta2_td_ref = beta2_td[0] if beta2_td.ndim == 2 else beta2_td
    beta2_diff = beta2_td_ref - beta2_pcfm

    coeffs = pcfm._beta_coeffs_from_profile(
        system,
        float(system.center_frequency or np.mean(freqs)),
    )
    beta2_eff_diag = None
    if cfg.use_beta2_eff and coeffs is not None:
        beta2_eff_diag = np.array([pcfm._beta2_eff(f, f, coeffs) for f in freqs], dtype=float)

    td_rows = [
        _summarize_array("freqs_Hz", freqs),
        _summarize_array("wavelengths_m", wavelengths),
        _summarize_array("channel_bandwidth_Hz", np.array([bandwidth_hz])),
        _summarize_array("launch_psd_W_per_Hz", launch_psd),
        _summarize_array("beta1_TD_s_per_m", beta1_td),
        _summarize_array("beta2_TD_s2_per_m", beta2_td),
        _summarize_array("Aeff_TD_m2", aeff_td),
        _summarize_array("gamma_TD_1_per_W_m", gamma_td),
        _summarize_array("launch_powers_W", launch_powers_w),
    ]
    td_meta = [
        f"  n_modes={system.n_modes}, n_channels={freqs.size}, "
        f"fiber_length_km={float(system.fiber_length) / 1e3:.2f}, "
        f"baud_rate_GBd={float(system.pulse.baud_rate) * 1e-9:.2f}",
    ]
    lg.info(
        _format_param_table(
            f"TD parameters (collision-coefficient method) for {profile_path}:",
            td_meta,
            td_rows,
        )
    )

    pcfm_rows = [
        _summarize_array("freqs_Hz", freqs),
        _summarize_array("wavelengths_m", wavelengths),
        _summarize_array("channel_bandwidth_Hz", np.array([bandwidth_hz])),
        _summarize_array("launch_psd_W_per_Hz", launch_psd),
        _summarize_array("beta2_PCFM_s2_per_m", beta2_pcfm),
        _summarize_array("Aeff_PCFM_m2", aeff_pcfm),
        _summarize_array("gamma_PCFM_1_per_W_m", gamma_pcfm),
        _summarize_array("launch_powers_W", launch_powers_w),
        _summarize_array("beta2_TD_minus_PCFM_s2_per_m", beta2_diff),
    ]
    if beta2_eff_diag is not None:
        pcfm_rows.append(_summarize_array("beta2_eff_diag_s2_per_m", beta2_eff_diag))
        pcfm_rows.append(
            _summarize_array(
                "beta2_eff_minus_beta2_s2_per_m",
                beta2_eff_diag - beta2_pcfm,
            )
        )
    pcfm_meta = [
        f"  use_beta2_eff={cfg.use_beta2_eff}, use_numeric_sci={cfg.use_numeric_sci}, "
        f"use_numeric_xci={cfg.use_numeric_xci}, degree={cfg.degree}",
    ]
    lg.info(
        _format_param_table(
            f"PCFM parameters (pcfm_gn) for {profile_path}:",
            pcfm_meta,
            pcfm_rows,
        )
    )
