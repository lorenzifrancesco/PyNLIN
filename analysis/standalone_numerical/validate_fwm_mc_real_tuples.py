"""Validate the Dar FWM MC coefficient on physical channel-tuple sequences.

Unlike the controlled single-tuple scaling study, this script never changes
dispersion, spacing, baud rate, or length to prescribe x and mu.  It loads all
of them from TOML and derives x and mu from each resulting physical tuple.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
        "text.usetex": False,
    }
)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.optimize import brentq

from pynlin.methods.td.fullband_mc import estimate_xpm_n1_local_taylor_mc
from pynlin.methods.td.fwm_kernel import FWMChannels
from pynlin.methods.td.fwm_mc import estimate_fwm_term_sum_dar_mc
from pynlin.methods.td.xhkm_mc import XPMTaylorDispersion, estimate_xhkm_sums_mc
from pynlin.system import System


OUT_MEDIA = Path("media/fwm/real-tuples")
OUT_RESULTS = Path("results/fwm/real-tuples")


def _beta0_grid(system: System, freqs: np.ndarray, beta1: np.ndarray) -> np.ndarray:
    """Return beta at channel centers from the configured global fiber model."""
    spline_factory = getattr(system.fiber, "beta_spline_omega", None)
    if (
        getattr(system.fiber, "_beta_profile", None) is not None
        and spline_factory is not None
    ):
        try:
            return np.asarray(
                spline_factory(s=0.0, k=3)(2.0 * np.pi * freqs), dtype=float
            )
        except (TypeError, ValueError):
            pass

    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    order = np.argsort(omega)
    beta_sorted = np.zeros_like(omega)
    beta1_sorted = np.asarray(beta1, dtype=float)[order]
    beta_sorted[1:] = np.cumsum(
        0.5 * (beta1_sorted[1:] + beta1_sorted[:-1]) * np.diff(omega[order])
    )
    beta0 = np.empty_like(beta_sorted)
    beta0[order] = beta_sorted
    return beta0


def _channels_for_tuple(
    freqs: np.ndarray,
    beta0: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    indices: tuple[int, int, int, int],
    beta3: np.ndarray | None = None,
    beta4: np.ndarray | None = None,
) -> FWMChannels:
    d, a, b, c = indices
    omega = 2.0 * np.pi * (freqs - freqs[d])
    # A common constant and linear beta term cancels under energy conservation.
    # Remove both before differencing to retain tiny ZDW mismatch residuals.
    beta0 = beta0 - beta0[d] - beta1[d] * omega
    beta1 = beta1 - beta1[d]
    return FWMChannels(
        omega_a=float(omega[a]),
        omega_b=float(omega[b]),
        omega_c=float(omega[c]),
        omega_d=0.0,
        beta0_a=float(beta0[a]),
        beta0_b=float(beta0[b]),
        beta0_c=float(beta0[c]),
        beta0_d=0.0,
        beta1_a=float(beta1[a]),
        beta1_b=float(beta1[b]),
        beta1_c=float(beta1[c]),
        beta1_d=float(beta1[d]),
        gvd_a=float(beta2[a]),
        gvd_b=float(beta2[b]),
        gvd_c=float(beta2[c]),
        gvd_d=float(beta2[d]),
        beta3_a=0.0 if beta3 is None else float(beta3[a]),
        beta3_b=0.0 if beta3 is None else float(beta3[b]),
        beta3_c=0.0 if beta3 is None else float(beta3[c]),
        beta3_d=0.0 if beta3 is None else float(beta3[d]),
        beta4_a=0.0 if beta4 is None else float(beta4[a]),
        beta4_b=0.0 if beta4 is None else float(beta4[b]),
        beta4_c=0.0 if beta4 is None else float(beta4[c]),
        beta4_d=0.0 if beta4 is None else float(beta4[d]),
    )


def _dispersion_grids(
    system: System, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one global beta spline and its first four derivatives."""
    spline_factory = getattr(system.fiber, "beta_spline_omega", None)
    if (
        getattr(system.fiber, "_beta_profile", None) is not None
        and spline_factory is not None
    ):
        spline = spline_factory(s=0.0, k=5)
        omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
        return tuple(
            np.asarray(spline.derivative(order)(omega), dtype=float)
            for order in range(5)
        )

    beta1_grid, beta2_grid = system.beta_grids(freqs=freqs)
    beta1 = np.asarray(beta1_grid[0], dtype=float)
    beta2 = np.asarray(beta2_grid[0], dtype=float)
    beta0 = _beta0_grid(system, freqs, beta1)
    zeros = np.zeros_like(beta0)
    return beta0, beta1, beta2, zeros, zeros


def _zdw_frequency(system: System) -> float:
    """Find the beta2 zero inside the measured beta-profile domain."""
    freq_profile = getattr(system.fiber, "_freq_profile", None)
    spline_factory = getattr(system.fiber, "beta_spline_omega", None)
    if freq_profile is None or spline_factory is None:
        raise ValueError("a measured beta profile is required for the ZDW sequence")
    freqs = np.sort(np.asarray(freq_profile, dtype=float))
    beta2_spline = spline_factory(s=0.0, k=5).derivative(2)
    values = np.asarray(beta2_spline(2.0 * np.pi * freqs), dtype=float)
    crossings = np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:]))
    if crossings.size == 0:
        raise ValueError("the measured beta profile has no ZDW")
    index = int(crossings[0])
    omega_root = brentq(
        beta2_spline,
        2.0 * np.pi * float(freqs[index]),
        2.0 * np.pi * float(freqs[index + 1]),
    )
    return float(omega_root / (2.0 * np.pi))


def _polynomial_channels(
    freqs: np.ndarray,
    *,
    reference_frequency: float,
    coefficients: tuple[float, float, float, float, float],
    order: int,
) -> FWMChannels:
    """Translate one global Taylor model into channel-local coefficients."""
    if order not in {2, 3, 4}:
        raise ValueError("dispersion order must be 2, 3, or 4")
    offsets = (
        2.0 * np.pi * (np.asarray(freqs, dtype=float) - float(reference_frequency))
    )
    coefficients = (0.0, 0.0, *coefficients[2:])
    derivatives = []
    for local_order in range(5):
        values = np.zeros_like(offsets)
        for global_order in range(local_order, order + 1):
            values += (
                float(coefficients[global_order])
                * offsets ** (global_order - local_order)
                / float(math.factorial(global_order - local_order))
            )
        derivatives.append(values)
    return _channels_for_tuple(
        np.asarray(freqs, dtype=float),
        derivatives[0],
        derivatives[1],
        derivatives[2],
        (0, 1, 2, 3),
        beta3=derivatives[3],
        beta4=derivatives[4],
    )


def physical_tuple_coordinates(
    channels: FWMChannels, *, baud_rate: float, length: float
) -> tuple[float, float, float, float]:
    """Return center mismatch, gradient norm, x, and mu for fixed channels."""
    output_offset = float(channels.delta_omega)
    beta1_d_output = (
        channels.beta1_d
        + channels.gvd_d * output_offset
        + 0.5 * channels.beta3_d * output_offset**2
        + (channels.beta4_d / 6.0) * output_offset**3
    )
    delta_beta_center = float(
        channels.delta_beta0
        - channels.beta1_d * output_offset
        - 0.5 * channels.gvd_d * output_offset**2
        - (channels.beta3_d / 6.0) * output_offset**3
        - (channels.beta4_d / 24.0) * output_offset**4
    )
    gradient = np.array(
        [
            channels.beta1_a - beta1_d_output,
            channels.beta1_b - beta1_d_output,
            -channels.beta1_c + beta1_d_output,
        ],
        dtype=float,
    )
    gradient_norm = float(np.linalg.norm(gradient))
    mismatch_scale = float(baud_rate) * gradient_norm
    x_grad = float(length) * mismatch_scale
    mu = delta_beta_center / mismatch_scale if mismatch_scale > 0.0 else float("nan")
    return delta_beta_center, gradient_norm, x_grad, mu


def tuple_sequences(
    band_slices: dict[str, slice], *, translation_step: int, max_span: int
) -> list[tuple[str, str, int, tuple[int, int, int, int]]]:
    """Build exact carrier-conserving translation and span sequences per band."""
    records = []
    for band, band_slice in band_slices.items():
        start, stop = int(band_slice.start), int(band_slice.stop)
        # Translation of (d,a,b,c) = (j,j-1,j+2,j+1).
        for d in range(start + 1, stop - 2, int(translation_step)):
            records.append(("translation", str(band), d, (d, d - 1, d + 2, d + 1)))

        # Increasing-span sequence around the central target channel.
        d = (start + stop - 1) // 2
        available = min(d - start, (stop - 1 - d) // 2)
        for span in range(1, min(int(max_span), available) + 1):
            records.append(
                ("span", str(band), span, (d, d - span, d + 2 * span, d + span))
            )
    return records


def profile_sweep_targets(
    system: System, *, step: int, xpm_shift: int, fwm_shift: int
) -> np.ndarray:
    """Return targets for which all configured shifted channels stay in-profile."""
    freq_profile = getattr(system.fiber, "_freq_profile", None)
    if freq_profile is None:
        raise ValueError("a measured beta profile is required for the spectrum sweep")
    spacing = float(system.wdm.spacing)
    profile = np.asarray(freq_profile, dtype=float)
    offsets = np.array([0, int(xpm_shift), int(fwm_shift), 2 * int(fwm_shift)])
    first = (
        np.ceil((float(np.min(profile)) - float(np.min(offsets)) * spacing) / spacing)
        * spacing
    )
    last = float(np.max(profile)) - float(np.max(offsets)) * spacing
    return np.arange(first, last + 1e-9 * spacing, int(step) * spacing)


def estimate_xpm_sector_ensemble(
    *,
    dispersion: XPMTaylorDispersion,
    alpha: float,
    length: float,
    spacing_over_baud: float,
    min_samples: int,
    max_samples: int,
    batch_size: int,
    seed: int,
    sigma_threshold: float,
    max_relative_error: float,
    max_stderr_over_n1: float,
    system: System | None = None,
) -> dict[str, float | bool | str]:
    """Adaptively estimate covariance-aware residual XPM sectors."""
    names = ("n_2pc", "n_3pca", "n_3pcb", "n_4pc", "n1")
    estimate = estimate_xhkm_sums_mc(
        beta2=0.0,
        dispersion=dispersion,
        alpha=alpha,
        length=length,
        channel_spacing_over_baud=spacing_over_baud,
        n_samples=max_samples,
        batch_size=batch_size,
        min_samples=min_samples,
        target_relative_stderr=max_relative_error,
        target_stderr_over_n1=max_stderr_over_n1,
        sigma_threshold=sigma_threshold,
        seed=seed,
        system=system,
    )

    result: dict[str, float | bool | str] = {
        "n_samples": float(estimate.n_samples),
        "stop_reason": str(estimate.metadata["stop_reason"]),
    }
    for name in names:
        mean = float(getattr(estimate, name))
        stderr = float(getattr(estimate, f"{name}_stderr"))
        result[name] = mean
        result[f"{name}_stderr"] = stderr
        if name != "n1":
            relative_error = stderr / mean if mean > 0.0 else float("inf")
            scaled_error = stderr / estimate.n1 if estimate.n1 > 0.0 else float("inf")
            result[f"{name}_resolved"] = bool(
                mean - float(sigma_threshold) * stderr > 0.0
                and (
                    relative_error <= float(max_relative_error)
                    or scaled_error <= float(max_stderr_over_n1)
                )
            )
    return result


def _empty_sector_fields() -> dict[str, float]:
    fields = {
        "xpm_sector_n1": np.nan,
        "xpm_sector_n1_stderr": np.nan,
        "xpm_sector_samples": np.nan,
        "xpm_sector_stop_code": np.nan,
    }
    for name in ("2pc", "3pca", "3pcb", "4pc"):
        fields[f"xpm_{name}"] = np.nan
        fields[f"xpm_{name}_stderr"] = np.nan
        fields[f"xpm_{name}_resolved"] = np.nan
    return fields


def _clip_polygon_half_plane(
    vertices: list[tuple[float, float]], normal: tuple[float, float], bound: float
) -> list[tuple[float, float]]:
    """Clip a convex polygon to ``normal dot point <= bound``."""
    if not vertices:
        return []
    nx, ny = normal
    clipped = []
    previous = vertices[-1]
    previous_value = nx * previous[0] + ny * previous[1] - bound
    for current in vertices:
        current_value = nx * current[0] + ny * current[1] - bound
        previous_inside = previous_value <= 1e-12
        current_inside = current_value <= 1e-12
        if previous_inside != current_inside:
            fraction = previous_value / (previous_value - current_value)
            clipped.append(
                (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
            )
        if current_inside:
            clipped.append(current)
        previous = current
        previous_value = current_value
    return clipped


def _passband_island_polygon(
    center_x: int, center_y: int, center_sum: int, half_bandwidth: float
) -> list[tuple[float, float]]:
    """Return the fixed-d passband island used by the collapse inset."""
    h = float(half_bandwidth)
    vertices = [
        (center_x - h, center_y - h),
        (center_x + h, center_y - h),
        (center_x + h, center_y + h),
        (center_x - h, center_y + h),
    ]
    vertices = _clip_polygon_half_plane(vertices, (1.0, 1.0), center_sum + h)
    return _clip_polygon_half_plane(vertices, (-1.0, -1.0), -(center_sum - h))


def draw_channel_choice_inset(
    parent_ax: plt.Axes,
    *,
    xpm_shift: int,
    fwm_shift: int,
    spacing_over_baud: float,
    channel_spacing: float,
    length: float,
    zdw_beta2: float,
    zdw_beta3: float,
    zdw_beta4: float,
    bounds: tuple[float, float, float, float] = (0.57, 0.08, 0.41, 0.88),
) -> plt.Axes:
    """Highlight the configured XPM and FWM tuples on one collapse-style inset."""
    half_bandwidth = 0.5 / float(spacing_over_baud)
    selected = (
        (0, int(xpm_shift), int(xpm_shift), "C0", "XPM"),
        (int(fwm_shift), int(fwm_shift), 2 * int(fwm_shift), "C1", "FWM"),
    )
    coordinates = [value for item in selected for value in item[:3]]
    coordinate_min = float(min(coordinates))
    coordinate_max = float(max(coordinates))
    center = 0.5 * (coordinate_min + coordinate_max)
    half_window = max(5.0, 0.5 * (coordinate_max - coordinate_min) + 1.0)
    low = int(np.floor(center - half_window))
    high = int(np.ceil(center + half_window))
    ax = parent_ax.inset_axes(bounds)
    ax.set_facecolor("white")
    for center_x in range(low, high + 1):
        for center_y in range(low, high + 1):
            sum_radius = 3.0 * half_bandwidth
            for center_sum in range(
                int(np.ceil(center_x + center_y - sum_radius)),
                int(np.floor(center_x + center_y + sum_radius)) + 1,
            ):
                polygon = _passband_island_polygon(
                    center_x, center_y, center_sum, half_bandwidth
                )
                if len(polygon) >= 3:
                    ax.add_patch(
                        Polygon(
                            polygon,
                            closed=True,
                            fill=False,
                            edgecolor="0.82",
                            linewidth=0.22,
                            zorder=1,
                        )
                    )
    contour_x = np.linspace(low - 0.5, high + 0.5, 420)
    contour_y = np.linspace(low - 0.5, high + 0.5, 420)
    X, Y = np.meshgrid(contour_x, contour_y)
    for center_x, center_y, center_sum, color, label in selected:
        polygon = _passband_island_polygon(
            center_x, center_y, center_sum, half_bandwidth
        )
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.42,
                linewidth=1.2,
                label=label,
                zorder=4,
            )
        )
        # Put the tuple's shifted/repeated channel exactly at the ZDW.  Constant
        # and linear beta terms cancel, so evaluate the stable global quartic
        # model in angular-frequency offsets from the ZDW.
        shift = center_y if label == "XPM" else center_x
        omega_d = -float(shift) * 2.0 * np.pi * float(channel_spacing)
        omega_a = omega_d + X * 2.0 * np.pi * float(channel_spacing)
        omega_b = omega_d + Y * 2.0 * np.pi * float(channel_spacing)
        omega_c = omega_d + (X + Y) * 2.0 * np.pi * float(channel_spacing)

        def beta_relative(omega: np.ndarray | float) -> np.ndarray:
            omega = np.asarray(omega, dtype=float)
            return (
                0.5 * float(zdw_beta2) * omega**2
                + (float(zdw_beta3) / 6.0) * omega**3
                + (float(zdw_beta4) / 24.0) * omega**4
            )

        mismatch = (
            beta_relative(omega_a)
            + beta_relative(omega_b)
            - beta_relative(omega_c)
            - beta_relative(omega_d)
        )
        efficiency = np.sinc(float(length) * mismatch / (2.0 * np.pi)) ** 2
        level = float(np.exp(-1.0))
        if float(np.min(efficiency)) < level < float(np.max(efficiency)):
            ax.contour(
                X,
                Y,
                efficiency,
                levels=[level],
                colors=[color],
                linestyles=["-" if label == "XPM" else "--"],
                linewidths=1.15,
                zorder=5,
            )
    ax.axvline(0.0, color="0.1", lw=0.8, zorder=3)
    ax.axhline(0.0, color="0.1", lw=0.8, zorder=3)
    ax.set_xlim(low - 0.5, high + 0.5)
    ax.set_ylim(low - 0.5, high + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$(f_a-f_d)/\Delta f$", fontsize=5.8, labelpad=1)
    ax.set_ylabel(r"$(f_b-f_d)/\Delta f$", fontsize=5.8, labelpad=1)
    ax.tick_params(labelsize=5.4, length=2.0, pad=1)
    ax.legend(loc="lower right", fontsize=5.8, frameon=True, framealpha=0.88)
    ax.set_title(r"$\eta=e^{-1}$ at ZDW: XPM solid, FWM dashed", fontsize=5.2, pad=1.5)
    for spine in ax.spines.values():
        spine.set_color("0.45")
        spine.set_linewidth(0.6)
    return ax


def compute_dataset(
    system: System,
    *,
    n_samples: int,
    seed: int,
    alpha: float,
    translation_step: int,
    max_span: int,
    zdw_half_window: int,
    zdw_step: int,
    spectrum_step: int,
    xpm_shift: int,
    fwm_shift: int,
    sector_min_samples: int,
    sector_max_samples: int,
    sector_batch_size: int,
    sector_step: int,
    sector_max_relative_error: float,
    sector_max_stderr_over_n1: float,
    sector_sigma_threshold: float,
) -> dict[str, np.ndarray]:
    freqs = np.asarray(system.wdm.frequency_grid(), dtype=float)
    beta0, beta1, beta2, beta3, beta4 = _dispersion_grids(system, freqs)
    baud_rate = float(system.pulse.baud_rate)
    length = float(system.fiber_length)
    band_slices = getattr(system.wdm, "_band_slices", {"all": slice(0, freqs.size)})

    rng = np.random.default_rng(int(seed))
    random_variables = 2.0 * np.pi * (rng.random((3, int(n_samples))) - 0.5)
    rows: list[dict[str, object]] = []
    for sequence, band, coordinate, indices in tuple_sequences(
        band_slices, translation_step=translation_step, max_span=max_span
    ):
        channels = _channels_for_tuple(
            freqs, beta0, beta1, beta2, indices, beta3=beta3, beta4=beta4
        )
        delta_beta, gradient_norm, x_grad, mu = physical_tuple_coordinates(
            channels, baud_rate=baud_rate, length=length
        )
        estimate = estimate_fwm_term_sum_dar_mc(
            channels=channels,
            baud_rate=baud_rate,
            length=length,
            n_samples=n_samples,
            alpha=alpha,
            random_variables=random_variables,
        )
        d, a, b, c = indices
        rows.append(
            {
                "sequence": sequence,
                "band": band,
                "coordinate": coordinate,
                "d": d,
                "a": a,
                "b": b,
                "c": c,
                "target_frequency": freqs[d],
                "tuple_center_frequency": freqs[d],
                "delta_beta_center": delta_beta,
                "gradient_norm": gradient_norm,
                "x": x_grad,
                "mu": mu,
                "beta2_center": beta2[d],
                "beta3_center": beta3[d],
                "beta4_center": beta4[d],
                "noise_coefficient": estimate.total / length**2,
                "noise_stderr": estimate.total_stderr / length**2,
                "noise_local_quadratic": np.nan,
                "noise_local_cubic": np.nan,
                "noise_local_quartic": estimate.total / length**2,
                "noise_profile_quartic": estimate.total / length**2,
                "xpm_noise": np.nan,
                "xpm_stderr": np.nan,
                "fwm_degenerate_noise": np.nan,
                "fwm_degenerate_stderr": np.nan,
                **_empty_sector_fields(),
                "support_fraction": float(estimate.metadata["support_fraction"]),
            }
        )

    zdw = _zdw_frequency(system)
    zdw_coefficients = _dispersion_grids(system, np.array([zdw]))
    spacing = float(system.wdm.spacing)
    offsets = np.arange(-int(zdw_half_window), int(zdw_half_window) + 1, int(zdw_step))
    for offset in offsets:
        center_frequency = zdw + float(offset) * spacing
        tuple_freqs = center_frequency + spacing * np.array(
            [-float(fwm_shift), 0.0, 0.0, float(fwm_shift)]
        )
        local_beta0, local_beta1, local_beta2, local_beta3, local_beta4 = (
            _dispersion_grids(system, tuple_freqs)
        )
        profile_channels = _channels_for_tuple(
            tuple_freqs,
            local_beta0,
            local_beta1,
            local_beta2,
            (0, 1, 2, 3),
            beta3=local_beta3,
            beta4=local_beta4,
        )
        center_coefficients = tuple(
            float(values[0])
            for values in _dispersion_grids(system, np.array([center_frequency]))
        )
        model_channels = tuple(
            _polynomial_channels(
                tuple_freqs,
                reference_frequency=center_frequency,
                coefficients=center_coefficients,
                order=order,
            )
            for order in (2, 3, 4)
        )
        quadratic_channels, cubic_channels, quartic_channels = model_channels
        delta_beta, gradient_norm, x_grad, mu = physical_tuple_coordinates(
            quartic_channels, baud_rate=baud_rate, length=length
        )
        estimates = []
        for current_channels in (*model_channels, profile_channels):
            estimates.append(
                estimate_fwm_term_sum_dar_mc(
                    channels=current_channels,
                    baud_rate=baud_rate,
                    length=length,
                    n_samples=n_samples,
                    alpha=alpha,
                    random_variables=random_variables,
                )
            )
        quadratic, cubic, quartic, profile = estimates
        logical_center = int(offset)
        rows.append(
            {
                "sequence": "zdw",
                "band": "ZDW window",
                "coordinate": center_frequency,
                "d": logical_center - fwm_shift,
                "a": logical_center,
                "b": logical_center,
                "c": logical_center + fwm_shift,
                "target_frequency": tuple_freqs[0],
                "tuple_center_frequency": center_frequency,
                "delta_beta_center": delta_beta,
                "gradient_norm": gradient_norm,
                "x": x_grad,
                "mu": mu,
                "beta2_center": local_beta2[1],
                "beta3_center": local_beta3[1],
                "beta4_center": local_beta4[1],
                "noise_coefficient": quartic.total / length**2,
                "noise_stderr": quartic.total_stderr / length**2,
                "noise_local_quadratic": quadratic.total / length**2,
                "noise_local_cubic": cubic.total / length**2,
                "noise_local_quartic": quartic.total / length**2,
                "noise_profile_quartic": profile.total / length**2,
                "xpm_noise": np.nan,
                "xpm_stderr": np.nan,
                "fwm_degenerate_noise": np.nan,
                "fwm_degenerate_stderr": np.nan,
                **_empty_sector_fields(),
                "support_fraction": float(quartic.metadata["support_fraction"]),
            }
        )

    spectrum_targets = profile_sweep_targets(
        system,
        step=spectrum_step,
        xpm_shift=xpm_shift,
        fwm_shift=fwm_shift,
    )
    for sweep_index, target_frequency in enumerate(spectrum_targets):
        repeated_frequency = target_frequency + float(fwm_shift) * spacing
        fwm_freqs = target_frequency + spacing * np.array(
            [0.0, float(fwm_shift), float(fwm_shift), 2.0 * float(fwm_shift)]
        )
        fwm_beta0, fwm_beta1, fwm_beta2, fwm_beta3, fwm_beta4 = _dispersion_grids(
            system, fwm_freqs
        )
        fwm_channels = _channels_for_tuple(
            fwm_freqs,
            fwm_beta0,
            fwm_beta1,
            fwm_beta2,
            (0, 1, 2, 3),
            beta3=fwm_beta3,
            beta4=fwm_beta4,
        )
        delta_beta, gradient_norm, x_grad, mu = physical_tuple_coordinates(
            fwm_channels, baud_rate=baud_rate, length=length
        )
        fwm_estimate = estimate_fwm_term_sum_dar_mc(
            channels=fwm_channels,
            baud_rate=baud_rate,
            length=length,
            n_samples=n_samples,
            alpha=alpha,
            random_variables=random_variables,
        )

        xpm_frequency = target_frequency + float(xpm_shift) * spacing
        xpm_freqs = np.array([target_frequency, xpm_frequency])
        xpm_beta0, xpm_beta1, xpm_beta2, xpm_beta3, xpm_beta4 = _dispersion_grids(
            system, xpm_freqs
        )
        xpm_omega = 2.0 * np.pi * (xpm_freqs - target_frequency)
        beta0_conditioned = xpm_beta0 - xpm_beta0[0] - xpm_beta1[0] * xpm_omega
        beta1_conditioned = xpm_beta1 - xpm_beta1[0]
        xpm_value, xpm_stderr = estimate_xpm_n1_local_taylor_mc(
            beta0_offsets=beta0_conditioned,
            beta1=beta1_conditioned,
            beta2=xpm_beta2,
            beta3=xpm_beta3,
            beta4=xpm_beta4,
            baud_rate=baud_rate,
            length=length,
            target=0,
            interferer=1,
            n_samples=n_samples,
            seed=None,
            alpha=alpha,
            random_variables=random_variables,
        )
        sector_fields = _empty_sector_fields()
        channel_grid_index = sweep_index * int(spectrum_step)
        if channel_grid_index % int(sector_step) == 0:
            dispersion = XPMTaylorDispersion(
                beta0=beta0_conditioned,
                beta1=beta1_conditioned,
                beta2=xpm_beta2,
                beta3=xpm_beta3,
                beta4=xpm_beta4,
                baud_rate=baud_rate,
            )
            sectors = estimate_xpm_sector_ensemble(
                dispersion=dispersion,
                alpha=alpha,
                length=length,
                spacing_over_baud=(xpm_frequency - target_frequency) / baud_rate,
                min_samples=sector_min_samples,
                max_samples=sector_max_samples,
                batch_size=sector_batch_size,
                seed=seed + 10_000_019,
                sigma_threshold=sector_sigma_threshold,
                max_relative_error=sector_max_relative_error,
                max_stderr_over_n1=sector_max_stderr_over_n1,
                system=system,
            )
            sector_fields = {
                "xpm_sector_n1": float(sectors["n1"]) / length**2,
                "xpm_sector_n1_stderr": float(sectors["n1_stderr"]) / length**2,
                "xpm_sector_samples": float(sectors["n_samples"]),
                "xpm_sector_stop_code": float(
                    1 if sectors["stop_reason"] == "target_precision" else 2
                ),
            }
            for short_name, result_name in (
                ("2pc", "n_2pc"),
                ("3pca", "n_3pca"),
                ("3pcb", "n_3pcb"),
                ("4pc", "n_4pc"),
            ):
                sector_fields[f"xpm_{short_name}"] = (
                    float(sectors[result_name]) / length**2
                )
                sector_fields[f"xpm_{short_name}_stderr"] = (
                    float(sectors[f"{result_name}_stderr"]) / length**2
                )
                sector_fields[f"xpm_{short_name}_resolved"] = float(
                    bool(sectors[f"{result_name}_resolved"])
                )
        rows.append(
            {
                "sequence": "spectrum",
                "band": "SMF-28 profile",
                "coordinate": repeated_frequency,
                "d": sweep_index,
                "a": sweep_index + fwm_shift,
                "b": sweep_index + fwm_shift,
                "c": sweep_index + 2 * fwm_shift,
                "target_frequency": target_frequency,
                "tuple_center_frequency": repeated_frequency,
                "delta_beta_center": delta_beta,
                "gradient_norm": gradient_norm,
                "x": x_grad,
                "mu": mu,
                "beta2_center": fwm_beta2[1],
                "beta3_center": fwm_beta3[1],
                "beta4_center": fwm_beta4[1],
                "noise_coefficient": fwm_estimate.total / length**2,
                "noise_stderr": fwm_estimate.total_stderr / length**2,
                "noise_local_quadratic": np.nan,
                "noise_local_cubic": np.nan,
                "noise_local_quartic": np.nan,
                "noise_profile_quartic": fwm_estimate.total / length**2,
                "xpm_noise": xpm_value / length**2,
                "xpm_stderr": xpm_stderr / length**2,
                "fwm_degenerate_noise": fwm_estimate.total / length**2,
                "fwm_degenerate_stderr": fwm_estimate.total_stderr / length**2,
                **sector_fields,
                "support_fraction": float(fwm_estimate.metadata["support_fraction"]),
            }
        )

    fields = (
        "sequence",
        "band",
        "coordinate",
        "d",
        "a",
        "b",
        "c",
        "target_frequency",
        "tuple_center_frequency",
        "delta_beta_center",
        "gradient_norm",
        "x",
        "mu",
        "beta2_center",
        "beta3_center",
        "beta4_center",
        "noise_coefficient",
        "noise_stderr",
        "noise_local_quadratic",
        "noise_local_cubic",
        "noise_local_quartic",
        "noise_profile_quartic",
        "xpm_noise",
        "xpm_stderr",
        "fwm_degenerate_noise",
        "fwm_degenerate_stderr",
        "xpm_sector_n1",
        "xpm_sector_n1_stderr",
        "xpm_sector_samples",
        "xpm_sector_stop_code",
        "xpm_2pc",
        "xpm_2pc_stderr",
        "xpm_2pc_resolved",
        "xpm_3pca",
        "xpm_3pca_stderr",
        "xpm_3pca_resolved",
        "xpm_3pcb",
        "xpm_3pcb_stderr",
        "xpm_3pcb_resolved",
        "xpm_4pc",
        "xpm_4pc_stderr",
        "xpm_4pc_resolved",
        "support_fraction",
    )
    data = {field: np.asarray([row[field] for row in rows]) for field in fields}
    data.update(
        {
            "baud_rate": np.array(baud_rate),
            "length": np.array(length),
            "channel_spacing": np.array(float(system.wdm.spacing)),
            "n_samples": np.array(int(n_samples)),
            "seed": np.array(int(seed)),
            "alpha": np.array(float(alpha)),
            "zdw_frequency": np.array(zdw),
            "zdw_beta2": np.array(float(zdw_coefficients[2][0])),
            "zdw_beta3": np.array(float(zdw_coefficients[3][0])),
            "zdw_beta4": np.array(float(zdw_coefficients[4][0])),
            "calculation": np.array("physical_smf28_fwm_tuple_sequences"),
            "noise_convention": np.array(
                "prefactor_free_N_times_T_squared_over_L_squared"
            ),
            "value_normalization": np.array(
                "all plotted MC coefficients are divided by L_squared"
            ),
            "physical_prefactors_included": np.array(False),
            "xpm_shift_channels": np.array(int(xpm_shift)),
            "fwm_shift_channels": np.array(int(fwm_shift)),
            "spectrum_step": np.array(int(spectrum_step)),
            "sector_min_samples": np.array(int(sector_min_samples)),
            "sector_max_samples": np.array(int(sector_max_samples)),
            "sector_batch_size": np.array(int(sector_batch_size)),
            "sector_step": np.array(int(sector_step)),
            "sector_max_relative_error": np.array(float(sector_max_relative_error)),
            "sector_max_stderr_over_n1": np.array(float(sector_max_stderr_over_n1)),
            "sector_sigma_threshold": np.array(float(sector_sigma_threshold)),
            "sector_estimator": np.array(
                "adaptive_covariance_aware_aggregate_residuals"
            ),
            "sector_dispersion_model": np.array("channel_local_taylor_beta4"),
            "sector_common_random_numbers_across_spectrum": np.array(True),
        }
    )
    return data


def save_dataset(data: dict[str, np.ndarray], results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    npz_path = results_dir / "fwm_mc_real_tuple_sequences.npz"
    csv_path = results_dir / "fwm_mc_real_tuple_sequences.csv"
    np.savez(npz_path, **data)
    row_fields = [
        key
        for key, value in data.items()
        if np.asarray(value).ndim == 1
        and np.asarray(value).size == np.asarray(data["sequence"]).size
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(row_fields)
        writer.writerows(zip(*(data[field] for field in row_fields), strict=True))
    return npz_path, csv_path


def load_ssfm_xpm_cache(
    path: Path,
    *,
    expected_length: float,
    expected_baud_rate: float,
    expected_xpm_shift: int,
) -> dict[str, np.ndarray]:
    """Load an SSFM XPM spectrum cache after checking plot compatibility."""
    with np.load(path, allow_pickle=False) as cache:
        data = {key: np.asarray(cache[key]) for key in cache.files}
    required = {
        "target_frequency_hz",
        "n1_ssfm_m2",
        "n1_ssfm_stderr_m2",
        "fiber_length_m",
        "baud_rate_hz",
        "xpm_shift_channels",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"SSFM XPM cache is missing fields: {sorted(missing)}")
    if int(data["xpm_shift_channels"]) != int(expected_xpm_shift):
        raise ValueError(
            "SSFM XPM cache channel shift does not match the plotted XPM line"
        )
    if not np.isclose(float(data["fiber_length_m"]), float(expected_length)):
        raise ValueError(
            "SSFM XPM cache fiber length does not match the plotted XPM line"
        )
    if not np.isclose(float(data["baud_rate_hz"]), float(expected_baud_rate)):
        raise ValueError("SSFM XPM cache baud rate does not match the plotted XPM line")
    return data


def load_ssfm_fwm_cache(
    path: Path,
    *,
    expected_length: float,
    expected_baud_rate: float,
    expected_fwm_shift: int,
) -> dict[str, np.ndarray]:
    """Load an SSFM FWM spectrum cache after checking plot compatibility."""
    with np.load(path, allow_pickle=False) as cache:
        data = {key: np.asarray(cache[key]) for key in cache.files}
    required = {
        "target_frequency_hz",
        "c_ssfm_m2",
        "c_ssfm_stderr_m2",
        "fiber_length_m",
        "baud_rate_hz",
        "fwm_shift_channels",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"SSFM FWM cache is missing fields: {sorted(missing)}")
    if int(data["fwm_shift_channels"]) != int(expected_fwm_shift):
        raise ValueError(
            "SSFM FWM cache channel shift does not match the plotted FWM line"
        )
    if not np.isclose(float(data["fiber_length_m"]), float(expected_length)):
        raise ValueError(
            "SSFM FWM cache fiber length does not match the plotted FWM line"
        )
    if not np.isclose(float(data["baud_rate_hz"]), float(expected_baud_rate)):
        raise ValueError("SSFM FWM cache baud rate does not match the plotted FWM line")
    return data


def plot_dataset(
    data: dict[str, np.ndarray],
    out_dir: Path,
    ssfm_xpm: dict[str, np.ndarray] | None = None,
    ssfm_fwm: dict[str, np.ndarray] | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sequence in ("translation", "span"):
        selected = np.asarray(data["sequence"]) == sequence
        if not np.any(selected):
            continue
        fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)
        for band in np.unique(np.asarray(data["band"])[selected]):
            mask = selected & (np.asarray(data["band"]) == band)
            order = np.argsort(np.asarray(data["coordinate"], dtype=int)[mask])
            coordinate = np.asarray(data["coordinate"], dtype=float)[mask][order]
            noise = np.asarray(data["noise_coefficient"], dtype=float)[mask][order]
            noise_error = np.asarray(data["noise_stderr"], dtype=float)[mask][order]
            axes[0].errorbar(
                coordinate,
                noise,
                yerr=noise_error,
                marker=".",
                ms=3,
                lw=0.8,
                capsize=1,
                label=str(band),
            )
            axes[1].plot(
                coordinate,
                np.asarray(data["x"], dtype=float)[mask][order],
                marker=".",
                ms=3,
            )
            axes[2].plot(
                coordinate,
                np.asarray(data["mu"], dtype=float)[mask][order],
                marker=".",
                ms=3,
            )

        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[0].set_ylabel(r"$N_{dabc}T^2/L^2$")
        axes[1].set_ylabel(r"$x=L B\|\nabla\Delta\beta\|_2$")
        axes[2].set_ylabel(r"$\mu$")
        axes[2].set_xlabel(
            "target channel index"
            if sequence == "translation"
            else "tuple span [channels]"
        )
        axes[0].legend(ncol=3, fontsize=8, frameon=False)
        for axis in axes:
            axis.grid(True, which="both", alpha=0.25)
        fig.suptitle(f"Physical SMF-28 FWM tuple {sequence} sequences")
        fig.tight_layout()
        path = out_dir / f"real_tuple_{sequence}_sequences.pdf"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    selected = np.asarray(data["sequence"]) == "spectrum"
    if np.any(selected):
        xpm_shift = int(data["xpm_shift_channels"])
        fwm_shift = int(data["fwm_shift_channels"])
        order = np.argsort(np.asarray(data["coordinate"], dtype=float)[selected])
        frequency = (
            np.asarray(data["target_frequency"], dtype=float)[selected][order] * 1e-12
        )
        xpm = np.asarray(data["xpm_noise"], dtype=float)[selected][order]
        xpm_error = np.asarray(data["xpm_stderr"], dtype=float)[selected][order]
        fwm = np.asarray(data["fwm_degenerate_noise"], dtype=float)[selected][order]
        fwm_error = np.asarray(data["fwm_degenerate_stderr"], dtype=float)[selected][
            order
        ]
        ratio = np.divide(fwm, xpm, out=np.full_like(fwm, np.nan), where=xpm > 0.0)
        fig, axes = plt.subplots(4, 1, figsize=(8.4, 9.0), sharex=True)

        def draw_coefficient_panel(axis):
            axis.errorbar(
                frequency,
                xpm,
                yerr=xpm_error,
                lw=0.8,
                label=rf"XPM $(d,d,d{xpm_shift:+d},d{xpm_shift:+d})$",
            )
            if ssfm_xpm is not None:
                ssfm_frequency = (
                    np.asarray(ssfm_xpm["target_frequency_hz"], dtype=float) * 1e-12
                )
                normalization = float(data["length"]) ** 2
                ssfm_value = (
                    np.asarray(ssfm_xpm["n1_ssfm_m2"], dtype=float) / normalization
                )
                ssfm_error = (
                    np.asarray(ssfm_xpm["n1_ssfm_stderr_m2"], dtype=float)
                    / normalization
                )
                valid = (
                    np.isfinite(ssfm_frequency)
                    & np.isfinite(ssfm_value)
                    & np.isfinite(ssfm_error)
                    & (ssfm_value > 0.0)
                    & (ssfm_error >= 0.0)
                )
                axis.errorbar(
                    ssfm_frequency[valid],
                    ssfm_value[valid],
                    yerr=ssfm_error[valid],
                    color="0.1",
                    marker="o",
                    mfc="white",
                    ms=4.0,
                    lw=0.8,
                    ls="none",
                    capsize=1.5,
                    label="scalar SSFM $N_1$",
                    zorder=5,
                )
            axis.errorbar(
                frequency,
                fwm,
                yerr=fwm_error,
                lw=0.8,
                label=rf"FWM $(d,d{fwm_shift:+d},d{fwm_shift:+d},d{2 * fwm_shift:+d})$",
            )
            if ssfm_fwm is not None:
                ssfm_frequency = (
                    np.asarray(ssfm_fwm["target_frequency_hz"], dtype=float) * 1e-12
                )
                normalization = float(data["length"]) ** 2
                ssfm_value = (
                    np.asarray(ssfm_fwm["c_ssfm_m2"], dtype=float) / normalization
                )
                ssfm_error = (
                    np.asarray(ssfm_fwm["c_ssfm_stderr_m2"], dtype=float)
                    / normalization
                )
                valid = (
                    np.isfinite(ssfm_frequency)
                    & np.isfinite(ssfm_value)
                    & np.isfinite(ssfm_error)
                    & (ssfm_value > 0.0)
                    & (ssfm_error >= 0.0)
                )
                axis.errorbar(
                    ssfm_frequency[valid],
                    ssfm_value[valid],
                    yerr=ssfm_error[valid],
                    color="0.1",
                    marker="D",
                    mfc="white",
                    ms=4.0,
                    lw=0.8,
                    ls="none",
                    capsize=1.5,
                    label="scalar SSFM FWM",
                    zorder=5,
                )
                if "c_total_m2" in ssfm_fwm:
                    total_value = (
                        np.asarray(ssfm_fwm["c_total_m2"], dtype=float) / normalization
                    )
                    total_error = (
                        np.asarray(ssfm_fwm["c_total_stderr_m2"], dtype=float)
                        / normalization
                    )
                    total_valid = (
                        np.isfinite(ssfm_frequency)
                        & np.isfinite(total_value)
                        & (total_value > 0.0)
                    )
                    total_order = np.argsort(ssfm_frequency[total_valid])
                    axis.errorbar(
                        ssfm_frequency[total_valid][total_order],
                        total_value[total_valid][total_order],
                        yerr=total_error[total_valid][total_order],
                        color="0.45",
                        marker="v",
                        ms=3.2,
                        lw=0.8,
                        ls=":",
                        capsize=1.5,
                        label=r"scalar SSFM total NLI ($2\gamma^2P^3$ norm.)",
                        zorder=4,
                    )
                if "c_interferer_m2" in ssfm_fwm:
                    interferer_value = (
                        np.asarray(ssfm_fwm["c_interferer_m2"], dtype=float)
                        / normalization
                    )
                    interferer_error = (
                        np.asarray(ssfm_fwm["c_interferer_stderr_m2"], dtype=float)
                        / normalization
                    )
                    interferer_valid = (
                        np.isfinite(ssfm_frequency)
                        & np.isfinite(interferer_value)
                        & (interferer_value > 0.0)
                    )
                    interferer_order = np.argsort(ssfm_frequency[interferer_valid])
                    axis.errorbar(
                        ssfm_frequency[interferer_valid][interferer_order],
                        interferer_value[interferer_valid][interferer_order],
                        yerr=interferer_error[interferer_valid][interferer_order],
                        color="0.25",
                        marker="^",
                        ms=3.2,
                        lw=0.8,
                        ls="--",
                        capsize=1.5,
                        label=r"scalar SSFM interferer NLI ($2\gamma^2P^3$ norm.)",
                        zorder=4,
                    )
            sector_n1 = np.asarray(data["xpm_sector_n1"], dtype=float)[selected][order]
            sector_n1_error = np.asarray(data["xpm_sector_n1_stderr"], dtype=float)[
                selected
            ][order]
            sector_evaluated = np.isfinite(sector_n1)
            axis.errorbar(
                frequency[sector_evaluated],
                sector_n1[sector_evaluated],
                yerr=float(data["sector_sigma_threshold"])
                * sector_n1_error[sector_evaluated],
                color="C0",
                marker=".",
                ms=2.5,
                lw=0.6,
                ls="--",
                label=r"XPM local-$\beta_{2\ldots4}$ adaptive MC",
            )

            blue_shades = plt.get_cmap("Blues")([0.88, 0.72, 0.57, 0.42])
            for name, label, color, marker in zip(
                ("2pc", "3pca", "3pcb", "4pc"),
                ("2PC", "3PCa", "3PCb", "4PC"),
                blue_shades,
                ("o", "s", "^", "D"),
                strict=True,
            ):
                mean = np.asarray(data[f"xpm_{name}"], dtype=float)[selected][order]
                stderr = np.asarray(data[f"xpm_{name}_stderr"], dtype=float)[selected][
                    order
                ]
                resolved = (
                    np.asarray(data[f"xpm_{name}_resolved"], dtype=float)[selected][
                        order
                    ]
                    == 1.0
                )
                axis.errorbar(
                    frequency[resolved],
                    mean[resolved],
                    yerr=float(data["sector_sigma_threshold"]) * stderr[resolved],
                    color=color,
                    marker=marker,
                    ms=2.8,
                    lw=0.7,
                    ls="none",
                    capsize=1,
                    label=label,
                )
                maximum_budget = (
                    np.asarray(data["xpm_sector_stop_code"], dtype=float)[selected][
                        order
                    ]
                    == 2.0
                )
                unresolved = (
                    sector_evaluated & maximum_budget & ~resolved & (mean > 0.0)
                )
                axis.scatter(
                    frequency[unresolved],
                    mean[unresolved],
                    color=color,
                    marker="x",
                    s=12,
                    linewidths=0.7,
                    label="unresolved at max budget" if name == "2pc" else None,
                )

        draw_coefficient_panel(axes[0])
        axes[1].plot(frequency, ratio)
        fwm_x = np.asarray(data["x"], dtype=float)[selected][order]
        fwm_mu = np.asarray(data["mu"], dtype=float)[selected][order]
        x_line = axes[2].plot(frequency, fwm_x, color="C1", label=r"$x$")[0]
        mu_axis = axes[2].twinx()
        mu_line = mu_axis.plot(frequency, fwm_mu, color="C3", label=r"$\mu$")[0]
        mu_axis.axhline(0.0, color="C3", lw=0.6, ls=":", alpha=0.7)
        axes[3].plot(
            frequency,
            np.asarray(data["beta2_center"], dtype=float)[selected][order] * 1e27,
        )
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
        axes[0].set_ylabel("normalized MC coefficient")
        axes[1].set_ylabel("FWM / XPM")
        axes[2].set_ylabel(r"FWM $x$")
        mu_axis.set_ylabel(r"FWM $\mu$", color="C3")
        mu_axis.tick_params(axis="y", colors="C3")
        mu_axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[2].legend(
            [x_line, mu_line], [r"$x$", r"$\mu$"], loc="best", fontsize=7, frameon=False
        )
        axes[3].set_ylabel(r"$\beta_2$ [$10^{-27}$ s$^2$/m]")
        axes[3].set_xlabel("target channel $d$ frequency [THz]")
        axes[0].legend(ncol=2, fontsize=6.4, frameon=False)
        draw_channel_choice_inset(
            axes[3],
            xpm_shift=xpm_shift,
            fwm_shift=fwm_shift,
            spacing_over_baud=float(data["channel_spacing"]) / float(data["baud_rate"]),
            channel_spacing=float(data["channel_spacing"]),
            length=float(data["length"]),
            zdw_beta2=float(data["zdw_beta2"]),
            zdw_beta3=float(data["zdw_beta3"]),
            zdw_beta4=float(data["zdw_beta4"]),
            bounds=(0.02, 0.08, 0.34, 0.82),
        )
        zdw_thz = float(data["zdw_frequency"]) * 1e-12
        for axis in axes:
            axis.axvline(zdw_thz, color="0.3", lw=0.8, ls="-.")
            axis.grid(True, which="both", alpha=0.25)
        fig.suptitle(r"SMF-28 shifted-channel MC coefficients normalized by $L^2$")
        fig.tight_layout()
        path = out_dir / "real_tuple_full_spectrum_xpm_fwm.pdf"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

        fig_top, axis_top = plt.subplots(figsize=(8.4, 4.6))
        draw_coefficient_panel(axis_top)
        axis_top.set_yscale("log")
        axis_top.set_ylabel("normalized MC coefficient")
        axis_top.set_xlabel("target channel $d$ frequency [THz]")
        axis_top.axvline(zdw_thz, color="0.3", lw=0.8, ls="-.")
        axis_top.grid(True, which="both", alpha=0.25)
        axis_top.legend(ncol=2, fontsize=6.4, frameon=False)
        fig_top.tight_layout()
        top_path = out_dir / "real_tuple_full_spectrum_coefficients.pdf"
        fig_top.savefig(top_path, dpi=300)
        plt.close(fig_top)
        paths.append(top_path)

    selected = np.asarray(data["sequence"]) == "zdw"
    if np.any(selected):
        order = np.argsort(np.asarray(data["coordinate"], dtype=float)[selected])
        frequency = np.asarray(data["coordinate"], dtype=float)[selected][order] * 1e-12
        zdw = float(data["zdw_frequency"]) * 1e-12
        fig, axes = plt.subplots(4, 1, figsize=(8.2, 9.0), sharex=True)
        for field, label, style in (
            ("noise_local_quadratic", r"global order 2", ":"),
            ("noise_local_cubic", r"global order 3", "--"),
            ("noise_local_quartic", r"global order 4", "-"),
            ("noise_profile_quartic", r"profile, local order 4", "-."),
        ):
            axes[0].plot(
                frequency,
                np.asarray(data[field], dtype=float)[selected][order],
                ls=style,
                label=label,
            )
        axes[1].plot(frequency, np.asarray(data["x"], dtype=float)[selected][order])
        axes[2].plot(frequency, np.asarray(data["mu"], dtype=float)[selected][order])
        axes[3].plot(
            frequency,
            np.asarray(data["beta2_center"], dtype=float)[selected][order] * 1e27,
        )
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[0].set_ylabel(r"$N_{dabc}T^2/L^2$")
        axes[1].set_ylabel(r"$x$")
        axes[2].set_ylabel(r"$\mu$")
        axes[3].set_ylabel(r"$\beta_2$ [$10^{-27}$ s$^2$/m]")
        axes[3].set_xlabel("repeated-pump channel frequency [THz]")
        axes[0].legend(fontsize=8, frameon=False)
        for axis in axes:
            axis.axvline(zdw, color="0.3", lw=0.8, ls="-.")
            axis.grid(True, which="both", alpha=0.25)
        fwm_shift = int(data["fwm_shift_channels"])
        fig.suptitle(
            rf"SMF-28 ZDW scan: $(d,d{fwm_shift:+d},d{fwm_shift:+d},d{2 * fwm_shift:+d})$"
        )
        fig.tight_layout()
        path = out_dir / "real_tuple_zdw_degenerate_sequence.pdf"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("input/studies.toml"))
    parser.add_argument("--out-dir", type=Path, default=OUT_MEDIA)
    parser.add_argument("--results-dir", type=Path, default=OUT_RESULTS)
    parser.add_argument(
        "--n-samples", type=int, default=None, help="Defaults to [fwm].mc_samples."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Defaults to [fwm].mc_seed."
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Defaults to [fwm].mc_alpha."
    )
    parser.add_argument("--translation-step", type=int, default=10)
    parser.add_argument("--max-span", type=int, default=30)
    parser.add_argument(
        "--zdw-half-window",
        type=int,
        default=80,
        help="25 GHz steps on each side of the ZDW.",
    )
    parser.add_argument(
        "--zdw-step",
        type=int,
        default=2,
        help="Decimation of the ZDW-centered virtual grid.",
    )
    parser.add_argument(
        "--spectrum-step", type=int, default=None, help="Override TOML grid decimation."
    )
    parser.add_argument(
        "--xpm-shift", type=int, default=None, help="Override TOML XPM channel shift."
    )
    parser.add_argument(
        "--fwm-shift",
        type=int,
        default=None,
        help="Override TOML repeated-pump channel shift.",
    )
    parser.add_argument("--sector-min-samples", type=int, default=None)
    parser.add_argument("--sector-max-samples", type=int, default=None)
    parser.add_argument("--sector-batch-size", type=int, default=None)
    parser.add_argument(
        "--sector-step", type=int, default=None, help="Override sector grid decimation."
    )
    parser.add_argument(
        "--ssfm-xpm-cache",
        type=Path,
        default=None,
        help="Optional cached SSFM XPM spectrum NPZ to overlay.",
    )
    parser.add_argument(
        "--ssfm-fwm-cache",
        type=Path,
        default=None,
        help="Optional cached SSFM FWM spectrum NPZ to overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    system = System.from_toml(args.config)
    raw_fwm = (system.raw_config or {}).get("fwm", {})
    spectrum_config = raw_fwm.get("real_tuple_spectrum", {})
    n_samples = int(
        raw_fwm.get("mc_samples", 10_000) if args.n_samples is None else args.n_samples
    )
    seed = int(raw_fwm.get("mc_seed", 1234) if args.seed is None else args.seed)
    alpha = float(raw_fwm.get("mc_alpha", 0.0) if args.alpha is None else args.alpha)
    spectrum_step = int(
        spectrum_config.get("spectrum_step", 2)
        if args.spectrum_step is None
        else args.spectrum_step
    )
    xpm_shift = int(
        spectrum_config.get("xpm_shift_channels", 3)
        if args.xpm_shift is None
        else args.xpm_shift
    )
    fwm_shift = int(
        spectrum_config.get("fwm_shift_channels", 3)
        if args.fwm_shift is None
        else args.fwm_shift
    )
    sector_min_samples = int(
        spectrum_config.get("sector_min_samples", 200_000)
        if args.sector_min_samples is None
        else args.sector_min_samples
    )
    sector_max_samples = int(
        spectrum_config.get("sector_max_samples", 2_400_000)
        if args.sector_max_samples is None
        else args.sector_max_samples
    )
    sector_batch_size = int(
        spectrum_config.get("sector_batch_size", 25_000)
        if args.sector_batch_size is None
        else args.sector_batch_size
    )
    sector_step = int(
        spectrum_config.get("sector_step", 10)
        if args.sector_step is None
        else args.sector_step
    )
    sector_max_relative_error = float(
        spectrum_config.get("sector_max_relative_error", 0.25)
    )
    sector_max_stderr_over_n1 = float(
        spectrum_config.get("sector_max_stderr_over_n1", 0.0025)
    )
    sector_sigma_threshold = float(spectrum_config.get("sector_sigma_threshold", 3.0))
    if (
        min(
            n_samples,
            args.translation_step,
            args.max_span,
            args.zdw_half_window,
            args.zdw_step,
            spectrum_step,
            sector_min_samples,
            sector_max_samples,
            sector_batch_size,
            sector_step,
        )
        < 1
    ):
        raise ValueError("sample counts, sequence steps, and windows must be positive")
    if sector_min_samples > sector_max_samples:
        raise ValueError("sector-min-samples cannot exceed sector-max-samples")
    if sector_max_samples % sector_batch_size or sector_min_samples % sector_batch_size:
        raise ValueError("sector sample limits must be multiples of sector-batch-size")
    if (
        sector_max_relative_error <= 0.0
        or sector_max_stderr_over_n1 <= 0.0
        or sector_sigma_threshold <= 0.0
    ):
        raise ValueError("sector uncertainty thresholds must be positive")
    if xpm_shift == 0 or fwm_shift == 0:
        raise ValueError("xpm and fwm channel shifts must be nonzero")

    cache_setting = (
        spectrum_config.get("ssfm_xpm_cache", "")
        if args.ssfm_xpm_cache is None
        else args.ssfm_xpm_cache
    )
    ssfm_xpm = None
    if cache_setting:
        cache_path = Path(cache_setting)
        if cache_path.exists():
            ssfm_xpm = load_ssfm_xpm_cache(
                cache_path,
                expected_length=float(system.fiber_length),
                expected_baud_rate=float(system.pulse.baud_rate),
                expected_xpm_shift=xpm_shift,
            )
        else:
            print(f"SSFM XPM cache not found; plotting without overlay: {cache_path}")

    fwm_cache_setting = (
        spectrum_config.get("ssfm_fwm_cache", "")
        if args.ssfm_fwm_cache is None
        else args.ssfm_fwm_cache
    )
    ssfm_fwm = None
    if fwm_cache_setting:
        fwm_cache_path = Path(fwm_cache_setting)
        if fwm_cache_path.exists():
            ssfm_fwm = load_ssfm_fwm_cache(
                fwm_cache_path,
                expected_length=float(system.fiber_length),
                expected_baud_rate=float(system.pulse.baud_rate),
                expected_fwm_shift=fwm_shift,
            )
        else:
            print(f"SSFM FWM cache not found; plotting without overlay: {fwm_cache_path}")

    data = compute_dataset(
        system,
        n_samples=n_samples,
        seed=seed,
        alpha=alpha,
        translation_step=args.translation_step,
        max_span=args.max_span,
        zdw_half_window=args.zdw_half_window,
        zdw_step=args.zdw_step,
        spectrum_step=spectrum_step,
        xpm_shift=xpm_shift,
        fwm_shift=fwm_shift,
        sector_min_samples=sector_min_samples,
        sector_max_samples=sector_max_samples,
        sector_batch_size=sector_batch_size,
        sector_step=sector_step,
        sector_max_relative_error=sector_max_relative_error,
        sector_max_stderr_over_n1=sector_max_stderr_over_n1,
        sector_sigma_threshold=sector_sigma_threshold,
    )
    npz_path, csv_path = save_dataset(data, args.results_dir)
    paths = plot_dataset(data, args.out_dir, ssfm_xpm=ssfm_xpm, ssfm_fwm=ssfm_fwm)
    print(f"loaded {args.config}: {len(system.wdm.frequency_grid())} channels")
    print(
        f"fixed B={float(data['baud_rate']) / 1e9:.4g} GBd, "
        f"spacing={float(data['channel_spacing']) / 1e9:.4g} GHz, "
        f"L={float(data['length']) / 1e3:.4g} km"
    )
    print(f"evaluated {np.asarray(data['sequence']).size} physical tuples")
    spectrum_mask = np.asarray(data["sequence"]) == "spectrum"
    evaluated_sectors = (
        np.isfinite(np.asarray(data["xpm_2pc"], dtype=float)) & spectrum_mask
    )
    if np.any(evaluated_sectors):
        summary = []
        for name in ("2pc", "3pca", "3pcb", "4pc"):
            resolved = np.asarray(data[f"xpm_{name}_resolved"], dtype=float) == 1.0
            summary.append(
                f"{name}={np.count_nonzero(resolved & evaluated_sectors)}/{np.count_nonzero(evaluated_sectors)}"
            )
        print("resolved XPM sectors: " + ", ".join(summary))
    print(f"SMF-28 profile ZDW={float(data['zdw_frequency']) / 1e12:.6f} THz")
    print(f"saved {npz_path}")
    print(f"saved {csv_path}")
    for path in paths:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
