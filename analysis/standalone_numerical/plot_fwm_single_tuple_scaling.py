from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from pynlin.methods.td.fwm_kernel import FWMChannels
from pynlin.methods.td.fwm_mc import estimate_fwm_term_sum_dar_mc


OUT_MEDIA = Path("media/fwm/single-tuple-scaling")
OUT_RESULTS = Path("results/fwm/single-tuple-scaling")
# Positive phase solving |A|^2 = sinc^2(phase / 2) = exp(-1).
E_MINUS_ONE_PHASE = 3.288545458955736


def tuple_island_fields(
    channel_positions: np.ndarray,
    spacing_over_baud: float,
    resolution: int = 300,
    window_periods: float = 10.0,
) -> dict[str, np.ndarray | float | tuple[float, float]]:
    """Build the round-region field and occupied-band mask for one FWM tuple."""
    positions = np.asarray(channel_positions, dtype=float).reshape(-1)
    if positions.size != 4 or not np.allclose(positions, np.rint(positions), atol=1e-12):
        raise ValueError("island diagrams require four integer channel positions (d,a,b,c)")
    d, a, b, c = np.rint(positions).astype(int)
    residual = int(a + b - c - d)
    if spacing_over_baud <= 0.0:
        raise ValueError("spacing_over_baud must be positive")
    normalized_carrier_residual = float(residual) * float(spacing_over_baud)
    if abs(normalized_carrier_residual) >= 2.0:
        raise ValueError(
            "tuple bands have no positive-measure frequency-conserving overlap: "
            f"|(a+b-c-d)*Delta_f/B|={abs(normalized_carrier_residual):.6g} >= 2"
        )

    relative_a = a - d
    relative_b = b - d
    relative_c = c - d
    if window_periods <= 0.0:
        raise ValueError("window_periods must be positive")
    half_window = 0.5 * float(window_periods)
    xlim = (relative_a - half_window, relative_a + half_window)
    ylim = (relative_b - half_window, relative_b + half_window)
    x = np.linspace(*xlim, int(resolution))
    y = np.linspace(*ylim, int(resolution))
    X, Y = np.meshgrid(x, y)
    round_x = np.rint(X).astype(int)
    round_y = np.rint(Y).astype(int)
    round_sum = np.rint(X + Y).astype(int)

    # Frequencies are normalized by channel spacing. Blank the guard intervals
    # between occupied bands, as in the existing round-regions figures.
    occupied_width = 1.0 / float(spacing_over_baud)
    blank_half_width = max(0.0, 0.5 * (1.0 - occupied_width))
    distance_x = np.abs(np.mod(X, 1.0) - 0.5)
    distance_y = np.abs(np.mod(Y, 1.0) - 0.5)
    distance_sum = np.abs(np.mod(X + Y, 1.0) - 0.5)
    occupied = (
        (distance_x >= blank_half_width)
        & (distance_y >= blank_half_width)
        & (distance_sum >= blank_half_width)
    )
    half_bandwidth = 0.5 * occupied_width
    # Fixed-d section of the three input passbands. Intersecting the a/b square
    # with the finite c-band diagonal strip produces a lozenge for residual 0
    # and a small triangle for residual +/-1.
    selected = (
        (np.abs(X - relative_a) < half_bandwidth)
        & (np.abs(Y - relative_b) < half_bandwidth)
        & (np.abs(X + Y - relative_c) < half_bandwidth)
    )
    return {
        "X": X,
        "Y": Y,
        "round_sum": round_sum,
        "occupied": occupied,
        "selected": selected,
        "xlim": xlim,
        "ylim": ylim,
        "relative_a": float(relative_a),
        "relative_b": float(relative_b),
        "relative_c": float(relative_c),
        "energy_residual": float(residual),
        "normalized_carrier_residual": normalized_carrier_residual,
        "support_margin_over_baud": 2.0 - abs(normalized_carrier_residual),
        "half_bandwidth_over_spacing": half_bandwidth,
        "blank_half_width": blank_half_width,
    }


def _clip_polygon_half_plane(
    vertices: list[tuple[float, float]],
    normal: tuple[float, float],
    bound: float,
) -> list[tuple[float, float]]:
    """Clip a convex polygon to ``normal dot point <= bound``."""
    if not vertices:
        return []
    nx, ny = normal
    clipped: list[tuple[float, float]] = []
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
    center_x: int,
    center_y: int,
    center_sum: int,
    half_bandwidth: float,
) -> list[tuple[float, float]]:
    """Return the exact a/b/c passband-intersection polygon."""
    h = float(half_bandwidth)
    vertices = [
        (center_x - h, center_y - h),
        (center_x + h, center_y - h),
        (center_x + h, center_y + h),
        (center_x - h, center_y + h),
    ]
    vertices = _clip_polygon_half_plane(
        vertices, (1.0, 1.0), center_sum + h
    )
    return _clip_polygon_half_plane(
        vertices, (-1.0, -1.0), -(center_sum - h)
    )


def draw_tuple_island_inset(
    parent_ax: plt.Axes,
    channel_positions: np.ndarray,
    spacing_over_baud: float,
    *,
    mu_values: np.ndarray | None = None,
    x_values: np.ndarray | None = None,
    beta2_sign: float = -1.0,
    mu_colors: dict[float, object] | None = None,
    dispersion_model: str = "constant-beta2",
    extra_zero_sum: float = 5.0,
) -> plt.Axes:
    """Draw a compact round-region map highlighting the plotted tuple."""
    # The guard gaps are only about 2% of a period for the default spacing.
    # Resolve them explicitly so adjacent polygon outlines do not merge.
    fields = tuple_island_fields(
        channel_positions,
        spacing_over_baud,
        resolution=1200,
    )
    X = np.asarray(fields["X"])
    Y = np.asarray(fields["Y"])
    xlim = fields["xlim"]
    ylim = fields["ylim"]

    ax = parent_ax.inset_axes([0.57, 0.51, 0.40, 0.45])
    ax.set_facecolor("white")
    half_bandwidth = float(fields["half_bandwidth_over_spacing"])
    center_x_min = int(np.ceil(xlim[0] - half_bandwidth))
    center_x_max = int(np.floor(xlim[1] + half_bandwidth))
    center_y_min = int(np.ceil(ylim[0] - half_bandwidth))
    center_y_max = int(np.floor(ylim[1] + half_bandwidth))
    for center_x in range(center_x_min, center_x_max + 1):
        for center_y in range(center_y_min, center_y_max + 1):
            sum_radius = 3.0 * half_bandwidth
            center_sum_min = int(np.ceil(center_x + center_y - sum_radius))
            center_sum_max = int(np.floor(center_x + center_y + sum_radius))
            for center_sum in range(center_sum_min, center_sum_max + 1):
                polygon = _passband_island_polygon(
                    center_x,
                    center_y,
                    center_sum,
                    half_bandwidth,
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

    relative_a = int(fields["relative_a"])
    relative_b = int(fields["relative_b"])
    relative_c = int(fields["relative_c"])
    selected_polygon = _passband_island_polygon(
        relative_a,
        relative_b,
        relative_c,
        half_bandwidth,
    )
    if len(selected_polygon) >= 3:
        ax.add_patch(
            Polygon(
                selected_polygon,
                closed=True,
                facecolor="#f4a3b8",
                edgecolor="#7a1238",
                linewidth=1.3,
                zorder=4,
            )
        )
    ax.axvline(0.0, color="0.1", lw=0.9, zorder=3)
    ax.axhline(0.0, color="0.1", lw=0.9, zorder=3)

    if mu_values is not None and x_values is not None:
        edge_mus = np.unique(
            [float(np.min(mu_values)), float(np.max(mu_values))]
        )
        mismatch_fields = {
            float(mu): _fixed_d_normalized_mismatch(
                fields,
                channel_positions,
                spacing_over_baud,
                float(mu),
                beta2_sign,
                dispersion_model,
                extra_zero_sum,
            )
            for mu in edge_mus
        }
        x_candidates = np.unique(np.asarray(x_values, dtype=float))
        x_candidates = x_candidates[x_candidates > 0.0]
        if x_candidates.size:
            common_lo = float(x_candidates[0])
            common_hi = float(x_candidates[-1])
            common_interval_exists = True
            for values in mismatch_fields.values():
                minimum = float(np.nanmin(values))
                maximum = float(np.nanmax(values))
                minimum_absolute = (
                    0.0
                    if minimum <= 0.0 <= maximum
                    else min(abs(minimum), abs(maximum))
                )
                maximum_absolute = max(abs(minimum), abs(maximum))
                if maximum_absolute <= 0.0:
                    common_interval_exists = False
                    break
                common_lo = max(
                    common_lo, E_MINUS_ONE_PHASE / maximum_absolute
                )
                if minimum_absolute > 0.0:
                    common_hi = min(
                        common_hi, E_MINUS_ONE_PHASE / minimum_absolute
                    )
            if common_interval_exists and common_lo < common_hi:
                overlay_x = float(np.sqrt(common_lo * common_hi))
            else:
                target_log_x = 0.5 * (
                    np.log(x_candidates[0]) + np.log(x_candidates[-1])
                )
                visibility = np.array(
                    [
                        sum(
                            (
                                0.0
                                if np.nanmin(values) <= 0.0 <= np.nanmax(values)
                                else min(
                                    abs(float(np.nanmin(values))),
                                    abs(float(np.nanmax(values))),
                                )
                            )
                            < E_MINUS_ONE_PHASE / x_value
                            < max(
                                abs(float(np.nanmin(values))),
                                abs(float(np.nanmax(values))),
                            )
                            for values in mismatch_fields.values()
                        )
                        for x_value in x_candidates
                    ]
                )
                best = np.flatnonzero(visibility == np.max(visibility))
                chosen = best[
                    np.argmin(np.abs(np.log(x_candidates[best]) - target_log_x))
                ]
                overlay_x = float(x_candidates[chosen])
            drew_contour = False
            for mu in edge_mus:
                values = mismatch_fields[float(mu)]
                level = E_MINUS_ONE_PHASE / overlay_x
                minimum = float(np.nanmin(values))
                maximum = float(np.nanmax(values))
                levels = [
                    signed_level
                    for signed_level in (-level, level)
                    if minimum < signed_level < maximum
                ]
                if not levels:
                    continue
                color = (
                    mu_colors[float(mu)]
                    if mu_colors is not None and float(mu) in mu_colors
                    else "0.15"
                )
                ax.contour(
                    X,
                    Y,
                    values,
                    levels=levels,
                    colors=[color] * len(levels),
                    linestyles=["-"] * len(levels),
                    linewidths=1.0,
                    zorder=5,
                )
                drew_contour = True
            if drew_contour:
                ax.set_title(
                    rf"$\eta=|A|^2=e^{{-1}}$ at $x={overlay_x:.3g}$"
                    "\n"
                    rf"$\mu_{{\min}}={edge_mus[0]:g},\ "
                    rf"\mu_{{\max}}={edge_mus[-1]:g}$",
                    fontsize=5.3,
                    pad=1.5,
                )

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#7a1238",
                lw=1.3,
                label=(
                    rf"$({relative_a},{relative_b},{relative_c})$"
                    "\n"
                    rf"$|r|\Delta f/B="
                    rf"{abs(float(fields['normalized_carrier_residual'])):.2f}<2$"
                ),
            ),
        ],
        loc="lower right",
        fontsize=5.8,
        frameon=True,
        framealpha=0.88,
        handlelength=1.0,
        borderpad=0.25,
    )
    ax.set(xlim=xlim, ylim=ylim, xlabel=r"$(f_a-f_d)/\Delta f$", ylabel=r"$(f_b-f_d)/\Delta f$")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.xaxis.label.set_size(6)
    ax.yaxis.label.set_size(6)
    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.7)
    return ax


def _fixed_d_normalized_mismatch(
    fields: dict[str, np.ndarray | float | tuple[float, float]],
    channel_positions: np.ndarray,
    spacing_over_baud: float,
    mu: float,
    beta2_sign: float,
    dispersion_model: str,
    extra_zero_sum: float,
) -> np.ndarray:
    """Evaluate normalized mismatch over the full inset fixed-d plane."""
    X = np.asarray(fields["X"], dtype=float)
    Y = np.asarray(fields["Y"], dtype=float)
    relative_a = float(fields["relative_a"])
    relative_b = float(fields["relative_b"])
    channels, _ = build_single_tuple_channels(
        channel_positions=channel_positions,
        spacing_over_baud=spacing_over_baud,
        baud_rate=1.0,
        length=1.0,
        x_grad=1.0,
        mu=mu,
        beta2_sign=beta2_sign,
        dispersion_model=dispersion_model,
        extra_zero_sum=extra_zero_sum,
    )

    omega_a = 2.0 * np.pi * float(spacing_over_baud) * (X - relative_a)
    omega_b = 2.0 * np.pi * float(spacing_over_baud) * (Y - relative_b)
    omega_c = omega_a + omega_b + channels.delta_omega

    beta_a = (
        channels.beta0_a
        + channels.beta1_a * omega_a
        + 0.5 * channels.gvd_a * omega_a**2
        + (channels.beta3_a / 6.0) * omega_a**3
    )
    beta_b = (
        channels.beta0_b
        + channels.beta1_b * omega_b
        + 0.5 * channels.gvd_b * omega_b**2
        + (channels.beta3_b / 6.0) * omega_b**3
    )
    mismatch = (
        beta_a
        + beta_b
        - channels.beta0_c
        - channels.beta1_c * omega_c
        - 0.5 * channels.gvd_c * omega_c**2
        - (channels.beta3_c / 6.0) * omega_c**3
        - channels.beta0_d
    )
    return mismatch


def _symmetric_random_variables(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    paired = int(n_samples) // 2
    base = 2.0 * np.pi * (rng.random((3, paired)) - 0.5)
    values = np.concatenate((base, base[[1, 0, 2]]), axis=1)
    if int(n_samples) % 2:
        extra = 2.0 * np.pi * (rng.random((3, 1)) - 0.5)
        extra[1] = extra[0]
        values = np.concatenate((values, extra), axis=1)
    return values


def build_single_tuple_channels(
    *,
    channel_positions: np.ndarray,
    spacing_over_baud: float,
    baud_rate: float,
    length: float,
    x_grad: float,
    mu: float,
    beta2_sign: float = -1.0,
    dispersion_model: str = "constant-beta2",
    extra_zero_sum: float = 5.0,
) -> tuple[FWMChannels, dict[str, object]]:
    """Construct one fixed tuple at prescribed mismatch scale and detuning.

    ``channel_positions`` follows ``(d,a,b,c)`` and is expressed in channel
    indices. Their angular-frequency centers are
    ``2*pi*position*spacing_over_baud*baud_rate``. ``constant-beta2`` uses a
    quadratic global beta model. ``cubic-zdw`` adds beta3 while keeping the
    extra fixed-d phase-matching branch at
    ``(Omega_a+Omega_b)/(2*pi*spacing) = extra_zero_sum``. For a nonzero center
    gradient, the dispersion magnitude is selected so that
    ``L*B*||grad Delta beta|| = x_grad``. Repeated-index tuples such as
    ``(0)111`` have zero center gradient and instead use
    ``L*B**2*|beta2(d)| = x_grad``.
    """
    positions = np.asarray(channel_positions, dtype=float).reshape(-1)
    if positions.size != 4:
        raise ValueError("channel_positions must contain (d,a,b,c)")
    if not np.allclose(positions, np.rint(positions), atol=1e-12):
        raise ValueError("channel positions must be integer channel indices")
    if baud_rate <= 0.0 or length <= 0.0 or spacing_over_baud <= 0.0 or x_grad <= 0.0:
        raise ValueError("baud rate, length, spacing, and x_grad must be positive")
    if dispersion_model not in {"constant-beta2", "cubic-zdw"}:
        raise ValueError("dispersion_model must be 'constant-beta2' or 'cubic-zdw'")
    if dispersion_model == "cubic-zdw" and extra_zero_sum == 0.0:
        raise ValueError("extra_zero_sum must be nonzero for cubic-zdw")

    carrier_index_residual = float(positions[1] + positions[2] - positions[3] - positions[0])
    carrier_frequency_residual = carrier_index_residual * float(spacing_over_baud) * float(baud_rate)
    if abs(carrier_frequency_residual) >= 2.0 * float(baud_rate):
        raise ValueError(
            "tuple bands have no positive-measure frequency-conserving overlap: "
            f"|f_a+f_b-f_c-f_d|={abs(carrier_frequency_residual):.6g} Hz >= 2B"
        )

    omega = 2.0 * np.pi * positions * float(spacing_over_baud) * float(baud_rate)
    omega_relative = omega - omega[0]
    _, omega_a, omega_b, omega_c = omega_relative
    beta3_per_beta2 = (
        -2.0
        / (
            float(extra_zero_sum)
            * 2.0
            * np.pi
            * float(spacing_over_baud)
            * float(baud_rate)
        )
        if dispersion_model == "cubic-zdw"
        else 0.0
    )
    omega_out = omega_a + omega_b - omega_c

    def beta1_per_beta2(value: float) -> float:
        return value + 0.5 * beta3_per_beta2 * value**2

    beta1_a_per = beta1_per_beta2(float(omega_a))
    beta1_b_per = beta1_per_beta2(float(omega_b))
    beta1_c_per = beta1_per_beta2(float(omega_c))
    beta1_out_per = beta1_per_beta2(float(omega_out))
    gradient_per_beta2 = np.array(
        [
            beta1_a_per - beta1_out_per,
            beta1_b_per - beta1_out_per,
            -beta1_c_per + beta1_out_per,
        ]
    )
    norm_per_beta2 = float(np.linalg.norm(gradient_per_beta2))
    if norm_per_beta2 == 0.0:
        scale_kind = "curvature"
        beta2 = float(np.sign(beta2_sign) or -1.0) * float(x_grad) / (
            float(length) * float(baud_rate) ** 2
        )
    else:
        scale_kind = "gradient"
        beta2 = float(np.sign(beta2_sign) or -1.0) * float(x_grad) / (
            float(length) * float(baud_rate) * norm_per_beta2
        )
    beta3 = beta3_per_beta2 * beta2
    beta0 = 0.5 * beta2 * omega_relative**2 + (beta3 / 6.0) * omega_relative**3
    beta1 = beta2 * omega_relative + 0.5 * beta3 * omega_relative**2
    beta2_local = beta2 + beta3 * omega_relative
    gradient = beta2 * gradient_per_beta2
    gradient_norm = float(np.linalg.norm(gradient))
    center_delta_omega = 2.0 * np.pi * carrier_frequency_residual
    beta_d_at_constrained_output = (
        beta0[0]
        + beta1[0] * center_delta_omega
        + 0.5 * beta2_local[0] * center_delta_omega**2
        + (beta3 / 6.0) * center_delta_omega**3
    )
    natural_mismatch = float(beta0[1] + beta0[2] - beta0[3] - beta_d_at_constrained_output)
    mismatch_scale = (
        float(baud_rate) * gradient_norm
        if scale_kind == "gradient"
        else float(baud_rate) ** 2 * abs(beta2)
    )
    desired_mismatch = float(mu) * mismatch_scale
    beta0[1] += desired_mismatch - natural_mismatch

    channels = FWMChannels(
        omega_a=float(omega[1]),
        omega_b=float(omega[2]),
        omega_c=float(omega[3]),
        omega_d=float(omega[0]),
        beta0_a=float(beta0[1]),
        beta0_b=float(beta0[2]),
        beta0_c=float(beta0[3]),
        beta0_d=float(beta0[0]),
        beta1_a=float(beta1[1]),
        beta1_b=float(beta1[2]),
        beta1_c=float(beta1[3]),
        beta1_d=float(beta1[0]),
        gvd_a=float(beta2_local[1]),
        gvd_b=float(beta2_local[2]),
        gvd_c=float(beta2_local[3]),
        gvd_d=float(beta2_local[0]),
        beta3_a=beta3,
        beta3_b=beta3,
        beta3_c=beta3,
        beta3_d=beta3,
    )
    return channels, {
        "beta2": beta2,
        "beta3": beta3,
        "gradient_norm": gradient_norm,
        "dgd_scale": gradient_norm,
        "curvature_scale": float(baud_rate) ** 2 * abs(beta2),
        "delta_beta0": desired_mismatch,
        "natural_mu": natural_mismatch / mismatch_scale,
        "x_scale": float(x_grad),
        "x_grad": float(length) * float(baud_rate) * gradient_norm,
        "x_curvature": float(length) * float(baud_rate) ** 2 * abs(beta2),
        "x_phase": float(length) * abs(desired_mismatch),
        "x_combined": float(x_grad) + float(length) * abs(desired_mismatch),
        "scale_kind": scale_kind,
        "dispersion_model": dispersion_model,
        "extra_zero_sum": float(extra_zero_sum),
        "mu": float(mu),
        "center_frequency_residual": float(
            (positions[1] + positions[2] - positions[3] - positions[0])
            * spacing_over_baud
            * baud_rate
        ),
        "center_angular_frequency_residual": channels.delta_omega,
        "carrier_index_residual": carrier_index_residual,
        "support_margin_hz": 2.0 * float(baud_rate) - abs(carrier_frequency_residual),
    }


def evaluate_single_tuple_case(
    *,
    channel_positions: np.ndarray,
    spacing_over_baud: float,
    baud_rate: float,
    length: float,
    x_grad: float,
    mu: float,
    n_samples: int,
    seed: int,
    beta2_sign: float = -1.0,
    dispersion_model: str = "constant-beta2",
    extra_zero_sum: float = 5.0,
    random_variables: np.ndarray | None = None,
) -> dict[str, object]:
    channels, geometry = build_single_tuple_channels(
        channel_positions=channel_positions,
        spacing_over_baud=spacing_over_baud,
        baud_rate=baud_rate,
        length=length,
        x_grad=x_grad,
        mu=mu,
        beta2_sign=beta2_sign,
        dispersion_model=dispersion_model,
        extra_zero_sum=extra_zero_sum,
    )
    if random_variables is None:
        random_variables = _symmetric_random_variables(n_samples, seed)
    estimate = estimate_fwm_term_sum_dar_mc(
        channels=channels,
        baud_rate=baud_rate,
        length=length,
        n_samples=n_samples,
        random_variables=random_variables,
    )
    return {
        **geometry,
        "mc_value": estimate.total,
        "mc_stderr": estimate.total_stderr,
        # The frequency-domain estimator already returns N*T^2.
        "normalized_n": estimate.total / float(length) ** 2,
        "normalized_stderr": estimate.total_stderr / float(length) ** 2,
        "support_fraction": float(estimate.metadata["support_fraction"]),
    }


def compute_curves(
    *,
    channel_positions: np.ndarray,
    spacing_over_baud: float,
    baud_rate: float,
    length: float,
    x_grid: np.ndarray,
    mu_values: np.ndarray,
    n_samples: int,
    n_seeds: int,
    seed: int,
    beta2_sign: float,
    dispersion_model: str = "constant-beta2",
    extra_zero_sum: float = 5.0,
) -> dict[str, np.ndarray]:
    rows = []
    seed_values = []
    seed_errors = []
    random_by_seed = [
        _symmetric_random_variables(n_samples, seed + seed_index)
        for seed_index in range(n_seeds)
    ]
    for mu in mu_values:
        for x_grad in x_grid:
            cases = [
                evaluate_single_tuple_case(
                    channel_positions=channel_positions,
                    spacing_over_baud=spacing_over_baud,
                    baud_rate=baud_rate,
                    length=length,
                    x_grad=float(x_grad),
                    mu=float(mu),
                    n_samples=n_samples,
                    seed=seed + seed_index,
                    beta2_sign=beta2_sign,
                    dispersion_model=dispersion_model,
                    extra_zero_sum=extra_zero_sum,
                    random_variables=random_by_seed[seed_index],
                )
                for seed_index in range(n_seeds)
            ]
            values = np.array([case["mc_value"] for case in cases], dtype=float)
            errors = np.array([case["mc_stderr"] for case in cases], dtype=float)
            if n_seeds > 1:
                seed_stderr = float(np.std(values, ddof=1) / np.sqrt(n_seeds))
                internal_stderr = float(np.sqrt(np.mean(errors**2) / n_seeds))
                total_stderr = float(np.hypot(seed_stderr, internal_stderr))
            else:
                total_stderr = float(errors[0])
            case = cases[0]
            rows.append(
                [
                    mu,
                    case["x_scale"],
                    case["x_grad"],
                    case["x_curvature"],
                    case["x_phase"],
                    case["x_combined"],
                    case["beta2"],
                    case["beta3"],
                    case["gradient_norm"],
                    case["delta_beta0"],
                    float(np.mean(values)),
                    total_stderr,
                    float(np.mean(values)) / length**2,
                    total_stderr / length**2,
                    case["support_fraction"],
                    case["center_frequency_residual"],
                    case["center_angular_frequency_residual"],
                    case["carrier_index_residual"],
                    case["support_margin_hz"],
                ]
            )
            seed_values.append(values)
            seed_errors.append(errors)
    matrix = np.asarray(rows, dtype=float)
    names = (
        "mu",
        "x_scale",
        "x_grad",
        "x_curvature",
        "x_phase",
        "x_combined",
        "beta2",
        "beta3",
        "gradient_norm",
        "delta_beta0",
        "mc_value",
        "mc_stderr",
        "normalized_n",
        "normalized_stderr",
        "support_fraction",
        "center_frequency_residual",
        "center_angular_frequency_residual",
        "carrier_index_residual",
        "support_margin_hz",
    )
    data = {name: matrix[:, index] for index, name in enumerate(names)}
    data.update(
        {
            "seed_values": np.asarray(seed_values),
            "seed_errors": np.asarray(seed_errors),
            "channel_positions": np.asarray(channel_positions, dtype=float),
            "spacing_over_baud": np.array(float(spacing_over_baud)),
            "baud_rate": np.array(float(baud_rate)),
            "symbol_time": np.array(1.0 / float(baud_rate)),
            "length": np.array(float(length)),
            "n_samples": np.array(int(n_samples)),
            "n_seeds": np.array(int(n_seeds)),
            "seed": np.array(int(seed)),
            "scale_kind": np.array(str(case["scale_kind"])),
            "dispersion_model": np.array(str(case["dispersion_model"])),
            "extra_zero_sum": np.array(float(case["extra_zero_sum"])),
            "mc_value_convention": np.array("N_times_T_squared"),
            "vertical_normalization": np.array("mc_value/L^2 = N*T^2/L^2"),
        }
    )
    return data


def save_dataset(data: dict[str, np.ndarray], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "single_tuple_scaling.npz"
    np.savez(npz_path, **data)
    csv_path = out_dir / "single_tuple_scaling.csv"
    fields = (
        "mu",
        "x_scale",
        "x_grad",
        "x_curvature",
        "x_phase",
        "x_combined",
        "beta2",
        "beta3",
        "gradient_norm",
        "delta_beta0",
        "mc_value",
        "mc_stderr",
        "normalized_n",
        "normalized_stderr",
        "support_fraction",
        "center_frequency_residual",
        "center_angular_frequency_residual",
        "carrier_index_residual",
        "support_margin_hz",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(zip(*(data[field] for field in fields), strict=True))
    return npz_path, csv_path


def _tail_slope(x: np.ndarray, y: np.ndarray) -> float:
    count = max(3, x.size // 3)
    order = np.argsort(x)[-count:]
    return float(np.polyfit(np.log(x[order]), np.log(y[order]), 1)[0])


def plot_curves(data: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    mu_values = np.unique(data["mu"])
    color_positions = (
        np.linspace(0.12, 0.88, mu_values.size)
        if mu_values.size > 1
        else np.array([0.5])
    )
    mu_colors = {
        float(mu): plt.get_cmap("viridis")(position)
        for mu, position in zip(mu_values, color_positions, strict=True)
    }
    scale_kind = str(np.asarray(data["scale_kind"]).item())
    if scale_kind == "curvature":
        scale_label = r"$x_{\mathrm{curv}}=LB^2|\beta_2|$"
        combined_label = r"$L(|\Delta\beta_0|+B^2|\beta_2|)$"
        short_scale_label = r"$x_{\mathrm{curv}}$"
    else:
        scale_label = r"$x_\nabla=LB\|\nabla_\Omega\Delta\beta\|_2$"
        combined_label = r"$L(|\Delta\beta_0|+B\|\nabla_\Omega\Delta\beta\|_2)$"
        short_scale_label = r"$x_\nabla$"

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for mu in mu_values:
        mask = data["mu"] == mu
        order = np.argsort(data["x_scale"][mask])
        x_scale = data["x_scale"][mask][order]
        x_combined = data["x_combined"][mask][order]
        y = data["normalized_n"][mask][order]
        error = data["normalized_stderr"][mask][order]
        label = rf"$\mu={mu:g}$, slope={_tail_slope(x_scale, y):.2f}"
        color = mu_colors[float(mu)]
        axes[0].errorbar(
            x_scale,
            y,
            yerr=error,
            color=color,
            marker="o",
            ms=3,
            lw=0.9,
            capsize=1,
            label=label,
        )
        axes[1].errorbar(
            x_combined,
            y,
            yerr=error,
            color=color,
            marker="o",
            ms=3,
            lw=0.9,
            capsize=1,
            label=rf"$\mu={mu:g}$",
        )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$N_{dabc}T^2/L^2$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_xlabel(scale_label)
    axes[1].set_xlabel(combined_label)
    axes[0].legend(loc="lower left", fontsize=7, frameon=False)
    draw_tuple_island_inset(
        axes[0],
        np.asarray(data["channel_positions"], dtype=float),
        float(data["spacing_over_baud"]),
        mu_values=mu_values,
        x_values=np.asarray(data["x_scale"], dtype=float),
        beta2_sign=float(np.sign(np.asarray(data["beta2"], dtype=float)[0])),
        mu_colors=mu_colors,
        dispersion_model=str(
            np.asarray(data.get("dispersion_model", np.array("constant-beta2"))).item()
        ),
        extra_zero_sum=float(
            np.asarray(data.get("extra_zero_sum", np.array(5.0))).item()
        ),
    )
    fig.tight_layout()
    path = out_dir / "single_tuple_collapse.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for mu in mu_values:
        mask = data["mu"] == mu
        order = np.argsort(data["x_scale"][mask])
        x = data["x_scale"][mask][order]
        y = data["normalized_n"][mask][order]
        color = mu_colors[float(mu)]
        axes[0].plot(
            x,
            x * y,
            color=color,
            marker="o",
            ms=3,
            lw=0.9,
            label=rf"$\mu={mu:g}$",
        )
        if x.size >= 3:
            slope = np.gradient(np.log(y), np.log(x))
            axes[1].plot(
                x,
                slope,
                color=color,
                marker="o",
                ms=3,
                lw=0.9,
                label=rf"$\mu={mu:g}$",
            )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(short_scale_label)
    axes[0].set_ylabel(short_scale_label + r" $N_{dabc}T^2/L^2$")
    axes[1].set_xscale("log")
    axes[1].axhline(-1.0, color="0.35", ls="--", lw=0.8)
    axes[1].axhline(-2.0, color="0.35", ls=":", lw=0.8)
    axes[1].set_xlabel(short_scale_label)
    axes[1].set_ylabel("local log-log slope")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = out_dir / "single_tuple_asymptotics.pdf"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)
    return paths


def _parse_float_list(value: str) -> np.ndarray:
    return np.array([float(part) for part in value.split(",") if part.strip()], dtype=float)


def natural_mu(
    channel_positions: np.ndarray,
    spacing_over_baud: float,
    beta2_sign: float,
    dispersion_model: str = "constant-beta2",
    extra_zero_sum: float = 5.0,
) -> float:
    """Return the unshifted detuning ratio for a fixed tuple and model."""
    _, geometry = build_single_tuple_channels(
        channel_positions=channel_positions,
        spacing_over_baud=spacing_over_baud,
        baud_rate=1.0,
        length=1.0,
        x_grad=1.0,
        mu=0.0,
        beta2_sign=beta2_sign,
        dispersion_model=dispersion_model,
        extra_zero_sum=extra_zero_sum,
    )
    return float(geometry["natural_mu"])


def _parse_mu_list(
    value: str,
    channel_positions: np.ndarray,
    spacing_over_baud: float,
    beta2_sign: float,
    dispersion_model: str = "constant-beta2",
    extra_zero_sum: float = 5.0,
) -> np.ndarray:
    values = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() == "natural":
            values.append(
                natural_mu(
                    channel_positions,
                    spacing_over_baud,
                    beta2_sign,
                    dispersion_model,
                    extra_zero_sum,
                )
            )
        else:
            values.append(float(token))
    return np.asarray(list(dict.fromkeys(values)), dtype=float)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled single-tuple FWM scaling sweep.")
    parser.add_argument("--channels", type=str, default="0,-1,2,1", help="Fixed (d,a,b,c) channel positions.")
    parser.add_argument("--spacing-over-baud", type=float, default=25.0 / 24.5)
    parser.add_argument("--baud-rate", type=float, default=24.5e9)
    parser.add_argument("--length", type=float, default=100e3)
    parser.add_argument("--x-min", type=float, default=3e-2)
    parser.add_argument("--x-max", type=float, default=3e2)
    parser.add_argument("--n-x", type=int, default=25)
    parser.add_argument(
        "--mu",
        type=str,
        default="natural,0,0.5,2,10",
        help="Detuning ratios; 'natural' keeps the unshifted constant-beta2 mismatch.",
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--beta2-sign", type=float, default=-1.0)
    parser.add_argument(
        "--dispersion-model",
        choices=("constant-beta2", "cubic-zdw"),
        default="constant-beta2",
    )
    parser.add_argument(
        "--extra-zero-sum",
        type=float,
        default=5.0,
        help="Cubic model branch (Omega_a+Omega_b)/(2*pi*Delta_f); ZDW is half this value.",
    )
    parser.add_argument("--results-dir", type=Path, default=OUT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=OUT_MEDIA)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    channels = _parse_float_list(args.channels)
    mu_values = _parse_mu_list(
        args.mu,
        channels,
        args.spacing_over_baud,
        args.beta2_sign,
        args.dispersion_model,
        args.extra_zero_sum,
    )
    if args.n_x < 3 or args.n_samples < 2 or args.n_seeds < 1:
        raise ValueError("n-x >= 3, n-samples >= 2, and n-seeds >= 1 are required")
    data = compute_curves(
        channel_positions=channels,
        spacing_over_baud=args.spacing_over_baud,
        baud_rate=args.baud_rate,
        length=args.length,
        x_grid=np.geomspace(args.x_min, args.x_max, args.n_x),
        mu_values=mu_values,
        n_samples=args.n_samples,
        n_seeds=args.n_seeds,
        seed=args.seed,
        beta2_sign=args.beta2_sign,
        dispersion_model=args.dispersion_model,
        extra_zero_sum=args.extra_zero_sum,
    )
    npz_path, csv_path = save_dataset(data, args.results_dir)
    paths = plot_curves(data, args.out_dir)
    print(f"saved {npz_path}")
    print(f"saved {csv_path}")
    for path in paths:
        print(f"saved {path}")
    for mu in mu_values:
        mask = data["mu"] == mu
        print(f"mu={mu:g}: high-x slope={_tail_slope(data['x_scale'][mask], data['normalized_n'][mask]):.4g}")


if __name__ == "__main__":
    main()
