"""Stage-labelled cache/file naming helpers for TD-NLIN artifacts."""

from __future__ import annotations

from pathlib import Path


def pulse_name(*, ipulse: int | None = None, pulse_shape: str | None = None) -> str:
    if pulse_shape is not None:
        return str(pulse_shape).strip().lower()
    if ipulse == 0:
        return "gaussian"
    if ipulse == 1:
        return "nyquist"
    raise ValueError(f"Unsupported pulse selector: ipulse={ipulse!r}, pulse_shape={pulse_shape!r}")


def s1_ref_nlin_curve_path(
    *,
    ipulse: int | None = None,
    pulse_shape: str | None = None,
    mode: str,
    gvda: float,
    gvdb: float,
    directory: str | Path = "results",
) -> Path:
    pulse = pulse_name(ipulse=ipulse, pulse_shape=pulse_shape)
    return Path(directory) / f"s1_ref_nlin_curve_{pulse}_{mode}_gvda{gvda}_gvdb{gvdb}.npz"


def s2a_lo_timeint_path(
    *,
    ipulse: int | None = None,
    pulse_shape: str | None = None,
    m_lo: int,
    directory: str | Path = "results",
) -> Path:
    pulse = pulse_name(ipulse=ipulse, pulse_shape=pulse_shape)
    return Path(directory) / f"s2a_lo_timeint_{pulse}_m{m_lo}.npz"


def s2b_lo_extrema_path(
    *,
    ipulse: int | None = None,
    pulse_shape: str | None = None,
    m_lo_truncation: int,
    fiber_length: float,
    lld_max: float,
    custom_fB_index: int | None = None,
    profile_tag: str | None = None,
    directory: str | Path = "results",
) -> Path:
    pulse = pulse_name(ipulse=ipulse, pulse_shape=pulse_shape)
    suffix = ""
    if custom_fB_index is not None:
        suffix = f"_customfB{int(custom_fB_index)}"
    if profile_tag is not None:
        suffix = f"_prof{profile_tag}{suffix}"
    return (
        Path(directory)
        / f"s2b_lo_extrema_{pulse}_mtrunc{m_lo_truncation}_L{fiber_length/1e3:.1f}km_lldmax{lld_max:.2f}{suffix}.npz"
    )


def _hz_tag(value_hz: float) -> str:
    return f"{value_hz/1e9:.3f}GHz".replace(".", "p")


def s3_pair_nlin_kernel_path(
    *,
    ipulse: int,
    fiber_type: str,
    br_hz: float,
    n_ch: int,
    fiber_length: float,
    spacing_hz: float | None = None,
    profile_tag: str | None = None,
    disp_tag: str | None = None,
    directory: str | Path = "results",
) -> Path:
    name = Path(directory) / f"s3_pair_nlin_kernel_ipulse{ipulse}_{fiber_type}"
    stem = name.name
    if profile_tag:
        stem = f"{stem}_{profile_tag}"
    stem = f"{stem}_L{fiber_length/1e3:.1f}km"
    stem = f"{stem}_br{_hz_tag(br_hz)}_n{n_ch}"
    if spacing_hz is not None:
        stem = f"{stem}_sp{_hz_tag(float(spacing_hz))}"
    if disp_tag:
        stem = f"{stem}_disp{disp_tag}"
    return Path(directory) / f"{stem}.npy"


def s3_chan_nlin_td_path(
    *,
    tag: str,
    use_kappa: bool,
    use_x_mode: bool,
    extra_tag: str | None = None,
    directory: str | Path = "results",
) -> Path:
    suffix = ""
    if extra_tag:
        safe_tag = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(extra_tag))
        safe_tag = safe_tag.strip("_")
        if safe_tag:
            suffix = f"_{safe_tag}"
    return Path(directory) / f"s3_chan_nlin_td_{tag}{suffix}_k{int(use_kappa)}_x{int(use_x_mode)}.npy"


def s2_beta1_grid_path(temp_dir: str | Path = "/tmp") -> Path:
    return Path(temp_dir) / "s2_beta1_grid.npy"


def s2_beta2_grid_path(temp_dir: str | Path = "/tmp") -> Path:
    return Path(temp_dir) / "s2_beta2_grid.npy"


def s2_fB_grid_path(temp_dir: str | Path = "/tmp") -> Path:
    return Path(temp_dir) / "s2_fB_grid.npy"
