from pathlib import Path
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.constants import c

from pynlin.pulses import PulseType, RaisedCosinePulse, RootRaisedCosinePulse
from pynlin.system import System
from pynlin.utils import watt2dBm
from pynlin.wdm import IrregularWDM

PROFILE_MAX_W = 10.0
_BINARY_MODES = {"off", "on"}
_MODEL_MODES = {"off", "cached", "recompute"}
_POWER_PROFILE_MODES = {
    "flat",
    "cached",
    "recompute",
    "cached_no_profile_launch",
    "recompute_no_profile_launch",
}

_METHOD_MODES = {"off", "cached", "recompute"}
_TD_TIME_INTEGRAL_BACKENDS = {"direct", "x0mm_fft"}
_FULLBAND_FWM_TUPLE_SELECTION_MODES = {"joint_reservoir", "exhaustive_support_mc"}
_STUDY_TYPES = {"full_system", "subset", "sweep"}


@dataclass(frozen=True)
class ProfilesConfig:
    mode: str = "recompute"
    path: Path = Path("results/pcfm_power_profiles.npy")
    launch_csv: Path | None = None


@dataclass(frozen=True)
class TDMethodConfig:
    mode: str = "cached"
    exclude_self_channel: bool = True
    m_lo_truncation: int = 40
    use_kappa: bool = True
    use_x_mode: bool = True
    time_integral_backend: str = "direct"


@dataclass(frozen=True)
class PCFMMethodConfig:
    mode: str = "cached"
    numeric_sci: bool = True
    numeric_xci: bool = False
    eq18_xci: bool = False
    degree: int = 9
    include_mci: bool = False
    plot_total_and_sci: bool = False


@dataclass(frozen=True)
class GNMethodConfig:
    mode: str = "off"
    direct_mode: str = "off"


# FIXME(refactor): The MC engine currently bundles two independent backends:
#   - "ssfm" / "td"  (TD collision-coeff → chi1/chi2 → NLIN)
#   - "fullband"      (prefactor-free MC over 3PC/4PC/FWM tuples)
# These should be split into separate MethodConfig classes when the
# study/runner framework is next reworked.
@dataclass(frozen=True)
class MCMethodConfig:
    mode: str = "off"
    engine: str = "ssfm"
    n_trials: int = 1
    rng_seed: int = 1234
    template: Path | None = None
    n_channels: int = 5
    # Fullband-MC-specific fields (used when engine == "fullband")
    channel_decimation: int = 1
    target_decimation: int = 1
    target_offset: int = 0
    target_limit: int | None = None
    xpm_samples: int = 10000
    fwm_samples: int = 5000
    fwm_frequency_samples: int = 50
    seed: int = 1234
    max_fwm_tuples_per_target: int | None = None
    fwm_tuple_selection: str = "joint_reservoir"
    workers: int = 1


@dataclass(frozen=True)
class MethodsConfig:
    td: TDMethodConfig = field(default_factory=TDMethodConfig)
    pcfm: PCFMMethodConfig = field(default_factory=PCFMMethodConfig)
    gn: GNMethodConfig = field(default_factory=GNMethodConfig)
    mc: MCMethodConfig = field(default_factory=MCMethodConfig)


@dataclass(frozen=True)
class SubsetConfig:
    mode: str = "center_window"
    center: str | int = "auto"
    half_width: int = 2
    cut_indices: tuple[int, ...] = ()
    interferer_indices: tuple[int, ...] = ()
    include_sci: bool = True


@dataclass(frozen=True)
class SweepConfig:
    variable: str = ""
    unit: str = ""
    values: tuple[float, ...] = ()


@dataclass(frozen=True)
class StudyConfig:
    name: str
    type: str
    methods: tuple[str, ...]
    out_dir: Path
    plot: bool = False
    subset: SubsetConfig | None = None
    sweep: SweepConfig | None = None


@dataclass(frozen=True)
class StudiesRuntimeConfig:
    profiles: ProfilesConfig
    methods: MethodsConfig
    studies: tuple[StudyConfig, ...]


def _flat_profiles_enabled(system: System) -> bool:
    raw = getattr(system, "raw_config", None)
    if not isinstance(raw, Mapping):
        return False
    pog = raw.get("pcfm")
    if isinstance(pog, Mapping):
        run = pog.get("run")
        if isinstance(run, Mapping):
            mode = run.get("power_profiles_mode")
            if isinstance(mode, str):
                mode = mode.strip().lower()
                if mode == "flat":
                    return True
                if mode in _POWER_PROFILE_MODES:
                    return False
    nlin_section = raw.get("nlin")
    if not isinstance(nlin_section, Mapping):
        return False
    return bool(nlin_section.get("flat_profiles") or nlin_section.get("flat_profile"))


def _load_pcfm_runtime_config(system: System) -> dict[str, object]:
    """Load workflow/runtime flags from [pcfm] in the system TOML."""
    defaults: dict[str, object] = {
        "profile_path": "results/pcfm_power_profiles.npy",
        "launch_csv_path": None,
        "pcfm_numeric_sci": True,
        "pcfm_numeric_xci": False,
        "pcfm_eq18_xci": False,
        "pcfm_degree": 9,
        "pcfm_include_mci": False,
        "td_exclude_self_channel": True,
        "td_m_lo_truncation": 40,
        "td_time_integral_backend": "direct",
        "plot_pcfm_total_and_sci": False,
        "power_profiles_mode": "recompute",
        "td_mode": "cached",
        "pcfm_mode": "cached",
        "gn_mode": "off",
        "gn_direct_mode": "off",
        "plot_mode": "on",
    }
    raw = getattr(system, "raw_config", None)
    if not isinstance(raw, Mapping):
        return defaults
    if "profiles" in raw or "methods" in raw:
        profiles = _load_profiles_config(system)
        methods = _load_methods_config(system)
        defaults.update(
            {
                "profile_path": str(profiles.path),
                "launch_csv_path": None if profiles.launch_csv is None else str(profiles.launch_csv),
                "power_profiles_mode": profiles.mode,
                "td_mode": methods.td.mode,
                "pcfm_mode": methods.pcfm.mode,
                "gn_mode": methods.gn.mode,
                "gn_direct_mode": methods.gn.direct_mode,
                "pcfm_numeric_sci": methods.pcfm.numeric_sci,
                "pcfm_numeric_xci": methods.pcfm.numeric_xci,
                "pcfm_eq18_xci": methods.pcfm.eq18_xci,
                "pcfm_degree": methods.pcfm.degree,
                "pcfm_include_mci": methods.pcfm.include_mci,
                "td_exclude_self_channel": methods.td.exclude_self_channel,
                "td_m_lo_truncation": methods.td.m_lo_truncation,
                "td_time_integral_backend": methods.td.time_integral_backend,
                "plot_pcfm_total_and_sci": methods.pcfm.plot_total_and_sci,
            }
        )
        return defaults
    pog = raw.get("pcfm")
    if not isinstance(pog, Mapping):
        return defaults
    paths = pog.get("paths")
    run = pog.get("run")
    recompute = pog.get("recompute")
    if isinstance(paths, Mapping):
        if "profile" in paths:
            defaults["profile_path"] = paths.get("profile")
        if "launch_csv" in paths:
            defaults["launch_csv_path"] = paths.get("launch_csv")
    if isinstance(run, Mapping):
        legacy_run_keys = [
            key
            for key in (
                "compute_gn",
                "compute_gn_direct",
                "plot",
                "recompute_all",
                "recompute_profiles",
                "recompute_td",
                "recompute_pcfm",
                "recompute_gn",
                "recompute_gn_direct",
                "profiles_mode",
                "use_profile_launch_powers",
            )
            if key in run
        ]
        if legacy_run_keys:
            raise ValueError(
                "Legacy PCFM run keys are no longer supported: "
                f"{legacy_run_keys}. Use *_mode keys in [pcfm.run]."
            )
        for key in (
            "pcfm_numeric_sci",
            "pcfm_numeric_xci",
            "pcfm_eq18_xci",
            "pcfm_degree",
            "pcfm_include_mci",
            "td_exclude_self_channel",
            "td_m_lo_truncation",
            "td_time_integral_backend",
            "plot_pcfm_total_and_sci",
            "power_profiles_mode",
            "td_mode",
            "pcfm_mode",
            "gn_mode",
            "gn_direct_mode",
            "plot_mode",
        ):
            if key in run:
                defaults[key] = run.get(key)
    if isinstance(recompute, Mapping):
        raise ValueError(
            "Legacy [pcfm.recompute] is no longer supported. "
            "Use *_mode keys in [pcfm.run]."
        )

    defaults["power_profiles_mode"] = _normalize_mode(
        "power_profiles_mode", defaults["power_profiles_mode"], _POWER_PROFILE_MODES
    )
    defaults["td_mode"] = _normalize_mode("td_mode", defaults["td_mode"], _MODEL_MODES)
    defaults["pcfm_mode"] = _normalize_mode("pcfm_mode", defaults["pcfm_mode"], _MODEL_MODES)
    defaults["gn_mode"] = _normalize_mode("gn_mode", defaults["gn_mode"], _MODEL_MODES)
    defaults["gn_direct_mode"] = _normalize_mode(
        "gn_direct_mode", defaults["gn_direct_mode"], _MODEL_MODES
    )
    defaults["plot_mode"] = _normalize_mode("plot_mode", defaults["plot_mode"], _BINARY_MODES)
    defaults["td_m_lo_truncation"] = _normalize_nonnegative_int(
        "td_m_lo_truncation", defaults["td_m_lo_truncation"]
    )
    defaults["td_time_integral_backend"] = _normalize_mode(
        "td_time_integral_backend", defaults["td_time_integral_backend"], _TD_TIME_INTEGRAL_BACKENDS
    )
    defaults["pcfm_degree"] = _normalize_nonnegative_int(
        "pcfm_degree", defaults["pcfm_degree"]
    )
    defaults["pcfm_include_mci"] = bool(defaults["pcfm_include_mci"])
    return defaults


def _to_optional_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return Path(text)


def _as_mapping(value: object, name: str) -> Mapping:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a TOML table.")
    return value


def _as_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a boolean; got {value!r}.")


def _as_int(value: object, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}.") from exc


def _as_float_tuple(value: object, name: str) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of numbers; got {value!r}.")
    return tuple(float(v) for v in value)


def _as_int_tuple(value: object, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of integers; got {value!r}.")
    return tuple(int(v) for v in value)


def _load_profiles_config(system: System) -> ProfilesConfig:
    raw = getattr(system, "raw_config", None)
    profiles = _as_mapping(raw.get("profiles") if isinstance(raw, Mapping) else None, "[profiles]")
    mode = _normalize_mode("profiles.mode", profiles.get("mode", "recompute"), _POWER_PROFILE_MODES)
    path = _to_optional_path(profiles.get("path", "results/pcfm_power_profiles.npy"))
    if path is None:
        raise ValueError("[profiles].path is required.")
    return ProfilesConfig(
        mode=mode,
        path=path,
        launch_csv=_to_optional_path(profiles.get("launch_csv")),
    )


def _load_methods_config(system: System) -> MethodsConfig:
    raw = getattr(system, "raw_config", None)
    methods = _as_mapping(raw.get("methods") if isinstance(raw, Mapping) else None, "[methods]")
    td = _as_mapping(methods.get("td"), "[methods.td]")
    pcfm = _as_mapping(methods.get("pcfm"), "[methods.pcfm]")
    gn = _as_mapping(methods.get("gn"), "[methods.gn]")
    mc = _as_mapping(methods.get("mc"), "[methods.mc]")

    return MethodsConfig(
        td=TDMethodConfig(
            mode=_normalize_mode("methods.td.mode", td.get("mode", "cached"), _METHOD_MODES),
            exclude_self_channel=bool(td.get("exclude_self_channel", True)),
            m_lo_truncation=_normalize_nonnegative_int(
                "methods.td.m_lo_truncation", td.get("m_lo_truncation", 40)
            ),
            use_kappa=bool(td.get("use_kappa", True)),
            use_x_mode=bool(td.get("use_x_mode", True)),
            time_integral_backend=_normalize_mode(
                "methods.td.time_integral_backend",
                td.get("time_integral_backend", "direct"),
                _TD_TIME_INTEGRAL_BACKENDS,
            ),
        ),
        pcfm=PCFMMethodConfig(
            mode=_normalize_mode("methods.pcfm.mode", pcfm.get("mode", "cached"), _METHOD_MODES),
            numeric_sci=bool(pcfm.get("numeric_sci", True)),
            numeric_xci=bool(pcfm.get("numeric_xci", False)),
            eq18_xci=bool(pcfm.get("eq18_xci", False)),
            degree=_normalize_nonnegative_int("methods.pcfm.degree", pcfm.get("degree", 9)),
            include_mci=bool(pcfm.get("include_mci", False)),
            plot_total_and_sci=bool(pcfm.get("plot_total_and_sci", False)),
        ),
        gn=GNMethodConfig(
            mode=_normalize_mode("methods.gn.mode", gn.get("mode", "off"), _METHOD_MODES),
            direct_mode=_normalize_mode(
                "methods.gn.direct_mode", gn.get("direct_mode", "off"), _METHOD_MODES
            ),
        ),
        mc=MCMethodConfig(
            mode=_normalize_mode("methods.mc.mode", mc.get("mode", "off"), _METHOD_MODES),
            engine=str(mc.get("engine", "ssfm")),
            n_trials=max(_as_int(mc.get("n_trials", 1), "methods.mc.n_trials"), 1),
            rng_seed=_as_int(mc.get("rng_seed", 1234), "methods.mc.rng_seed"),
            template=_to_optional_path(mc.get("template")),
            n_channels=max(_as_int(mc.get("n_channels", 5), "methods.mc.n_channels"), 1),
            channel_decimation=_normalize_nonnegative_int(
                "methods.mc.channel_decimation", mc.get("channel_decimation", 1)
            ),
            target_decimation=_normalize_nonnegative_int(
                "methods.mc.target_decimation", mc.get("target_decimation", 1)
            ),
            target_offset=_normalize_nonnegative_int(
                "methods.mc.target_offset", mc.get("target_offset", 0)
            ),
            target_limit=_as_int(mc.get("target_limit"), "methods.mc.target_limit")
            if mc.get("target_limit") is not None else None,
            xpm_samples=max(_as_int(mc.get("xpm_samples", 10000), "methods.mc.xpm_samples"), 1),
            fwm_samples=max(_as_int(mc.get("fwm_samples", 5000), "methods.mc.fwm_samples"), 1),
            fwm_frequency_samples=max(
                _as_int(mc.get("fwm_frequency_samples", 50), "methods.mc.fwm_frequency_samples"), 1
            ),
            seed=_as_int(mc.get("seed", 1234), "methods.mc.seed"),
            max_fwm_tuples_per_target=_as_int(mc.get("max_fwm_tuples_per_target"), "methods.mc.max_fwm_tuples_per_target")
            if mc.get("max_fwm_tuples_per_target") is not None else None,
            fwm_tuple_selection=_normalize_mode(
                "methods.mc.fwm_tuple_selection",
                mc.get("fwm_tuple_selection", "joint_reservoir"),
                _FULLBAND_FWM_TUPLE_SELECTION_MODES,
            ),
            workers=max(_as_int(mc.get("workers", 1), "methods.mc.workers"), 0),
        ),
    )


def _load_subset_config(data: object, name: str) -> SubsetConfig | None:
    if data is None:
        return None
    subset = _as_mapping(data, f"[studies.{name}.subset]")
    return SubsetConfig(
        mode=str(subset.get("mode", "center_window")),
        center=subset.get("center", "auto"),
        half_width=max(_as_int(subset.get("half_width", 2), f"studies.{name}.subset.half_width"), 0),
        cut_indices=_as_int_tuple(subset.get("cut_indices"), f"studies.{name}.subset.cut_indices"),
        interferer_indices=_as_int_tuple(
            subset.get("interferer_indices"), f"studies.{name}.subset.interferer_indices"
        ),
        include_sci=bool(subset.get("include_sci", True)),
    )


def _load_sweep_config(data: object, name: str) -> SweepConfig | None:
    if data is None:
        return None
    sweep = _as_mapping(data, f"[studies.{name}.sweep]")
    return SweepConfig(
        variable=str(sweep.get("variable", "")),
        unit=str(sweep.get("unit", "")),
        values=_as_float_tuple(sweep.get("values"), f"studies.{name}.sweep.values"),
    )


def _load_studies_config(system: System) -> tuple[StudyConfig, ...]:
    raw = getattr(system, "raw_config", None)
    studies = _as_mapping(raw.get("studies") if isinstance(raw, Mapping) else None, "[studies]")
    if not studies:
        return (
            StudyConfig(
                name="full_system",
                type="full_system",
                methods=("td", "pcfm"),
                out_dir=Path("results/studies/full_system"),
                plot=True,
            ),
        )

    out: list[StudyConfig] = []
    for name, value in studies.items():
        study = _as_mapping(value, f"[studies.{name}]")
        study_type = _normalize_mode(f"studies.{name}.type", study.get("type"), _STUDY_TYPES)
        methods = study.get("methods", [])
        if not isinstance(methods, (list, tuple)) or not methods:
            raise ValueError(f"studies.{name}.methods must be a non-empty list.")
        method_names = tuple(str(method).strip().lower() for method in methods)
        unknown = sorted(set(method_names) - {"td", "pcfm", "gn", "mc"})
        if unknown:
            raise ValueError(f"studies.{name}.methods contains unknown methods: {unknown}")
        out.append(
            StudyConfig(
                name=str(name),
                type=study_type,
                methods=method_names,
                out_dir=Path(study.get("out_dir", f"results/studies/{name}")),
                plot=bool(study.get("plot", False)),
                subset=_load_subset_config(study.get("subset"), str(name)),
                sweep=_load_sweep_config(study.get("sweep"), str(name)),
            )
        )
    return tuple(out)


def load_studies_runtime_config(system: System) -> StudiesRuntimeConfig:
    """Load the method-agnostic studies runtime config from a system TOML."""
    return StudiesRuntimeConfig(
        profiles=_load_profiles_config(system),
        methods=_load_methods_config(system),
        studies=_load_studies_config(system),
    )


def _normalize_mode(name: str, value: object, allowed: set[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string in {sorted(allowed)}; got {value!r}.")
    mode = value.strip().lower()
    if mode not in allowed:
        raise ValueError(f"Invalid {name}={value!r}; expected one of {sorted(allowed)}.")
    return mode


def _normalize_nonnegative_int(name: str, value: object) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}.") from exc
    if out < 0:
        raise ValueError(f"{name} must be non-negative; got {out}.")
    return out


def _resolve_scaling_run_flags(
    system: System,
    *,
    pcfm_numeric_xci: bool | None = None,
    pcfm_eq18_xci: bool | None = None,
    recompute_td: bool | None = None,
    recompute_pcfm: bool | None = None,
    exclude_self_channel: bool | None = None,
) -> dict[str, bool]:
    """Resolve scaling-script runtime booleans from [pcfm.run] with optional overrides."""
    runtime_cfg = _load_pcfm_runtime_config(system)
    return {
        "pcfm_numeric_xci": (
            bool(runtime_cfg["pcfm_numeric_xci"])
            if pcfm_numeric_xci is None
            else bool(pcfm_numeric_xci)
        ),
        "pcfm_eq18_xci": (
            bool(runtime_cfg["pcfm_eq18_xci"])
            if pcfm_eq18_xci is None
            else bool(pcfm_eq18_xci)
        ),
        "recompute_td": (
            runtime_cfg["td_mode"] == "recompute"
            if recompute_td is None
            else bool(recompute_td)
        ),
        "recompute_pcfm": (
            runtime_cfg["pcfm_mode"] == "recompute"
            if recompute_pcfm is None
            else bool(recompute_pcfm)
        ),
        "exclude_self_channel": (
            bool(runtime_cfg["td_exclude_self_channel"])
            if exclude_self_channel is None
            else bool(exclude_self_channel)
        ),
    }


def _select_scaling_channel(system: System) -> tuple[int, str]:
    """Pick a representative CUT for scaling sweeps.

    For irregular multi-band grids, choose the center of the populated band whose
    midpoint is closest to the overall spectral centroid. This avoids selecting
    a band-edge channel when bands are concatenated or separated by gaps.
    """
    freqs = np.asarray(system.wdm.frequency_grid(), dtype=float).reshape(-1)
    if freqs.size == 0:
        raise ValueError("Cannot select a scaling channel from an empty WDM grid.")

    target_freq = float(np.mean(freqs))
    band_slices = getattr(system.wdm, "_band_slices", None)
    if isinstance(band_slices, Mapping) and band_slices:
        best_score = None
        best_idx = None
        best_label = "overall"
        for name, slc in band_slices.items():
            start = 0 if slc.start is None else int(slc.start)
            stop = int(slc.stop) if slc.stop is not None else int(freqs.size)
            if stop <= start:
                continue
            idx = np.arange(start, stop, dtype=int)
            band_freqs = freqs[idx]
            band_center_idx = int(idx[idx.size // 2])
            band_center_freq = float(freqs[band_center_idx])
            spans_target = float(np.min(band_freqs)) <= target_freq <= float(np.max(band_freqs))
            score = (
                0 if spans_target else 1,
                abs(band_center_freq - target_freq),
                abs(idx.size - freqs.size),
                band_center_idx,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_idx = band_center_idx
                best_label = str(name)
        if best_idx is not None:
            return best_idx, best_label

    center_idx = int(np.argmin(np.abs(freqs - target_freq)))
    return center_idx, "overall"


def _channel_nonoverlap_spacing_hz(system: System) -> float:
    """Return the minimum adjacent-channel spacing that avoids spectral overlap.

    The spacing sweep should stay in the regime where neighboring WDM channels
    do not overlap spectrally. For Nyquist/Gaussian pulses we use the symbol
    rate as the effective occupied bandwidth. For raised-cosine families we
    include the roll-off expansion of the support.
    """
    baud_rate_hz = float(system.pulse.baud_rate)
    pulse_cfg = getattr(system, "pulse_config", None)
    pulse_type = getattr(pulse_cfg, "type", None)
    rolloff = getattr(pulse_cfg, "rolloff", None)
    if rolloff is None:
        rolloff = getattr(system.pulse, "rolloff", None)
    rolloff = 0.0 if rolloff is None else max(float(rolloff), 0.0)

    if pulse_type in (PulseType.RAISED_COSINE, PulseType.ROOT_RAISED_COSINE) or isinstance(
        system.pulse, (RaisedCosinePulse, RootRaisedCosinePulse)
    ):
        return baud_rate_hz * (1.0 + rolloff)
    return baud_rate_hz


def _wdm_nonoverlap_max_spacing_hz(system: System) -> float | None:
    """Return the largest spacing that preserves non-overlap between WDM bands.

    For irregular multi-band grids, increasing the global spacing stretches each
    band from its fixed start wavelength. This yields a hard upper bound beyond
    which adjacent bands overlap in frequency. Regular single-band grids do not
    have this constraint, so ``None`` is returned.
    """
    wdm = system.wdm
    if not isinstance(wdm, IrregularWDM):
        return None

    ordered_specs = sorted(wdm.band_specs.items(), key=lambda kv: kv[1].start_nm)
    if len(ordered_specs) < 2:
        return None

    max_spacing_hz = float("inf")
    for idx, (_, upper_spec) in enumerate(ordered_specs[:-1]):
        upper_start_hz = float(c / (upper_spec.start_nm * 1e-9))
        upper_span_channels = int(upper_spec.n_channels) - 1
        if upper_span_channels <= 0:
            continue
        for _, lower_spec in ordered_specs[idx + 1 :]:
            lower_start_hz = float(c / (lower_spec.start_nm * 1e-9))
            gap_hz = upper_start_hz - lower_start_hz
            if gap_hz <= 0.0:
                return 0.0
            max_spacing_hz = min(max_spacing_hz, gap_hz / upper_span_channels)

    if not np.isfinite(max_spacing_hz):
        return None
    return max_spacing_hz


def _format_pulse_title_token(system: System) -> str:
    pulse_cfg = getattr(system, "pulse_config", None)
    pulse_type = getattr(pulse_cfg, "type", None)
    label_map = {
        PulseType.GAUSSIAN: "Gaussian",
        PulseType.NYQUIST: "Nyquist",
        PulseType.RAISED_COSINE: "RC",
        PulseType.ROOT_RAISED_COSINE: "RRC",
    }
    pulse_label = (
        label_map.get(pulse_type)
        if pulse_type is not None
        else system.pulse.__class__.__name__.replace("Pulse", "")
    )
    rolloff = getattr(pulse_cfg, "rolloff", None)
    if rolloff is None:
        rolloff = getattr(system.pulse, "rolloff", None)
    if rolloff is not None and pulse_type in (
        PulseType.NYQUIST,
        PulseType.RAISED_COSINE,
        PulseType.ROOT_RAISED_COSINE,
    ):
        return f"pulse={pulse_label}, rho={float(rolloff):.2f}"
    return f"pulse={pulse_label}"


def _format_launch_power_title_token(
    launch_powers_w: np.ndarray | None,
    *,
    channel_idx: int,
) -> str | None:
    if launch_powers_w is None:
        return None
    launch = np.asarray(launch_powers_w, dtype=float).reshape(-1)
    if launch.size == 0 or channel_idx < 0 or channel_idx >= launch.size:
        return None
    launch_dbm = np.asarray(watt2dBm(np.maximum(launch, 1e-18)), dtype=float).reshape(-1)
    cut_dbm = float(launch_dbm[channel_idx])
    if np.allclose(launch_dbm, cut_dbm, atol=1e-9, rtol=0.0):
        return f"Pch={cut_dbm:.2f} dBm"
    return f"Pch(CUT)={cut_dbm:.2f} dBm"


def _format_scaling_plot_title(
    system: System,
    *,
    sweep_axis: str,
    channel_idx: int,
    band_label: str,
    launch_powers_w: np.ndarray | None = None,
    channel_freq_thz: float | None = None,
    channel_freq_range_thz: tuple[float, float] | None = None,
) -> str:
    head = [f"Center channel idx={int(channel_idx)}"]
    if band_label:
        head.append(f"band={band_label}")
    if channel_freq_range_thz is not None:
        f_lo, f_hi = channel_freq_range_thz
        head.append(f"f={float(f_lo):.3f}-{float(f_hi):.3f} THz")
    elif channel_freq_thz is not None:
        head.append(f"f={float(channel_freq_thz):.3f} THz")

    fixed = []
    if sweep_axis != "length" and system.fiber_length is not None:
        fixed.append(f"L={float(system.fiber_length) * 1e-3:.1f} km")
    if sweep_axis != "baud" and system.baud_rate is not None:
        fixed.append(f"R={float(system.baud_rate) * 1e-9:.1f} GBd")
    if sweep_axis != "spacing" and getattr(system.wdm, "spacing", None) is not None:
        fixed.append(f"df={float(system.wdm.spacing) * 1e-9:.2f} GHz")
    fixed.append(_format_pulse_title_token(system))
    launch_token = _format_launch_power_title_token(
        launch_powers_w,
        channel_idx=channel_idx,
    )
    if launch_token is not None:
        fixed.append(launch_token)

    title = ", ".join(head)
    if fixed:
        title += "\n" + ", ".join(fixed)
    return title
