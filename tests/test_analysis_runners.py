from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.config import MCMethodConfig, PCFMMethodConfig, ProfilesConfig, TDMethodConfig
from analysis.runners.methods import TDResult, run_mc, run_pcfm, run_td
from analysis.runtime.context import build_run_context
from analysis.system_nlin import _validate_smf_mmf_configs
from pynlin.constellation_stats import qam_mu0
from pynlin.fiber import SMFiber
from pynlin.methods.mc import compute_chi1_chi2, nlin_from_chi
from pynlin.methods.pcfm import PcfmConfig, compute_pcfm_nlin
from pynlin.pulses import GaussianPulse
from pynlin.system import System
from pynlin.wdm import Amplification, RegularWDM


def _minimal_system() -> System:
    fiber = SMFiber(beta2=20e-27, effective_area=80e-12, length=50e3)
    fiber.beta1 = 0.0
    wdm = RegularWDM(spacing=50e9, num_channels=2, center_frequency=193.1e12)
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=64, samples_per_symbol=8)
    amp = Amplification(n_pumps=0, raman_gain=0.0, pumps=None)
    return System(fiber=fiber, wdm=wdm, pulse=pulse, amplification=amp, source=Path("test.toml"))


def test_run_pcfm_matches_direct_kernel_on_flat_context(tmp_path):
    system = _minimal_system()
    profiles = ProfilesConfig(mode="flat", path=tmp_path / "flat.npy", launch_csv=None)
    context = build_run_context(
        system=system,
        config_path=Path("test.toml"),
        out_dir=tmp_path,
        profiles=profiles,
        cache_prefix="pcfm_direct_compare",
    )
    method_cfg = PCFMMethodConfig(
        mode="recompute",
        numeric_sci=False,
        numeric_xci=False,
        degree=3,
    )

    result = run_pcfm(context, method_cfg, cache_scope="unit")
    direct = compute_pcfm_nlin(
        system=system,
        profile_path=context.profile_path,
        launch_powers_w=context.launch_powers_w,
        config=PcfmConfig(degree=3, use_numeric_sci=False, use_numeric_xci=False),
    )

    assert np.allclose(result.launch_nlin_w, direct)
    assert np.allclose(result.output_nlin_w, direct * context.output_over_launch_ratio)


def test_run_pcfm_emits_eq18_xci_when_enabled(tmp_path):
    system = _minimal_system()
    profiles = ProfilesConfig(mode="flat", path=tmp_path / "flat.npy", launch_csv=None)
    context = build_run_context(
        system=system,
        config_path=Path("test.toml"),
        out_dir=tmp_path,
        profiles=profiles,
        cache_prefix="pcfm_eq18",
    )
    method_cfg = PCFMMethodConfig(
        mode="recompute",
        numeric_sci=False,
        numeric_xci=False,
        eq18_xci=True,
        degree=3,
    )

    result = run_pcfm(context, method_cfg, cache_scope="unit")

    assert result.eq18_xci_launch_w is not None
    assert result.eq18_xci_output_w is not None
    assert result.eq18_xci_output_w.shape == result.output_nlin_w.shape


def test_run_mc_matches_direct_chi_reconstruction_on_flat_context(tmp_path):
    system = _minimal_system()
    profiles = ProfilesConfig(mode="flat", path=tmp_path / "flat.npy", launch_csv=None)
    context = build_run_context(
        system=system,
        config_path=Path("test.toml"),
        out_dir=tmp_path,
        profiles=profiles,
        cache_prefix="mc_direct_compare",
    )
    td_cfg = TDMethodConfig(mode="cached", use_kappa=False, use_x_mode=True, exclude_self_channel=True)
    collision_coeffs = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=float)
    td_result = TDResult(
        collision_coeffs=collision_coeffs,
        launch_nlin_w=np.zeros(2, dtype=float),
        output_nlin_w=np.zeros(2, dtype=float),
    )

    result = run_mc(context, td_result, td_cfg, MCMethodConfig(mode="recompute"))
    chi1, chi2, prefactor = compute_chi1_chi2(
        system,
        collision_coeffs,
        context.launch_powers_w,
        use_kappa=td_cfg.use_kappa,
        use_x_mode=td_cfg.use_x_mode,
        exclude_self_channel=td_cfg.exclude_self_channel,
    )
    direct_nlin = nlin_from_chi(chi1, chi2, prefactor, qam_mu0(16))

    assert np.allclose(result.chi1, chi1)
    assert np.allclose(result.chi2, chi2)
    assert np.allclose(result.prefactor, prefactor)
    assert np.allclose(result.nlin_16qam_output_w, direct_nlin * context.output_over_launch_ratio)


def test_run_td_passes_time_integral_backend(monkeypatch, tmp_path):
    system = _minimal_system()
    profiles = ProfilesConfig(mode="flat", path=tmp_path / "flat.npy", launch_csv=None)
    context = build_run_context(
        system=system,
        config_path=Path("test.toml"),
        out_dir=tmp_path,
        profiles=profiles,
        cache_prefix="td_backend",
    )
    seen = {}

    def fake_collision_coeffs_system_uwb(*args, **kwargs):
        seen["backend"] = kwargs["time_integral_backend"]
        return np.ones((1, 2, 1, 2), dtype=float)

    monkeypatch.setattr(
        "analysis.runners.methods.collision_coeffs_system_uwb",
        fake_collision_coeffs_system_uwb,
    )

    result = run_td(
        context,
        TDMethodConfig(mode="recompute", time_integral_backend="x0mm_fft"),
        cache_scope="unit",
    )

    assert seen["backend"] == "x0mm_fft"
    assert result.collision_coeffs.shape == (1, 2, 1, 2)


def test_system_nlin_rejects_same_smf_mmf_config_path(tmp_path):
    system = _minimal_system()
    config_path = tmp_path / "same.toml"
    config_path.write_text("")

    with pytest.raises(ValueError, match="distinct config files"):
        _validate_smf_mmf_configs(config_path, config_path, system, system)
