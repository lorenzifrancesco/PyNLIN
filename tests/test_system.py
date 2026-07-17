from pathlib import Path

import numpy as np
import pytest

from pynlin.pulses import PulseType
from pynlin.methods.td.cache import (
    s1_ref_nlin_curve_path,
    s2a_lo_timeint_path,
    s2b_lo_extrema_path,
    s3_chan_nlin_td_path,
    s3_pair_nlin_kernel_path,
)
from pynlin.methods.td.reference_curves import load_s1_ref_dataset, save_s1_ref_nlin_curve
from pynlin.system import System, _interp_with_linear_extrapolation


def _input_path(name: str) -> Path:
    """Return an absolute path to a file under the input directory."""
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "input" / name


def test_system_from_active_studies_toml():
    cfg_path = _input_path("studies.toml")
    system = System.from_toml(cfg_path)

    assert system.fiber.fiber_type == "SM"
    assert system.fiber.n_modes == 1
    assert system.wdm.num_channels == 600
    assert system.pulse_config.type == PulseType.NYQUIST
    assert system.pulse.baud_rate == pytest.approx(24.5e9)
    assert system.amplification.n_pumps == 3
    assert system.amplification.raman_gain == pytest.approx(0.0)
    assert system.pump_specs and len(system.pump_specs) == 3

    # Numerics are embedded in [numerics] in the active studies TOML.
    assert system.numerics is not None
    assert system.numerics.gvd == pytest.approx(-20e-27)


def test_active_studies_toml_beta_grids_are_finite():
    system = System.from_toml(_input_path("studies.toml"))
    freqs = system.wdm.frequency_grid()

    beta1, beta2 = system.beta_grids(freqs=freqs)

    assert beta1.shape == (system.n_modes, freqs.size)
    assert beta2.shape == (system.n_modes, freqs.size)
    assert np.all(np.isfinite(beta1))
    assert np.all(np.isfinite(beta2))


def test_profile_interpolation_extrapolates_linearly_at_both_ends():
    result = _interp_with_linear_extrapolation(
        np.array([-1.0, 0.5, 3.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 3.0, 7.0]),
    )

    assert np.allclose(result, [-1.0, 2.0, 11.0])


def test_pcfm_struct_merges_bands_and_singular_band_section():
    system = System.from_toml(_input_path("studies.toml"))

    assert system.n_channels == 600
    assert hasattr(system.wdm, "_band_slices")
    assert set(system.wdm._band_slices) == {"S", "C", "L"}


def test_stage_labelled_cache_names():
    assert s1_ref_nlin_curve_path(pulse_shape="nyquist", mode="perfect", gvda=0.0, gvdb=0.0).name == (
        "s1_ref_nlin_curve_nyquist_perfect_gvda0.0_gvdb0.0.npz"
    )
    assert s2a_lo_timeint_path(ipulse=1, m_lo=2).name == "s2a_lo_timeint_nyquist_m2.npz"
    assert s2b_lo_extrema_path(ipulse=1, m_lo_truncation=3, fiber_length=100e3, lld_max=25.0).name == (
        "s2b_lo_extrema_nyquist_mtrunc3_L100.0km_lldmax25.00.npz"
    )
    assert s3_pair_nlin_kernel_path(
        ipulse=1,
        fiber_type="smf",
        br_hz=50e9,
        n_ch=150,
        fiber_length=100e3,
        spacing_hz=100e9,
        disp_tag="abc123",
    ).name == "s3_pair_nlin_kernel_ipulse1_smf_L100.0km_br50p000GHz_n150_sp100p000GHz_dispabc123.npy"
    assert s3_chan_nlin_td_path(tag="flat_profile", use_kappa=False, use_x_mode=True).name == (
        "s3_chan_nlin_td_flat_profile_k0_x1.npy"
    )


def test_s1_reference_dataset_stores_normalized_curve(tmp_path):
    path = tmp_path / "ref.npz"
    llw_grid = np.array([0.1, 1.0, 10.0])
    raw_curve = np.array([4.0, 8.0, 12.0])
    save_s1_ref_nlin_curve(
        path,
        llw_grid=llw_grid,
        raw_nlin_curve=raw_curve,
        fiber_length=2.0,
        baud_rate=5.0,
        pulse_shape="nyquist",
        mode="perfect",
        gvda=0.0,
        gvdb=0.0,
        n_samples_numeric=3,
    )
    dataset = load_s1_ref_dataset(path=path, mode="perfect", gvda=0.0, gvdb=0.0)
    assert np.allclose(dataset["llw_grid"], llw_grid)
    assert np.allclose(dataset["raw_nlin_curve"], raw_curve)
    assert np.allclose(dataset["ref_nlin_curve"], raw_curve / 100.0)
    assert dataset["x_norm"] == pytest.approx(10.0)
