from pathlib import Path

import numpy as np
import pytest

from pynlin.pulses import GaussianPulse, PulseType
from pynlin.nlin.cache_names import (
    s1_ref_nlin_curve_path,
    s2a_lo_timeint_path,
    s2b_lo_extrema_path,
    s3_chan_nlin_td_path,
    s3_pair_nlin_kernel_path,
)
from pynlin.nlin.reference_curves import load_s1_ref_dataset, save_s1_ref_nlin_curve
from pynlin.system import System
from pynlin.wdm import RegularWDM


def _input_path(name: str) -> Path:
    """Return an absolute path to a file under the input directory."""
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "input" / name


@pytest.mark.parametrize(
    "fname,fiber_type,n_modes",
    [("smf_struct.toml", "SM", 1), ("mmf_struct.toml", "MM", 4)],
)
def test_system_from_toml_structured(fname, fiber_type, n_modes):
    cfg_path = _input_path(fname)
    system = System.from_toml(cfg_path)

    # Fiber
    assert system.fiber.fiber_type == fiber_type
    assert system.fiber.n_modes == n_modes

    # WDM
    assert isinstance(system.wdm, RegularWDM)
    assert system.wdm.num_channels == 200

    # Pulse
    assert isinstance(system.pulse, GaussianPulse)
    assert system.pulse_config.type == PulseType.GAUSSIAN
    assert system.pulse.baud_rate == pytest.approx(33e9)

    # Amplification
    assert system.amplification.n_pumps == 2
    assert system.amplification.raman_gain == pytest.approx(0.0)
    assert system.pump_specs and len(system.pump_specs) == 2

    # Numerics should be picked up automatically from numerical_config.toml
    assert system.numerics is not None
    assert system.numerics.gvd == pytest.approx(-20e-27)


def test_constant_dispersion_beta1_extrapolates_beyond_csv_range():
    system = System.from_toml(_input_path("pcfm_struct.toml"))
    freqs = system.wdm.frequency_grid()

    beta1, _ = system.beta_grids(freqs=freqs)
    beta1 = beta1.reshape(-1)
    diffs = np.diff(beta1)

    # With constant beta2 and uniform channel spacing, beta1(f) should keep a
    # nonzero linear slope in frequency even below the CSV support instead of
    # clamping flat.
    assert np.all(np.abs(diffs) > 0.0)
    coeffs = np.polyfit(freqs, beta1, 1)
    fitted = np.polyval(coeffs, freqs)
    assert np.allclose(beta1, fitted, rtol=0.0, atol=1e-20)


def test_pcfm_struct_merges_bands_and_singular_band_section():
    system = System.from_toml(_input_path("pcfm_struct.toml"))

    assert system.n_channels == 150
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
        spacing_hz=100e9,
        disp_tag="abc123",
    ).name == "s3_pair_nlin_kernel_ipulse1_smf_br50p000GHz_n150_sp100p000GHz_dispabc123.npy"
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
