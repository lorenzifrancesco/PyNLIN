from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.standalone_numerical.plot_fwm_single_tuple_scaling import (
    _fixed_d_normalized_mismatch,
    _parse_mu_list,
    _passband_island_polygon,
    _symmetric_random_variables,
    build_single_tuple_channels,
    compute_curves,
    evaluate_single_tuple_case,
    natural_mu,
    plot_curves,
    save_dataset,
    tuple_island_fields,
)


POSITIONS = np.array([0.0, -1.0, 2.0, 1.0])
REPEATED_POSITIONS = np.array([0.0, 1.0, 1.0, 1.0])


def test_tuple_island_highlights_energy_conserving_region():
    fields = tuple_island_fields(POSITIONS, 25.0 / 24.5, resolution=401)
    selected = np.asarray(fields["selected"], dtype=bool)
    X = np.asarray(fields["X"])
    Y = np.asarray(fields["Y"])
    center = np.unravel_index(
        np.argmin((X - fields["relative_a"]) ** 2 + (Y - fields["relative_b"]) ** 2),
        X.shape,
    )

    assert fields["energy_residual"] == 0.0
    assert np.any(selected)
    assert selected[center]
    assert (fields["relative_c"], fields["relative_a"], fields["relative_b"]) == (1.0, -1.0, 2.0)
    assert fields["xlim"][1] - fields["xlim"][0] == 10.0
    assert fields["ylim"][1] - fields["ylim"][0] == 10.0


def test_tuple_island_accepts_carrier_mismatch_with_spectral_overlap():
    fields = tuple_island_fields(np.array([0.0, -1.0, 3.0, 1.0]), 25.0 / 24.5)

    assert fields["energy_residual"] == 1.0
    assert fields["support_margin_over_baud"] > 0.0
    assert np.any(fields["selected"])


def test_exact_island_polygons_distinguish_lozenges_and_edge_triangles():
    half_bandwidth = 0.5 / (25.0 / 24.5)

    lozenge = _passband_island_polygon(0, 0, 0, half_bandwidth)
    upper_triangle = _passband_island_polygon(0, 0, 1, half_bandwidth)
    lower_triangle = _passband_island_polygon(0, 0, -1, half_bandwidth)

    assert len(lozenge) == 6
    assert len(upper_triangle) == 3
    assert len(lower_triangle) == 3


def test_tuple_island_rejects_tuple_without_spectral_overlap():
    with pytest.raises(ValueError, match="no positive-measure"):
        tuple_island_fields(np.array([0.0, -1.0, 4.0, 1.0]), 25.0 / 24.5)

    # At equality the four closed supports meet only at an edge point, which
    # has zero measure and therefore contributes zero to the MC integral.
    with pytest.raises(ValueError, match="no positive-measure"):
        tuple_island_fields(np.array([0.0, -1.0, 4.0, 1.0]), 1.0)


def test_single_tuple_builder_sets_gradient_and_detuning_scales():
    baud_rate = 24.5e9
    length = 100e3
    channels, geometry = build_single_tuple_channels(
        channel_positions=POSITIONS,
        spacing_over_baud=25.0 / 24.5,
        baud_rate=baud_rate,
        length=length,
        x_grad=12.0,
        mu=0.5,
    )

    assert geometry["x_grad"] == 12.0
    np.testing.assert_allclose(length * baud_rate * geometry["gradient_norm"], 12.0)
    np.testing.assert_allclose(channels.delta_beta0 / (baud_rate * geometry["gradient_norm"]), 0.5)
    np.testing.assert_allclose(geometry["x_phase"], 6.0)
    np.testing.assert_allclose(geometry["x_combined"], 18.0)
    assert geometry["center_frequency_residual"] == 0.0
    assert geometry["center_angular_frequency_residual"] == 0.0


def test_repeated_tuple_uses_curvature_scale():
    baud_rate = 24.5e9
    length = 100e3
    channels, geometry = build_single_tuple_channels(
        channel_positions=REPEATED_POSITIONS,
        spacing_over_baud=25.0 / 24.5,
        baud_rate=baud_rate,
        length=length,
        x_grad=12.0,
        mu=0.5,
    )

    assert geometry["scale_kind"] == "curvature"
    assert geometry["gradient_norm"] == 0.0
    assert geometry["x_scale"] == 12.0
    assert geometry["x_grad"] == 0.0
    np.testing.assert_allclose(geometry["x_curvature"], 12.0)
    np.testing.assert_allclose(length * baud_rate**2 * abs(channels.gvd_a), 12.0)
    constrained_center_mismatch = (
        channels.delta_beta0
        - channels.beta1_d * channels.delta_omega
        - 0.5 * channels.gvd_d * channels.delta_omega**2
    )
    np.testing.assert_allclose(
        constrained_center_mismatch / (baud_rate**2 * abs(channels.gvd_a)), 0.5
    )
    np.testing.assert_allclose(geometry["x_phase"], 6.0)
    np.testing.assert_allclose(geometry["x_combined"], 18.0)
    assert geometry["carrier_index_residual"] == 1.0
    assert geometry["support_margin_hz"] > 0.0
    assert natural_mu(REPEATED_POSITIONS, 25.0 / 24.5, -1.0) == 0.0
    np.testing.assert_array_equal(
        _parse_mu_list("natural,0,1", REPEATED_POSITIONS, 25.0 / 24.5, -1.0),
        np.array([0.0, 1.0]),
    )


def test_natural_mu_reproduces_unshifted_constant_beta2_tuple():
    mu = natural_mu(POSITIONS, 25.0 / 24.5, -1.0)
    channels, geometry = build_single_tuple_channels(
        channel_positions=POSITIONS,
        spacing_over_baud=25.0 / 24.5,
        baud_rate=24.5e9,
        length=100e3,
        x_grad=10.0,
        mu=mu,
        beta2_sign=-1.0,
    )
    omega = np.array(
        [channels.omega_d, channels.omega_a, channels.omega_b, channels.omega_c]
    )
    beta2 = channels.gvd_a
    unshifted_beta0 = 0.5 * beta2 * omega**2

    np.testing.assert_allclose(
        channels.delta_beta0,
        unshifted_beta0[1] + unshifted_beta0[2] - unshifted_beta0[3] - unshifted_beta0[0],
        rtol=1e-13,
    )
    np.testing.assert_allclose(mu, -5.234897265966793)
    np.testing.assert_allclose(geometry["x_phase"] / geometry["x_grad"], abs(mu))


def test_normalized_n_is_invariant_at_fixed_dimensionless_parameters():
    random_variables = _symmetric_random_variables(512, 1234)
    common = dict(
        channel_positions=POSITIONS,
        spacing_over_baud=25.0 / 24.5,
        x_grad=8.0,
        mu=1.5,
        n_samples=512,
        seed=1234,
        random_variables=random_variables,
    )
    first = evaluate_single_tuple_case(baud_rate=10e9, length=2e3, **common)
    second = evaluate_single_tuple_case(baud_rate=20e9, length=5e3, **common)

    np.testing.assert_allclose(first["normalized_n"], second["normalized_n"], rtol=2e-13)
    np.testing.assert_allclose(
        first["normalized_stderr"], second["normalized_stderr"], rtol=2e-13
    )
    np.testing.assert_allclose(
        first["mc_value"] / (2e3) ** 2,
        first["normalized_n"],
    )


def test_repeated_tuple_normalization_is_dimensionless():
    random_variables = _symmetric_random_variables(512, 1357)
    common = dict(
        channel_positions=REPEATED_POSITIONS,
        spacing_over_baud=25.0 / 24.5,
        x_grad=8.0,
        mu=1.5,
        n_samples=512,
        seed=1357,
        random_variables=random_variables,
    )
    first = evaluate_single_tuple_case(baud_rate=10e9, length=2e3, **common)
    second = evaluate_single_tuple_case(baud_rate=20e9, length=5e3, **common)

    np.testing.assert_allclose(first["normalized_n"], second["normalized_n"], rtol=2e-13)
    np.testing.assert_allclose(
        first["normalized_stderr"], second["normalized_stderr"], rtol=2e-13
    )


def test_cubic_zdw_model_has_requested_extra_zero_branch():
    positions = np.array([0.0, 1.0, 1.0, 2.0])
    spacing_over_baud = 1.0
    mu = natural_mu(
        positions,
        spacing_over_baud,
        -1.0,
        dispersion_model="cubic-zdw",
        extra_zero_sum=5.0,
    )
    channels, geometry = build_single_tuple_channels(
        channel_positions=positions,
        spacing_over_baud=spacing_over_baud,
        baud_rate=1.0,
        length=1.0,
        x_grad=1.0,
        mu=mu,
        dispersion_model="cubic-zdw",
        extra_zero_sum=5.0,
    )
    fields = {
        "X": np.array([[0.0, 2.0, 2.0, 1.0]]),
        "Y": np.array([[2.0, 0.0, 3.0, 1.0]]),
        "relative_a": 1.0,
        "relative_b": 1.0,
    }
    mismatch = _fixed_d_normalized_mismatch(
        fields,
        positions,
        spacing_over_baud,
        mu,
        -1.0,
        "cubic-zdw",
        5.0,
    )

    assert geometry["dispersion_model"] == "cubic-zdw"
    np.testing.assert_allclose(
        -2.0 * channels.gvd_d / (channels.beta3_d * 2.0 * np.pi),
        5.0,
    )
    np.testing.assert_allclose(mismatch[0, :3], 0.0, atol=1e-14)
    assert abs(mismatch[0, 3]) > 1e-3


def test_offset_carriers_conserve_frequency_sample_by_sample():
    positions = np.array([0.0, -1.0, 3.0, 1.0])
    baud_rate = 24.5e9
    random_variables = _symmetric_random_variables(1024, 2468)
    channels, geometry = build_single_tuple_channels(
        channel_positions=positions,
        spacing_over_baud=25.0 / 24.5,
        baud_rate=baud_rate,
        length=100e3,
        x_grad=10.0,
        mu=0.0,
    )
    ra, rb, rc = random_variables
    rd = ra + rb - rc + channels.delta_omega / baud_rate
    accepted = (rd > -np.pi) & (rd < np.pi)
    omega_a = channels.omega_a + ra * baud_rate
    omega_b = channels.omega_b + rb * baud_rate
    omega_c = channels.omega_c + rc * baud_rate
    omega_d = channels.omega_d + rd * baud_rate

    assert geometry["carrier_index_residual"] == 1.0
    assert geometry["support_margin_hz"] > 0.0
    assert np.any(accepted)
    np.testing.assert_allclose(
        omega_a[accepted] + omega_b[accepted] - omega_c[accepted] - omega_d[accepted],
        0.0,
        atol=2e-4,
    )


def test_single_tuple_dataset_and_plots(tmp_path):
    data = compute_curves(
        channel_positions=POSITIONS,
        spacing_over_baud=25.0 / 24.5,
        baud_rate=24.5e9,
        length=100e3,
        x_grid=np.geomspace(0.1, 10.0, 5),
        mu_values=np.array([0.0, 2.0]),
        n_samples=128,
        n_seeds=1,
        seed=4321,
        beta2_sign=-1.0,
    )
    npz_path, csv_path = save_dataset(data, tmp_path / "results")
    plot_paths = plot_curves(data, tmp_path / "plots")

    assert data["normalized_n"].size == 10
    assert npz_path.exists()
    assert csv_path.exists()
    assert all(path.exists() for path in plot_paths)
    saved = np.load(npz_path)
    assert saved["mc_value_convention"].item() == "N_times_T_squared"
    assert saved["vertical_normalization"].item() == "mc_value/L^2 = N*T^2/L^2"
    assert saved["scale_kind"].item() == "gradient"
    assert np.all(saved["center_frequency_residual"] == 0.0)
    assert np.all(saved["center_angular_frequency_residual"] == 0.0)
    assert np.all(saved["support_margin_hz"] > 0.0)


def test_zero_mismatch_edge_does_not_break_efficiency_overlay(tmp_path):
    data = compute_curves(
        channel_positions=np.array([0.0, 1.0, 0.0, 1.0]),
        spacing_over_baud=25.0 / 24.5,
        baud_rate=24.5e9,
        length=100e3,
        x_grid=np.geomspace(0.1, 10.0, 3),
        mu_values=np.array([0.0, 10.0]),
        n_samples=32,
        n_seeds=1,
        seed=2468,
        beta2_sign=-1.0,
    )

    assert np.all(data["normalized_n"][data["mu"] == 0.0] > 0.0)
    assert all(path.exists() for path in plot_curves(data, tmp_path))
