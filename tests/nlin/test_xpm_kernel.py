import time

import numpy as np

from pynlin.fiber import SMFiber
import pynlin.methods.td.collision as collision_mod
from pynlin.methods.td.collision import ensure_i_low_dataset
from pynlin.methods.td.reference_curves import load_s1_ref_dataset, save_s1_ref_nlin_curve
from pynlin.methods.td.time_integrals import (
    X0mm_space_integral,
    compute_all_collisions_time_integrals,
    m_th_time_integral_general,
)
from pynlin.methods.td.xpm_kernel import (
    classify_collision,
    collision_class_masks,
    compute_x0mm_fft,
    compute_x0mm_time_integrals_fft,
    compute_xpm_coefficient_direct,
    compute_xpm_kernel_fft,
    compute_xpm_time_integrand_direct,
    sample_correlation,
    sliding_overlap_fft,
)
from pynlin.pulses import GaussianPulse, NyquistPulse


def _relative_max_error(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    return float(np.max(np.abs(actual - expected)) / scale)


def test_sliding_overlap_uses_positive_delay_convention():
    dt = 0.25
    t = (np.arange(9) - 4) * dt
    C = np.exp(-t**2)
    delay = 2 * dt
    D = np.exp(-(t - delay) ** 2)

    lags, Q = sliding_overlap_fft(C, D, dt)

    sampled = sample_correlation(lags, Q, np.array([delay]))[0]
    direct = np.sum(C * np.interp(t - delay, t, D, left=0.0, right=0.0)) * dt

    np.testing.assert_allclose(sampled, direct, rtol=1e-12, atol=1e-12)


def test_x0mm_fft_matches_legacy_general_time_integral_gaussian():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=128, samples_per_symbol=32)
    z = np.linspace(0.0, 1000.0, 9)
    m_values = np.arange(-3, 4)
    dgd = 2.0e-13
    gvda = 0.0
    gvdb = 0.0

    fft_values = compute_x0mm_time_integrals_fft(
        pulse,
        z,
        m_values,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )
    legacy_values = np.stack(
        [m_th_time_integral_general(int(m), z, pulse, dgd, gvda, gvdb) for m in m_values]
    )

    assert _relative_max_error(fft_values.real, legacy_values.real) < 1e-2
    assert np.max(np.abs(fft_values.imag)) < 1e-8 * np.max(np.abs(fft_values.real))


def test_x0mm_fft_speedup_report(capsys):
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=128, samples_per_symbol=32)
    z = np.linspace(0.0, 1000.0, 9)
    m_values = np.arange(-15, 16)
    dgd = 2.0e-13
    gvda = 0.0
    gvdb = 0.0

    t0 = time.perf_counter()
    legacy_values = np.stack(
        [m_th_time_integral_general(int(m), z, pulse, dgd, gvda, gvdb) for m in m_values]
    )
    legacy_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    fft_values = compute_x0mm_time_integrals_fft(
        pulse,
        z,
        m_values,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )
    fft_s = time.perf_counter() - t0

    rel_err = _relative_max_error(fft_values.real, legacy_values.real)
    speedup = legacy_s / max(fft_s, 1e-12)
    with capsys.disabled():
        print(
            f"\nX0mm FFT-correlation speedup: "
            f"legacy_direct={legacy_s:.3f}s, fft_correlation={fft_s:.3f}s, "
            f"speedup={speedup:.2f}x, max_rel_error={rel_err:.3e}"
        )

    assert rel_err < 1e-2


def test_x0mm_fft_matches_legacy_space_integral_gaussian():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=128, samples_per_symbol=32)
    z = np.linspace(0.0, 1000.0, 11)
    m_values = np.arange(-2, 3)
    dgd = -1.5e-13
    gvda = 0.0
    gvdb = 0.0

    x_fft = compute_x0mm_fft(
        pulse,
        z,
        m_values,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )
    legacy_I = np.stack(
        [m_th_time_integral_general(int(m), z, pulse, dgd, gvda, gvdb) for m in m_values]
    )
    x_legacy = X0mm_space_integral(z, legacy_I, axis=1)

    assert _relative_max_error(x_fft.real, x_legacy.real) < 1e-2


def test_compute_all_collisions_fft_backend_matches_direct_gaussian():
    fiber = SMFiber(length=300.0, beta2=0.0)
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=128, samples_per_symbol=32)
    dgd = 2.0e-13
    gvda = 0.0
    gvdb = 0.0

    z_fft, I_fft, m_fft = compute_all_collisions_time_integrals(
        fiber,
        pulse,
        dgd,
        gvda,
        gvdb,
        use_multiprocessing=False,
        partial_collisions_margin=1,
        time_integral_backend="x0mm_fft",
    )
    I_direct = np.stack(
        [m_th_time_integral_general(int(m), z_fft, pulse, dgd, gvda, gvdb) for m in m_fft]
    )

    assert z_fft.ndim == 1
    assert I_fft.shape == (len(m_fft), z_fft.size)
    assert _relative_max_error(I_fft.real, I_direct.real) < 1e-2


def test_compute_all_collisions_fft_backend_matches_direct_nyquist_small_grid():
    fiber = SMFiber(length=100.0, beta2=0.0)
    pulse = NyquistPulse(baud_rate=10e9, num_symbols=220, samples_per_symbol=16, rolloff=0.0)
    dgd = 1.0e-13
    gvda = 0.0
    gvdb = 0.0

    z_fft, I_fft, m_fft = compute_all_collisions_time_integrals(
        fiber,
        pulse,
        dgd,
        gvda,
        gvdb,
        use_multiprocessing=False,
        partial_collisions_margin=1,
        time_integral_backend="x0mm_fft",
    )
    I_direct = np.stack(
        [m_th_time_integral_general(int(m), z_fft, pulse, dgd, gvda, gvdb) for m in m_fft]
    )

    assert I_fft.shape == (len(m_fft), z_fft.size)
    assert _relative_max_error(I_fft.real, I_direct.real) < 2e-2


def test_i_low_dataset_fft_backend_writes_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        collision_mod,
        "s2a_lo_timeint_path",
        lambda **kwargs: tmp_path / f"s2a_m{kwargs['m_lo']}.npz",
    )
    path = ensure_i_low_dataset(
        m_lo=1,
        ipulse=0,
        baud_rate=10e9,
        fiber_length=100.0,
        max_lld=0.05,
        recompute=True,
        time_integral_backend="x0mm_fft",
        n_lld_samples=3,
    )

    with np.load(path, allow_pickle=False) as dataset:
        assert set(["I_low_values", "lld_range", "z"]).issubset(dataset.files)
        assert dataset["I_low_values"].shape == (3, 3)
        assert str(dataset["time_integral_backend"].item()) == "x0mm_fft"


def test_s1_reference_dataset_records_time_integral_backend(tmp_path):
    path = tmp_path / "s1.npz"
    save_s1_ref_nlin_curve(
        path,
        llw_grid=np.array([1.0, 2.0]),
        raw_nlin_curve=np.array([3.0, 4.0]),
        fiber_length=10.0,
        baud_rate=2.0,
        pulse_shape="gaussian",
        mode="perfect",
        gvda=0.0,
        gvdb=0.0,
        n_samples_numeric=2,
        time_integral_backend="x0mm_fft",
    )

    dataset = load_s1_ref_dataset(
        path=path,
        pulse_shape="gaussian",
        mode="perfect",
        gvda=0.0,
        gvdb=0.0,
        time_integral_backend="x0mm_fft",
    )

    assert dataset["time_integral_backend"] == "x0mm_fft"
    try:
        load_s1_ref_dataset(
            path=path,
            pulse_shape="gaussian",
            mode="perfect",
            gvda=0.0,
            gvdb=0.0,
            time_integral_backend="direct",
        )
    except ValueError as exc:
        assert "time_integral_backend" in str(exc)
    else:
        raise AssertionError("backend mismatch was accepted")


def test_x0mm_fft_matches_legacy_general_time_integral_nyquist_small_grid():
    pulse = NyquistPulse(baud_rate=10e9, num_symbols=220, samples_per_symbol=16, rolloff=0.0)
    z = np.linspace(0.0, 500.0, 5)
    m_values = np.array([-1, 0, 1])
    dgd = 1.0e-13
    gvda = 0.0
    gvdb = 0.0

    fft_values = compute_x0mm_time_integrals_fft(
        pulse,
        z,
        m_values,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )
    legacy_values = np.stack(
        [m_th_time_integral_general(int(m), z, pulse, dgd, gvda, gvdb) for m in m_values]
    )

    assert _relative_max_error(fft_values.real, legacy_values.real) < 2e-2


def test_direct_generic_reduces_to_x0mm_time_integral():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=128, samples_per_symbol=32)
    z = np.linspace(0.0, 1000.0, 7)
    m = -2
    dgd = 2.0e-13
    gvda = 0.0
    gvdb = 0.0

    direct = compute_xpm_time_integrand_direct(
        pulse,
        z,
        h=0,
        k=m,
        m=m,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )
    legacy = m_th_time_integral_general(m, z, pulse, dgd, gvda, gvdb)

    assert _relative_max_error(direct.real, legacy.real) < 1e-2


def test_generic_xpm_fft_matches_direct_entries_gaussian():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=128, samples_per_symbol=32)
    z = np.linspace(0.0, 700.0, 7)
    h_values = np.array([-1, 0, 1])
    r_values = np.array([-1, 0, 1])
    m_values = np.array([-2, -1, 0, 1])
    dgd = 1.5e-13
    gvda = 0.0
    gvdb = 0.0

    result = compute_xpm_kernel_fft(
        pulse,
        z,
        h_values,
        r_values,
        m_values,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )

    for h, r, m in [(-1, 0, -1), (0, 1, -2), (1, -1, 1), (1, 1, 0)]:
        h_idx = int(np.where(h_values == h)[0][0])
        r_idx = int(np.where(r_values == r)[0][0])
        m_idx = int(np.where(m_values == m)[0][0])
        direct = compute_xpm_coefficient_direct(
            pulse,
            z,
            h=h,
            k=m + r,
            m=m,
            dgd=dgd,
            gvda=gvda,
            gvdb=gvdb,
        )

        np.testing.assert_allclose(result.X[h_idx, r_idx, m_idx], direct, rtol=2e-2, atol=1e-15)


def test_generic_xpm_reproduces_x0mm_slice():
    pulse = GaussianPulse(baud_rate=10e9, num_symbols=128, samples_per_symbol=32)
    z = np.linspace(0.0, 700.0, 7)
    m_values = np.arange(-2, 3)
    dgd = -1.2e-13
    gvda = 0.0
    gvdb = 0.0

    table = compute_xpm_kernel_fft(
        pulse,
        z,
        h_values=np.array([0]),
        r_values=np.array([0]),
        m_values=m_values,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )
    x0mm = compute_x0mm_fft(
        pulse,
        z,
        m_values,
        dgd=dgd,
        gvda=gvda,
        gvdb=gvdb,
    )

    np.testing.assert_allclose(table.X[0, 0, :], x0mm, rtol=1e-12, atol=1e-12)


def test_collision_classifier_basic_cases():
    assert classify_collision(0, 0, -3) == "2pc"
    assert classify_collision(0, 2, -3) == "3pc"  # h=0, k!=m, first kind
    assert classify_collision(2, 0, -3) == "3pc"  # h!=0, k=m, second kind
    assert classify_collision(2, 0, 2) == "3pc"  # h=m!=k edge case
    assert classify_collision(2, 3, -1) == "3pc"  # h=k!=m edge case
    assert classify_collision(2, 4, -1) == "4pc"


def test_collision_class_masks_shapes_and_exhaustive():
    h_values = np.array([-1, 0, 1])
    r_values = np.array([-1, 0, 1])
    m_values = np.array([-1, 0, 1])

    masks = collision_class_masks(h_values, r_values, m_values)

    assert set(masks) == {"2pc", "3pc", "4pc"}
    for mask in masks.values():
        assert mask.shape == (h_values.size, r_values.size, m_values.size)
    total = masks["2pc"].astype(int) + masks["3pc"].astype(int) + masks["4pc"].astype(int)
    assert np.all(total == 1)
