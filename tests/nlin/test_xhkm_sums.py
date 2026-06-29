import numpy as np
import pytest

from pynlin.methods.td.reference_curves import (
    load_xhkm_sum_reference_curves,
    save_xhkm_sum_reference_curves,
)
from pynlin.methods.td.xhkm_sums import compute_xhkm_sums


def test_xhkm_sums_single_2pc_sector():
    X = np.array([[[3.0 + 4.0j, 1.0 + 0.0j]]])

    sums = compute_xhkm_sums(X, h_values=np.array([0]), r_values=np.array([0]), m_values=np.array([-1, 0]))

    assert sums.n1 == pytest.approx(26.0)
    assert sums.n2 == pytest.approx(26.0)
    assert sums.n_2pc == pytest.approx(26.0)
    assert sums.n_3pc_total == pytest.approx(0.0)
    assert sums.n_3pca == pytest.approx(0.0)
    assert sums.n_3pcb == pytest.approx(0.0)
    assert sums.n_3pc_other == pytest.approx(0.0)
    assert sums.n_3pc_k_eq_m == pytest.approx(0.0)
    assert sums.n_4pc == pytest.approx(0.0)
    assert sums.n_k_neq_m == pytest.approx(0.0)
    assert sums.n2_over_n1 == pytest.approx(1.0)


def test_xhkm_sums_separates_n2_and_k_neq_m():
    h_values = np.array([0, 1])
    r_values = np.array([0, 2])
    m_values = np.array([-1, 0])
    X = np.zeros((2, 2, 2), dtype=complex)
    X[0, 0, 0] = 2.0  # 2PC
    X[1, 0, 1] = 3.0  # k=m 3PC
    X[0, 1, 0] = 4.0  # 3PCa: h=0, k!=m

    sums = compute_xhkm_sums(X, h_values, r_values, m_values)

    assert sums.n1 == pytest.approx(29.0)
    assert sums.n2 == pytest.approx(13.0)
    assert sums.n_2pc == pytest.approx(4.0)
    assert sums.n_3pc_k_eq_m == pytest.approx(9.0)
    assert sums.n_3pca == pytest.approx(16.0)
    assert sums.n_3pcb == pytest.approx(9.0)
    assert sums.n_3pc_other == pytest.approx(0.0)
    assert sums.n_4pc == pytest.approx(0.0)
    assert sums.n_k_neq_m == pytest.approx(16.0)
    assert sums.n_3pc_total == pytest.approx(25.0)


def test_xhkm_sums_separates_other_3pc_and_4pc():
    h_values = np.array([0, 1, 2])
    r_values = np.array([-1, 0, 2])
    m_values = np.array([-1, 0])
    X = np.zeros((3, 3, 2), dtype=complex)
    X[1, 2, 0] = 2.0  # 3PC other: h=k, h!=0, k!=m
    X[2, 0, 1] = 3.0  # 4PC: h=2, k=-1, m=0

    sums = compute_xhkm_sums(X, h_values, r_values, m_values)

    assert sums.n_3pc_other == pytest.approx(4.0)
    assert sums.n_4pc == pytest.approx(9.0)
    assert sums.n_3pc_total == pytest.approx(4.0)


def test_xhkm_sums_validates_indices_and_shape():
    X = np.zeros((1, 1, 1), dtype=complex)

    with pytest.raises(ValueError, match="exactly one zero"):
        compute_xhkm_sums(X, h_values=np.array([1]), r_values=np.array([0]), m_values=np.array([0]))
    with pytest.raises(ValueError, match="exactly one zero"):
        compute_xhkm_sums(X, h_values=np.array([0]), r_values=np.array([0, 0]), m_values=np.array([0]))
    with pytest.raises(ValueError, match="does not match"):
        compute_xhkm_sums(X, h_values=np.array([0]), r_values=np.array([0]), m_values=np.array([0, 1]))


def test_xhkm_sum_reference_curves_roundtrip(tmp_path):
    path = tmp_path / "xhkm.npz"
    llw = np.array([1.0, 2.0])
    save_xhkm_sum_reference_curves(
        path,
        llw_grid=llw,
        raw_n1=np.array([4.0, 8.0]),
        raw_n2=np.array([2.0, 4.0]),
        raw_n_2pc=np.array([1.0, 2.0]),
        raw_n_3pc_total=np.array([2.5, 5.0]),
        raw_n_3pca=np.array([0.5, 1.0]),
        raw_n_3pcb=np.array([1.0, 2.0]),
        raw_n_3pc_other=np.array([1.0, 2.0]),
        raw_n_3pc_k_eq_m=np.array([1.0, 2.0]),
        raw_n_4pc=np.array([0.5, 1.0]),
        raw_n_k_neq_m=np.array([2.0, 4.0]),
        fiber_length=10.0,
        baud_rate=2.0,
        pulse_shape="gaussian",
        mode="perfect",
        gvda=0.0,
        gvdb=1.0,
        h_values=np.array([-1, 0, 1]),
        r_values=np.array([0, 1]),
        partial_collisions_margin=3,
        n_samples_numeric=2,
    )

    dataset = load_xhkm_sum_reference_curves(
        path,
        pulse_shape="gaussian",
        mode="perfect",
        gvda=0.0,
        gvdb=1.0,
        h_values=np.array([-1, 0, 1]),
        r_values=np.array([0, 1]),
    )

    assert np.allclose(dataset["ref_n1"], np.array([4.0, 8.0]) / 400.0)
    assert np.allclose(dataset["ref_n_3pca"], np.array([0.5, 1.0]) / 400.0)
    assert np.allclose(dataset["ref_n_4pc"], np.array([0.5, 1.0]) / 400.0)
    assert np.allclose(dataset["n2_over_n1"], np.array([0.5, 0.5]))
    assert dataset["time_integral_backend"] == "xhkm_fft"
    assert dataset["calculation"] == "prefactor_free_dar_n1_n2_from_xhkm"

    with pytest.raises(ValueError, match="incompatible h_values"):
        load_xhkm_sum_reference_curves(path, h_values=np.array([0]))
