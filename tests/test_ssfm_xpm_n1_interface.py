from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.methods.ssfm_interface import (
    _build_scalar_xpm_runtime_config,
    _paired_trial_powers,
    compute_scalar_ssfm_xpm_n1,
    fit_scalar_ssfm_xpm_sweep,
)


class _WDM:
    def frequency_grid(self):
        return np.array([193.1e12, 193.15e12])


def _system():
    fiber = SimpleNamespace(
        beta2=20e-27,
        gamma=1.3e-3,
        effective_area=80e-12,
        length=50e3,
    )
    return SimpleNamespace(
        pulse=SimpleNamespace(baud_rate=32e9),
        wdm=_WDM(),
        fiber=fiber,
        fiber_length=fiber.length,
        effective_area=fiber.effective_area,
    )


def _trial(seed, power, symbol_hash="same"):
    return {
        "seed": seed,
        "cut_tx_symbol_sha256": symbol_hash,
        "n_symbols_used": 100,
        "center_nli_power_w": power,
    }


def test_scalar_runtime_overrides_incompatible_template_values(tmp_path):
    template = {
        "modulation": {
            "symbol_rate_gbd": 99.0,
            "samples_per_symbol": 2,
            "pulse_shape": "rrc",
        },
        "wdm": {"n_channels": 5, "power_per_channel_w": 9.0},
        "fiber": {"nonlinearity": 7.0, "loss_db_per_m": 1.0},
        "solver": {"propagation_backend": "gnlse"},
        "output": {"show_plot": True},
    }

    cfg = _build_scalar_xpm_runtime_config(
        template_cfg=template,
        system=_system(),
        cut_power_w=0.6e-3,
        interferer_power_w=0.4e-3,
        channel_idx=0,
        interferer_idx=1,
        out_dir=tmp_path,
        scenario_tag="case",
        n_trials=3,
        rng_seed=17,
        n_symbols=256,
        samples_per_symbol=8,
        max_step_m=500.0,
        max_nonlinear_phase_deg=0.1,
    )

    assert cfg["modulation"]["symbol_rate_gbd"] == 32.0
    assert cfg["modulation"]["symbol_distribution"] == "circular_complex_gaussian"
    assert cfg["modulation"]["pulse_shape"] == "nyquist_rect"
    assert cfg["wdm"]["n_channels"] == 3
    assert cfg["wdm"]["channel_powers_w"] == [0.0, 0.6e-3, 0.4e-3]
    assert "power_per_channel_w" not in cfg["wdm"]
    assert cfg["fiber"]["nonlinearity"] == pytest.approx(1.3e-3)
    assert cfg["fiber"]["loss_db_per_m"] == 0.0
    assert cfg["solver"]["propagation_backend"] == "custom_ssfm"
    assert cfg["output"]["save_plots"] is False


def test_scalar_runtime_accepts_virtual_frequencies_and_higher_dispersion(tmp_path):
    cfg = _build_scalar_xpm_runtime_config(
        template_cfg={},
        system=_system(),
        cut_power_w=0.6e-3,
        interferer_power_w=0.4e-3,
        channel_idx=0,
        interferer_idx=1,
        out_dir=tmp_path,
        scenario_tag="virtual",
        n_trials=2,
        rng_seed=1,
        n_symbols=128,
        samples_per_symbol=8,
        max_step_m=100.0,
        max_nonlinear_phase_deg=0.2,
        target_frequency_hz=200e12,
        interferer_frequency_hz=200.05e12,
        dispersion_betas_ps_per_m=[-0.02, 1e-4, 2e-6],
    )

    assert cfg["wdm"]["channel_spacing_thz"] == pytest.approx(0.05)
    assert cfg["wdm"]["channel_powers_w"] == [0.0, 0.6e-3, 0.4e-3]
    assert cfg["fiber"]["betas"] == [-0.02, 1e-4, 2e-6]


def test_paired_payload_validation_rejects_different_cut_symbols():
    full = {"trial_results": {"trials": [_trial(1, 2.0, "full")]}}
    cut = {"trial_results": {"trials": [_trial(1, 1.0, "cut")]}}

    with pytest.raises(ValueError, match="CUT symbols"):
        _paired_trial_powers(full, cut)


def test_scalar_runner_uses_paired_difference_and_scalar_prefactor(tmp_path):
    def run_fn(path):
        is_full = Path(path).stem.endswith("_full")
        powers = [5e-9, 7e-9] if is_full else [1e-9, 1e-9]
        return {
            "trial_results": {
                "trials": [_trial(11 + i, power) for i, power in enumerate(powers)]
            }
        }

    result = compute_scalar_ssfm_xpm_n1(
        {
            "template_cfg": {},
            "out_dir": tmp_path,
            "run_fn": run_fn,
        },
        system=_system(),
        cut_power_w=0.6e-3,
        interferer_power_w=0.4e-3,
        channel_idx=0,
        interferer_idx=1,
        n_trials=2,
        rng_seed=11,
        sweep_tag="synthetic",
        n_symbols=256,
        samples_per_symbol=8,
    )

    np.testing.assert_allclose(result.excess_power_trials_w, [4e-9, 6e-9])
    denominator = 4.0 * (1.3e-3) ** 2 * 0.6e-3 * (0.4e-3) ** 2
    assert result.n1_ssfm_m2 == pytest.approx(5e-9 / denominator)
    assert (tmp_path / "ssfm" / "runtime" / "synthetic_full.toml").exists()
    assert (tmp_path / "ssfm" / "runtime" / "synthetic_cut_only.toml").exists()


def test_power_sweep_fit_recovers_trialwise_quadratic_coefficient():
    powers = np.array([1.0, 2.0, 3.0])
    gamma = 0.5
    cut_power = 2.0
    results = []
    for power in powers:
        results.append(
            SimpleNamespace(
                excess_power_trials_w=np.array(
                    [3.0 * power**2 + 1.0, 5.0 * power**2 - 2.0]
                ),
                gamma_w_inv_m=gamma,
            )
        )

    fit = fit_scalar_ssfm_xpm_sweep(
        results, powers, cut_power_w=cut_power
    )

    assert fit.slope_w_inv == pytest.approx(4.0)
    assert fit.intercept_w == pytest.approx(-0.5)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.n1_ssfm_m2 == pytest.approx(4.0 / (4.0 * gamma**2 * cut_power))
