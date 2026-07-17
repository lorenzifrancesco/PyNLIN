from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.methods.ssfm_interface import (
    _build_scalar_fwm_runtime_config,
    compute_scalar_ssfm_fwm_excess,
    fit_scalar_ssfm_fwm_sweep,
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


def _builder_kwargs(tmp_path, **overrides):
    kwargs = dict(
        template_cfg={},
        system=_system(),
        idler_power_w=0.2e-3,
        pump_power_w=0.3e-3,
        source_power_w=0.4e-3,
        target_frequency_hz=200e12,
        spacing_hz=80e9,
        out_dir=tmp_path,
        scenario_tag="case",
        n_trials=3,
        rng_seed=17,
        n_symbols=256,
        samples_per_symbol=8,
        max_step_m=500.0,
        max_nonlinear_phase_deg=0.1,
        dispersion_betas_ps_per_m=[-0.02, 1e-4, 2e-6],
    )
    kwargs.update(overrides)
    return kwargs


def test_fwm_runtime_places_powers_on_upper_slots(tmp_path):
    cfg = _build_scalar_fwm_runtime_config(**_builder_kwargs(tmp_path))

    assert cfg["wdm"]["n_channels"] == 5
    assert cfg["wdm"]["channel_powers_w"] == [0.0, 0.0, 0.2e-3, 0.3e-3, 0.4e-3]
    assert cfg["wdm"]["channel_spacing_thz"] == pytest.approx(0.08)
    assert cfg["fiber"]["betas"] == [-0.02, 1e-4, 2e-6]
    assert cfg["fiber"]["loss_db_per_m"] == 0.0
    assert cfg["fiber"]["nonlinearity"] == pytest.approx(1.3e-3)
    assert cfg["solver"]["propagation_backend"] == "custom_ssfm"
    assert cfg["modulation"]["symbol_distribution"] == "circular_complex_gaussian"


def test_fwm_runtime_enforces_five_channel_nyquist_oversampling(tmp_path):
    cfg = _build_scalar_fwm_runtime_config(
        **_builder_kwargs(tmp_path, samples_per_symbol=2)
    )
    # Outermost slot offset is 2 spacings: 2*(2*80/32) + 1 = 11.
    assert cfg["modulation"]["samples_per_symbol"] == 11


def test_fwm_runtime_rejects_dark_idler_and_missing_betas(tmp_path):
    with pytest.raises(ValueError, match="idler power"):
        _build_scalar_fwm_runtime_config(**_builder_kwargs(tmp_path, idler_power_w=0.0))
    with pytest.raises(ValueError, match="dispersion betas"):
        _build_scalar_fwm_runtime_config(
            **_builder_kwargs(tmp_path, dispersion_betas_ps_per_m=None)
        )


def _case_powers(case):
    return {
        "full": [10e-9, 12e-9],
        "pump_control": [4e-9, 5e-9],
        "source_control": [3e-9, 4e-9],
        "idler_only": [1e-9, 1e-9],
    }[case]


def _stub_run_fn(calls=None):
    def run_fn(path):
        stem = Path(path).stem
        case = stem.split("_", 1)[1]
        if calls is not None:
            calls.append(case)
        return {
            "trial_results": {
                "trials": [
                    _trial(11 + i, power)
                    for i, power in enumerate(_case_powers(case))
                ]
            }
        }

    return run_fn


def _excess_kwargs(tmp_path, **overrides):
    kwargs = dict(
        system=_system(),
        idler_power_w=0.2e-3,
        pump_power_w=0.3e-3,
        source_power_w=0.4e-3,
        target_frequency_hz=200e12,
        spacing_hz=80e9,
        n_trials=2,
        rng_seed=11,
        sweep_tag="s00",
        n_symbols=256,
        samples_per_symbol=8,
        dispersion_betas_ps_per_m=[-0.02, 1e-4, 2e-6],
    )
    kwargs.update(overrides)
    return kwargs


def test_fwm_excess_uses_inclusion_exclusion_and_prefactor(tmp_path):
    ctx = {"template_cfg": {}, "out_dir": tmp_path, "run_fn": _stub_run_fn()}

    result = compute_scalar_ssfm_fwm_excess(ctx, **_excess_kwargs(tmp_path))

    np.testing.assert_allclose(result.excess_power_trials_w, [4e-9, 4e-9])
    denominator = 2.0 * (1.3e-3) ** 2 * (0.3e-3) ** 2 * 0.4e-3
    assert result.c_ssfm_m2 == pytest.approx(4e-9 / denominator)
    runtime_dir = tmp_path / "ssfm" / "runtime"
    for case in ("full", "pump_control", "source_control", "idler_only"):
        assert (runtime_dir / f"s00_{case}.toml").exists()


def test_fwm_excess_reuses_precomputed_idler_only_payload(tmp_path):
    calls = []
    ctx = {"template_cfg": {}, "out_dir": tmp_path, "run_fn": _stub_run_fn(calls)}
    idler_payload = {
        "trial_results": {
            "trials": [
                _trial(11 + i, power)
                for i, power in enumerate(_case_powers("idler_only"))
            ]
        }
    }

    result = compute_scalar_ssfm_fwm_excess(
        ctx, **_excess_kwargs(tmp_path, idler_only_payload=idler_payload)
    )

    assert "idler_only" not in calls
    assert len(calls) == 3
    np.testing.assert_allclose(result.excess_power_trials_w, [4e-9, 4e-9])


def test_fwm_excess_rejects_unpaired_control_symbols(tmp_path):
    def run_fn(path):
        case = Path(path).stem.split("_", 1)[1]
        symbol_hash = "other" if case == "source_control" else "same"
        return {
            "trial_results": {
                "trials": [_trial(11, 1e-9, symbol_hash)]
            }
        }

    ctx = {"template_cfg": {}, "out_dir": tmp_path, "run_fn": run_fn}
    with pytest.raises(ValueError, match="CUT symbols"):
        compute_scalar_ssfm_fwm_excess(ctx, **_excess_kwargs(tmp_path, n_trials=1))


def test_fwm_sweep_fit_recovers_trialwise_cubic_coefficient():
    scales = np.array([1.0, 2.0, 3.0])
    pump_powers = 2.0 * scales
    source_powers = 3.0 * scales
    gamma = 0.5
    results = []
    for pump, source in zip(pump_powers, source_powers, strict=True):
        x = pump**2 * source
        results.append(
            SimpleNamespace(
                excess_power_trials_w=np.array([3.0 * x + 1.0, 5.0 * x - 2.0]),
                gamma_w_inv_m=gamma,
            )
        )

    fit = fit_scalar_ssfm_fwm_sweep(results, pump_powers, source_powers)

    assert fit.slope_w_inv2 == pytest.approx(4.0)
    assert fit.intercept_w == pytest.approx(-0.5)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.c_ssfm_m2 == pytest.approx(4.0 / (2.0 * gamma**2))


def test_fwm_sweep_fit_rejects_degenerate_power_points():
    results = [
        SimpleNamespace(
            excess_power_trials_w=np.array([1.0, 2.0]), gamma_w_inv_m=0.5
        )
        for _ in range(2)
    ]
    with pytest.raises(ValueError, match="distinct"):
        fit_scalar_ssfm_fwm_sweep(
            results, np.array([1.0, 1.0]), np.array([2.0, 2.0])
        )
