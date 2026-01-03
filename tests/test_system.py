from pathlib import Path

import pytest

from pynlin.pulses import GaussianPulse, PulseType
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
