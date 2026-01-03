from pathlib import Path

from pynlin.system import System
from pynlin.utils import (
    NumericalConfig,
    OpticalBands,
    PulseShape,
    _toml_load,
    get_next_filename,
    load_nc_toml_to_struct,
    load_toml_to_struct,
    save_struct_to_toml,
)

def load_system(filepath: Path | str, numerical_path: Path | str | None = None) -> System:
    """
    Preferred entry point: load a full System object from TOML.

    This is a convenience shim for legacy callers that imported io_utils; it
    does not expose the old Config class defined here previously.
    """
    return System.from_toml(filepath, numerical_path=numerical_path)
