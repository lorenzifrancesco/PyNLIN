"""
Expose selected helpers from the legacy flat nlin.py module so sibling
submodules (collision, noise, etc.) can import them when autodoc loads.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

# Load the standalone nlin.py module under a distinct name to avoid clashing
# with this package namespace.
_nlin_path = Path(__file__).resolve().parent.parent / "nlin.py"
_spec = spec_from_file_location("pynlin.nlin_module", _nlin_path)
if _spec and _spec.loader:
    _nlin_module = module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_nlin_module)
        m_th_time_integral_general = _nlin_module.m_th_time_integral_general  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        # Optional heavy deps (e.g., h5py) may be missing; keep a stub for tests/docs.
        def m_th_time_integral_general(*args, **kwargs):  # type: ignore[override]
            import numpy as np

            z = kwargs.get("z")
            if z is None and args:
                z = args[1] if len(args) > 1 else 0
            return np.zeros_like(z)
else:
    # Fallback stub to keep autodoc imports alive if the file is missing.
    def m_th_time_integral_general(*args, **kwargs):  # type: ignore[override]
        import numpy as np

        z = kwargs.get("z")
        if z is None and args:
            # best-effort: assume second positional arg is z
            z = args[1] if len(args) > 1 else 0
        return np.zeros_like(z)


__all__ = [
    "m_th_time_integral_general",
]
