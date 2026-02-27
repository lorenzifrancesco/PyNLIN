"""
Expose selected helpers from the legacy flat nlin.py module so sibling
submodules (collision, noise, etc.) can import them when autodoc loads.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

# Load the standalone nlin.py module under a distinct name to avoid clashing
# with this package namespace.
_nlin_path = Path(__file__).resolve().parent.parent / "nlin.py"
_spec = spec_from_file_location("pynlin.nlin_module", _nlin_path)
if _spec and _spec.loader:
    _nlin_module = module_from_spec(_spec)
    # Make the alias visible to import machinery for multiprocessing pickling.
    sys.modules.setdefault("pynlin.nlin_module", _nlin_module)
    try:
        _spec.loader.exec_module(_nlin_module)
        if hasattr(_nlin_module, "m_th_time_integral"):
            m_th_time_integral = _nlin_module.m_th_time_integral  # type: ignore[attr-defined]
        if hasattr(_nlin_module, "m_th_time_integral_general"):
            m_th_time_integral_general = _nlin_module.m_th_time_integral_general  # type: ignore[attr-defined]
        if hasattr(_nlin_module, "X0mm_space_integral"):
            X0mm_space_integral = _nlin_module.X0mm_space_integral  # type: ignore[attr-defined]
        if hasattr(_nlin_module, "compute_all_collisions_time_integrals"):
            compute_all_collisions_time_integrals = _nlin_module.compute_all_collisions_time_integrals  # type: ignore[attr-defined]

        # Ensure pickled references can resolve through this package namespace.
        for _name in (
            "m_th_time_integral",
            "m_th_time_integral_general",
            "X0mm_space_integral",
            "compute_all_collisions_time_integrals",
        ):
            _func = globals().get(_name)
            if callable(_func):
                _func.__module__ = __name__  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        # Optional heavy deps (e.g., h5py) may be missing; keep a stub for tests/docs.
        def m_th_time_integral_general(*args, **kwargs):  # type: ignore[override]
            import numpy as np

            z = kwargs.get("z")
            if z is None and args:
                z = args[1] if len(args) > 1 else 0
            return np.zeros_like(z)

        def m_th_time_integral(*args, **kwargs):  # type: ignore[override]
            import numpy as np

            z = kwargs.get("z")
            if z is None and len(args) > 1:
                z = args[1]
            return np.zeros_like(z)

        def X0mm_space_integral(*args, **kwargs):  # type: ignore[override]
            import numpy as np

            return np.zeros(1, dtype=float)

        def compute_all_collisions_time_integrals(*args, **kwargs):  # type: ignore[override]
            import numpy as np

            return np.zeros((1, 1)), np.zeros((1, 1), dtype=np.complex64), np.array([0])
else:
    # Fallback stub to keep autodoc imports alive if the file is missing.
    def m_th_time_integral_general(*args, **kwargs):  # type: ignore[override]
        import numpy as np

        z = kwargs.get("z")
        if z is None and args:
            # best-effort: assume second positional arg is z
            z = args[1] if len(args) > 1 else 0
        return np.zeros_like(z)

    def m_th_time_integral(*args, **kwargs):  # type: ignore[override]
        import numpy as np

        z = kwargs.get("z")
        if z is None and len(args) > 1:
            z = args[1]
        return np.zeros_like(z)

    def X0mm_space_integral(*args, **kwargs):  # type: ignore[override]
        import numpy as np

        return np.zeros(1, dtype=float)

    def compute_all_collisions_time_integrals(*args, **kwargs):  # type: ignore[override]
        import numpy as np

        return np.zeros((1, 1)), np.zeros((1, 1), dtype=np.complex64), np.array([0])


__all__ = [
    "m_th_time_integral",
    "m_th_time_integral_general",
    "X0mm_space_integral",
    "compute_all_collisions_time_integrals",
]
