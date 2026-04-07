"""Compatibility Sphinx config for builds pointed at ``docs/``.

This repo keeps the canonical Sphinx sources in ``docs/source``. Some users
invoke ``sphinx-build`` with ``docs`` as the source directory, which makes
Sphinx look for ``docs/conf.py``. This shim loads the canonical config and
adjusts path-like settings so both entrypoints work.
"""

from __future__ import annotations

from pathlib import Path
import runpy


_HERE = Path(__file__).resolve().parent
_SOURCE_CONF = _HERE / "source" / "conf.py"

# Load the canonical Sphinx configuration with its own __file__ context.
globals().update(runpy.run_path(str(_SOURCE_CONF)))

# When Sphinx is pointed at docs/, docnames are rooted there rather than at
# docs/source/, so the root document and relative asset/template paths need to
# be rebased.
root_doc = "source/index"
templates_path = ["source/_templates"]
html_static_path = ["source/_static"]
exclude_patterns = ["build", "_build", "logs"]
