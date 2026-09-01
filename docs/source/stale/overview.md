# Overview

## What this project contains

- `src/pynlin`: the installable Python package with models, solvers, and
  utilities.
- `analysis/`: research scripts and utilities for nonlinearity
  estimation, Raman optimization, FWM studies, and plotting helpers.
- `docs/`: Sphinx documentation (this site). Build with
  `make -C docs html` from the repo root.

## How to navigate

- API docs for the installable package live under `api/` in the sidebar.
- Analysis scripts and helpers are documented under `analysis_api/`;
  these are importable but may depend on optional scientific packages.
- For figures or numerical experiments, look at the CLI scripts (usually
  guarded by `if __name__ == "__main__":`) or the plotting modules.

## Development notes

- Keep imports side-effect free; put runtime work under `__main__`
  blocks.
- When adding heavy dependencies needed only for analysis, consider
  mocking them in `docs/source/conf.py` for autodoc.
- Use `python -m ...` from the repo root (with the root on `PYTHONPATH`)
  to run analysis modules cleanly.
