# analysis entrypoints

This page covers the main standalone analysis scripts that are intended to be
run directly from the repository root rather than imported as library modules.

## PCFM workflow drivers

- `analysis/studies.py`: canonical TOML-driven study runner for full-system, subset, and sweep studies.
- `analysis/cli.py`: compatibility entrypoint that dispatches to the named studies runner.
- PCFM scaling studies are now expressed as `[studies.<name>]` entries in TOML using `type = "sweep"`.

## System-level studies

- `analysis/uwb_nlin.py`: UWB SMF case-study driver, including Raman-profile generation and GSNR/NLIN plotting.
- `analysis/system_nlin.py`: multimode/system TD-NLIN workflow and timing instrumentation.
- `analysis/psd_system.py`: PSD, bispectrum, and fourth-order proxy diagnostics for pulse/constellation settings.
- `analysis/mc_nlin.py`: MC method benchmark: plot chi1/chi2 decomposition results from studies.
- `analysis/benchmark.py`: runtime benchmarking for TD-NLIN precompute and reduction stages.

## Additional utilities

- `analysis/fitting_power_profiles.py`: profile-fitting experiments and approximation comparisons.
- `analysis/fwm/*.py`: FWM brute-force, efficiency, and phase-matching studies.
- `analysis/raman/*.py`: Raman-response, plane/omega diagnostics, undepleted fits, and convergence monitors.
