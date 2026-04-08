# analysis entrypoints

This page covers the main standalone analysis scripts that are intended to be
run directly from the repository root rather than imported as library modules.

## PCFM workflow drivers

- `analysis/pcfm_nlin.py`: end-to-end PCFM/TD workflow runner combining profile handling, TD aggregation, PCFM/GN evaluation, plotting, and CSV export.
- `analysis/pcfm_scaling.py`: interactive or CLI launcher for the main PCFM scaling sweeps.
- `analysis/pcfm_baud_scaling.py`: baud-rate scaling study for TD vs. PCFM quantities.
- `analysis/pcfm_channel_spacing_scaling.py`: channel-spacing sweep with WDM-grid rebuilding and asymptotic comparisons.
- `analysis/pcfm_spacing_scaling.py`: spacing-ratio sweep variant for regular and irregular WDM configurations.
- `analysis/pcfm_length_scaling.py`: span-length scaling study.
- `analysis/pcfm_debug_scan.py`: diagnostic scan for normalization, dispersion, and CUT/interferer checks.
- `analysis/pcfm_expansion.py`: exact-vs-asymptotic low-`B/abs(Delta f)` expansion plots saved under `media/pcfm/`.

## System-level studies

- `analysis/uwb_nlin.py`: UWB SMF case-study driver, including Raman-profile generation and GSNR/NLIN plotting.
- `analysis/system_nlin.py`: multimode/system TD-NLIN workflow and timing instrumentation.
- `analysis/psd_system.py`: PSD, bispectrum, and fourth-order proxy diagnostics for pulse/constellation settings.
- `analysis/dar_nlin.py`: simplified Dar et al. benchmark comparing TD and PCFM on a compact SMF system.
- `analysis/benchmark.py`: runtime benchmarking for TD-NLIN precompute and reduction stages.

## Additional utilities

- `analysis/fitting_power_profiles.py`: profile-fitting experiments and approximation comparisons.
- `analysis/fwm/*.py`: FWM brute-force, efficiency, and phase-matching studies.
- `analysis/raman/*.py`: Raman-response, plane/omega diagnostics, undepleted fits, and convergence monitors.
