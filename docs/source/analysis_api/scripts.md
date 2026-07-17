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
- `analysis/standalone_numerical/plot_xhkm_sum_curves.py`: plot prefactor-free
  Dar-style `N1`/`N2` curves computed from generic FFT `X[h,r,m]` collision tensors.
- `analysis/standalone_numerical/generate_xhkm_sum_curves.py`: generate the
  higher-support Xhkm demo datasets and plots, using `h,r=-5..5`, `m` margin
  `10`, and a small truncation annotation on each plot.
- `analysis/standalone_numerical/generate_xhkm_extended_range.py`: extend the
  Nyquist Xhkm curves up to $L/L_W=5000$ with adaptive $z$ resolution, and
  plot the comparison between no-dispersion and $L/L_D=1,5,10$ cases.
  The plot marks the region where the finite $z$ grid may not resolve the
  collision widths.

## Additional utilities

- `analysis/fitting_power_profiles.py`: profile-fitting experiments and approximation comparisons.
- `analysis/fwm/*.py`: FWM brute-force, efficiency, and phase-matching studies.
- `analysis/fwm/fwm_efficiency/*.py`: Raman-assisted FWM efficiency, plane/omega diagnostics, undepleted fits, and convergence monitors.
