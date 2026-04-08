# analysis.sandbox.smf-scripts package

This directory is a legacy sandbox workspace rather than a normal Python
package: its filesystem name contains a hyphen, so Sphinx autodoc cannot import
it under a valid module path. The scripts are still useful as reference and are
summarized below.

## Scripts

- `constellations.py`: modulation-order and constellation-noise exploratory plots.
- `plotter_hybrid.py`: hybrid signal/noise plotting utilities.
- `plotter_noise.py`: channel and power-sweep noise diagnostics.
- `plotter_profiles.py`: signal, ASE, and pump profile plotting helpers.
- `plotter_single_chan.py`: single-channel collision and noise visualizations.
- `pulse_energy.py`: pulse-energy and PSD-style sandbox checks.
