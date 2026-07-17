 # PyNLIN
 A Python package and scripts for the evaluation of nonlinear interference noise in fiber transmissions

# Installation

## End users
Just clone the repository and `pip install` it.
```bash
git clone https://github.com/geeanlooca/PyNLIN.git
cd PyNLIN
pip install .
```


## Development

### Set up the environment

#### Conda
I usually like to install the core numerical packages from conda directly, and let `pip` manage the rest of the dependencies.

```bash
conda create -n <env> python=3.10 --yes
conda activate <env>
conda install numpy scipy matplotlib h5py
```

#### `venv`

```bash
python -m <env>
source <env>/bin/activate # <env>\Scripts\activate.bat under Windows
```

### Install the package
For development purposes, the package should be installed in the editable mode. Changes you make to the package are immediatly reflected on the installed version and consequently on the scripts using the package.

From the root of the repository:
```bash
make install
```
or
```bash
pip install -e .[dev]
```

# Running Analysis Studies

The recommended way to run simulations and comparisons is the TOML-driven
studies runner:

```bash
python -m analysis.studies --config input/studies.toml
```

`analysis/studies.py` reads one TOML file containing both the physical system
definition and the analysis runtime configuration. The example
`input/studies.toml` is the canonical template. It defaults to flat profiles and
cached methods so the CLI is runnable as a smoke test; switch modes to
`recompute` when you intentionally want to regenerate expensive outputs.

## Configuration Layout

A studies TOML has two parts.

Physical system sections are loaded by `pynlin.system.System`:

```toml
[fiber]
[pulse]
[wdm]
[wdm.bands.<name>]
[amplification]
[[amplification.pumps]]
[numerics]
```

Analysis sections control profiles, methods, and named studies:

```toml
[profiles]
[methods.td]
[methods.pcfm]
[methods.gn]
[methods.mc]
[studies.<name>]
[studies.<name>.subset]
[studies.<name>.sweep]
```

Core method implementations live under `pynlin.methods`:

- `pynlin.methods.td`: time-domain collision-coefficient NLIN.
- `pynlin.methods.pcfm`: PCFM/GN kernels.
- `pynlin.methods.mc`: TD chi1/chi2 reconstruction and a **fullband** backend
  that evaluates 3PC/4PC tuples directly by Monte Carlo sampling. The
  historical `engine = "ssfm"` label selects TD reconstruction; it does not
  run an SSFM.

`analysis` is orchestration only: it resolves profiles, builds cache keys,
runs selected methods, writes CSV/NPY outputs, and creates plots.

## Profiles

The `[profiles]` section controls the signal power profile used by TD/PCFM/GN:

```toml
[profiles]
mode = "recompute"
path = "results/pcfm_power_profiles.npy"
launch_csv = "results/pcfm_launch_power.csv"
```

Supported `mode` values:

- `flat`: write a synthetic flat profile and use launch powers from the TOML or `launch_csv`.
- `cached`: load `path`; fail if it does not exist; use launch powers from the profile.
- `recompute`: compute Raman/ISRS profiles at `path`; use launch powers from the profile.
- `cached_no_profile_launch`: load `path`, but use TOML/CSV launch powers instead of profile launch powers.
- `recompute_no_profile_launch`: recompute `path`, but use TOML/CSV launch powers instead of profile launch powers.

Use `flat` for quick sweeps and regression checks. Use `recompute` when Raman
profiles should be part of the study.

## Method Configuration

Each method has a global configuration. Individual studies choose which methods
to run using `methods = [...]`.

```toml
[methods.td]
mode = "recompute"          # off | cached | recompute
exclude_self_channel = true
m_lo_truncation = 40
use_kappa = true
use_x_mode = true

[methods.pcfm]
mode = "recompute"          # off | cached | recompute
numeric_sci = true
numeric_xci = false
eq18_xci = true
degree = 9
include_mci = false

[methods.mc]
mode = "off"                # off | cached | recompute
engine = "ssfm"             # historical TD-reconstruction label | fullband
n_trials = 1
rng_seed = 1234
# Historical TD-reconstruction backend (engine = "ssfm"):
template = ""
n_channels = 5
# Fullband MC backend (engine = "fullband"):
channel_decimation = 1      # keep every N-th channel for MC targets
target_decimation = 1       # stride over decimated-grid targets
target_offset = 0           # starting offset for target selection
target_limit =              # max targets (null = all)
xpm_samples = 20000         # XPM tuple samples per target
fwm_samples = 10000         # FWM tuple samples per target
max_fwm_tuples_per_target = # cap on FWM tuple evaluations (null = no cap)
fwm_tuple_selection = "phase_proxy"  # phase_proxy | random
```

`cached` reuses existing method outputs when the cache path exists. `recompute`
forces regeneration. `off` disables the method even if a study lists it.

Two MC backends are available:

- **`engine = "ssfm"`** (historical label, not an SSFM): reconstructs
  modulation-dependent NLIN from TD collision coefficients via chi1/chi2. This
  backend **requires TD** — the
  study must include `"td"` in its methods list (or be run after TD). Suitable
  for systems where TD collision-coefficient computation is feasible.

- **`engine = "fullband"`**: evaluates XPM (3PC) and FWM (4PC) contributions
  directly by Monte Carlo sampling of interferer tuples on a decimated
  frequency grid. Does **not** require TD — works independently on very wide
  systems (OESCLU, >1000 channels) where TD collision coefficients are
  prohibitively expensive. The prefactor-free sums are converted to an
  approximate NLIN power via `γ²·(16/81)·Pj·Pavg·S`. Tuning `xpm_samples`,
  `fwm_samples`, `channel_decimation`, and `max_fwm_tuples_per_target`
  controls the accuracy vs. runtime trade-off.

For a real scalar split-step comparison of prefactor-free XPM $N_1$, use
`analysis/standalone_numerical/validate_ssfm_xpm_n1.py`. It runs paired
CUT-only/CUT-plus-interferer cases through `gnlse-python`, requires
$\Delta f\geq2B$ to exclude interferer-generated spectral overlap, fits the
low-power $P_i^2$ slope, and compares it directly with Dar MC.
`validate_ssfm_xpm_spectrum.py` repeats that experiment at sparse physical WDM
frequencies and writes the cache optionally overlaid by
`validate_fwm_mc_real_tuples.py`.

## Study Types

### Full System

`full_system` runs the study workflow over all channels. With `methods = ["mc"]`
and `engine = "fullband"`, it runs the fullband MC on the decimated grid,
interpolates to the full channel plan, estimates NLIN power and GSNR, and
produces plots with green diamond markers for the fullband MC data:

```toml
[studies.full_system]
type = "full_system"
methods = ["td", "pcfm"]
plot = true
```

This is the closest replacement for the old PCFM workflow scripts.

### Subset

`subset` evaluates selected CUT/interferer channels and writes compact CSV
summaries. When `engine = "fullband"`, the subset study runs the fullband MC
runner directly on the decimated grid and reports NLIN power per channel:

```toml
[studies.center_refined]
type = "subset"
methods = ["td", "pcfm", "mc"]
out_dir = "results/studies/center_refined"

[studies.center_refined.subset]
mode = "center_window"
center = "auto"
half_width = 2
include_sci = true
```

Subset modes:

- `center_window`: choose one CUT and include neighboring interferers.
- `explicit`: use explicit `cut_indices` and `interferer_indices`.

Example explicit subset:

```toml
[studies.custom_subset]
type = "subset"
methods = ["td", "pcfm"]
out_dir = "results/studies/custom_subset"

[studies.custom_subset.subset]
mode = "explicit"
cut_indices = [299]
interferer_indices = [297, 298, 299, 300, 301]
include_sci = false
```

### Sweep

`sweep` changes one system parameter and records center-channel results:

```toml
[studies.length_sweep]
type = "sweep"
methods = ["td", "pcfm", "mc"]
out_dir = "results/studies/length_sweep"

[studies.length_sweep.sweep]
variable = "fiber.length"
unit = "m"
values = [50000, 75000, 100000, 125000]
```

Supported sweep variables:

- `length` or `fiber.length`
- `baud` or `pulse.baud_rate`
- `spacing` or `wdm.spacing`

Sweeps currently use flat profiles per sweep point. This keeps parameter sweeps
fast and avoids recomputing Raman profiles for every value unless a dedicated
profile workflow is added for that study.

## Outputs

Each study writes under its `out_dir`.

Subset studies produce files such as:

```text
td_subset.csv
pcfm_subset.csv
subset_summary.csv
```

Sweep studies produce:

```text
sweep_<variable>.csv
flat_profile_sweep_<variable>_<value>.npy
pcfm_<cache-tag>.npy
fullband_mc_<cache-tag>.npz               # when engine = "fullband"
```

Full-system studies produce:

```text
results/gsnr_nli.pdf
results/nlin_power.pdf
results/fullband_mc_<profile>_<cache-tag>.npz   # XPM, FWM, total per target
```

The sweep CSV includes:

- swept value and label
- selected channel index and band label
- TD NLIN power
- PCFM NLIN power
- MC `chi1`, `chi2`, `prefactor`, and 16-QAM reconstruction when enabled
- fullband MC NLIN power when `engine = "fullband"`
- output/launch signal-power ratio

TD collision/NLIN cache files are named with method settings and content hashes
so repeated runs can reuse expensive intermediate results.

## Typical Commands

Run the configured studies:

```bash
python -m analysis.studies --config input/studies.toml
```

Use the compatibility CLI entrypoint:

```bash
python analysis/cli.py --config input/studies.toml
```

Run only a custom study by creating a small TOML with one `[studies.<name>]`
section and passing it to the same command.

For quick development, set:

```toml
[profiles]
mode = "flat"

[methods.td]
mode = "cached"

[methods.pcfm]
mode = "cached"
```

For a fullband-MC-only study (no TD, no PCFM):

```toml
[profiles]
mode = "flat"

[methods.td]
mode = "off"

[methods.pcfm]
mode = "off"

[methods.mc]
mode = "recompute"
engine = "fullband"
channel_decimation = 4
xpm_samples = 20000
fwm_samples = 10000

[studies.mc_only]
type = "full_system"
methods = ["mc"]
plot = true
```

For publication-quality recomputation, set profile and method modes to
`recompute` and verify the output directory is clean or intentionally reused.

The fullband MC diagnostic (`.npz`) can be inspected from Python:

```python
import numpy as np
d = np.load("results/fullband_mc_<profile>_<tag>.npz")
print(d.files)              # ['xpm', 'fwm', 'total', 'target_indices', ...]
xpm_frac = d['xpm'].sum() / d['total'].sum()
print(f"XPM contribution: {xpm_frac:.1%}")
```


# Singularity images

Packaging the code in a Singularity image allows us to run code using PyNLIN on the Department's SLURM cluster.

There are two main ways in which you can run build and run a Singularity image:

1. Install Singularity on your local machine, build the image, copy it to the cluster, and submit a job using the image.
2. Build the image using the remote builder and pull the image directly on the cluster to avoid wasting too much time on uploading the image.

> :warning: **The image pulls the latest commit on the `main` branch directly from GitHub. Local edits or commits not pushed to GitHub will not be reflected in the resulting image file

## Local build

Once you have Singularity installed, just run

```bash
sudo singularity build --force singularity.sif singularity.def
```
The resulting `.sif` image file can be used to run python scripts locally using

```bash
singularity exec singularity.sif python <script>.py
```
or uploaded to the cluster.
An example `.slurm` file to run a job on the cluster is provided in the `slurm/` directory of this repository.

## Remote build


## Building 
