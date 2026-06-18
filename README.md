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
- `pynlin.methods.mc`: chi1/chi2 reconstruction from TD collision coefficients.

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
n_trials = 1
rng_seed = 1234
```

`cached` reuses existing method outputs when the cache path exists. `recompute`
forces regeneration. `off` disables the method even if a study lists it.

MC requires TD because it reconstructs modulation-dependent NLIN from TD
collision coefficients.

## Study Types

### Full System

`full_system` runs the end-to-end TD/PCFM workflow over all channels and can
produce plots:

```toml
[studies.full_system]
type = "full_system"
methods = ["td", "pcfm"]
plot = true
```

This is the closest replacement for the old PCFM workflow scripts.

### Subset

`subset` evaluates selected CUT/interferer channels and writes compact CSV
summaries:

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
```

The sweep CSV includes:

- swept value and label
- selected channel index and band label
- TD NLIN power
- PCFM NLIN power
- MC `chi1`, `chi2`, `prefactor`, and 16-QAM reconstruction when enabled
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

For publication-quality recomputation, set profile and method modes to
`recompute` and verify the output directory is clean or intentionally reused.


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
