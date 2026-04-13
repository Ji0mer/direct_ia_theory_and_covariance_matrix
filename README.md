# Direct IA Theory and Covariance Matrix

CosmoSIS pipeline for direct intrinsic-alignment (IA) forecasting. This repository generates mock projected correlation functions, builds their covariance matrices, writes FITS data products, and runs standard or accelerated forecast chains from those products.

## Overview

This repository is built around three closely related workflows:

1. Generate mock `wgp`, `wpp`, and `wgg` data vectors.
2. Build covariance matrices and package the result into a 2-point FITS file.
3. Run forecast pipelines from the generated FITS input, either in full mode or with a fixed-background accelerated path.

The `direct_ia/` tree vendors the theory, projection, likelihood, and utility modules needed to run the pipeline as a self-contained project.

## Pipeline At A Glance

### Mock generation

The mock-generation examples (`examples/generate-data.ini` and `examples/generate-data-photoz.ini`) follow this logic:

1. Load cached nonlinear and linear matter power spectra plus background distances (`read_pk`, `read_pk_lin`, `growth`).
2. Build galaxy and IA power-spectrum ingredients (`fast_pt`, `pk_to_cl_gg`, `IA`, `flatten_gi`, `flatten_ii`).
3. Project into configuration-space observables:
   - spectroscopic path: `wgp`, `wpp`, `wgg`
   - photo-z path: `wgp_photoz`, `wpp_photoz`, `wgg_photoz`
4. Apply photo-z factors, assemble the covariance matrix, write the FITS product, and run `ia_like` as a consistency check.

### Standard forecast

The standard forecast examples (`examples/params-forecast.ini` and `examples/params-forecast-photoz.ini`) read an existing FITS file, recompute the theory prediction for the configured IA and bias parameters, and evaluate the likelihood with `ia_like`.

### Fast forecast

The accelerated examples (`examples/params-forecast-fast.ini` and `examples/params-forecast-photoz-fast.ini`) replace the slowest repeated calculations with exact fixed-background modules in `scripts/accelerated_forecast/`.

Use the fast path when cosmology, distances, survey setup, `n(z)`, and photo-z assumptions stay fixed while IA and galaxy-bias parameters vary. The first run creates cache and template products under `intermediate_dir`; repeated runs with compatible fixed inputs are much faster.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `direct_ia/` | Core theory, projection, likelihood, and utility modules vendored into this repository |
| `scripts/` | Pipeline-specific modules for projection, covariance, FITS output, photo-z, and accelerated forecasts |
| `examples/` | Main runnable CosmoSIS entry points |
| `generate/` | Alternate generation pipeline configs |
| `forecast/` | Alternate forecast pipeline configs |
| `fits_data/` | Template and generated FITS files |
| `nz_data/` | Redshift-distribution inputs |
| `output/` | Cached power spectra, intermediates, logs, and run outputs |
| `setup.sh` | Environment bootstrap for this repository |

## Requirements

You will need:

- a working CosmoSIS installation
- `cosmosis-standard-library`
- a conda or virtual environment used by that CosmoSIS install
- a bash shell on Linux or WSL

Install the following Python packages into the same environment:

- `numpy`
- `scipy`
- `matplotlib`
- `astropy`
- `fitsio`
- `mcfit`
- `hankl`

Dependency notes:

- `fitsio` is used by the FITS I/O path in `read_pk`, `photoz_factor`, `ialike`, and `makefits`.
- `mcfit` is required by `direct_ia/projection/projected_corrs_legendre/legendre_interface.py`.
- `hankl` is required at runtime by the photo-z correlation modules in `scripts/photoz/` and by the accelerated photo-z forecast path.

## Installation

### Expected default layout

`setup.sh` assumes the following directory structure by default:

```text
/home/jiomer/research/
|-- cosmosis/
`-- direct_ia_theory_and_covariance_matrix/
```

### Install Python dependencies

If your tree matches the default layout, activate the CosmoSIS environment, install the missing packages, and source the repository setup script:

```bash
source /home/jiomer/anaconda3/etc/profile.d/conda.sh
conda activate /home/jiomer/research/cosmosis/env
pip install numpy scipy matplotlib astropy fitsio mcfit hankl

cd /home/jiomer/research/direct_ia_theory_and_covariance_matrix
source setup.sh
```

If you prefer conda packages where available:

```bash
conda install -c conda-forge numpy scipy matplotlib astropy fitsio mcfit
pip install hankl
```

### What `setup.sh` exports

After activation, `setup.sh` sources `cosmosis-configure` and exports:

- `COSMOSIS_LIB`
- `IA_LIB`
- `DATA_DIR`

If your layout differs from the default, set the paths before sourcing `setup.sh`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONDA_SH` | `$HOME/anaconda3/etc/profile.d/conda.sh` | Conda initialization script |
| `COSMOSIS_ROOT` | `../cosmosis` | CosmoSIS checkout root |
| `COSMOSIS_ENV` | `$COSMOSIS_ROOT/env` | CosmoSIS conda environment |
| `COSMOSIS_LIB` | `$COSMOSIS_ROOT/cosmosis-standard-library` | `cosmosis-standard-library` checkout |

Example:

```bash
export CONDA_SH=/path/to/conda.sh
export COSMOSIS_ROOT=/path/to/cosmosis
export COSMOSIS_ENV=/path/to/cosmosis/env
export COSMOSIS_LIB=/path/to/cosmosis/cosmosis-standard-library
source setup.sh
```

## Quick Start

| Task | Command |
| --- | --- |
| Generate spectroscopic mocks | `cosmosis examples/generate-data.ini` |
| Generate photo-z mocks | `cosmosis examples/generate-data-photoz.ini` |
| Run standard spectroscopic forecast | `cosmosis examples/params-forecast.ini` |
| Run standard photo-z forecast | `cosmosis examples/params-forecast-photoz.ini` |
| Run accelerated spectroscopic forecast | `cosmosis examples/params-forecast-fast.ini` |
| Run accelerated photo-z forecast | `cosmosis examples/params-forecast-photoz-fast.ini` |

For a smoke test, keep `sampler = test` in the relevant `.ini`. Switch to `multinest`, `polychord`, or another sampler only after the pipeline runs cleanly.

## What You Will Usually Edit

Most routine changes happen in:

- `examples/*.ini`
- `examples/values-generate.ini`
- `examples/values-generate-photoz.ini`
- `examples/values-forecast.ini`
- `examples/values-forecast-tatt-test.ini`

Typical edits include:

- FITS input and output file names
- `n(z)` inputs in `nz_data/`
- sample and survey names
- survey area, number density, and shape noise
- IA model choices and IA amplitude parameters
- galaxy-bias parameters
- fast-run cache locations via `intermediate_dir`

## Inputs And Outputs

The bundled examples expect repository-local inputs such as:

- `output/pk_fid/` for cached power spectra and distance tables
- `nz_data/` for redshift distributions
- `fits_data/` for template and generated FITS products

Generated text outputs, intermediates, and caches are written under `output/`.

## Notes

- Covariance generation is part of the mock-generation workflow, not a separate post-processing step.
- The generation pipelines also run the likelihood module as a consistency check.
- This repository already carries the migrated `direct_ia` modules, so `direct_ia_theory` is provenance rather than a runtime dependency.
- If you need to rebuild the compiled Limber projection module, inspect `direct_ia/projection/projected_corrs_limber/`.
- `output/` can grow quickly when running many forecast scans or cache-heavy fast runs.

## Contact

Zepei Yang (`yang.zep@northeastern.edu`)
