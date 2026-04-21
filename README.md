# Direct IA Theory and Covariance Matrix

CosmoSIS pipeline for direct intrinsic-alignment (IA) forecasts, mock data generation, covariance construction, FITS packaging, and accelerated forecast runs.

This repository is organized around a small set of working pipelines:

1. Generate `wgp`, `wpp`, and `wgg` mock measurements from cached matter power spectra and chosen `n(z)`.
2. Build the corresponding covariance matrix and package everything into a 2-point FITS file.
3. Run standard or accelerated forecast chains from those FITS products.

The `direct_ia/` tree vendors the core theory, projection, likelihood, and utility modules so the project can run as a self-contained IA forecast repository.

## Current Workflow

### 1. Mock generation

The main generation entry points are:

- `generate/generate-data.ini`
- `generate/generate-data-photoz.ini`

These pipelines do the following:

1. Load cached linear and nonlinear matter power spectra plus distances from `output/pk_fid/`.
2. Replace the survey redshift distributions with the selected files from `nz_data/`.
3. Build galaxy and IA power-spectrum ingredients with `fast_pt`, `pk_to_cl_gg`, `IA`, `flatten_gi`, and `flatten_ii`.
4. Project into configuration-space observables:
   - spectroscopic path: `wgp`, `wpp`, `wgg`
   - photo-z path: `wgp_photoz`, `wpp_photoz`, `wgg_photoz`
5. Apply any photo-z correction factors.
6. Build the covariance matrix through the selected `scripts/covmat/` module.
7. Write a FITS data product with `makefits`.
8. Run `ia_like` as a consistency check.

### 2. Forecast runs

The main accelerated forecast entry points are:

- `forecast/params-forecast-fast.ini`
- `forecast/params-forecast-photoz-fast.ini`

These pipelines reuse fixed background information and exact cached basis products in `scripts/accelerated_forecast/` to speed up repeated forecasts when cosmology, distances, and survey setup are held fixed.

The `examples/` directory still contains runnable reference configurations, including:

- `examples/generate-data.ini`
- `examples/generate-data-photoz.ini`
- `examples/params-forecast.ini`
- `examples/params-forecast-photoz.ini`
- `examples/params-forecast-fast.ini`
- `examples/params-forecast-photoz-fast.ini`

In practice, the `generate/` and `forecast/` trees are the better place to look first for the currently maintained working copies.

## Covariance Modules

The covariance calculation lives in `scripts/covmat/`.

### Default projected covariance

- `scripts/covmat/cov_equation_final.py`
- helper kernels: `scripts/covmat/dht_simpson.py`

This is the default covariance path used by the checked-in generation pipelines unless you change the `[covmat] file = ...` line in the `.ini`.

### Alternate `n(z)`-aware covariance

- `scripts/covmat/cov_equation_nz_final.py`
- helper kernels: `scripts/covmat/dht_simpson_nz.py`

This path uses the same saved bin-averaged Bessel kernels from `output/avg_jn/`, but evaluates the covariance with redshift-dependent weights and noise terms. To switch to it, change the `[covmat]` module file in the relevant generation `.ini`.

### Saved bin-averaged Bessel kernels

Both covariance paths can reuse the tracked kernel cache under:

- `output/avg_jn/`

This avoids regenerating the bin-averaged Bessel functions every run. The cache must match the active `rbins`.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `direct_ia/` | Core theory, projection, likelihood, and utilities vendored into this repository |
| `scripts/` | Pipeline-specific modules for covariance, FITS output, photo-z handling, accelerated forecasts, and auxiliary processing |
| `generate/` | Main mock-generation configs and values files |
| `forecast/` | Main forecast configs and values files |
| `examples/` | Reference and alternate runnable CosmoSIS configs |
| `fits_data/` | Input and generated FITS products |
| `nz_data/` | Redshift-distribution inputs |
| `output/` | Cached spectra, covariance kernels, intermediates, and run outputs |
| `setup.sh` | Environment bootstrap for this repository |

## Requirements

You will need:

- a working CosmoSIS installation
- `cosmosis-standard-library`
- a conda or virtual environment used by that CosmoSIS install
- Linux or WSL with bash available

Install the Python dependencies into the same environment:

- `numpy`
- `scipy`
- `matplotlib`
- `astropy`
- `fitsio`
- `mcfit`
- `hankl`

Dependency notes:

- `fitsio` is used in FITS I/O and likelihood paths.
- `mcfit` is required by `direct_ia/projection/projected_corrs_legendre/legendre_interface.py`.
- `hankl` is used by the photo-z correlation modules and accelerated photo-z basis pipeline.

## Installation

### Expected layout

If you use the repository defaults, `setup.sh` expects the CosmoSIS checkout to live next to this repository:

```text
<workspace-parent>/
|-- cosmosis/
`-- direct_ia_theory_and_covariance_matrix/
```

### Activate the environment

The most portable approach is to set the paths you want explicitly:

```bash
export REPO_ROOT=/path/to/direct_ia_theory_and_covariance_matrix
export COSMOSIS_ROOT=/path/to/cosmosis
export COSMOSIS_ENV=$COSMOSIS_ROOT/env
export COSMOSIS_LIB=$COSMOSIS_ROOT/cosmosis-standard-library
export CONDA_SH=/path/to/conda.sh

source "$CONDA_SH"
conda activate "$COSMOSIS_ENV"

cd "$REPO_ROOT"
source setup.sh
```

If your shell already initializes conda, you can skip `source "$CONDA_SH"` and just run `conda activate "$COSMOSIS_ENV"` before sourcing `setup.sh`.

### Install missing packages

```bash
pip install numpy scipy matplotlib astropy fitsio mcfit hankl
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

If you do not set overrides, `setup.sh` uses repo-relative defaults such as `../cosmosis` for the CosmoSIS root. If your layout differs, set the variables first:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONDA_SH` | unset | Conda initialization script |
| `COSMOSIS_ROOT` | `../cosmosis` | CosmoSIS checkout root |
| `COSMOSIS_ENV` | `$COSMOSIS_ROOT/env` | CosmoSIS conda environment |
| `COSMOSIS_LIB` | `$COSMOSIS_ROOT/cosmosis-standard-library` | `cosmosis-standard-library` checkout |

Example:

```bash
export REPO_ROOT=/path/to/direct_ia_theory_and_covariance_matrix
export CONDA_SH=/path/to/conda.sh
export COSMOSIS_ROOT=/path/to/cosmosis
export COSMOSIS_ENV=/path/to/cosmosis/env
export COSMOSIS_LIB=/path/to/cosmosis/cosmosis-standard-library
cd "$REPO_ROOT"
source setup.sh
```

## Quick Start

| Task | Command |
| --- | --- |
| Generate spectroscopic mocks | `cosmosis generate/generate-data.ini` |
| Generate photo-z mocks | `cosmosis generate/generate-data-photoz.ini` |
| Run accelerated spectroscopic forecast | `cosmosis forecast/params-forecast-fast.ini` |
| Run accelerated photo-z forecast | `cosmosis forecast/params-forecast-photoz-fast.ini` |
| Run reference standard spectroscopic forecast | `cosmosis examples/params-forecast.ini` |
| Run reference standard photo-z forecast | `cosmosis examples/params-forecast-photoz.ini` |

For smoke tests, set `sampler = test` in the target `.ini`. Switch to `multinest`, `polychord`, or another production sampler only after the pipeline runs cleanly.

## Files You Will Usually Edit

Most routine work happens in:

- `generate/generate-data.ini`
- `generate/generate-data-photoz.ini`
- `generate/values-generate.ini`
- `generate/values-generate-photoz.ini`
- `forecast/params-forecast-fast.ini`
- `forecast/params-forecast-photoz-fast.ini`
- `forecast/values-forecast.ini`
- `forecast/values-forecast-photoz.ini`
- the `[covmat]` file choice inside the generation `.ini`

Typical edits include:

- FITS input and output file names
- `n(z)` replacements in `nz_data/`
- sample and survey names
- IA amplitude and IA-model options
- galaxy-bias parameters
- survey area, number density, and shape noise
- accelerated cache locations such as `intermediate_dir`

## Inputs and Outputs

The pipeline expects repository-local inputs such as:

- `output/pk_fid/` for cached power spectra and distances
- `output/avg_jn/` for saved bin-averaged Bessel kernels
- `nz_data/` for redshift distributions
- `fits_data/` for FITS templates and generated products

Generated text outputs, intermediates, and caches are written under `output/`.

## Git and Data Policy

This repository is meant to keep code and lightweight configuration changes in git while avoiding noisy data-heavy commits.

Current local-data rules:

- `output/` is ignored except for the tracked caches in `output/pk_fid/` and `output/avg_jn/`
- `fits_data/DESI/` is ignored
- `nz_data/DESI_samples/` is ignored
- `fits_data/Roman_fits/` is ignored
- `nz_data/Roman_z_bins/` is ignored

If you add new large local survey products, prefer keeping them under ignored directories rather than committing them directly.

## Notes

- Covariance generation is part of the mock-generation workflow, not a separate post-processing step.
- The generation pipelines also run the likelihood module as a built-in consistency check.
- The accelerated forecast modules live in `scripts/accelerated_forecast/` and assume fixed background quantities and reusable cached templates.
- If you need to rebuild the compiled Limber projection code, inspect `direct_ia/projection/projected_corrs_limber/`.
- `output/` can grow quickly during scans and cache-heavy fast runs.

## Contact

Zepei Yang (`yang.zep@northeastern.edu`)
