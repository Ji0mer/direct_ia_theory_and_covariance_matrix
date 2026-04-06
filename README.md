# Direct IA Forecast Pipeline

This repository contains a CosmoSIS pipeline for direct intrinsic-alignment forecast work. It can:

- generate mock projected-correlation data vectors
- compute covariance matrices for `wgp`, `wpp`, and `wgg`
- export FITS data products
- run forecasts from existing FITS files

The `direct_ia/` directory contains the theory, projection, likelihood, and utility modules required by this pipeline. These were migrated from `direct_ia_theory` so this repository can run as a self-contained workflow.

Contact: Zepei Yang (`yang.zep@northeastern.edu`)

## Repository layout

- `direct_ia/`: migrated theory, likelihood, projection, and utility modules
- `scripts/`: pipeline-specific modules such as `replace_nz`, covariance, FITS writing, and photo-z projections
- `scripts/accelerated_forecast/`: exact fixed-cosmology acceleration modules for forecast pipelines
- `examples/`: runnable CosmoSIS configs
- `fits_data/`: template and generated FITS files
- `nz_data/`: redshift-distribution inputs
- `output/`: cached inputs, intermediates, and run outputs

## Setup

This README assumes you already have a working CosmoSIS environment and `cosmosis-standard-library`.

Edit `setup.sh` for your machine and set:

- `COSMOSIS_LIB`
- `IA_LIB`
- `DATA_DIR`

Then source it:

```bash
source setup.sh
```

If you need to rebuild the compiled Limber projection module, see `direct_ia/projection/projected_corrs_limber/`.

## Quick start

Generate a spectroscopic mock:

```bash
cosmosis examples/generate-data.ini
```

Generate a photo-z mock:

```bash
cosmosis examples/generate-data-photoz.ini
```

Run a spectroscopic forecast:

```bash
cosmosis examples/params-forecast.ini
```

Run a photo-z forecast:

```bash
cosmosis examples/params-forecast-photoz.ini
```

Run the fast spectroscopic forecast:

```bash
cosmosis examples/params-forecast-fast.ini
```

Run the fast photo-z forecast:

```bash
cosmosis examples/params-forecast-photoz-fast.ini
```

For a smoke test, keep `sampler = test` in the relevant `.ini` file.

## Fast forecasts

This repository includes exact fixed-background acceleration paths for the current fast forecast pipelines under `scripts/accelerated_forecast/`.

- `examples/params-forecast-fast.ini` is the fast spectroscopic forecast example.
- `examples/params-forecast-photoz-fast.ini` is the fast photo-z forecast example.
- These fast examples are intended for the common forecast case where cosmology, distances, `n(z)`, and photo-z settings are fixed while IA and galaxy-bias parameters vary.

The acceleration strategy is different for the two forecast types:

- Spectroscopic forecast: cached FAST-PT and `wgg` terms plus exact projected IA basis templates for `wgp` and `wpp`.
- Photo-z forecast: exact projected basis templates for `wgg`, `wgp`, and `wpp`, so the hot path only combines precomputed templates.

These modules are designed to preserve the forecast outputs of the original pipelines for fixed background inputs rather than introducing an emulator or approximate surrogate model.

Current fast-path modules include:

- `scripts/accelerated_forecast/nlbias_exact.py`
- `scripts/accelerated_forecast/wgg_exact.py`
- `scripts/accelerated_forecast/photoz_basis_exact.py`
- `scripts/accelerated_forecast/projected_ia_basis_exact.py`

Notes:

- The first run is slower because it builds cache and template files under the example `intermediate_dir`.
- Repeated runs with the same fixed cosmology and survey inputs are much faster.
- The fast examples support both `nla` and `tatt` IA settings through the acceleration modules.
- The public repository's original `tatt` path currently has an upstream FAST-PT grid mismatch in some configurations; the accelerated modules work around that internally for the forecast hot path.
- If you change cosmology, distances, `n(z)`, or photo-z settings, treat the next run as a new cold start.
- You do not need to manually delete old cache files when switching cosmology. The accelerated modules key caches on the fixed-background inputs and will build a new cache automatically.
- Old caches are not reused for a different cosmology, but they do remain on disk until you remove them yourself.
- These fast examples are intended for fixed-cosmology forecast scans over IA and bias parameters. Do not use them to sample cosmological parameters within the same run.

## Inputs and configuration

The example pipelines expect local inputs already present in this repository:

- cached power spectra and distance tables under `output/pk_fid/`
- redshift-distribution files under `nz_data/`
- FITS templates or existing FITS data vectors under `fits_data/`

Most user changes happen in:

- `examples/*.ini`
- `examples/values-generate.ini`
- `examples/values-forecast.ini`

Typical edits include file paths, sample names, `n(z)` choices, survey area, number density, shape noise, IA parameters, bias parameters, and photo-z settings.

For accelerated runs, the most important knobs are:

- `examples/values-forecast.ini` or another forecast values file
- the `intermediate_dir` in the example `.ini`
- the `ia_model` setting in the accelerated theory section
- cache and template directories inside `scripts/accelerated_forecast/*`

## Notes

- Covariance calculation is a core part of the mock-generation workflow, not a separate post-processing step.
- The generation pipelines also run the likelihood module as a consistency check.
- Output and cache directories can become large.
- `direct_ia_theory` should be treated as module provenance, not as a runtime dependency for this repository.
