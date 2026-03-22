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

For a smoke test, keep `sampler = test` in the relevant `.ini` file.

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

## Notes

- Covariance calculation is a core part of the mock-generation workflow, not a separate post-processing step.
- The generation pipelines also run the likelihood module as a consistency check.
- Output and cache directories can become large.
- `direct_ia_theory` should be treated as module provenance, not as a runtime dependency for this repository.
