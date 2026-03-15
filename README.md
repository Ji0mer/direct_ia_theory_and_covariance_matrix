# Direct IA Forecast Pipeline

This repository contains the project-specific CosmoSIS pipeline used to generate direct intrinsic-alignment (IA) data vectors, build covariance matrices, and run parameter forecasts in the echoIA-style FITS format.

It is designed to sit on top of:

- `direct_ia_theory` for the base IA theory and standard projection utilities
- `cosmosis-standard-library` for the main CosmoSIS modules
- a local CosmoSIS environment with the required Python dependencies

The current workflow supports:

- mock 3x2pt data-vector generation for spectroscopic samples
- mock 3x2pt data-vector generation with a simple Gaussian photo-z model
- forecast runs against generated FITS data products
- custom cached implementations of several expensive theory modules

Contact: Zepei Yang (`yang.zep@northeastern.edu`)

## Repository role

The upstream `direct_ia_theory` repository provides the core theory pipeline. This repository adds the pieces needed to make that pipeline usable for direct IA forecast production:

- replacement `n(z)` utilities
- covariance-matrix construction
- FITS export in the echoIA-style format
- photo-z projection wrappers
- cached versions of `fast_pt`, nonlinear-bias, projection, and TATT-related modules
- example configurations for both mock generation and forecast inference

## Requirements

Before using this repository, make sure the following are available locally:

1. CosmoSIS
2. `cosmosis-standard-library`
3. [`direct_ia_theory`](https://github.com/ssamuroff/direct_ia_theory)
4. A Python environment with packages used by the custom scripts, including `numpy`, `scipy`, `matplotlib`, `astropy`, and `fitsio`
5. MultiNest if you want to run the forecast examples without changing the sampler

## Environment setup

Edit [`setup.sh`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/setup.sh) so that it matches your local installation paths. In particular, confirm these variables:

- `COSMOSIS_LIB`: path to `cosmosis-standard-library`
- `IA_LIB`: path to the `direct_ia_theory` repository
- `IA_LIB1`: path to this repository
- `DATA_DIR`: directory where generated FITS products should be written

Then source the environment:

```bash
source setup.sh
```

The checked-in `setup.sh` is machine-specific and assumes a local conda environment plus a preconfigured CosmoSIS install. Adjust it before running anything on a different system.

## Quick start

### 1. Generate a spectroscopic mock data vector

```bash
cosmosis examples/generate-data.ini
```

This example:

- loads matter power spectra from `direct_ia_theory/output/pk_fid/`
- replaces the sample redshift distributions with the files defined in `[replace_nz]`
- computes `wgp`, `wpp`, and `wgg`
- builds a covariance matrix
- writes an echoIA-style FITS file through `scripts/makefits/makefits.py`

The default output FITS path is set by `[makefits]` in [`examples/generate-data.ini`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/examples/generate-data.ini).

### 2. Generate a mock with photo-z smearing

```bash
cosmosis examples/generate-data-photoz.ini
```

This follows the same overall structure, but swaps in the custom modules in [`scripts/photoz`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/photoz) and uses the `[photoz_errors]` block in [`examples/values-generate.ini`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/examples/values-generate.ini).

### 3. Run a forecast against a generated FITS file

For the spectroscopic example:

```bash
cosmosis examples/params-forecast.ini
```

For the photo-z example:

```bash
cosmosis examples/params-generate-data-photoz.ini
```

These configs read an existing FITS data vector from `fits_data/` and run the IA likelihood. By default they are configured to use `multinest`. For a quick smoke test, switch `sampler = test` in the relevant `.ini` file.

## Key configuration knobs

The most important settings usually live in the example `.ini` files and value files:

- `examples/generate-data.ini`
- `examples/generate-data-photoz.ini`
- `examples/params-forecast.ini`
- `examples/params-generate-data-photoz.ini`
- `examples/values-generate.ini`
- `examples/values-forecast.ini`

Common parameters to edit:

- sample naming:
  - `default_sample`
  - `default_survey`
- IA model selection:
  - `[IA] ia_model = nla` or `tatt`
- redshift distributions:
  - `[replace_nz] nz_shape_all`, `nz_dens_all`, `nz_shape`, `nz_dens`
- covariance settings:
  - `[covmat] zeff`, `sigma_e`, `nbar_shape`, `nbar_dens`, `area_shape`, `area_dens`, `rmin`, `rmax`, `nr`
- FITS output path:
  - `[makefits] save_fits`
- IA and galaxy-bias parameters:
  - `[intrinsic_alignment_parameters]`
  - `[bias_forecast_sample_density]`
- photo-z settings:
  - `[photoz_errors] sigmaz`, `Pi_max`, `N_pi`, `Pi_mask_max`

## Repository layout

- [`examples`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/examples): runnable CosmoSIS configuration files
- [`fits_data`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/fits_data): example input and generated FITS data products
- [`nz_data`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/nz_data): sample redshift-distribution files
- [`output`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/output): cached intermediate products and run outputs
- [`scripts/add_nz`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/add_nz): redshift-distribution replacement utility
- [`scripts/covmat`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/covmat): covariance-matrix construction
- [`scripts/makefits`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/makefits): writes the final FITS file containing `wgp`, `wpp`, `wgg`, and `COVMAT`
- [`scripts/photoz`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/photoz): projected-correlation modules with photo-z smearing
- [`scripts/nonlinear_bias`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/nonlinear_bias): cached nonlinear-bias and power-spectrum utilities
- [`scripts/projection`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/projection): cached projection modules
- [`scripts/structure`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/structure): cached `fast_pt` implementation
- [`scripts/tatt`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/tatt): custom TATT interfaces
- [`scripts/synthetic_parameter`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/synthetic_parameter): notebook used to derive synthetic sample parameters

## Main custom modules

### `scripts/add_nz`

Replaces the redshift distributions in a reference setup with the target sample `n(z)` files.

### `scripts/covmat`

Builds the covariance matrix for the projected observables. The output ordering written to the FITS file is:

```text
wgp, wpp, wgg
```

### `scripts/makefits`

Collects the projected observables and covariance matrix from the CosmoSIS datablock and writes them to an echoIA-compatible FITS file with extensions for:

- `nz_shape`
- `nz_density`
- `wgp`
- `wpp`
- `wgg`
- `COVMAT`

### `scripts/photoz`

Applies a simple Gaussian photo-z model to the projected correlations used in the photometric workflow.

### `scripts/nonlinear_bias`, `scripts/projection`, `scripts/structure`, `scripts/tatt`

These modules mirror the role of the corresponding components in `direct_ia_theory`, but cache intermediate calculations to reduce repeated runtime cost in forecast runs.

## Typical workflow

1. Source [`setup.sh`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/setup.sh).
2. Choose the relevant example config in [`examples`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/examples).
3. Edit the sample definition, `n(z)` files, IA model, and covariance settings.
4. Run `cosmosis` on a generation config to create a FITS data vector.
5. Point a forecast config at that FITS product and run inference.

## Notes

- The example configs assume existing power-spectrum inputs in `direct_ia_theory/output/pk_fid/`.
- Several paths in the repository are hard-coded through environment variables, so path consistency matters.
- The forecast examples are configured for one-bin use cases by default.
- Generated products and cached intermediates can become large; keep an eye on `output/`.
