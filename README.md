# Direct IA Forecast Pipeline

This repository contains the project-specific CosmoSIS pipeline used to generate direct intrinsic-alignment (IA) mock data vectors, compute covariance matrices, and run forecast analyses in an echoIA-style FITS format.

It extends [`direct_ia_theory`](https://github.com/ssamuroff/direct_ia_theory) with:

- custom `n(z)` replacement utilities
- dedicated covariance-matrix calculation code for projected IA observables
- FITS export helpers
- photo-z projected-correlation modules
- cached versions of several expensive theory components

Contact: Zepei Yang (`yang.zep@northeastern.edu`)

## Setup

Edit [`setup.sh`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/setup.sh) to match your local paths, especially:

- `COSMOSIS_LIB`
- `IA_LIB`
- `IA_LIB1`
- `DATA_DIR`

Then source the environment:

```bash
source setup.sh
```

The checked-in `setup.sh` is machine-specific and should be treated as a template.

## Quick start

Generate a spectroscopic mock:

```bash
cosmosis examples/generate-data.ini
```

Generate a mock with photo-z smearing:

```bash
cosmosis examples/generate-data-photoz.ini
```

Run a forecast from an existing FITS data vector:

```bash
cosmosis examples/params-forecast.ini
```

For the photometric forecast example:

```bash
cosmosis examples/params-generate-data-photoz.ini
```

If you only want a quick smoke test, switch `sampler = test` in the relevant `.ini` file.

## Covariance matrix pipeline

Covariance-matrix computation is a central part of this repository, not just a helper step. The code under [`scripts/covmat`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/covmat) builds covariance matrices for the projected `wgp`, `wpp`, and `wgg` observables and is used directly by the mock-data generation workflow.

## Main files

- [`examples`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/examples): runnable CosmoSIS configs
- [`fits_data`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/fits_data): example and generated FITS products
- [`nz_data`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/nz_data): redshift-distribution inputs
- [`output`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/output): cached intermediates and run outputs
- [`scripts/covmat`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/covmat): covariance-matrix calculation for `wgp`, `wpp`, and `wgg`
- [`scripts/makefits`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/makefits): FITS writer
- [`scripts/photoz`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/photoz): photo-z projected-correlation modules
- [`scripts/nonlinear_bias`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/nonlinear_bias), [`scripts/projection`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/projection), [`scripts/structure`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/structure), [`scripts/tatt`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/scripts/tatt): cached theory modules

## Notes

- The example configs assume existing power-spectrum inputs from `direct_ia_theory/output/pk_fid/`.
- Most runtime customization happens in the files under [`examples`](/projects/blazek_group_storage/zepei/ia_forecast/direct_ia_to_public/examples).
- Generated products and cached outputs can become large.
