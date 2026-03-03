# alphaearth-to-sar-cross-model-prediction

Offline workflow to spatially align AlphaEarth embeddings with Sentinel-1 SAR measurements, then run sanity EDA for model-building readiness.

## What this does

This project provides a single script that performs two steps:

1. Spatial join: nearest-neighbor match between AlphaEarth points and SAR points using a KD-tree, keeping matches within 10 meters.
2. Sanity EDA: missing-value checks, SAR histograms, and PCA diagnostics on embedding columns.

## Project structure

- `offline_spatial_join_and_sanity_eda.py`: main pipeline script.
- `DataSources/`: input CSV files.
- `eda_outputs/`: generated EDA plots.
- `alphaearth_sentinel1_merged.csv`: merged output table created by the script.

## Input expectations

The script auto-detects:

- AlphaEarth CSV using embedding-like columns such as `A00`, `A01`, ...
- Sentinel-1 CSV using SAR-like columns containing `VV`/`VH`
- Latitude/longitude columns (or parses GeoJSON-like geometry columns such as `.geo`)

Place input CSVs under the project directory (for example in `DataSources/`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib
```

## Run

From the project root:

```bash
python offline_spatial_join_and_sanity_eda.py
```

## Outputs

- `alphaearth_sentinel1_merged.csv`: joined AlphaEarth + SAR dataset
- `eda_outputs/hist_vv.png`
- `eda_outputs/hist_vh.png`
- `eda_outputs/hist_vv_vh_ratio.png`
- `eda_outputs/pca_cumulative_explained_variance.png`

The script also prints a concise console summary of join quality and EDA diagnostics.
