# alphaearth-to-sar-cross-model-prediction

Offline workflow to spatially align AlphaEarth embeddings with Sentinel-1 SAR measurements, run sanity EDA, and execute Phase 2 regression modeling on the combined 2024 dataset.

## What this does

This project provides:

1. Spatial join: nearest-neighbor match between AlphaEarth points and SAR points using a KD-tree, keeping matches within 10 meters.
2. Sanity EDA: missing-value checks, SAR histograms, and PCA diagnostics on embedding columns.
3. Phase 2 regression modeling: build a smaller balanced subset from the combined CSV and evaluate predictive capacity for `S1_VV`, `S1_VH`, and `S1_VV_div_VH`.

## Project structure

- `offline_spatial_join_and_sanity_eda.py`: main pipeline script.
- `phase2_regression_modeling.py`: balanced subset creation and regression analysis.
- `DataSources/`: input CSV files.
- `eda_outputs/`: generated EDA plots.
- `phase2_outputs/`: generated Phase 2 metrics, plots, and feature-importance tables.
- `alphaearth_sentinel1_merged.csv`: merged output table created by the script.

## Input expectations

The script auto-detects:

- AlphaEarth CSV using embedding-like columns such as `A00`, `A01`, ...
- Sentinel-1 CSV using SAR-like columns containing `VV`/`VH`
- Latitude/longitude columns (or parses GeoJSON-like geometry columns such as `.geo`)

Place input CSVs under the project directory (for example in `DataSources/`).

For Phase 2, the primary combined input is `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv`, which contains 2,880 rows:

- 4 regions x 720 rows each
- 9 Dynamic World labels x 320 rows each overall
- 80 samples in every `region x dw_label` cell

Additional project data currently includes:

- per-region sample CSVs for `sf_bay_urban`, `iowa_ag`, `amazon_forest`, and `california_coast`
- Sentinel-2 context GeoTIFFs for the same four regions

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

Phase 2 regression modeling:

```bash
MPLCONFIGDIR=/tmp/matplotlib ./.venv_ds/bin/python phase2_regression_modeling.py
```

## Outputs

- `alphaearth_sentinel1_merged.csv`: joined AlphaEarth + SAR dataset
- `eda_outputs/hist_vv.png`
- `eda_outputs/hist_vh.png`
- `eda_outputs/hist_vv_vh_ratio.png`
- `eda_outputs/pca_cumulative_explained_variance.png`
- `DataSources/alphaearth_s1_dw_samples_balanced_subset_2024.csv`
- `phase2_outputs/regression_metrics.csv`
- `phase2_outputs/regional_metrics_S1_VV.csv`
- `phase2_outputs/phase2_summary.md`

The script also prints a concise console summary of join quality and EDA diagnostics.
