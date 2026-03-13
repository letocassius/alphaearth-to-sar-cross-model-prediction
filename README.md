# alphaearth-to-sar-cross-model-prediction

This repository contains the current offline workflow for testing whether AlphaEarth embedding vectors can predict Sentinel-1 SAR measurements across multiple land-cover contexts, then analyzing where those predictions fail.

## Repository layout

```text
alphaearth-to-sar-cross-model-prediction/
├── DataSources/              # input CSVs and other source data
├── scripts/                  # active analysis and reporting scripts
├── outputs/
│   ├── eda/                  # offline spatial join + sanity EDA outputs
│   └── full_dataset/         # current Phase 2 and Phase 3 machine outputs
├── reports/                  # human-facing PDF summaries
├── archive/
│   └── legacy_subset/        # archived outputs from the older subset pipeline
└── README.md
```

## Active workflow

### 1. Offline spatial join and sanity EDA

Script: `scripts/offline_spatial_join_and_sanity_eda.py`

- auto-detects AlphaEarth and Sentinel-1 CSVs
- performs the nearest-neighbor spatial join
- saves the merged table and sanity-check plots to `outputs/eda/`

Main outputs:

- `outputs/eda/alphaearth_sentinel1_merged.csv`
- `outputs/eda/`

### 2. Phase 2 full-dataset modeling

Script: `scripts/phase2_full_dataset_lightgbm_experiments.py`

- uses `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv`
- trains LightGBM models on all 2,880 rows
- compares `embedding_only` vs `embedding_plus_context`
- evaluates `S1_VV`, `S1_VH`, and `S1_VV_div_VH`
- saves metrics, diagnostics, predictions, and plots to `outputs/full_dataset/`

Core outputs:

- `outputs/full_dataset/full_dataset_lightgbm_metrics.csv`
- `outputs/full_dataset/full_dataset_lightgbm_stability.csv`
- `outputs/full_dataset/full_dataset_lightgbm_regional_metrics.csv`
- `outputs/full_dataset/full_dataset_lightgbm_land_use_metrics.csv`
- `outputs/full_dataset/test_predictions_*`
- `outputs/full_dataset/feature_importance_*`

### 3. Phase 3 failure-mode analysis

Script: `scripts/phase3_failure_analysis.py`

- selects the best held-out full-dataset model for each target
- analyzes held-out residuals by land use and by region
- computes Moran's I spatial autocorrelation diagnostics within each region
- generates a failure-mode outlier catalog for the primary `S1_VV` target
- exports GeoJSON residual layers and Phase 3 figures to `outputs/full_dataset/`

Core outputs:

- `outputs/full_dataset/phase3_best_model_selection.csv`
- `outputs/full_dataset/phase3_best_model_predictions.csv`
- `outputs/full_dataset/phase3_land_use_error.csv`
- `outputs/full_dataset/phase3_regional_error.csv`
- `outputs/full_dataset/phase3_region_land_use_error.csv`
- `outputs/full_dataset/phase3_spatial_autocorrelation.csv`
- `outputs/full_dataset/phase3_outliers_S1_VV.csv`
- `outputs/full_dataset/phase3_residuals_S1_VV.geojson`

### 4. Report generation

Primary script: `scripts/build_project_reports.py`

- builds three PDF summaries in one run
- writes a Phase 2 summary PDF
- writes a Phase 3 summary PDF
- writes a cumulative project summary PDF
- stores the PDFs in both `outputs/full_dataset/` and `reports/`

Compatibility wrapper: `scripts/build_phase2_full_dataset_pdf_report.py`

- calls the same multi-report pipeline

Generated reports:

- `reports/phase2_modeling_summary_report.pdf`
- `reports/phase3_failure_analysis_summary_report.pdf`
- `reports/project_summary_report.pdf`
- `reports/phase2_full_dataset_lightgbm_report.pdf`
  This is a legacy alias that now points to the cumulative project summary.

## Archived material

Older subset-based outputs are preserved under:

- `archive/legacy_subset/phase2_outputs/`
- `archive/legacy_subset/alphaearth_to_sar_consolidated_summary_report.pdf`

These files are kept for reference only and are not part of the current workflow.

## Study regions and data assets

The repository currently includes per-region data for:

- `amazon_forest`
- `california_coast`
- `iowa_ag`
- `sf_bay_urban`

Available source files include:

- AlphaEarth plus SAR sample CSVs for each region
- the combined all-regions modeling CSV
- Sentinel-1 SAR tabular data

## Environment setup

Use Python 3.10+ and install the core dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib optuna lightgbm
```

## How to run

Run the spatial join and sanity EDA:

```bash
python3 scripts/offline_spatial_join_and_sanity_eda.py
```

Run the Phase 2 full-dataset modeling step:

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase2_full_dataset_lightgbm_experiments.py
```

Run the Phase 3 failure analysis:

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase3_failure_analysis.py
```

Build all reports:

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/build_project_reports.py
```

The compatibility command still works:

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/build_phase2_full_dataset_pdf_report.py
```

## Notes

- The active scripts resolve paths from their own file location, so they can be run from the repository root or another working directory.
- The current machine-generated outputs live under `outputs/full_dataset/`.
- The human-facing PDFs live under `reports/`.
