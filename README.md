# alphaearth-to-sar-cross-model-prediction

This repository contains an offline workflow for testing whether AlphaEarth embedding vectors can predict Sentinel-1 SAR measurements across multiple land-cover contexts.

## Repository contents

- `offline_spatial_join_and_sanity_eda.py`: auto-detects AlphaEarth and Sentinel-1 CSVs, performs a nearest-neighbor spatial join, and saves sanity-check visualizations.
- `phase2_regression_modeling.py`: builds a balanced modeling subset from the combined 2024 dataset, audits the data, and evaluates ridge and tuned LightGBM regressors for SAR targets.
- `build_phase2_pdf_report.py`: renders the saved Phase 2 metrics, diagnostics, and plots into a PDF report.
- `phase2_full_dataset_lightgbm_experiments.py`: runs a separate full-dataset LightGBM ablation on all 2,880 rows without changing the existing Phase 2 outputs.
- `build_phase2_full_dataset_pdf_report.py`: renders the saved full-dataset ablation metrics and diagnostics into a PDF report.
- `build_consolidated_summary_report.py`: renders a single summary PDF across the subset benchmark and the full-dataset ablation.
- `alphaearth_to_sar_consolidated_summary_report.pdf`: the current top-level summary report for the repository.
- `DataSources/`: bundled CSV and GeoTIFF inputs for the four study regions.

## Workflow

### 1. Offline spatial join and sanity EDA

`offline_spatial_join_and_sanity_eda.py`:

- finds an AlphaEarth table by detecting embedding columns such as `A00` to `A63`
- finds a Sentinel-1 table by detecting SAR-like `VV` and `VH` columns
- detects coordinates from latitude and longitude columns or parses GeoJSON-style geometry fields such as `.geo`
- matches AlphaEarth points to the nearest SAR point with a maximum distance of 10 meters
- saves a merged table plus basic diagnostics for missingness, SAR distributions, and embedding PCA

Generated outputs include:

- `alphaearth_sentinel1_merged.csv`
- `eda_outputs/`

### 2. Phase 2 regression modeling

`phase2_regression_modeling.py` uses `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv` as the modeling input. The dataset contains 2,880 rows:

- 4 regions
- 9 Dynamic World labels
- 80 samples in each `region x dw_label` cell

The script:

- samples a balanced subset with 20 rows per `region x dw_label` cell
- defines spatial blocks within each region for grouped validation
- trains ridge and tuned LightGBM regressors
- predicts `S1_VV`, `S1_VH`, and `S1_VV_div_VH`
- tunes LightGBM with Optuna Bayesian search and early-stopping-informed tree counts
- reports held-out test metrics plus region-level and land-use-level diagnostics
- saves prediction plots, residual histograms, feature-importance tables, CV search results, and optimization metadata

Generated outputs include:

- `DataSources/alphaearth_s1_dw_samples_balanced_subset_2024.csv`
- `phase2_outputs/dataset_audit.json`
- `phase2_outputs/regression_metrics.csv`
- `phase2_outputs/regional_metrics.csv`
- `phase2_outputs/land_use_metrics.csv`
- `phase2_outputs/cv_results_*_lightgbm.csv`
- `phase2_outputs/best_params_*_lightgbm.json`
- `phase2_outputs/phase2_summary.md`
- `phase2_outputs/phase2_model_performance_report.pdf`

Generate the PDF report after modeling with:

```bash
python3 build_phase2_pdf_report.py
```

### 3. Full-dataset LightGBM ablation

`phase2_full_dataset_lightgbm_experiments.py` uses all 2,880 rows and compares:

- `embedding_only`
- `embedding_plus_context` using region plus Dynamic World probabilities

The script:

- preserves the existing Phase 2 files and outputs
- tunes LightGBM for each target and feature-set ablation
- runs repeated grouped-CV stability checks across multiple seeds
- reports held-out metrics plus regional and land-use diagnostics
- saves all outputs to `phase2_full_dataset_outputs/`

Generated outputs include:

- `phase2_full_dataset_outputs/full_dataset_lightgbm_metrics.csv`
- `phase2_full_dataset_outputs/full_dataset_lightgbm_stability.csv`
- `phase2_full_dataset_outputs/full_dataset_lightgbm_regional_metrics.csv`
- `phase2_full_dataset_outputs/full_dataset_lightgbm_land_use_metrics.csv`
- `phase2_full_dataset_outputs/phase2_full_dataset_lightgbm_report.pdf`

Render the full-dataset PDF report with:

```bash
python3 build_phase2_full_dataset_pdf_report.py
```

### 4. Consolidated summary report

Build the top-level summary report that synthesizes the subset benchmark and the full-dataset ablation:

```bash
python3 build_consolidated_summary_report.py
```

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
- Sentinel-2 context GeoTIFFs for the same four regions

## Environment setup

Use Python 3.10+ and install the core dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib optuna lightgbm
```

## How to run

Run the spatial join and EDA step from the repository root:

```bash
python3 offline_spatial_join_and_sanity_eda.py
```

Run the Phase 2 modeling step:

```bash
python3 phase2_regression_modeling.py
```

If Matplotlib needs a writable config directory in your environment, run:

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 phase2_regression_modeling.py
```

Then render the PDF report:

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 build_phase2_pdf_report.py
```

Run the full-dataset ablation:

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 phase2_full_dataset_lightgbm_experiments.py
MPLCONFIGDIR=/tmp/matplotlib python3 build_phase2_full_dataset_pdf_report.py
MPLCONFIGDIR=/tmp/matplotlib python3 build_consolidated_summary_report.py
```

## Notes

- The scripts are designed for local, offline execution against files already present in the repository tree.
- Output directories such as `eda_outputs/`, `phase2_outputs/`, and `phase2_full_dataset_outputs/` are created when the scripts are run.
