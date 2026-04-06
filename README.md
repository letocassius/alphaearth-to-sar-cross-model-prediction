# alphaearth-to-sar-cross-model-prediction

End-to-end workflow for testing whether AlphaEarth optical embeddings can predict Sentinel-1 SAR backscatter, then diagnosing where that relationship holds, where it breaks down, and how derived polarization targets behave.

The repository now documents both:

- the upstream Google Earth Engine workflow used to generate the study data
- the downstream local Python workflow used to model, analyze, and report results

## Project Scope

The final modeling table is a balanced 2,880-row sample across four regions and nine Dynamic World classes. Each row contains:

- 64 AlphaEarth embedding bands `A00` to `A63`
- Sentinel-1 SAR targets `S1_VV`, `S1_VH`, and `S1_VV_div_VH`
- Dynamic World label probabilities and modal class
- point coordinates, region name, and year

The repository evaluates whether the embedding can recover:

- `S1_VV` in dB
- `S1_VH` in dB
- `S1_VV_div_VH`, which in this dataset is the dB-space polarization difference `S1_VV - S1_VH`

The current workflow reports:

- LightGBM baselines on the core SAR targets
- Ridge follow-up baselines for polarization-difference experiments
- Derived linear `VV/VH` ratio baselines, including a log-ratio formulation
- Failure analysis by land use and region
- Cross-modal similarity analysis between embedding space and SAR space
- Data sufficiency diagnostics

## Reproducibility Status

This repository supports two different reproducibility standards:

1. Exact downstream reproduction from the committed merged table.
   This path is supported today. If you start from `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv`, you can regenerate the local model outputs, analysis artifacts, and reports produced by the Python workflow in this repository.
2. Approximate upstream-to-downstream reconstruction from Earth Engine source collections.
   This path is also documented, but it is not exact in every detail because not every historical artifact was preserved.

What is exact:

- The downstream local analysis and report-generation workflow is fully represented in this repository.
- The merged-table Earth Engine export workflow in `gee/export_dynamic_world_joined_samples.js` comes directly from the recorded Earth Engine code path used for the study.

What is approximate or partially reconstructed:

- The standalone `AlphaEarth_Embeddings_2024.csv` and `Sentinel1_SAR_2024.csv` export scripts are reconstructed convenience workflows derived from the merged workflow. They should reproduce equivalent point tables over the same sampled points, but they should not be treated as guaranteed byte-for-byte originals unless the exact historical export scripts are recovered.
- Sentinel-2 context GeoTIFFs are intentionally not tracked in git. Phase 4 metrics still run without them, but the qualitative chip figure is only fully reproducible after regenerating those TIFFs.
- The exact historical Python version and fully pinned lockfile were not preserved. The repository includes a minimal dependency manifest, not a frozen environment specification.

Bottom line:

- If your goal is to reproduce the committed analysis outputs from the committed merged CSV, this repository is sufficient.
- If your goal is to reproduce the entire project from raw upstream data with exact environment matching and byte-for-byte parity for every intermediate artifact, additional historical materials are still required.

## Repository Layout

- `gee/`
  Google Earth Engine scripts for regenerating upstream data products.
- `DataSources/`
  Committed CSV inputs used by the local workflow.
  Sentinel-2 context GeoTIFFs are expected here under `DataSources/sentinel2_context/` when regenerated locally.
- `scripts/`
  Local Python analysis and report-generation entrypoints, organized by project phase.
- `outputs/full_dataset/`
  Machine-readable results, figures, and intermediate report artifacts for the full 2,880-sample dataset.
- `outputs/eda/`
  Spatial-join and sanity-EDA outputs.
- `reports/`
  Human-facing deliverables, including the phase PDFs, the combined PDF summary, and `final_report.md`.

## Reproducibility Checklist

Use this checklist before running anything:

- Work from the repository root: `alphaearth-to-sar-cross-model-prediction/`
- Create and activate a Python virtual environment
- Install `requirements.txt`
- Confirm that `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv` exists if you want the standard downstream workflow
- Confirm that `DataSources/sentinel2_context/*.tif` exists only if you want the full Phase 4 qualitative figure rather than the placeholder version
- Run scripts in the documented order; later phases depend on files produced by earlier phases
- Treat `outputs/` and `reports/` as generated artifacts that can be overwritten by reruns

## Upstream GEE Workflow

The Earth Engine workflow uses the following collections:

- AlphaEarth embeddings: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- Sentinel-1 SAR: `COPERNICUS/S1_GRD`
- Dynamic World: `GOOGLE/DYNAMICWORLD/V1`
- Sentinel-2 context imagery: `COPERNICUS/S2_SR_HARMONIZED`

The recorded Earth Engine configuration is:

- `YEAR = 2024`
- `START = 2024-01-01`
- `END = 2025-01-01`
- `SCALE = 10`
- `POINTS_PER_CLASS = 80`
- sampling seed `42`
- export folder `GEE_AlphaEarth_SAR_Project`

The four study AOIs are:

- `sf_bay_urban`: rectangle `[-122.60, 37.15, -121.70, 37.95]`
- `iowa_ag`: rectangle `[-95.90, 41.60, -94.90, 42.40]`
- `amazon_forest`: rectangle `[-60.50, -3.60, -59.50, -2.60]`
- `california_coast`: rectangle `[-121.90, 34.70, -120.90, 35.50]`

The upstream export logic is:

1. Build a yearly AlphaEarth mosaic over each AOI.
2. Build a Sentinel-1 median composite over each AOI using:
   - `instrumentMode = IW`
   - both `VV` and `VH` polarization present
   - `resolution_meters = 10`
3. Derive `S1_VV_div_VH` as `VV - VH` in dB space.
4. Build a Dynamic World yearly product using:
   - modal `label`
   - mean per-class probability bands
5. Build a common valid-data mask from AlphaEarth, Sentinel-1, and Dynamic World.
6. Run `stratifiedSample` over Dynamic World labels with up to 80 points per class per region.
7. Attach `longitude`, `latitude`, `region`, and `year`.
8. Export per-region CSVs, the combined CSV, and per-region Sentinel-2 RGB GeoTIFFs.

## GEE Scripts

Run these in the Earth Engine Code Editor or adapt them to your preferred Earth Engine client workflow:

- `gee/export_dynamic_world_joined_samples.js`
  Exact merged-table export workflow.
  Produces:
  - `alphaearth_s1_dw_samples_<region>_2024.csv`
  - `alphaearth_s1_dw_samples_all_regions_2024.csv`
- `gee/export_sentinel2_context_tiffs.js`
  Exports per-region RGB GeoTIFFs:
  - `sentinel2_context_<region>_2024.tif`
- `gee/export_alphaearth_embeddings_2024.js`
  Reconstructed AlphaEarth-only convenience export over the same sampled points.
- `gee/export_sentinel1_sar_2024.js`
  Reconstructed Sentinel-1-only convenience export over the same sampled points.

After exporting from Earth Engine:

1. Place the merged and per-region CSVs in `DataSources/`.
2. Create `DataSources/sentinel2_context/`.
3. Place the exported TIFFs there, preserving filenames such as `sentinel2_context_sf_bay_urban_2024.tif`.

Expected filenames for the standard local workflow:

- `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv`
- `DataSources/alphaearth_s1_dw_samples_sf_bay_urban_2024.csv`
- `DataSources/alphaearth_s1_dw_samples_iowa_ag_2024.csv`
- `DataSources/alphaearth_s1_dw_samples_amazon_forest_2024.csv`
- `DataSources/alphaearth_s1_dw_samples_california_coast_2024.csv`

Optional but supported inputs:

- `DataSources/AlphaEarth_Embeddings_2024.csv`
- `DataSources/Sentinel1_SAR_2024.csv`
- `DataSources/sentinel2_context/sentinel2_context_<region>_2024.tif`

## Local Python Environment

The original project did not preserve an exact Python version or fully pinned dependency lockfile. The repository includes a minimal dependency manifest in `requirements.txt`, so environment setup is reproducible at the package-family level but not yet frozen at the exact-version level.

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 --version
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `lightgbm` and `rasterio` are compiled dependencies. If installation fails on your machine, the failure is environmental rather than a missing project file.
- If matplotlib needs a writable config directory, run the plotting scripts with `MPLCONFIGDIR=/tmp/matplotlib`.
- The README does not claim exact dependency pinning. If you need strong environment reproducibility, create a lockfile after a successful install on your target platform.

## Local Workflow

### Option A: Start from committed merged data

Use this path if `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv` already exists and you want to reproduce the downstream local analysis.

This is the canonical reproducible path for this repository.

Preconditions:

- You are running from the repository root
- `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv` exists
- Your Python environment is installed

Execution contract:

- `phase2_full_dataset_lightgbm_experiments.py` produces the tuned model outputs and parameter JSON files used by later phases
- `phase3_failure_analysis.py` depends on Phase 2 outputs
- `phase4_cross_modal_similarity_analysis.py` depends on the merged dataset and optionally on Sentinel-2 TIFFs
- `phase5_data_sufficiency_analysis.py` depends on the LightGBM parameter files written by Phase 2
- `build_project_reports.py` depends on the CSV and JSON outputs written by Phases 2 to 5

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase2_full_dataset_lightgbm_experiments.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase3_failure_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase4_cross_modal_similarity_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase5_data_sufficiency_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/build_project_reports.py
```

### Option B: Rebuild from separate AlphaEarth and Sentinel-1 point tables

Use this path if you exported `AlphaEarth_Embeddings_2024.csv` and `Sentinel1_SAR_2024.csv` separately and want to regenerate the offline nearest-neighbor join audit.

This path is for the spatial-join audit only. It does not replace the standard downstream workflow built around the committed merged CSV.

Preconditions:

- The AlphaEarth and Sentinel-1 CSVs exist somewhere under the repository tree
- Their schemas still expose enough embedding columns and SAR-like column names for automatic detection

Important limitation:

- The join script auto-discovers candidate CSV files heuristically. If you place many unrelated CSVs under the repository, inspect the selected inputs before treating the join as authoritative.

```bash
python3 scripts/offline_spatial_join_and_sanity_eda.py
```

That script:

- searches the repo tree for AlphaEarth and Sentinel-1 CSV inputs
- joins them with a nearest-neighbor KD-tree using a 10 m threshold
- writes `outputs/eda/alphaearth_sentinel1_merged.csv`
- writes histogram and PCA diagnostics to `outputs/eda/`

### Full report regeneration

For a clean rerun from committed merged data, use exactly this order:

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase2_full_dataset_lightgbm_experiments.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase3_failure_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase4_cross_modal_similarity_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase5_data_sufficiency_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/build_project_reports.py
```

## Script Order

1. `scripts/offline_spatial_join_and_sanity_eda.py`
   Builds or audits a merged AlphaEarth + Sentinel-1 point table when starting from separate exports.
2. `scripts/phase2_full_dataset_lightgbm_experiments.py`
   Runs the main modeling pipeline on the full dataset.
3. `scripts/phase3_failure_analysis.py`
   Summarizes residual failures by land use, region, and spatial pattern.
4. `scripts/phase4_cross_modal_similarity_analysis.py`
   Measures embedding-space versus SAR-space similarity.
   If `DataSources/sentinel2_context/` is missing, all metrics still run, but the qualitative chip figure becomes a placeholder.
5. `scripts/phase5_data_sufficiency_analysis.py`
   Measures coverage, redundancy, and learning-curve behavior for the current sampling design.
   This script requires the `best_params_*_lightgbm.json` files produced by Phase 2.
6. `scripts/build_project_reports.py`
   Regenerates the phase PDFs and the combined summary report.
   This script requires the machine-readable outputs produced by Phases 2 through 5.

## What Each Script Reads and Writes

### `scripts/phase2_full_dataset_lightgbm_experiments.py`

Reads:

- `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv`

Writes:

- `outputs/full_dataset/full_dataset_lightgbm_metrics.csv`
- `outputs/full_dataset/full_dataset_lightgbm_stability.csv`
- `outputs/full_dataset/full_dataset_lightgbm_regional_metrics.csv`
- `outputs/full_dataset/full_dataset_lightgbm_land_use_metrics.csv`
- `outputs/full_dataset/full_dataset_polarization_difference_metrics.csv`
- `outputs/full_dataset/full_dataset_ratio_baseline_metrics.csv`
- `outputs/full_dataset/best_params_*_lightgbm.json`
- prediction CSVs, feature-importance CSVs, and plots under `outputs/full_dataset/`

### `scripts/phase3_failure_analysis.py`

Reads:

- `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv`
- Phase 2 outputs under `outputs/full_dataset/`

Writes:

- `outputs/full_dataset/phase3_summary.json`
- `outputs/full_dataset/phase3_*`

### `scripts/phase4_cross_modal_similarity_analysis.py`

Reads:

- `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv`
- optionally `DataSources/sentinel2_context/*.tif`

Writes:

- `outputs/full_dataset/phase4_summary.json`
- `outputs/full_dataset/phase4_*`

### `scripts/phase5_data_sufficiency_analysis.py`

Reads:

- `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv`
- Phase 2 `best_params_*_lightgbm.json`

Writes:

- `outputs/full_dataset/data_sufficiency_*`
- `reports/data_sufficiency_summary_report.pdf`

### `scripts/build_project_reports.py`

Reads:

- Phase 2 to Phase 5 machine-readable outputs under `outputs/full_dataset/`

Writes:

- `reports/phase2_modeling_summary_report.pdf`
- `reports/phase3_failure_analysis_summary_report.pdf`
- `reports/phase4_cross_modal_similarity_summary_report.pdf`
- `reports/project_summary_report.pdf`
- mirrored copies under `outputs/full_dataset/`

## Key Outputs

Primary machine-readable outputs are written to `outputs/full_dataset/`.

Core files include:

- `full_dataset_lightgbm_metrics.csv`
- `full_dataset_polarization_difference_metrics.csv`
- `full_dataset_ratio_baseline_metrics.csv`
- `phase3_summary.json`
- `phase4_summary.json`
- `data_sufficiency_summary_report.pdf`

Human-facing reports are written to both `reports/` and `outputs/full_dataset/`.

## Recommended Reading Order

If you are new to the repository, start with:

1. `reports/final_report.md`
2. `reports/project_summary_report.pdf`
3. `outputs/full_dataset/phase3_summary.json`
4. `outputs/full_dataset/phase4_summary.json`
5. `outputs/full_dataset/data_sufficiency_summary_report.pdf`
6. `scripts/build_project_reports.py`

## Main Reports

- `reports/phase2_modeling_summary_report.pdf`
- `reports/phase3_failure_analysis_summary_report.pdf`
- `reports/phase4_cross_modal_similarity_summary_report.pdf`
- `reports/data_sufficiency_summary_report.pdf`
- `reports/project_summary_report.pdf`

The current `project_summary_report.pdf` summarizes:

- how Ridge and LightGBM behave on each prediction target
- whether direct polarization-difference prediction matches the structural baseline `VV_hat - VH_hat`
- how the derived linear `VV/VH` ratio behaves under direct-ratio versus log-ratio modeling
- where errors concentrate by land use and region
- how strongly embedding-space similarity aligns with SAR-space similarity
- whether the current balanced sample design is sufficient for the observed modeling performance

## Notes On Target Definitions

- `S1_VV` and `S1_VH` are stored in dB.
- `S1_VV_div_VH` is not a raw linear ratio in the source table. It is the stored polarization difference `S1_VV - S1_VH`.
- The repository also includes ratio-baseline experiments on a derived linear `VV/VH` target, including a Ridge log-ratio baseline.

## What Is Still Needed For Stronger Reproducibility

You do not need to provide anything for the standard downstream workflow from the committed merged CSV.

You would only need to provide additional artifacts if you want a stronger reproducibility guarantee than the repository currently supports. The missing items are:

- the exact historical Python version used for the original runs
- a fully pinned lockfile or exported environment file
- the exact historical standalone Earth Engine export scripts for `AlphaEarth_Embeddings_2024.csv` and `Sentinel1_SAR_2024.csv`, if you want those specific convenience exports reproduced byte-for-byte
- regenerated Sentinel-2 context TIFFs, if you want the full Phase 4 qualitative figure rather than the placeholder path
