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

The local analysis and report-generation workflow is fully represented in this repository.

The upstream Earth Engine export workflow is represented by the scripts in `gee/`. Those scripts are sufficient to regenerate the study tables and the omitted Sentinel-2 context rasters. Two caveats apply:

- The committed merged-table export workflow is exact, because it comes directly from the recorded Earth Engine code.
- The standalone `AlphaEarth_Embeddings_2024.csv` and `Sentinel1_SAR_2024.csv` exports are reconstructed convenience scripts derived from the merged workflow. They reproduce equivalent point tables over the same sampled points, but they should not be treated as guaranteed byte-for-byte originals unless you still have the exact historical export code.

Large Sentinel-2 context GeoTIFFs are intentionally not tracked in git. Regenerate Sentinel-2 TIFFs with the provided GEE code before running the full phase 4 qualitative figure workflow.

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

## Local Python Environment

The original project did not preserve an exact Python version or fully pinned dependency lockfile. The repository now includes a minimal dependency manifest in `requirements.txt`.

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If matplotlib needs a writable config directory, run the plotting scripts with `MPLCONFIGDIR=/tmp/matplotlib`.

## Local Workflow

### Option A: Start from committed merged data

Use this path if `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv` already exists and you only need to regenerate model outputs and reports.

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase2_full_dataset_lightgbm_experiments.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase3_failure_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase4_cross_modal_similarity_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase5_data_sufficiency_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/build_project_reports.py
```

### Option B: Rebuild from separate AlphaEarth and Sentinel-1 point tables

Use this path if you exported `AlphaEarth_Embeddings_2024.csv` and `Sentinel1_SAR_2024.csv` separately and want to regenerate the offline nearest-neighbor join audit.

```bash
python3 scripts/offline_spatial_join_and_sanity_eda.py
```

That script:

- searches the repo tree for AlphaEarth and Sentinel-1 CSV inputs
- joins them with a nearest-neighbor KD-tree using a 10 m threshold
- writes `outputs/eda/alphaearth_sentinel1_merged.csv`
- writes histogram and PCA diagnostics to `outputs/eda/`

### Full report regeneration

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
6. `scripts/build_project_reports.py`
   Regenerates the phase PDFs and the combined summary report.

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
