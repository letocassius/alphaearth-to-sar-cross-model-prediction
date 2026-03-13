# alphaearth-to-sar-cross-model-prediction

Offline workflow for testing whether AlphaEarth optical embeddings can predict Sentinel-1 SAR backscatter, then diagnosing where that relationship breaks down.

## Workflow

1. `scripts/offline_spatial_join_and_sanity_eda.py`
   Builds the merged AlphaEarth + Sentinel-1 table and basic EDA outputs.
2. `scripts/phase2_full_dataset_lightgbm_experiments.py`
   Runs full-dataset LightGBM modeling on all 2,880 balanced samples.
3. `scripts/phase3_failure_analysis.py`
   Summarizes residual failures by land use, region, and spatial pattern.
4. `scripts/phase4_cross_modal_similarity_analysis.py`
   Measures embedding-space versus SAR-space similarity.
5. `scripts/build_project_reports.py`
   Generates the phase PDFs and the combined project summary.

## Key Paths

- `DataSources/`: source CSVs and Sentinel-2 context rasters
- `outputs/full_dataset/`: metrics, predictions, plots, and machine-generated reports
- `reports/`: human-facing PDF summaries

## Main Reports

- `reports/phase2_modeling_summary_report.pdf`
- `reports/phase3_failure_analysis_summary_report.pdf`
- `reports/phase4_cross_modal_similarity_summary_report.pdf`
- `reports/project_summary_report.pdf`

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib optuna lightgbm rasterio

python3 scripts/offline_spatial_join_and_sanity_eda.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase2_full_dataset_lightgbm_experiments.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase3_failure_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase4_cross_modal_similarity_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/build_project_reports.py
```
