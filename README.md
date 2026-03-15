# alphaearth-to-sar-cross-model-prediction

Offline workflow for testing whether AlphaEarth optical embeddings can predict Sentinel-1 SAR backscatter, then diagnosing where that relationship holds, where it breaks down, and how derived polarization targets behave.

## Project Scope

The repository works from a merged AlphaEarth + Sentinel-1 sample table and evaluates whether a 64-dimensional embedding can recover:

- `S1_VV` in dB
- `S1_VH` in dB
- `S1_VV_div_VH`, which in this dataset is the dB-space polarization difference `S1_VV - S1_VH`

The current workflow uses the full balanced dataset of 2,880 samples and reports results for:

- LightGBM baselines on the core SAR targets
- Ridge follow-up baselines for polarization-difference experiments
- Derived linear `VV/VH` ratio baselines, including a log-ratio formulation
- Failure analysis by land use and region
- Cross-modal similarity analysis between embedding space and SAR space

## Workflow

1. `scripts/offline_spatial_join_and_sanity_eda.py`
   Builds or audits the merged AlphaEarth + Sentinel-1 table and produces sanity-check outputs.
2. `scripts/phase2_full_dataset_lightgbm_experiments.py`
   Runs the main modeling pipeline on the full dataset.
   Outputs include:
   - LightGBM experiments for `S1_VV`, `S1_VH`, and `S1_VV_div_VH`
   - Polarization-difference comparison blocks
   - Derived linear `VV/VH` ratio baselines
3. `scripts/phase3_failure_analysis.py`
   Summarizes residual failures by land use, region, and spatial pattern.
4. `scripts/phase4_cross_modal_similarity_analysis.py`
   Measures embedding-space versus SAR-space similarity.
5. `scripts/build_project_reports.py`
   Regenerates the phase PDFs and the combined executive summary report.

## Key Outputs

Primary machine-readable outputs are written to `outputs/full_dataset/`.

Core files include:

- `full_dataset_lightgbm_metrics.csv`
- `full_dataset_polarization_difference_metrics.csv`
- `full_dataset_ratio_baseline_metrics.csv`
- `phase3_summary.json`
- `phase4_summary.json`

Human-facing reports are written to both `reports/` and `outputs/full_dataset/`.

## Main Reports

- `reports/phase2_modeling_summary_report.pdf`
- `reports/phase3_failure_analysis_summary_report.pdf`
- `reports/phase4_cross_modal_similarity_summary_report.pdf`
- `reports/project_summary_report.pdf`

The current `project_summary_report.pdf` is results-first and summarizes:

- how Ridge and LightGBM behave on each prediction target
- whether direct polarization-difference prediction matches the structural baseline `VV_hat - VH_hat`
- how the derived linear `VV/VH` ratio behaves under direct-ratio versus log-ratio modeling
- where errors concentrate by land use and region
- how strongly embedding-space similarity aligns with SAR-space similarity

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib optuna lightgbm rasterio
```

If matplotlib needs a writable config directory, run the modeling and reporting scripts with `MPLCONFIGDIR=/tmp/matplotlib`.

## Run

```bash
python3 scripts/offline_spatial_join_and_sanity_eda.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase2_full_dataset_lightgbm_experiments.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase3_failure_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/phase4_cross_modal_similarity_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python3 scripts/build_project_reports.py
```

## Notes On Target Definitions

- `S1_VV` and `S1_VH` are stored in dB.
- `S1_VV_div_VH` is not a raw linear ratio in the source table. It is the stored polarization difference `S1_VV - S1_VH`.
- The repository now also includes separate ratio-baseline experiments on a derived linear `VV/VH` target for comparison, including a Ridge log-ratio baseline.
