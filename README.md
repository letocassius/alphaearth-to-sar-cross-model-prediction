# alphaearth-to-sar-cross-model-prediction

Reproducible code and data for two workflows:

1. a full-dataset study of whether AlphaEarth embeddings predict Sentinel-1 SAR targets across four regions
2. a single-image reconstruction study of whether Sentinel-1 can reproduce the full 64-band AlphaEarth embedding image over downtown San Francisco plus the Golden Gate Bridge

This README is intentionally narrow. It only documents the workflows, files, and commands that are still kept in the repository.

## Repository Layout

### `DataSources/`

Canonical local inputs for the two supported workflows.

Full-dataset study:

- `alphaearth_s1_dw_samples_all_regions_2024.csv`
- `alphaearth_s1_dw_samples_sf_bay_urban_2024.csv`
- `alphaearth_s1_dw_samples_iowa_ag_2024.csv`
- `alphaearth_s1_dw_samples_amazon_forest_2024.csv`
- `alphaearth_s1_dw_samples_california_coast_2024.csv`

Single-image SAR reconstruction:

- `single_image_sar_reconstruction/sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif`
- `single_image_sar_reconstruction/sentinel1_small_vv_vh_sf_downtown_golden_gate_2024.tif`
- `single_image_sar_reconstruction/sentinel2_natural_color_sf_downtown_golden_gate_2024.tif`

### `gee/`

Google Earth Engine scripts for regenerating the upstream inputs.

- `export_dynamic_world_joined_samples.js`
  Exports the merged four-region point table used by the full-dataset study.
- `export_sentinel2_context_tiffs.js`
  Exports optional Sentinel-2 context images for the full-dataset study.
- `export_single_image_sar_reconstruction_stack.js`
  Exports the downtown San Francisco plus Golden Gate single-image SAR + AlphaEarth stack.
- `export_single_image_sentinel2_context.js`
  Exports the matching Sentinel-2 natural-color image for the single-image reconstruction AOI.

### `scripts/`

Supported local Python entrypoints.

Full-dataset study:

- `phase2_full_dataset_lightgbm_experiments.py`
- `phase3_failure_analysis.py`
- `phase4_cross_modal_similarity_analysis.py`
- `phase5_data_sufficiency_analysis.py`
- `build_project_reports.py`

Single-image SAR reconstruction:

- `run_single_image_sar_reconstruction.py`
- `build_single_image_sar_reconstruction_figures.py`
- `build_sentinel2_vs_alphaearth_pca_figure.py`

### `outputs/`

Generated artifacts. The two relevant output trees are:

- `outputs/full_dataset/`
- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/`

### `reports/`

Human-facing summaries generated from the outputs.

The two report entrypoints currently kept in the repo are:

- `reports/final_report.md`
- `reports/single_image_sar_reconstruction_sf_downtown_golden_gate_report.md`

## Environment Setup

Run all commands from the repository root.

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:

- `rasterio` and `lightgbm` are compiled dependencies.
- if matplotlib needs a writable cache, prefix commands with `MPLCONFIGDIR=/tmp/matplotlib`

## Workflow A: Full-Dataset Study

This is the main tabular study using the committed merged CSV.

### What it does

It models whether AlphaEarth embeddings predict:

- `S1_VV`
- `S1_VH`
- `S1_VV_div_VH`

Then it performs:

- failure analysis
- cross-modal similarity analysis
- data sufficiency analysis
- report generation

### Required input

You need:

- `DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv`

Optional:

- per-region CSVs already present in `DataSources/`
- Sentinel-2 context TIFFs if you want the qualitative Phase 4 image chips

### Exact run order

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/phase2_full_dataset_lightgbm_experiments.py
MPLCONFIGDIR=/tmp/matplotlib python scripts/phase3_failure_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python scripts/phase4_cross_modal_similarity_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python scripts/phase5_data_sufficiency_analysis.py
MPLCONFIGDIR=/tmp/matplotlib python scripts/build_project_reports.py
```

### Main outputs

Generated machine-readable outputs go to:

- `outputs/full_dataset/`

Generated report files include:

- `reports/phase2_modeling_summary_report.pdf`
- `reports/phase3_failure_analysis_summary_report.pdf`
- `reports/phase4_cross_modal_similarity_summary_report.pdf`
- `reports/data_sufficiency_summary_report.pdf`
- `reports/project_summary_report.pdf`
- `reports/final_report.md`

## Workflow B: Single-Image SAR Reconstruction

This workflow asks a different question from the full-dataset study.

Instead of predicting one scalar SAR target from AlphaEarth embeddings, it tries to reconstruct the entire 64-band AlphaEarth embedding image from Sentinel-1 alone over one compact AOI.

### AOI

The current single-image AOI covers:

- downtown San Francisco
- the Golden Gate Bridge

### Required inputs

You need all three files below:

- `DataSources/single_image_sar_reconstruction/sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif`
- `DataSources/single_image_sar_reconstruction/sentinel1_small_vv_vh_sf_downtown_golden_gate_2024.tif`
- `DataSources/single_image_sar_reconstruction/sentinel2_natural_color_sf_downtown_golden_gate_2024.tif`

### Exact reconstruction command

This command reproduces the single-image reconstruction currently documented in the repository:

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/run_single_image_sar_reconstruction.py \
  --full-stack-glob 'sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif' \
  --sar-path DataSources/single_image_sar_reconstruction/sentinel1_small_vv_vh_sf_downtown_golden_gate_2024.tif \
  --sample-probability 0.002 \
  --output-dir outputs/single_image_sar_reconstruction_sf_downtown_golden_gate \
  --report-path reports/single_image_sar_reconstruction_sf_downtown_golden_gate_report.md
```

### Exact figure-generation commands

After the reconstruction run completes, generate the supporting figures with:

```bash
MPLCONFIGDIR=/tmp/matplotlib python scripts/build_single_image_sar_reconstruction_figures.py
MPLCONFIGDIR=/tmp/matplotlib python scripts/build_sentinel2_vs_alphaearth_pca_figure.py
```

### Rebuild the PDF report

From the `reports/` directory:

```bash
pandoc single_image_sar_reconstruction_sf_downtown_golden_gate_report.md \
  -o single_image_sar_reconstruction_sf_downtown_golden_gate_report.pdf \
  --pdf-engine=pdflatex
```

### Main outputs

Core machine-readable outputs:

- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/full_image_metrics_by_band.csv`
- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/heldout_metrics_by_band.csv`
- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/run_metadata.json`
- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/predicted_tiles/predicted_alphaearth_from_sar_small_stack_sf_downtown_golden_gate_2024.tif`

Core figures:

- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/sentinel2_vs_alphaearth_pca_rgb.png`
- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/true_embedding_all_band_large.png`
- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/reproduced_embedding_all_band_large.png`
- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/residual_heatmap_all_band_large.png`
- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/residual_summary_by_band.png`

Report files:

- `reports/single_image_sar_reconstruction_sf_downtown_golden_gate_report.md`
- `reports/single_image_sar_reconstruction_sf_downtown_golden_gate_report.pdf`

## Regenerating the Upstream Inputs in Earth Engine

### Full-dataset point table

Run:

- `gee/export_dynamic_world_joined_samples.js`

Then place the exported CSVs under `DataSources/` with the filenames already used in this repository.

### Single-image SAR reconstruction inputs

Run:

- `gee/export_single_image_sar_reconstruction_stack.js`
- `gee/export_single_image_sentinel2_context.js`

Then place the exported TIFFs under:

- `DataSources/single_image_sar_reconstruction/`

with these exact filenames:

- `sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif`
- `sentinel1_small_vv_vh_sf_downtown_golden_gate_2024.tif`
- `sentinel2_natural_color_sf_downtown_golden_gate_2024.tif`

## What Was Removed

This repository previously contained older exploratory workflows that are no longer documented as supported:

- the sampled Sentinel-2 single-image pixel-fraction experiment
- the standalone AlphaEarth-only and Sentinel-1-only Earth Engine convenience exports
- stale reports tied to earlier large-area reconstruction attempts

Those files were removed so the repository matches the workflows described above.

## Reproducibility Contract

If someone clones the repository and starts from the files named in this README, the commands in this README are sufficient to regenerate the supported local outputs without guessing hidden steps.

The main assumptions are:

- Python and the packages in `requirements.txt` install successfully on the local machine
- Earth Engine access is already configured for the user if they need to regenerate upstream TIFFs or CSVs
- `pandoc` and a LaTeX engine such as `pdflatex` are available if they want PDF report regeneration
