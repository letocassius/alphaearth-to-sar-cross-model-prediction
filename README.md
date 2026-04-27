# AlphaEarth-to-Sentinel-1 SAR Reconstruction

This project tests whether **AlphaEarth satellite embeddings** can reconstruct **Sentinel-1 SAR backscatter** at the pixel level.

The workflow uses Google Earth Engine to export a colocated GeoTIFF containing Sentinel-1 SAR bands and 64-dimensional AlphaEarth embeddings. A Python pipeline then samples pixels, trains a LightGBM model, reconstructs a full predicted SAR image (`SARhat`), and evaluates residuals spatially and statistically.

## Research Question

**To what extent can AlphaEarth embeddings reconstruct Sentinel-1 SAR signals at the pixel level, and how does reconstruction performance vary across SAR bands and spatial regions?**

## Workflow Overview

The project has two parts:

```text
1. Google Earth Engine export
   Sentinel-1 + AlphaEarth → stacked GeoTIFF

2. Python pipeline
   stacked GeoTIFF → sampled dataset → LightGBM model → SARhat → metrics + figures
```

## Step 1: Export the GeoTIFF from Google Earth Engine

Open the Earth Engine Code Editor:

```text
https://code.earthengine.google.com/
```

Paste and run:

```text
earth_engine/export_single_image_sar_reconstruction_stack.js
```

Then go to the **Tasks** tab, click **Run**, and export the stacked GeoTIFF to Google Drive.

Download the output file:

```text
sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif
```

Place it here:

```text
data/raw/sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif
```

The stacked GeoTIFF should contain 67 bands:

```text
Bands 1-3:  S1_VV, S1_VH, S1_VV_div_VH
Bands 4-67: A00, A01, ..., A63
```

The raw GeoTIFF is intentionally excluded from GitHub because it is large.

## Step 2: Set Up the Python Environment

Using conda is recommended because `rasterio` and `lightgbm` have compiled dependencies.

```bash
conda env create -f environment.yml
conda activate alphaearth-sar
```

Alternative pip setup:

```bash
pip install -r requirements.txt
```

On macOS, LightGBM may require OpenMP:

```bash
brew install libomp
```

## Step 3: Run the Pipeline

From the project root:

```bash
python run_pipeline.py
```

Or pass a custom GeoTIFF path:

```bash
python run_pipeline.py --full-stack-path data/raw/YOUR_FILE.tif
```

## What the Pipeline Does

The model learns:

```text
X = 64-dimensional AlphaEarth embedding
Y = Sentinel-1 SAR values
```

Pipeline steps:

1. Load the colocated Sentinel-1 + AlphaEarth GeoTIFF.
2. Randomly sample a percentage of valid SAR pixels.
3. Save sampled pixel row/column locations.
4. Pair each SAR pixel target with its 64-dimensional AlphaEarth embedding.
5. Train a LightGBM model to learn `F(X) = Y`.
6. Iterate across the full image and predict SAR values for unsampled pixels.
7. Save the reconstructed full-scene SAR image as `SARhat`.
8. Compare actual SAR vs. predicted SARhat using residual metrics and figures.

## Main Outputs

Generated outputs are written to:

```text
data/processed/
reports/figures/
models/
```

Most important outputs:

```text
data/processed/gap_fill_metrics_by_band.csv
reports/figures/residual_heatmap_sar_large.png
reports/figures/residual_summary_by_sar_band.png
reports/figures/true_sar_all_band_large.png
reports/figures/sarhat_all_band_large.png
data/processed/sar_hat_from_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif
```

The reconstructed SARhat GeoTIFF is excluded from GitHub because it is large and can be regenerated.

## Current Results

Gap-fill performance was evaluated only on pixels not copied from the training dataset.

| Band | R² | RMSE | MAE | Pearson r |
|---|---:|---:|---:|---:|
| S1_VV | 0.968 | 1.112 | 0.711 | 0.984 |
| S1_VH | 0.974 | 1.257 | 0.900 | 0.987 |
| S1_VV_div_VH | 0.789 | 1.211 | 0.890 | 0.889 |

## Interpretation

The model reconstructs the raw Sentinel-1 SAR bands very accurately, with R² values above 0.96 for both VV and VH. This suggests that AlphaEarth embeddings encode substantial information related to radar backscatter in this urban coastal scene.

The derived polarization feature, `S1_VV_div_VH`, is harder to reconstruct, with an R² of approximately 0.79. This is expected because derived SAR relationships are more sensitive to nonlinear scattering behavior and noise.

## GitHub Notes

This repository is designed to stay lightweight and reproducible.

Committed:

```text
source code
Earth Engine export script
README / environment files
small final metrics
selected report figures
```

Excluded:

```text
raw GeoTIFFs
SARhat GeoTIFFs
sampled pixel CSVs
trained model files
large intermediate outputs
```

To reproduce the full output set, run:

```bash
python run_pipeline.py
```
