# Single-Image Pixel Fraction Report

## Scope

This report summarizes the single-image experiment built from one Google Earth Engine Sentinel-2 export paired with AlphaEarth annual embeddings at the same pixel locations.

The experiment asks a narrower question than the main SAR workflow:

- can the 64-dimensional AlphaEarth embedding predict one Sentinel-2 pixel value
- how does performance change as the fraction of training pixels increases

This run uses:

- input CSV: `DataSources/single_image_pixel_fraction/sentinel2_alphaearth_pixel_pairs_2024_n5000.csv`
- input TIFF: `DataSources/single_image_pixel_fraction/sentinel2_rgb_2024.tif`
- target pixel band: `B4`
- model: `StandardScaler + Ridge(alpha=1.0)`
- random train/test split: `80% / 20%`

## Source Artifacts Used

- `outputs/single_image_pixel_fraction/sentinel2_alphaearth_pixel_pairs_2024_n5000_B4_fraction_metrics.csv`
- `outputs/single_image_pixel_fraction/sentinel2_alphaearth_pixel_pairs_2024_n5000_B4_test_predictions.csv`
- `outputs/single_image_pixel_fraction/sentinel2_alphaearth_pixel_pairs_2024_n5000_B4_run_metadata.json`
- `outputs/single_image_pixel_fraction/sentinel2_alphaearth_pixel_pairs_2024_n5000_B4_test_predictions_fraction_0p1_overlay.png`

## Dataset Summary

After dropping rows with missing values, the experiment used `4,544` sampled pixels:

- training rows: `3,635`
- test rows: `909`

The Earth Engine export originally targeted `5,000` random pixels, so the final usable count is slightly smaller after validity filtering and NA removal.

## Fraction Sweep Results

| Training fraction | Train rows | Test rows | R^2 | RMSE | MAE | Pearson r |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.01` | 50 | 909 | 0.365 | 0.0426 | 0.0219 | 0.619 |
| `0.05` | 182 | 909 | 0.578 | 0.0348 | 0.0183 | 0.770 |
| `0.10` | 364 | 909 | 0.621 | 0.0329 | 0.0170 | 0.797 |
| `0.25` | 909 | 909 | 0.666 | 0.0309 | 0.0169 | 0.817 |
| `0.50` | 1,818 | 909 | 0.676 | 0.0304 | 0.0176 | 0.823 |
| `1.00` | 3,635 | 909 | 0.668 | 0.0308 | 0.0182 | 0.821 |

## Main Findings

Three points stand out.

### 1. Small fractions already carry useful signal

Even with only `1%` of the training set, the embedding predicts held-out `B4` values better than chance with:

- `R^2 = 0.365`
- `Pearson r = 0.619`

That means the annual AlphaEarth embedding is already encoding meaningful information about the visible red band at the pixel level.

### 2. Performance improves quickly, then starts to plateau

The largest gains happen between `1%` and `25%` of the available training rows:

- `R^2` rises from `0.365` to `0.666`
- `RMSE` drops from `0.0426` to `0.0309`

After that, the curve flattens. Moving from `25%` to `50%` provides only a small gain, and the `100%` run is slightly worse than `50%` on this split.

### 3. The best result on this split is at `50%`

For this specific train/test split and this linear model:

- best `R^2`: `0.676` at `50%`
- best `RMSE`: `0.0304` at `50%`
- best `Pearson r`: `0.823` at `50%`

The small drop at `100%` should not be overinterpreted. It likely reflects ordinary split variance and model bias rather than a real penalty from using more data.

## Interpretation

This experiment supports a practical conclusion: for a single Sentinel-2 scene paired with AlphaEarth annual embeddings, a moderate subset of sampled pixels is enough to recover much of the available predictive signal for `B4`.

That matters for the next stage of your workflow because it suggests you do not need dense supervision over the entire image to start learning an embedding-to-pixel mapping. A fraction in the `10%` to `25%` range already gets close to the best observed result in this run.

At the same time, this is still a single-image, single-band, single-split experiment using a linear model. It is best treated as a feasibility check, not as a final estimate of generalization performance.

## Qualitative Output

The held-out prediction overlay for the `0.10` fraction run is saved at:

- `outputs/single_image_pixel_fraction/sentinel2_alphaearth_pixel_pairs_2024_n5000_B4_test_predictions_fraction_0p1_overlay.png`

## Replication Path

The exact Earth Engine script used for this workflow is committed at:

- `gee/export_single_image_alphaearth_pixel_pairs.js`

To reproduce the local run, place the Earth Engine exports at:

- `DataSources/single_image_pixel_fraction/sentinel2_alphaearth_pixel_pairs_2024_n5000.csv`
- `DataSources/single_image_pixel_fraction/sentinel2_rgb_2024.tif`

The TIFF is intentionally not tracked in git because it is too large for normal repository use.

That figure shows:

- the Sentinel-2 RGB context image
- true held-out `B4` values
- predicted held-out `B4` values
- residuals

## Recommended Next Steps

- repeat the same experiment for `B8`, `B11`, and `B12` to see whether the embedding is more aligned with NIR or SWIR than with visible red
- run repeated random splits to replace the single-split estimate with a stability range
- add a summary plot of `R^2` and `RMSE` versus training fraction
- if the goal is image reconstruction, test nonlinear models after this linear baseline
