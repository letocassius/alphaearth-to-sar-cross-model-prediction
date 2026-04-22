# Single-Image AlphaEarth-to-SAR Reconstruction Report

## Objective

This experiment tests whether colocated AlphaEarth embeddings can reproduce a Sentinel-1 SAR image over one compact scene.

The model direction is now `F(AlphaEarth embedding) = SAR`. Each sampled SAR pixel supplies the target value `Y`, and the colocated 64-dimensional AlphaEarth vector supplies the input features `X`.

The chosen area is a small San Francisco scene covering downtown San Francisco and the Golden Gate Bridge. That AOI contains urban structure, water, coastline, and bridge geometry while remaining small enough for full-scene reconstruction.

## Inputs

- SAR image: `DataSources/single_image_sar_reconstruction/sentinel1_small_vv_vh_sf_downtown_golden_gate_2024.tif`
- Colocated SAR + AlphaEarth stack: `DataSources/single_image_sar_reconstruction/sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif`
- Spatial extent: approximately `668 x 1894` pixels at `10 m`
- Predictors: `A00` to `A63`
- Reconstruction targets: `S1_VV`, `S1_VH`, `S1_VV_div_VH`

The stack already contains the required colocated inputs, so no new data export was needed for this AOI. A new Earth Engine export is only needed if the AOI, year, SAR preprocessing, or AlphaEarth version changes.

## Modeling Setup

A subset of valid Sentinel-1 pixels was sampled and saved with image-array coordinates so the same pixels can be identified during the full-scene fill step.

- Sampling strategy: `random`
- Sampling percentage: `0.200%`
- Sampled rows: `2511`
- Train rows: `2008`
- Test rows: `503`
- Model: `MultiOutputRegressor(LGBMRegressor)`
- SARhat fill rule: `observed SAR retained at training pixels; model predictions used for all other valid pixels`

The sampled pixel coordinate table is:

- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/sampled_pixel_locations.csv`

The sampled modeling table with coordinates, split labels, all 64 AlphaEarth features, and SAR targets is:

- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/sampled_alphaearth_to_sar_dataset.csv`

## Reconstruction Procedure

The workflow was:

1. Fetch and use the local Sentinel-1 image for the target SAR bands.
2. Use the colocated full stack to fetch the AlphaEarth embedding at each SAR pixel.
3. Sample valid SAR pixels and save their row/column locations.
4. Train `F(X) = Y`, where `X` is the 64-band AlphaEarth embedding and `Y` is the three-band SAR vector.
5. Iterate over every valid SAR pixel in the scene.
6. Copy observed SAR values at training pixels and predict all unsampled pixels from their AlphaEarth embeddings.
7. Save the completed SAR reconstruction as a GeoTIFF and compare it with the actual SAR image.

The reconstructed full SAR image is:

- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/sar_hat_from_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif`

## Quantitative Results

### Held-Out Sample Performance

- Mean R^2 across SAR bands: `0.924`
- Mean RMSE across SAR bands: `1.0831`
- Mean Pearson r across SAR bands: `0.961`

| Band | Count | R^2 | RMSE | MAE | Pearson r |
|---|---:|---:|---:|---:|---:|
| `S1_VV` | 503 | 0.970 | 1.0462 | 0.6740 | 0.985 |
| `S1_VH` | 503 | 0.978 | 1.1291 | 0.8437 | 0.989 |
| `S1_VV_div_VH` | 503 | 0.825 | 1.0739 | 0.8306 | 0.909 |

### Gap-Fill Performance

These metrics evaluate only pixels not copied from the training dataset.

- Mean R^2 across SAR bands: `0.910`
- Mean RMSE across SAR bands: `1.1934`
- Mean Pearson r across SAR bands: `0.953`

| Band | Count | R^2 | RMSE | MAE | Pearson r |
|---|---:|---:|---:|---:|---:|
| `S1_VV` | 1263184 | 0.968 | 1.1121 | 0.7106 | 0.984 |
| `S1_VH` | 1263184 | 0.974 | 1.2567 | 0.9003 | 0.987 |
| `S1_VV_div_VH` | 1263184 | 0.789 | 1.2114 | 0.8895 | 0.889 |

### Full SARhat Performance

These metrics compare the completed SARhat image against the actual SAR image over all valid pixels.

- Mean R^2 across SAR bands: `0.910`
- Mean RMSE across SAR bands: `1.1925`
- Mean Pearson r across SAR bands: `0.953`

| Band | Count | R^2 | RMSE | MAE | Pearson r |
|---|---:|---:|---:|---:|---:|
| `S1_VV` | 1265192 | 0.968 | 1.1112 | 0.7094 | 0.984 |
| `S1_VH` | 1265192 | 0.974 | 1.2557 | 0.8989 | 0.987 |
| `S1_VV_div_VH` | 1265192 | 0.790 | 1.2104 | 0.8881 | 0.889 |

## Interpretation

The AlphaEarth embeddings contain strong supervised signal for reconstructing direct Sentinel-1 backscatter in this scene. The held-out and gap-fill metrics are the most important checks because they evaluate pixels not used to fit the model.

The `S1_VV_div_VH` target remains the hardest band because it is a derived polarization relationship rather than a direct backscatter channel. That pattern is consistent with the broader tabular results in this repository.

## Context Figure

The following figure shows the Sentinel-2 natural-color scene beside a PCA visualization of the AlphaEarth embedding field. It is included only as spatial context for the features used by the model.

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/sentinel2_vs_alphaearth_pca_rgb.png){ width=100% }

## Full-Scene SARhat

The next three figures show the actual SAR image, the reconstructed SARhat image, and the spatial residual intensity measured as per-pixel RMSE across the three SAR bands.

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/true_sar_all_band_large.png){ width=100% }

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/sarhat_all_band_large.png){ width=100% }

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/residual_heatmap_sar_large.png){ width=100% }

The residual map is the main geospatial diagnostic. It shows where the AlphaEarth-to-SAR mapping transfers cleanly and where SAR behavior is driven by geometry or scattering effects that are not fully represented by the embedding.

## Residual Summary

The following graph summarizes reconstruction quality by SAR band.

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/residual_summary_by_sar_band.png){ width=70% }

The secondary diagnostic compares held-out sample performance, gap-fill performance, and full SARhat performance.

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/heldout_vs_full_r2.png){ width=70% }

## Conclusion

For this downtown San Francisco plus Golden Gate Bridge AOI, AlphaEarth embeddings can reproduce much of the Sentinel-1 SAR image through supervised regression.

The current local data is sufficient for this corrected single-image workflow. More data would be useful if the goal shifts from a single-scene demonstration to a robust operational model: additional AOIs, seasons, incidence-angle regimes, and urban/coastal geometries would reduce the risk that the fitted mapping is scene-specific.
