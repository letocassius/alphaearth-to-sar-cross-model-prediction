# Single-Image SAR-to-AlphaEarth Reconstruction Report

## Objective

This experiment tests whether a compact Sentinel-1 SAR scene can be used to reconstruct the AlphaEarth embedding field over the same area.

The goal is not to predict one optical band or one embedding band in isolation. The goal is to take SAR-only input and reproduce the full 64-band AlphaEarth embedding image over one complete scene.

The chosen area is a small San Francisco scene covering downtown San Francisco and the Golden Gate Bridge. That AOI is large enough to contain complex urban, water, coastline, and bridge structure, but still small enough to support fast end-to-end reconstruction.

## Inputs

- SAR image: `DataSources/single_image_sar_reconstruction/sentinel1_small_vv_vh_sf_downtown_golden_gate_2024.tif`
- Full truth stack: `DataSources/single_image_sar_reconstruction/sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif`
- Spatial extent: approximately `668 x 1894` pixels at `10 m`
- SAR predictors: `S1_VV`, `S1_VH`, `S1_VV_div_VH`
- Reconstruction targets: `A00` to `A63`

The truth stack contains both the SAR bands and the AlphaEarth embedding bands. That makes it possible to train on sampled pixels and then compare the reproduced image against the true AlphaEarth image everywhere in the AOI.

## Modeling Setup

To keep the run fast and stable, the model was trained on a small random sample of valid pixels rather than on the full scene.

- Sampled rows: `2511`
- Train rows: `2008`
- Test rows: `503`
- Model: `PolynomialFeatures(degree=2) + StandardScaler + Ridge(alpha=10.0)`

This model choice is deliberately simple. The point of this report is to measure whether the SAR signal is strong enough to recover the embedding field over an entire image, not to maximize leaderboard performance with a heavy model.

## Reconstruction Procedure

The workflow was:

1. Sample valid pixels from the downtown-plus-Golden-Gate stack.
2. Fit one multi-output regression model from the three SAR inputs to the 64 AlphaEarth target bands.
3. Apply the fitted model to every valid pixel in the scene.
4. Write the reproduced 64-band embedding image as a GeoTIFF.
5. Compare the reproduced image against the true AlphaEarth image band-by-band and pixel-by-pixel.

The full reproduced image is:

- `outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/predicted_tiles/predicted_alphaearth_from_sar_small_stack_sf_downtown_golden_gate_2024.tif`

## Quantitative Results

### Held-Out Sample Performance

- Mean R^2 across 64 bands: `0.491`
- Median R^2 across 64 bands: `0.495`
- Mean RMSE across 64 bands: `0.0478`
- Mean Pearson r across 64 bands: `0.686`
- Best band: `A27` with `R^2 = 0.916`
- Worst band: `A20` with `R^2 = 0.113`

### Full-Image Reconstruction Performance

- Mean R^2 across 64 bands: `0.480`
- Median R^2 across 64 bands: `0.484`
- Mean RMSE across 64 bands: `0.0501`
- Mean Pearson r across 64 bands: `0.671`
- Best band: `A27` with `R^2 = 0.889`
- Worst band: `A28` with `R^2 = 0.075`

## Interpretation

The central result is that the model reproduces the AlphaEarth embedding field moderately well from Sentinel-1 alone, but not uniformly across all embedding dimensions.

The held-out and full-image metrics are close. That matters. It means the reconstruction quality seen on the test sample is not collapsing when the model is applied to the whole image. In other words, the model is not just memorizing sampled pixels. It is learning a scene-level mapping that transfers across the AOI.

At the same time, the reconstruction is incomplete. A mean full-image `R^2` of `0.480` says that SAR-only input can explain a meaningful fraction of AlphaEarth variation, but it does not recover the full optical embedding space. Some bands are strongly reproducible, while others remain weak.

That pattern is consistent with the broader project result: SAR and AlphaEarth are related, but they are not interchangeable representations.

## Single All-Band Reproduced Image

The figure below is the most important visualization in this report.

Because the AlphaEarth image has 64 bands, it cannot be shown directly as one normal RGB image. To create one single interpretable image from all bands, the true 64-band AlphaEarth field was projected into 3 dimensions using PCA, and the reproduced embedding was projected using the same PCA basis.

To make that distinction explicit, the following figure compares a real Sentinel-2 natural-color image with the AlphaEarth PCA RGB visualization for the same AOI.

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/sentinel2_vs_alphaearth_pca_rgb.png){ width=100% }

The next three figures show the same all-band representation at a much larger scale:

- the true all-band AlphaEarth image
- the SAR-reproduced all-band AlphaEarth image
- the residual intensity, measured as per-pixel RMSE across all 64 bands

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/true_embedding_all_band_large.png){ width=100% }

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/reproduced_embedding_all_band_large.png){ width=100% }

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/residual_heatmap_all_band_large.png){ width=100% }

This figure is the best single summary of whether the scene was reproduced. The reproduced image preserves broad spatial organization and major structural transitions, while the residual map shows where the SAR-only model fails to match the full embedding.

## Residual Summary

The following graph summarizes reconstruction quality by embedding band.

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/residual_summary_by_band.png){ width=70% }

This plot separates two ideas:

- `R^2` indicates how much bandwise variation is recovered
- `RMSE` and `MAE` indicate the absolute residual size

Bands with high `R^2` and low residual error are the bands most reproducible from SAR. Bands with low `R^2` and higher residual error are the dimensions where the SAR-to-embedding mapping is weakest.

## Additional Diagnostics

The secondary diagnostics are intentionally reduced here. Their main role is just to confirm that sampled performance and full-image performance stay close.

![](../outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/heldout_vs_full_r2.png){ width=70% }

That comparison supports the same conclusion as the headline metrics: the model generalizes reasonably consistently from the sample to the full scene, but the quality ceiling is band-dependent and clearly limited.

## Conclusion

For this downtown San Francisco plus Golden Gate Bridge AOI, Sentinel-1 can reproduce a meaningful part of the AlphaEarth embedding image using a small training sample and a lightweight regression model.

That is a useful result. It shows that SAR contains enough information to reconstruct substantial structure in the AlphaEarth space over a full image, not just at isolated sampled pixels.

But the reproduction is only partial. The embedding is not fully recoverable from SAR alone, and the residual structure shows that some AlphaEarth dimensions depend on information outside what this SAR input can provide.

The practical takeaway is:

- full-image SAR-to-AlphaEarth reconstruction is feasible
- it works well enough to preserve broad structure
- it is not accurate enough to claim that Sentinel-1 fully reproduces the AlphaEarth representation
