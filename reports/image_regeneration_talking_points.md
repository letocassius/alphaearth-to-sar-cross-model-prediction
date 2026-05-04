# Image Regeneration Summary and Talking Points

## Report Summary

This part of the project evaluates whether one satellite representation can be used to regenerate another image-like product over the same San Francisco downtown / Golden Gate scene. The main regeneration task predicts Sentinel-1 SAR imagery from 64-dimensional AlphaEarth embeddings. The output image, called `SARhat`, is compared against the true Sentinel-1 SAR image using held-out sample metrics, full-image metrics, gap-fill metrics, and residual heatmaps.

The regenerated SAR image preserves the major spatial structure of the original scene. Water, urban blocks, bridges, coastlines, islands, parks, and hilly terrain remain visually recognizable in the predicted image. Quantitatively, the raw SAR bands are reconstructed very well: full-image R2 is 0.968 for `S1_VV` and 0.974 for `S1_VH`, with Pearson correlations of 0.984 and 0.987. The derived `S1_VV_div_VH` band is harder to regenerate, but still shows useful predictive skill, with full-image R2 of 0.790 and Pearson r of 0.889.

The residual heatmaps show that errors are not evenly distributed. Lower residuals appear across large homogeneous regions such as open water and broad land-cover zones, while higher residuals appear around sharp edges and complex surfaces, especially bridges, dense urban blocks, shorelines, ports, and small islands. This pattern suggests that the model captures broad spatial and land-cover structure well, but has more difficulty with fine-scale radar scattering, high-contrast boundaries, and localized infrastructure.

The experiment also includes a reproduced AlphaEarth embedding image, visualized as PCA RGB from all 64 bands. This output shows that the regenerated image retains recognizable scene organization: water, urban grids, islands, bridges, and terrain zones remain spatially coherent. However, because no numeric metric table was generated for the embedding-regeneration view, this part should be discussed as a qualitative visual comparison unless additional metrics are added.

## Main Talking Points

### 1. The regenerated SAR image is visually coherent.

The `SARhat` image is not random texture. It reconstructs the layout of the Bay Area scene in a spatially meaningful way. Major geographic features such as the Golden Gate Bridge, Bay Bridge, Treasure Island, Alcatraz, San Francisco's street grid, coastlines, and open-water regions remain recognizable.

Talking point: The model learned enough from AlphaEarth embeddings to regenerate an image that preserves the scene's physical geography, not just aggregate statistics.

### 2. Raw SAR bands regenerate much better than the derived ratio.

`S1_VV` and `S1_VH` are the strongest regenerated channels. Their full-image R2 values are 0.968 and 0.974, and their correlations with the true SAR bands are above 0.98.

The `S1_VV_div_VH` ratio is weaker, with full-image R2 of 0.790. This is still meaningful, but it shows that derived polarization relationships are harder to reproduce than the original backscatter channels.

Talking point: AlphaEarth contains strong information about raw SAR backscatter, but the derived ratio exposes more complex scattering behavior and noise.

### 3. Regeneration performance is stable across evaluation settings.

The model performs similarly on held-out pixels, gap-fill pixels, and the full image. For `S1_VV`, R2 stays around 0.968 to 0.970. For `S1_VH`, R2 stays around 0.974 to 0.978. This consistency suggests the model is not only memorizing training pixels.

Talking point: The regenerated image appears to generalize across the full scene, rather than only matching the sampled training locations.

### 4. Residuals reveal where regeneration is hardest.

The residual heatmap highlights larger errors around spatially complex regions: bridges, coastlines, port areas, dense urban texture, and high-contrast land-water boundaries. These are places where small spatial mismatches or radar-specific scattering effects can create larger errors.

Talking point: The model gets the broad image right, but the hardest areas are fine-scale structures and abrupt transitions.

### 5. Open water and broad land-cover regions are easier to regenerate.

Compared with urban infrastructure and coastlines, broad open-water areas show lower residual structure. Large-scale surface classes are easier for the model because they are more spatially consistent and less dominated by small objects.

Talking point: Regeneration is strongest where the target image has smooth or consistent spatial patterns.

### 6. The reproduced embedding image supports a qualitative cross-modal story.

The reproduced embedding visualization, shown as PCA RGB across 64 bands, retains major scene organization. Water, built-up land, islands, and bridges remain visible, suggesting that SAR contains information related to the structure captured in AlphaEarth embeddings.

Because this part currently lacks a metric table, it should be presented as visual evidence rather than a quantified result.

Talking point: The reverse-direction image suggests cross-modal structure exists in both directions, but the SAR-from-AlphaEarth task is the stronger quantitative result.

## Short Presentation Script

For the image-regeneration part, we used AlphaEarth embeddings as input and regenerated Sentinel-1 SAR imagery over the San Francisco downtown and Golden Gate scene. The predicted SAR image, or `SARhat`, preserves the major spatial features of the true SAR image, including water, bridges, islands, coastline, and dense urban structure. Quantitatively, the model performs very well on the raw SAR channels, with full-image R2 of 0.968 for `S1_VV` and 0.974 for `S1_VH`. The derived `S1_VV_div_VH` ratio is harder to regenerate, with full-image R2 of 0.790, but it still shows meaningful correlation with the true image.

The residual maps are useful because they show where the regeneration breaks down. Errors concentrate around bridges, shorelines, ports, and dense urban areas, which are exactly the places where radar scattering is more complex and small spatial differences matter more. Overall, the image-regeneration results show that AlphaEarth embeddings preserve enough cross-modal information to reconstruct much of the SAR image structure, especially for raw backscatter bands, while fine-scale infrastructure and derived polarization ratios remain more challenging.

## Figure Callouts

Use `true_sar_all_band_large.png` to show the real Sentinel-1 SAR reference image.

Use `sarhat_all_band_large.png` to show the regenerated SAR image from AlphaEarth embeddings.

Use `residual_heatmap_sar_large.png` to show where the model makes larger spatial errors.

Use `heldout_vs_full_r2.png` to show that held-out, gap-fill, and full-image R2 values are consistent.

Use `true_embedding_all_band_large.png` and `reproduced_embedding_all_band_large.png` only as qualitative visual comparisons unless embedding-regeneration metrics are added.

## Cautions for the Report

Do not describe the result as universal image generation. This is scene-specific reconstruction from colocated satellite features.

Do not imply the model understands SAR physics directly. It learns a statistical mapping from AlphaEarth embeddings to SAR bands.

Do not overclaim the reproduced embedding image without numeric metrics. It is useful visual evidence, but the strongest quantitative conclusion is SAR regeneration from AlphaEarth.

