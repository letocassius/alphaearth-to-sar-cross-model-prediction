# Main Findings: AlphaEarth-to-Sentinel-1 SAR Reconstruction

## Study Context

This analysis tests whether Google AlphaEarth embeddings can reconstruct Sentinel-1 SAR backscatter for a single San Francisco downtown / Golden Gate scene. The model uses 64 AlphaEarth embedding dimensions (`A00` to `A63`) to predict three SAR targets: `S1_VV`, `S1_VH`, and the derived ratio `S1_VV_div_VH`.

The reconstruction model was trained on 2,008 sampled pixels and evaluated on 503 held-out pixels. Full-scene and gap-fill metrics were then computed across more than 1.26 million valid pixels.

## Main Findings

### 1. AlphaEarth embeddings strongly reconstruct raw Sentinel-1 SAR backscatter.

The strongest result is that AlphaEarth embeddings can predict the two raw Sentinel-1 polarization bands with high accuracy. On held-out pixels, the model reached an R2 of 0.970 for `S1_VV` and 0.978 for `S1_VH`, with Pearson correlations of 0.985 and 0.989. The same pattern holds across the full image and gap-fill evaluation, where both raw bands remain above 0.967 R2.

This suggests that the AlphaEarth embedding space contains substantial information related to radar backscatter, even though SAR is a different sensing modality from the optical and geospatial signals typically associated with embedding products.

### 2. VH is the easiest SAR band to reconstruct in this scene.

Across held-out, full-scene, and gap-fill evaluations, `S1_VH` is the best-performing target. In the gap-fill evaluation, `S1_VH` achieved an R2 of 0.974 and Pearson r of 0.987. This indicates that the model captures much of the spatial variation in cross-polarized backscatter for the San Francisco coastal urban scene.

Although `S1_VH` has a slightly higher RMSE and MAE than `S1_VV`, its variance explained and correlation are consistently higher, making it the strongest target by overall fit.

### 3. The derived VV/VH ratio is meaningfully predictable but harder than the raw bands.

The `S1_VV_div_VH` target performs lower than the raw polarization channels. It reaches an R2 of 0.825 on held-out pixels and 0.789 in the gap-fill evaluation, compared with R2 values above 0.96 for `S1_VV` and `S1_VH`.

This does not mean the ratio failed. Pearson correlation remains high at about 0.909 on held-out pixels and 0.889 in the gap-fill evaluation. However, the lower R2 indicates that the ratio contains more complex or noisier structure than either raw channel alone. This is expected because polarization ratios can amplify local scattering differences, nonlinear effects, and measurement noise.

### 4. Reconstruction generalizes from sampled pixels to the full scene with little performance loss.

The held-out sample and full-scene metrics are closely aligned. Mean held-out R2 is 0.924, while mean full-scene R2 is 0.910 and mean gap-fill R2 is 0.910. The raw SAR bands show especially stable performance between held-out and full-scene evaluations.

This stability suggests that the model is not only fitting the small sampled dataset. It is learning a relationship that extends across the broader image. The largest performance drop appears in `S1_VV_div_VH`, which again points to the ratio target as the main source of residual uncertainty.

### 5. A small number of AlphaEarth dimensions carry a large share of the SAR signal.

The feature-importance report shows that `A27`, `A63`, and `A25` are the top three overall predictors. Together, they account for 41.8% of average SHAP importance across the three SAR targets. `A27` is the strongest feature overall, ranking first by both SHAP importance and mean absolute Pearson correlation.

The top ten dimensions are:

| Rank | Feature | Mean SHAP Share | Mean Absolute Pearson r |
| ---: | --- | ---: | ---: |
| 1 | A27 | 0.220 | 0.822 |
| 2 | A63 | 0.130 | 0.769 |
| 3 | A25 | 0.068 | 0.737 |
| 4 | A11 | 0.039 | 0.700 |
| 5 | A34 | 0.032 | 0.700 |
| 6 | A17 | 0.023 | 0.707 |
| 7 | A30 | 0.010 | 0.756 |
| 8 | A24 | 0.018 | 0.653 |
| 9 | A44 | 0.015 | 0.666 |
| 10 | A26 | 0.029 | 0.565 |

This concentration of importance implies that the SAR-relevant information in AlphaEarth is not evenly distributed across all 64 dimensions. Instead, a subset of embedding dimensions appears especially aligned with radar-relevant surface structure.

### 6. The most important features differ by SAR target.

The target-level analysis shows that feature importance is not identical across `S1_VV`, `S1_VH`, and `S1_VV_div_VH`.

For `S1_VV`, the most important features include `A27`, `A11`, `A63`, `A17`, and `A30`. For `S1_VH`, the strongest features are `A27`, `A63`, `A11`, `A17`, and `A25`. For the ratio target, `A27` and `A25` dominate, followed by features such as `A24`, `A32`, and `A03`.

This pattern suggests that AlphaEarth dimensions encode multiple aspects of the surface that relate differently to each SAR response. Some dimensions, especially `A27`, are broadly useful across targets, while others appear more target-specific.

## Key Evidence Table

| Evaluation | Target | R2 | RMSE | MAE | Pearson r |
| --- | --- | ---: | ---: | ---: | ---: |
| Held-out | S1_VV | 0.970 | 1.046 | 0.674 | 0.985 |
| Held-out | S1_VH | 0.978 | 1.129 | 0.844 | 0.989 |
| Held-out | S1_VV_div_VH | 0.825 | 1.074 | 0.831 | 0.909 |
| Gap-fill | S1_VV | 0.968 | 1.112 | 0.711 | 0.984 |
| Gap-fill | S1_VH | 0.974 | 1.257 | 0.900 | 0.987 |
| Gap-fill | S1_VV_div_VH | 0.789 | 1.211 | 0.890 | 0.889 |

## Suggested Summary Paragraph

Overall, the results show that AlphaEarth embeddings contain strong cross-modal information for reconstructing Sentinel-1 SAR backscatter in the San Francisco downtown / Golden Gate scene. A LightGBM model trained on 64 AlphaEarth dimensions accurately reproduced the raw `S1_VV` and `S1_VH` bands, with held-out R2 values above 0.96 and full-scene gap-fill R2 values near 0.97. The derived `S1_VV_div_VH` ratio was more difficult to reconstruct, but still showed meaningful predictive skill, with held-out R2 of 0.825 and Pearson r above 0.90. Feature-importance analysis indicates that SAR predictability is concentrated in a small subset of embedding dimensions, led by `A27`, `A63`, and `A25`, which together account for 41.8% of average SHAP importance. These findings support the conclusion that AlphaEarth embeddings encode radar-relevant surface information, while also showing that derived polarization relationships remain more challenging than raw SAR backscatter.

## Limitations

These findings are scene-specific. The experiment uses one geographic area, one time period, one SAR image stack, and a random pixel split within the same scene. The results therefore demonstrate within-scene reconstruction skill, not global generalization. A stronger follow-up study would test the model across multiple cities, land-cover types, seasons, and SAR acquisition conditions.

