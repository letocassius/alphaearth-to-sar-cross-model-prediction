# Final Report

## Scope

This report consolidates the repository's late-stage analysis into a single narrative document.
It focuses on the two downstream interpretation phases:

- Phase 3: failure analysis of the best-performing prediction models
- Phase 4: cross-modal similarity analysis between AlphaEarth embedding space and SAR space

The analysis uses the balanced full dataset of 2,880 samples spanning four regions and nine Dynamic World land-cover classes.

## Source Artifacts Used

The report is based on the machine-readable outputs already present in the repository:

- `outputs/full_dataset/phase3_summary.json`
- `outputs/full_dataset/phase3_best_model_selection.csv`
- `outputs/full_dataset/phase3_land_use_error.csv`
- `outputs/full_dataset/phase3_regional_error.csv`
- `outputs/full_dataset/phase4_summary.json`
- `outputs/full_dataset/phase4_knn_overlap_overall.csv`
- `outputs/full_dataset/phase4_knn_overlap_by_land_use.csv`
- `outputs/full_dataset/phase4_distance_correlation.csv`

## Executive Summary

The repository's strongest predictive results are achieved on the direct SAR backscatter targets, not on the polarization-difference target. The best model for `S1_VH` is an embeddings-only configuration with `R^2 = 0.942`, `RMSE = 1.182`, and `Pearson r = 0.971`. The best model for `S1_VV` is also embeddings-only with `R^2 = 0.902`, `RMSE = 1.255`, and `Pearson r = 0.951`.

The polarization-difference target `S1_VV_div_VH` is materially harder. Its best model requires embeddings plus context and reaches `R^2 = 0.698`, `RMSE = 1.155`, and `Pearson r = 0.836`. That gap indicates the AlphaEarth embedding captures substantial information about direct SAR behavior but is less aligned with the more structured and interaction-heavy signal represented by `VV - VH`.

Failure analysis shows that performance degradation is not uniform. For `S1_VV`, the hardest land-cover classes are `built`, `water`, and `shrub_and_scrub`, with the largest absolute errors concentrated in built environments. Regionally, `california_coast` is the only region flagged as significantly different in the phase summary, even though `sf_bay_urban` posts the largest mean absolute errors for both `S1_VV` and `S1_VH`.

Cross-modal similarity analysis shows a weak global relationship between embedding geometry and SAR geometry. The overall distance correlation is only `Pearson r = 0.195` and `Spearman rho = 0.151`, while nearest-neighbor overlap remains close to zero for most samples. Alignment improves in specific regimes, especially `california_coast` and the `water` class, but the dominant result is that local neighborhoods in embedding space rarely match local neighborhoods in SAR space.

Taken together, the results support a clear conclusion: AlphaEarth embeddings are useful for predicting aggregate SAR targets through supervised regression, but they should not be treated as a generally SAR-isomorphic representation at the neighborhood-structure level.

## Phase 3: Failure Analysis

### Best Model Selection

The phase 3 model-selection table identifies one best model per target:

| Target | Best model | Feature set | R^2 | RMSE | MAE |
| --- | --- | --- | ---: | ---: | ---: |
| `S1_VH` | Embeddings only | `embedding_only` | 0.942 | 1.182 | 0.858 |
| `S1_VV` | Embeddings only | `embedding_only` | 0.902 | 1.255 | 0.825 |
| `S1_VV_div_VH` | Embeddings + context | `embedding_plus_context` | 0.698 | 1.155 | 0.798 |

Two patterns matter here:

1. The raw backscatter targets are well recovered without adding region or Dynamic World context.
2. The polarization-difference target needs contextual metadata to reach its best performance, which suggests that the embedding alone is not encoding enough structure to resolve that target consistently.

### Land-Cover Failure Modes

For `S1_VV`, the hardest classes by mean absolute error are:

- `built`: `MAE = 1.227`, `bias = -0.333`
- `water`: `MAE = 1.011`, `bias = 0.020`
- `shrub_and_scrub`: `MAE = 0.902`, `bias = -0.054`

The `built` class also has the largest maximum absolute error at `12.077`, which is far above the maxima for most other classes. That is consistent with heterogeneous urban scattering, mixed pixels, and local geometry effects that are difficult to capture from optical embeddings alone.

For `S1_VV_div_VH`, the class ordering changes somewhat, but `water` and `built` remain challenging:

- `water`: `MAE = 1.187`
- `built`: `MAE = 1.145`
- `snow_and_ice`: `MAE = 0.783`

One notable contrast is that `trees` becomes the easiest polarization-difference class with `MAE = 0.484`, implying that vegetation structure may produce more stable relative-polarization behavior than urban or water surfaces in this dataset.

### Regional Failure Modes

Regional errors show that the urban region is generally hardest on the direct SAR targets:

- `S1_VH`: `sf_bay_urban` has the highest `MAE = 0.991`
- `S1_VV`: `sf_bay_urban` has the highest `MAE = 1.026`

For `S1_VV_div_VH`, the hardest region is instead:

- `iowa_ag`: `MAE = 0.867`, `bias = -0.294`

This shift matters. It suggests that the factors driving polarization-difference error are not the same as the factors driving raw-backscatter error. The direct targets struggle most in urban heterogeneity, while the difference target appears especially sensitive to agricultural structure and context.

The phase summary flags `california_coast` as the primary significant region. Combined with the phase 4 results, that points to California as a region where the relationship between optical structure and SAR structure is more coherent, even when residual patterns remain statistically notable.

## Phase 4: Cross-Modal Similarity Analysis

### Overall Neighborhood Overlap

The nearest-neighbor overlap analysis asks whether the samples that are close in embedding space are also close in SAR space.

The answer is mostly no.

Overall overlap statistics are:

| k | Mean overlap | Median overlap | P90 overlap |
| ---: | ---: | ---: | ---: |
| 5 | 0.0195 | 0.0 | 0.0 |
| 10 | 0.0319 | 0.0 | 0.1 |
| 20 | 0.0533 | 0.0 | 0.15 |

Even at `k = 20`, the median overlap remains zero. Most samples do not share local neighbors across the two spaces. That result is stronger than a generic statement that alignment is "imperfect"; it indicates that the two geometries are fundamentally different for most of the dataset.

### Distance Correlation

The global pairwise distance correlation is weak:

- Overall `Pearson r = 0.195`
- Overall `Spearman rho = 0.151`

This means that even beyond exact neighbor overlap, the broader metric structure of the two spaces is only loosely related.

At the region level, however, the picture is more nuanced:

- `california_coast`: `Pearson r = 0.689`, `Spearman rho = 0.676`
- `amazon_forest`: `Pearson r = 0.530`, `Spearman rho = 0.506`
- `sf_bay_urban`: `Pearson r = 0.508`, `Spearman rho = 0.519`
- `iowa_ag`: `Pearson r = 0.275`, `Spearman rho = 0.272`

California Coast stands out as the strongest region for geometric agreement, while Iowa Agriculture is notably weaker.

At the land-cover level, `water` is the clearest aligned class:

- `water`: `Pearson r = 0.574`, `Spearman rho = 0.585`

The weakest classes include:

- `built`: `Pearson r = 0.115`, `Spearman rho = 0.092`
- `bare`: `Pearson r = 0.145`, `Spearman rho = 0.130`
- `trees`: `Pearson r = 0.172`, `Spearman rho = 0.195`
- `crops`: `Pearson r = 0.186`, `Spearman rho = 0.118`

This class-level pattern is consistent with the phase 3 error analysis. Water exhibits comparatively coherent cross-modal structure, while urban and other heterogeneous classes do not.

### Land-Cover Overlap Patterns

The overlap-by-land-use table reinforces the same conclusion.

At `k = 10`, the best classes are:

- `water`: `mean overlap = 0.0978`, `median = 0.1`
- `snow_and_ice`: `mean overlap = 0.0428`
- `flooded_vegetation`: `mean overlap = 0.0356`

The worst classes are:

- `crops`: `mean overlap = 0.0141`
- `shrub_and_scrub`: `mean overlap = 0.0153`
- `grass`: `mean overlap = 0.0156`

Only `water` achieves a nonzero median overlap at `k = 10`. That is a strong indication that optical and SAR neighborhood structure become meaningfully comparable only in a limited subset of surface types.

## Combined Interpretation

The combined phase 3 and phase 4 evidence supports three main conclusions.

### 1. Prediction is easier than representation matching

The models can predict `S1_VV` and `S1_VH` very well from AlphaEarth embeddings, but that does not imply that the embedding space itself mirrors SAR space. Supervised regression can recover a target from information distributed globally across features even when local geometry is mismatched.

### 2. Urban and heterogeneous surfaces are the main failure regime

`built` is one of the hardest classes in the residual analysis and one of the weakest classes in the cross-modal correlation analysis. This consistency makes it the clearest regime where optical embeddings underrepresent the factors that matter for SAR behavior.

### 3. Alignment is regime-dependent rather than universal

Water, flooded vegetation, and California Coast exhibit much stronger optical-SAR agreement than the global average. Any future method that tries to exploit AlphaEarth embeddings for SAR-adjacent tasks should likely be stratified by region or land-cover regime instead of assuming one global mapping quality.

## Recommendations

### Repository Use

- Use `S1_VV` and `S1_VH` as the main headline results for predictive performance.
- Treat `S1_VV_div_VH` as a more difficult secondary target that benefits from added context.
- Use the phase 4 outputs to qualify any claim that AlphaEarth is "close" to SAR in a representation-learning sense.

### Next Analysis Steps

- Add regime-specific modeling experiments for `built` and `iowa_ag`, since those are the clearest failure domains.
- Test interaction-aware models or stratified models for polarization difference rather than relying on one pooled global model.
- Evaluate whether region-specific calibration or land-cover-conditioned models can preserve strong predictive performance while reducing the large-tail errors seen in `built`.
- If cross-modal retrieval is a goal, do not rely on the raw AlphaEarth geometry alone; add explicit alignment or metric-learning steps.

## Final Takeaway

The repository demonstrates that AlphaEarth optical embeddings contain strong predictive signal for Sentinel-1 backscatter, especially for `S1_VV` and `S1_VH`. However, the same repository also shows that this predictive usefulness should not be confused with geometric equivalence between modalities. Cross-modal neighborhood structure is weak overall and improves only in selected regimes such as water and California Coast. The practical implication is that AlphaEarth embeddings are a useful supervised input for SAR prediction, but they are not a general drop-in proxy for SAR similarity structure.
