#!/usr/bin/env python3
"""
Phase 3 failure-mode analysis for the full-dataset AlphaEarth-to-SAR models.

This script consumes the saved held-out predictions from the best-performing
full-dataset LightGBM feature set for each target, then produces:
- residual summary tables
- land-use and region failure diagnostics
- Moran's I spatial autocorrelation diagnostics by region
- primary-target failure mode plots
- a primary-target outlier catalog
- GeoJSON residual layers for mapping
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.neighbors import NearestNeighbors


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "DataSources" / "alphaearth_s1_dw_samples_all_regions_2024.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "full_dataset"
METRICS_PATH = OUTPUT_DIR / "full_dataset_lightgbm_metrics.csv"
PRIMARY_TARGET = "S1_VV"
TARGETS = ["S1_VV", "S1_VH", "S1_VV_div_VH"]
DW_PROB_COLS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]
DW_LABEL_NAMES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}
FEATURE_LABELS = {
    "embedding_only": "Embeddings only",
    "embedding_plus_context": "Embeddings + context",
}
REGION_ORDER = ["amazon_forest", "california_coast", "iowa_ag", "sf_bay_urban"]


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def evaluate_residuals(frame: pd.DataFrame) -> dict[str, float]:
    residual = frame["residual"].to_numpy()
    abs_residual = np.abs(residual)
    return {
        "n": int(len(frame)),
        "mae": float(abs_residual.mean()),
        "rmse": rmse(residual),
        "bias": float(residual.mean()),
        "median_abs_error": float(np.median(abs_residual)),
        "p90_abs_error": float(np.quantile(abs_residual, 0.90)),
        "max_abs_error": float(abs_residual.max()),
    }


def build_source_frame() -> pd.DataFrame:
    source = pd.read_csv(DATA_PATH)
    source["dw_label"] = source["dw_label"].astype(int)
    source["dw_label_name"] = source["dw_label"].map(DW_LABEL_NAMES)
    source["label_confidence"] = source.apply(lambda row: float(row[row["dw_label_name"]]), axis=1)
    source["dominant_class"] = source[DW_PROB_COLS].idxmax(axis=1)
    source["dominant_class_prob"] = source[DW_PROB_COLS].max(axis=1)
    keep_cols = [
        "system:index",
        "region",
        "dw_label",
        "dw_label_name",
        "latitude",
        "longitude",
        "label_confidence",
        "dominant_class",
        "dominant_class_prob",
        *DW_PROB_COLS,
    ]
    return source[keep_cols].copy()


def load_best_model_predictions(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(METRICS_PATH)
    best_rows = metrics.loc[metrics.groupby("target")["r2"].idxmax()].sort_values("target").reset_index(drop=True)

    prediction_frames: list[pd.DataFrame] = []
    for row in best_rows.itertuples():
        pred_path = OUTPUT_DIR / f"test_predictions_{row.feature_set}_{row.target}.csv"
        pred = pd.read_csv(pred_path)
        pred["model_label"] = FEATURE_LABELS[row.feature_set]
        pred["abs_residual"] = pred["residual"].abs()
        prediction_frames.append(pred)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    merged = predictions.merge(source, on="system:index", how="left", validate="many_to_one", suffixes=("", "_source"))
    merged["region"] = merged["region_source"].fillna(merged["region"])
    merged["dw_label_name"] = merged["dw_label_name_source"].fillna(merged["dw_label_name"])
    merged["latitude"] = merged["latitude_source"].fillna(merged["latitude"])
    merged["longitude"] = merged["longitude_source"].fillna(merged["longitude"])
    merged = merged.drop(columns=[col for col in merged.columns if col.endswith("_source")])
    return best_rows, merged


def build_group_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in frame.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(evaluate_residuals(part))
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_knn_coordinates(frame: pd.DataFrame) -> np.ndarray:
    latitude = frame["latitude"].to_numpy(dtype=float)
    longitude = frame["longitude"].to_numpy(dtype=float)
    mean_lat = np.deg2rad(latitude.mean())
    x = longitude * 111.32 * np.cos(mean_lat)
    y = latitude * 110.57
    return np.column_stack([x, y])


def morans_i_knn(
    frame: pd.DataFrame,
    value_col: str = "residual",
    neighbors: int = 8,
    permutations: int = 499,
    seed: int = 42,
) -> dict[str, float]:
    n_obs = len(frame)
    if n_obs < 4:
        return {"morans_i": np.nan, "p_value": np.nan}

    coords = prepare_knn_coordinates(frame)
    k = min(neighbors + 1, n_obs)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    weights = np.zeros((n_obs, n_obs), dtype=float)
    for row_idx in range(n_obs):
        neighbor_ids = indices[row_idx, 1:]
        neighbor_distances = distances[row_idx, 1:]
        inverse_distance = 1.0 / np.maximum(neighbor_distances, 1e-9)
        inverse_distance /= inverse_distance.sum()
        weights[row_idx, neighbor_ids] = inverse_distance

    values = frame[value_col].to_numpy(dtype=float)
    centered = values - values.mean()
    denominator = float(np.square(centered).sum())
    weight_sum = float(weights.sum())
    if denominator == 0.0 or weight_sum == 0.0:
        return {"morans_i": np.nan, "p_value": np.nan}

    observed = float((n_obs / weight_sum) * ((centered @ weights @ centered) / denominator))

    rng = np.random.default_rng(seed)
    permuted_scores = []
    for _ in range(permutations):
        permuted = rng.permutation(centered)
        score = float((n_obs / weight_sum) * ((permuted @ weights @ permuted) / denominator))
        permuted_scores.append(score)
    permuted_scores_arr = np.asarray(permuted_scores)
    p_value = float((np.sum(np.abs(permuted_scores_arr) >= abs(observed)) + 1) / (permutations + 1))
    return {"morans_i": observed, "p_value": p_value}


def save_geojson_layers(frame: pd.DataFrame) -> None:
    for target, part in frame.groupby("target", sort=False):
        features = []
        for _, row in part.iterrows():
            properties = {
                "system_index": row["system:index"],
                "region": row["region"],
                "dw_label_name": row["dw_label_name"],
                "feature_set": row["feature_set"],
                "model_label": row["model_label"],
                "actual": float(row["actual"]),
                "predicted": float(row["predicted"]),
                "residual": float(row["residual"]),
                "abs_residual": float(row["abs_residual"]),
            }
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["longitude"]), float(row["latitude"])],
                    },
                    "properties": properties,
                }
            )
        out_path = OUTPUT_DIR / f"phase3_residuals_{target}.geojson"
        out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))


def save_primary_residual_map(frame: pd.DataFrame, spatial_df: pd.DataFrame) -> None:
    primary = frame[frame["target"] == PRIMARY_TARGET].copy()
    color_limit = float(np.quantile(primary["abs_residual"], 0.98))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.subplots_adjust(top=0.90, left=0.06, right=0.92, bottom=0.08, hspace=0.22, wspace=0.18)
    fig.suptitle("Phase 3 Residual Maps for S1_VV", fontsize=20, fontweight="bold")

    for ax, region in zip(axes.flatten(), REGION_ORDER):
        part = primary[primary["region"] == region].copy()
        moran_row = spatial_df[(spatial_df["target"] == PRIMARY_TARGET) & (spatial_df["region"] == region)]
        im = ax.scatter(
            part["longitude"],
            part["latitude"],
            c=part["residual"],
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
            s=28,
            edgecolor="none",
        )
        title = region.replace("_", " ")
        if not moran_row.empty:
            title += f"\nMoran's I={moran_row.iloc[0]['morans_i']:.3f}, p={moran_row.iloc[0]['p_value']:.3f}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.15)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.90)
    cbar.set_label("Residual (predicted - actual)")
    plt.savefig(OUTPUT_DIR / "phase3_residual_maps_S1_VV.png", dpi=220)
    plt.close(fig)


def save_primary_land_use_diagnostics(land_use_df: pd.DataFrame) -> None:
    primary = land_use_df[land_use_df["target"] == PRIMARY_TARGET].sort_values("mae", ascending=True).copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(top=0.88, left=0.18, right=0.96, bottom=0.12, wspace=0.30)
    fig.suptitle("Phase 3 Land-use Diagnostics for S1_VV", fontsize=18, fontweight="bold")

    axes[0].barh(primary["dw_label_name"], primary["mae"], color="#2563eb")
    axes[0].set_title("MAE by land use")
    axes[0].set_xlabel("MAE")

    colors = ["#dc2626" if value > 0 else "#059669" for value in primary["bias"]]
    axes[1].barh(primary["dw_label_name"], primary["bias"], color=colors)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Signed bias by land use")
    axes[1].set_xlabel("Mean residual")

    plt.savefig(OUTPUT_DIR / "phase3_land_use_diagnostics_S1_VV.png", dpi=220)
    plt.close(fig)


def save_primary_region_land_use_heatmap(region_land_use_df: pd.DataFrame) -> None:
    primary = region_land_use_df[region_land_use_df["target"] == PRIMARY_TARGET].copy()
    label_order = (
        primary.groupby("dw_label_name")["mae"].mean().sort_values(ascending=False).index.tolist()
    )
    pivot = (
        primary.pivot(index="region", columns="dw_label_name", values="mae")
        .reindex(index=REGION_ORDER, columns=label_order)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_title("Phase 3 Region x Land-use MAE Heatmap for S1_VV", fontsize=16, fontweight="bold")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[label.replace("_", "\n") for label in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), labels=[region.replace("_", " ") for region in pivot.index])
    for row_idx in range(len(pivot.index)):
        for col_idx in range(len(pivot.columns)):
            ax.text(col_idx, row_idx, f"{pivot.iloc[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MAE")
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase3_region_land_use_heatmap_S1_VV.png", dpi=220)
    plt.close(fig)


def save_primary_residual_boxplot(frame: pd.DataFrame, land_use_df: pd.DataFrame) -> None:
    primary = frame[frame["target"] == PRIMARY_TARGET].copy()
    label_order = (
        land_use_df[land_use_df["target"] == PRIMARY_TARGET]
        .sort_values("mae", ascending=False)["dw_label_name"]
        .tolist()
    )
    series = [primary.loc[primary["dw_label_name"] == label, "residual"].to_numpy() for label in label_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(series, vert=False, tick_labels=label_order, patch_artist=True)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Phase 3 Residual Spread by Land Use for S1_VV", fontsize=16, fontweight="bold")
    ax.set_xlabel("Residual (predicted - actual)")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase3_residual_boxplot_S1_VV.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    source = build_source_frame()
    best_models, best_predictions = load_best_model_predictions(source)
    best_predictions.to_csv(OUTPUT_DIR / "phase3_best_model_predictions.csv", index=False)

    land_use_summary = build_group_summary(best_predictions, ["target", "feature_set", "model_label", "dw_label_name"])
    regional_summary = build_group_summary(best_predictions, ["target", "feature_set", "model_label", "region"])
    region_land_use_summary = build_group_summary(
        best_predictions,
        ["target", "feature_set", "model_label", "region", "dw_label_name"],
    )

    residual_distribution_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for target, part in best_predictions.groupby("target", sort=False):
        residual = part["residual"].to_numpy()
        abs_residual = part["abs_residual"].to_numpy()
        residual_distribution_rows.append(
            {
                "target": target,
                "feature_set": part["feature_set"].iloc[0],
                "model_label": part["model_label"].iloc[0],
                "n": int(len(part)),
                "residual_mean": float(residual.mean()),
                "residual_std": float(residual.std(ddof=0)),
                "residual_skew": float(skew(residual, bias=False)),
                "residual_excess_kurtosis": float(kurtosis(residual, fisher=True, bias=False)),
                "p90_abs_residual": float(np.quantile(abs_residual, 0.90)),
                "p95_abs_residual": float(np.quantile(abs_residual, 0.95)),
                "max_abs_residual": float(abs_residual.max()),
            }
        )
        for region, region_part in part.groupby("region", sort=False):
            stats = morans_i_knn(region_part)
            spatial_rows.append(
                {
                    "target": target,
                    "feature_set": part["feature_set"].iloc[0],
                    "model_label": part["model_label"].iloc[0],
                    "region": region,
                    "n": int(len(region_part)),
                    "morans_i": stats["morans_i"],
                    "p_value": stats["p_value"],
                    "significant_at_0_05": bool(stats["p_value"] < 0.05) if not np.isnan(stats["p_value"]) else False,
                }
            )

    outliers = (
        best_predictions[best_predictions["target"] == PRIMARY_TARGET]
        .sort_values("abs_residual", ascending=False)
        .head(20)
        .copy()
    )
    outliers["outlier_rank"] = np.arange(1, len(outliers) + 1)
    outlier_cols = [
        "outlier_rank",
        "system:index",
        "region",
        "dw_label_name",
        "label_confidence",
        "dominant_class",
        "dominant_class_prob",
        "latitude",
        "longitude",
        "actual",
        "predicted",
        "residual",
        "abs_residual",
    ]
    outliers = outliers[outlier_cols]

    residual_distribution = pd.DataFrame(residual_distribution_rows).sort_values("target")
    spatial_summary = pd.DataFrame(spatial_rows).sort_values(["target", "region"])
    best_models = best_models[["target", "feature_set", "r2", "rmse", "mae", "pearson_r"]].copy()
    best_models["model_label"] = best_models["feature_set"].map(FEATURE_LABELS)
    best_models = best_models[["target", "feature_set", "model_label", "r2", "rmse", "mae", "pearson_r"]]

    best_models.to_csv(OUTPUT_DIR / "phase3_best_model_selection.csv", index=False)
    land_use_summary.sort_values(["target", "mae"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "phase3_land_use_error.csv",
        index=False,
    )
    regional_summary.sort_values(["target", "mae"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "phase3_regional_error.csv",
        index=False,
    )
    region_land_use_summary.sort_values(["target", "region", "mae"], ascending=[True, True, False]).to_csv(
        OUTPUT_DIR / "phase3_region_land_use_error.csv",
        index=False,
    )
    residual_distribution.to_csv(OUTPUT_DIR / "phase3_residual_distribution.csv", index=False)
    spatial_summary.to_csv(OUTPUT_DIR / "phase3_spatial_autocorrelation.csv", index=False)
    outliers.to_csv(OUTPUT_DIR / "phase3_outliers_S1_VV.csv", index=False)

    save_geojson_layers(best_predictions)
    save_primary_residual_map(best_predictions, spatial_summary)
    save_primary_land_use_diagnostics(land_use_summary)
    save_primary_region_land_use_heatmap(region_land_use_summary)
    save_primary_residual_boxplot(best_predictions, land_use_summary)

    summary_payload = {
        "primary_target": PRIMARY_TARGET,
        "targets": TARGETS,
        "best_models": json.loads(best_models.to_json(orient="records")),
        "primary_hardest_land_use": land_use_summary[land_use_summary["target"] == PRIMARY_TARGET]
        .sort_values("mae", ascending=False)
        .head(3)[["dw_label_name", "mae", "bias"]]
        .to_dict(orient="records"),
        "primary_significant_regions": spatial_summary[
            (spatial_summary["target"] == PRIMARY_TARGET) & (spatial_summary["significant_at_0_05"])
        ]["region"].tolist(),
    }
    (OUTPUT_DIR / "phase3_summary.json").write_text(json.dumps(summary_payload, indent=2))

    print("Saved Phase 3 outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
