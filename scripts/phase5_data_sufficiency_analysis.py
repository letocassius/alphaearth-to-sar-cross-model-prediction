#!/usr/bin/env python3
"""
Data sufficiency analysis for the AlphaEarth -> SAR regression project.

This script reuses the existing dataset preparation, feature construction,
metric functions, and model settings from Phase 2. It produces:

- learning curves for Ridge and LightGBM
- repeated grouped-CV stability summaries
- subgroup coverage and subgroup performance diagnostics
- redundancy / effective sample size checks
- a standalone PDF summary report with charts and interpretation
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neighbors import NearestNeighbors

from build_project_reports import draw_dataframe_page, draw_image_grid_page, draw_text_page
from phase2_full_dataset_lightgbm_experiments import (
    FEATURE_LABELS,
    OUTPUT_DIR,
    RANDOM_STATE,
    STABILITY_SEEDS,
    TARGET_COLS,
    assign_spatial_blocks,
    build_feature_frames,
    ensure_output_dirs,
    evaluate_predictions,
    load_dataset,
    ridge_params,
    train_ridge_model,
)


REPORTS_DIR = OUTPUT_DIR.parent.parent / "reports"
ANALYSIS_PREFIX = "data_sufficiency"
LEARNING_FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
LEARNING_SEEDS = [13, 23, 33]
LEARNING_TEST_SIZE = 0.30
SPATIAL_BLOCK_SAMPLE_CAP = 800


def output_path(name: str) -> Path:
    return OUTPUT_DIR / f"{ANALYSIS_PREFIX}_{name}"


def report_copy_path(name: str) -> Path:
    return REPORTS_DIR / f"{ANALYSIS_PREFIX}_{name}"


def load_lightgbm_params(feature_set: str, target: str) -> Dict[str, Any]:
    path = OUTPUT_DIR / f"best_params_{feature_set}_{target}_lightgbm.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing LightGBM parameter file: {path}")
    return json.loads(path.read_text())


def sample_training_subset(
    train_df: pd.DataFrame,
    fraction: float,
    seed: int,
) -> pd.Index:
    if math.isclose(fraction, 1.0):
        return train_df.index
    sampled = train_df.groupby("region_dw_label", group_keys=False).apply(
        lambda part: part.sample(n=max(1, int(round(len(part) * fraction))), random_state=seed)
    )
    return pd.Index(sampled.index)


def build_models(feature_set: str, target: str) -> Dict[str, Any]:
    lightgbm_params = load_lightgbm_params(feature_set, target)
    return {
        "ridge": {"label": "Ridge", "factory": lambda: train_ridge_model},
        "lightgbm": {"label": "LightGBM", "factory": lambda: LGBMRegressor(**lightgbm_params)},
    }


def fit_model(model_name: str, feature_set: str, target: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    if model_name == "ridge":
        return train_ridge_model(X_train, y_train)
    if model_name == "lightgbm":
        return LGBMRegressor(**load_lightgbm_params(feature_set, target)).fit(X_train, y_train)
    raise ValueError(f"Unknown model {model_name}")


def run_learning_curves(
    full_df: pd.DataFrame,
    feature_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    train_idx, test_idx = train_test_split(
        full_df.index,
        test_size=LEARNING_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=full_df["region_dw_label"],
    )
    train_df = full_df.loc[train_idx].copy()
    test_df = full_df.loc[test_idx].copy()

    rows: List[Dict[str, Any]] = []
    for feature_set, X_full in feature_frames.items():
        X_train_full = X_full.loc[train_idx].copy()
        X_test = X_full.loc[test_idx].copy()
        for target in TARGET_COLS:
            y_train_full = train_df[target]
            y_test = test_df[target]
            for model_name in ["ridge", "lightgbm"]:
                for fraction in LEARNING_FRACTIONS:
                    for seed in LEARNING_SEEDS:
                        subset_idx = sample_training_subset(train_df, fraction, seed)
                        X_subset = X_train_full.loc[subset_idx]
                        y_subset = y_train_full.loc[subset_idx]
                        model = fit_model(model_name, feature_set, target, X_subset, y_subset)
                        train_pred = model.predict(X_subset)
                        valid_pred = model.predict(X_test)
                        train_metrics = evaluate_predictions(y_subset, train_pred)
                        valid_metrics = evaluate_predictions(y_test, valid_pred)
                        rows.append(
                            {
                                "feature_set": feature_set,
                                "target": target,
                                "model_name": model_name,
                                "fraction": fraction,
                                "seed": seed,
                                "n_train": int(len(X_subset)),
                                "train_r2": train_metrics["r2"],
                                "train_rmse": train_metrics["rmse"],
                                "valid_r2": valid_metrics["r2"],
                                "valid_rmse": valid_metrics["rmse"],
                            }
                        )
    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(output_path("learning_curve_raw.csv"), index=False)
    summary_df = (
        curve_df.groupby(["feature_set", "target", "model_name", "fraction"], as_index=False)
        .agg(
            n_train=("n_train", "mean"),
            train_r2_mean=("train_r2", "mean"),
            train_r2_std=("train_r2", "std"),
            valid_r2_mean=("valid_r2", "mean"),
            valid_r2_std=("valid_r2", "std"),
            train_rmse_mean=("train_rmse", "mean"),
            valid_rmse_mean=("valid_rmse", "mean"),
        )
        .fillna(0.0)
    )
    summary_df.to_csv(output_path("learning_curve_summary.csv"), index=False)
    return summary_df


def save_learning_curve_plots(curve_summary: pd.DataFrame) -> list[Path]:
    image_paths: list[Path] = []
    for target in TARGET_COLS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
        fig.suptitle(f"Learning Curves: {target}", fontsize=18, fontweight="bold")
        panel_order = [
            ("embedding_only", "ridge"),
            ("embedding_only", "lightgbm"),
            ("embedding_plus_context", "ridge"),
            ("embedding_plus_context", "lightgbm"),
        ]
        for ax, (feature_set, model_name) in zip(axes.flatten(), panel_order):
            part = curve_summary[
                (curve_summary["target"] == target)
                & (curve_summary["feature_set"] == feature_set)
                & (curve_summary["model_name"] == model_name)
            ].sort_values("fraction")
            train_x = part["n_train"].to_numpy()
            ax.plot(train_x, part["train_r2_mean"], marker="o", label="Train R2", color="#1d4ed8")
            ax.plot(train_x, part["valid_r2_mean"], marker="o", label="Validation R2", color="#dc2626")
            ax.fill_between(
                train_x,
                part["valid_r2_mean"] - part["valid_r2_std"],
                part["valid_r2_mean"] + part["valid_r2_std"],
                color="#fecaca",
                alpha=0.4,
            )
            ax.set_title(f"{FEATURE_LABELS[feature_set]} | {model_name.title()}")
            ax.set_xlabel("Training samples")
            ax.set_ylabel("R2")
            ax.grid(alpha=0.25)
            ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = output_path(f"learning_curve_{target}.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        image_paths.append(path)

        rmse_fig, rmse_ax = plt.subplots(figsize=(10, 6))
        for feature_set, color in [("embedding_only", "#2563eb"), ("embedding_plus_context", "#059669")]:
            for model_name, linestyle in [("ridge", "-"), ("lightgbm", "--")]:
                part = curve_summary[
                    (curve_summary["target"] == target)
                    & (curve_summary["feature_set"] == feature_set)
                    & (curve_summary["model_name"] == model_name)
                ].sort_values("fraction")
                rmse_ax.plot(
                    part["n_train"],
                    part["valid_rmse_mean"],
                    marker="o",
                    linestyle=linestyle,
                    color=color,
                    label=f"{FEATURE_LABELS[feature_set]} | {model_name.title()}",
                )
        rmse_ax.set_title(f"Validation RMSE vs Training Size: {target}")
        rmse_ax.set_xlabel("Training samples")
        rmse_ax.set_ylabel("Validation RMSE")
        rmse_ax.grid(alpha=0.25)
        rmse_ax.legend(fontsize=8)
        rmse_fig.tight_layout()
        rmse_path = output_path(f"learning_curve_rmse_{target}.png")
        rmse_fig.savefig(rmse_path, dpi=180)
        plt.close(rmse_fig)
        image_paths.append(rmse_path)
    return image_paths


def run_repeated_grouped_cv(
    full_df: pd.DataFrame,
    feature_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    strat_labels = full_df["region_dw_label"]
    groups = full_df["spatial_block"]

    for feature_set, X_full in feature_frames.items():
        for target in TARGET_COLS:
            y_full = full_df[target]
            for model_name in ["ridge", "lightgbm"]:
                for seed in STABILITY_SEEDS:
                    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
                    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_full, strat_labels, groups), start=1):
                        X_train = X_full.iloc[train_idx]
                        X_valid = X_full.iloc[valid_idx]
                        y_train = y_full.iloc[train_idx]
                        y_valid = y_full.iloc[valid_idx]
                        model = fit_model(model_name, feature_set, target, X_train, y_train)
                        pred = model.predict(X_valid)
                        metrics = evaluate_predictions(y_valid, pred)
                        rows.append(
                            {
                                "feature_set": feature_set,
                                "target": target,
                                "model_name": model_name,
                                "seed": seed,
                                "fold": fold,
                                **metrics,
                                "n_valid": int(len(valid_idx)),
                            }
                        )
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(output_path("cv_stability_raw.csv"), index=False)
    summary_df = (
        cv_df.groupby(["feature_set", "target", "model_name"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            pearson_r_mean=("pearson_r", "mean"),
            folds=("fold", "count"),
        )
        .fillna(0.0)
    )
    summary_df.to_csv(output_path("cv_stability_summary.csv"), index=False)
    return summary_df


def select_best_models(cv_summary: pd.DataFrame) -> pd.DataFrame:
    best_rows = (
        cv_summary.sort_values(["target", "r2_mean"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()
    )
    best_rows.to_csv(output_path("best_models_by_target.csv"), index=False)
    return best_rows


def plot_cv_stability(cv_summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    for ax, target in zip(axes, TARGET_COLS):
        part = cv_summary[cv_summary["target"] == target].copy()
        labels = [
            f"{FEATURE_LABELS[row.feature_set]}\n{row.model_name.title()}"
            for row in part.itertuples()
        ]
        ax.bar(labels, part["r2_mean"], yerr=part["r2_std"], color="#93c5fd", edgecolor="#1d4ed8", capsize=4)
        ax.set_title(f"Repeated grouped-CV stability: {target}")
        ax.set_ylabel("R2 mean +/- std")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    fig.tight_layout()
    path = output_path("cv_stability_r2.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_coverage_tables(full_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_region = full_df.groupby("region", as_index=False).size().rename(columns={"size": "n_samples"})
    by_class = full_df.groupby("dw_label_name", as_index=False).size().rename(columns={"size": "n_samples"})
    by_region_class = (
        full_df.groupby(["region", "dw_label_name"], as_index=False)
        .size()
        .rename(columns={"size": "n_samples"})
        .sort_values(["region", "dw_label_name"])
    )
    by_region.to_csv(output_path("coverage_by_region.csv"), index=False)
    by_class.to_csv(output_path("coverage_by_land_cover.csv"), index=False)
    by_region_class.to_csv(output_path("coverage_by_region_land_cover.csv"), index=False)
    return by_region, by_class, by_region_class


def plot_coverage(by_region: pd.DataFrame, by_class: pd.DataFrame, by_region_class: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(by_region["region"], by_region["n_samples"], color="#60a5fa", edgecolor="#1d4ed8")
    ax.set_title("Sample coverage by region")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_path("coverage_by_region.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(by_class["dw_label_name"], by_class["n_samples"], color="#6ee7b7", edgecolor="#047857")
    ax.set_title("Sample coverage by Dynamic World class")
    ax.set_xlabel("Rows")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_path("coverage_by_land_cover.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    pivot = by_region_class.pivot(index="dw_label_name", columns="region", values="n_samples").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_title("Coverage heatmap: region x Dynamic World class")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = output_path("coverage_heatmap_region_land_cover.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def evaluate_best_models_by_subgroup(
    full_df: pd.DataFrame,
    feature_frames: Dict[str, pd.DataFrame],
    best_models: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, test_idx = train_test_split(
        full_df.index,
        test_size=LEARNING_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=full_df["region_dw_label"],
    )
    train_df = full_df.loc[train_idx].copy()
    test_df = full_df.loc[test_idx].copy()

    region_rows: list[Dict[str, Any]] = []
    class_rows: list[Dict[str, Any]] = []

    for row in best_models.itertuples():
        feature_set = row.feature_set
        target = row.target
        model_name = row.model_name
        X_full = feature_frames[feature_set]
        X_train = X_full.loc[train_idx]
        X_test = X_full.loc[test_idx]
        model = fit_model(model_name, feature_set, target, X_train, train_df[target])
        pred = model.predict(X_test)
        frame = test_df[["region", "dw_label_name"]].copy()
        frame["actual"] = test_df[target].to_numpy()
        frame["predicted"] = pred

        for region, part in frame.groupby("region"):
            metrics = evaluate_predictions(part["actual"], part["predicted"])
            region_rows.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "model_name": model_name,
                    "region": region,
                    "n_samples": int(len(part)),
                    **metrics,
                }
            )
        for label, part in frame.groupby("dw_label_name"):
            metrics = evaluate_predictions(part["actual"], part["predicted"])
            class_rows.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "model_name": model_name,
                    "dw_label_name": label,
                    "n_samples": int(len(part)),
                    **metrics,
                }
            )

    region_df = pd.DataFrame(region_rows).sort_values(["target", "r2"])
    class_df = pd.DataFrame(class_rows).sort_values(["target", "r2"])
    region_df.to_csv(output_path("best_model_region_performance.csv"), index=False)
    class_df.to_csv(output_path("best_model_land_cover_performance.csv"), index=False)
    return region_df, class_df


def plot_subgroup_performance(region_df: pd.DataFrame, class_df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for target in TARGET_COLS:
        part = region_df[region_df["target"] == target]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(part["region"], part["r2"], color="#c4b5fd", edgecolor="#6d28d9")
        ax.set_title(f"Best-model held-out R2 by region: {target}")
        ax.set_ylabel("R2")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = output_path(f"subgroup_region_r2_{target}.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        part = class_df[class_df["target"] == target].sort_values("r2")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(part["dw_label_name"], part["r2"], color="#fdba74", edgecolor="#c2410c")
        ax.set_title(f"Best-model held-out R2 by Dynamic World class: {target}")
        ax.set_xlabel("R2")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = output_path(f"subgroup_land_cover_r2_{target}.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def compute_redundancy_diagnostics(full_df: pd.DataFrame, feature_frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    block_counts = full_df.groupby("spatial_block").size().sort_values(ascending=False)
    region_class_counts = full_df.groupby("region_dw_label").size().sort_values(ascending=False)
    effective_n_spatial = float((block_counts.sum() ** 2) / np.square(block_counts).sum())
    effective_n_region_class = float((region_class_counts.sum() ** 2) / np.square(region_class_counts).sum())

    embeddings = feature_frames["embedding_only"].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    nearest_neighbor_cosine = 1.0 - distances[:, 1]

    sampled_blocks = []
    rng = np.random.default_rng(RANDOM_STATE)
    for block, part in full_df.groupby("spatial_block"):
        if len(part) < 2:
            continue
        sample_idx = part.index.to_numpy()
        if len(sample_idx) > SPATIAL_BLOCK_SAMPLE_CAP:
            sample_idx = rng.choice(sample_idx, size=SPATIAL_BLOCK_SAMPLE_CAP, replace=False)
        block_embed = feature_frames["embedding_only"].loc[sample_idx].to_numpy(dtype=float)
        similarity = cosine_similarity(block_embed)
        upper = similarity[np.triu_indices_from(similarity, k=1)]
        sampled_blocks.append(
            {
                "spatial_block": block,
                "n_samples": int(len(sample_idx)),
                "mean_pairwise_cosine": float(np.mean(upper)) if len(upper) else np.nan,
            }
        )
    block_similarity_df = pd.DataFrame(sampled_blocks).sort_values("mean_pairwise_cosine", ascending=False)
    block_similarity_df.to_csv(output_path("spatial_block_similarity.csv"), index=False)

    diagnostics = {
        "n_rows": int(len(full_df)),
        "n_spatial_blocks": int(block_counts.shape[0]),
        "mean_rows_per_spatial_block": float(block_counts.mean()),
        "max_rows_per_spatial_block": int(block_counts.max()),
        "spatial_block_effective_n": effective_n_spatial,
        "n_region_class_cells": int(region_class_counts.shape[0]),
        "mean_rows_per_region_class_cell": float(region_class_counts.mean()),
        "region_class_effective_n": effective_n_region_class,
        "mean_nearest_neighbor_cosine": float(np.mean(nearest_neighbor_cosine)),
        "p95_nearest_neighbor_cosine": float(np.quantile(nearest_neighbor_cosine, 0.95)),
        "exact_duplicate_embedding_rows": int(pd.DataFrame(embeddings).duplicated().sum()),
    }
    output_path("redundancy_summary.json").write_text(json.dumps(diagnostics, indent=2))
    return diagnostics


def plot_redundancy(full_df: pd.DataFrame, redundancy: Dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    block_counts = full_df.groupby("spatial_block").size().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(block_counts, bins=20, color="#bfdbfe", edgecolor="#1d4ed8")
    ax.set_title("Rows per spatial block")
    ax.set_xlabel("Rows in block")
    ax.set_ylabel("Count of blocks")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_path("redundancy_spatial_block_histogram.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    embeddings = build_feature_frames(full_df)["embedding_only"].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    nearest_neighbor_cosine = 1.0 - distances[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(nearest_neighbor_cosine, bins=30, color="#fde68a", edgecolor="#b45309")
    ax.axvline(redundancy["mean_nearest_neighbor_cosine"], color="black", linestyle="--", linewidth=1)
    ax.set_title("Nearest-neighbor cosine similarity across embeddings")
    ax.set_xlabel("Cosine similarity to nearest neighbor")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_path("redundancy_nearest_neighbor_cosine.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def build_learning_curve_takeaways(curve_summary: pd.DataFrame, cv_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for target in TARGET_COLS:
        target_curves = curve_summary[curve_summary["target"] == target]
        best_curve = (
            target_curves.sort_values(["valid_r2_mean", "fraction"], ascending=[False, False])
            .groupby(["feature_set", "model_name"], as_index=False)
            .first()
            .sort_values("valid_r2_mean", ascending=False)
            .iloc[0]
        )
        best_combo = target_curves[
            (target_curves["feature_set"] == best_curve["feature_set"])
            & (target_curves["model_name"] == best_curve["model_name"])
        ].sort_values("fraction")
        r2_at_80 = float(best_combo.loc[np.isclose(best_combo["fraction"], 0.8), "valid_r2_mean"].iloc[0])
        r2_at_100 = float(best_combo.loc[np.isclose(best_combo["fraction"], 1.0), "valid_r2_mean"].iloc[0])
        gain_tail = r2_at_100 - r2_at_80
        variability = float(
            cv_summary[
                (cv_summary["target"] == target)
                & (cv_summary["feature_set"] == best_curve["feature_set"])
                & (cv_summary["model_name"] == best_curve["model_name"])
            ]["r2_std"].iloc[0]
        )
        if gain_tail >= 0.02:
            data_need = "More data likely helps"
        elif gain_tail >= 0.01:
            data_need = "Marginal gains still likely"
        else:
            data_need = "Curve is near plateau"
        rows.append(
            {
                "target": target,
                "best_feature_set": best_curve["feature_set"],
                "best_model": best_curve["model_name"],
                "best_valid_r2": float(best_curve["valid_r2_mean"]),
                "tail_gain_80_to_100": gain_tail,
                "best_cv_r2_std": variability,
                "assessment": data_need,
            }
        )
    takeaway_df = pd.DataFrame(rows)
    takeaway_df.to_csv(output_path("learning_curve_takeaways.csv"), index=False)
    return takeaway_df


def build_final_recommendation(
    takeaways: pd.DataFrame,
    region_df: pd.DataFrame,
    class_df: pd.DataFrame,
    redundancy: Dict[str, Any],
) -> list[str]:
    lines: list[str] = [
        "This analysis asks whether the current 2,880-row dataset is large enough to support stable conclusions for the three main SAR targets.",
        "",
    ]

    for row in takeaways.itertuples():
        region_spread = float(
            region_df[region_df["target"] == row.target]["r2"].max()
            - region_df[region_df["target"] == row.target]["r2"].min()
        )
        class_spread = float(
            class_df[class_df["target"] == row.target]["r2"].max()
            - class_df[class_df["target"] == row.target]["r2"].min()
        )
        if row.assessment == "Curve is near plateau" and row.best_valid_r2 >= 0.85 and row.best_cv_r2_std <= 0.05:
            verdict = "More data is not the primary bottleneck."
        elif row.assessment == "Curve is near plateau":
            verdict = "More generic rows are unlikely to change the result much; target quality or features are the tighter bottleneck."
        else:
            verdict = "Additional data is likely to produce measurable gains."
        lines.append(
            f"- {row.target}: best curve uses {FEATURE_LABELS[row.best_feature_set]} with {row.best_model.title()}; held-out R2 at full size is {row.best_valid_r2:.3f}, tail gain from 80% to 100% is {row.tail_gain_80_to_100:.3f}, CV std is {row.best_cv_r2_std:.3f}, region spread is {region_spread:.3f}, class spread is {class_spread:.3f}. {verdict}"
        )

    lines.extend(
        [
            "",
            f"- Spatial clustering reduces the nominal sample count. The 2,880 rows occupy {redundancy['n_spatial_blocks']} spatial blocks, with a cluster-size effective sample size of about {redundancy['spatial_block_effective_n']:.0f}.",
            f"- Region x class balancing is stronger than raw spatial independence: the region/class effective sample size is about {redundancy['region_class_effective_n']:.0f}.",
            f"- Embedding redundancy is moderate rather than extreme: mean nearest-neighbor cosine similarity is {redundancy['mean_nearest_neighbor_cosine']:.3f} and the 95th percentile is {redundancy['p95_nearest_neighbor_cosine']:.3f}.",
            "",
        ]
    )

    vv_status = takeaways[takeaways["target"] == "S1_VV"].iloc[0]
    vh_status = takeaways[takeaways["target"] == "S1_VH"].iloc[0]
    diff_status = takeaways[takeaways["target"] == "S1_VV_div_VH"].iloc[0]

    if vv_status["assessment"] == "Curve is near plateau" and vh_status["assessment"] == "Curve is near plateau":
        overall_mode = "VV and VH look closer to a feature-limited or target-limited regime than a purely data-limited regime."
    else:
        overall_mode = "VV and VH still show some sensitivity to sample size, so the project is not fully saturated."
    if diff_status["assessment"] != "Curve is near plateau":
        diff_mode = "The polarization-difference target is the clearest place where more data should help."
    else:
        diff_mode = "The polarization-difference target is weaker, but the remaining gap may be feature-driven as much as sample-driven."

    lines.extend(
        [
            overall_mode,
            diff_mode,
            "If more data is added, the highest-value additions are more independent spatial samples, more diverse regions, and targeted examples from difficult land-cover conditions and extreme polarization-difference cases rather than simply duplicating the current balanced design.",
        ]
    )
    return lines


def build_pdf_report(
    curve_takeaways: pd.DataFrame,
    cv_summary: pd.DataFrame,
    by_region: pd.DataFrame,
    by_class: pd.DataFrame,
    by_region_class: pd.DataFrame,
    best_models: pd.DataFrame,
    region_perf: pd.DataFrame,
    class_perf: pd.DataFrame,
    redundancy: Dict[str, Any],
) -> Path:
    report_path = output_path("summary_report.pdf")
    top_lines = [
        "Executive Summary",
        "",
        "Current dataset size: 2,880 rows across four regions and nine Dynamic World classes.",
        "",
    ]
    for row in curve_takeaways.itertuples():
        top_lines.append(
            f"- {row.target}: best observed learning-curve result is {row.best_model.title()} with {FEATURE_LABELS[row.best_feature_set]} at validation R2={row.best_valid_r2:.3f}; assessment: {row.assessment}."
        )
    top_lines.extend(
        [
            "",
            f"- Spatial-block effective sample size is about {redundancy['spatial_block_effective_n']:.0f} out of 2,880 nominal rows, so the dataset is materially clustered.",
            f"- Mean nearest-neighbor embedding cosine similarity is {redundancy['mean_nearest_neighbor_cosine']:.3f}, which indicates moderate redundancy but not collapse into near-duplicates.",
            "",
            "Recommendation",
            "",
        ]
    )
    top_lines.extend(build_final_recommendation(curve_takeaways, region_perf, class_perf, redundancy))

    with plt.rc_context({"figure.max_open_warning": 0}):
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(report_path) as pdf:
            draw_text_page("Data Sufficiency Analysis", top_lines, pdf)
            draw_dataframe_page(
                "Learning Curve Summary",
                curve_takeaways.assign(
                    best_feature_set=curve_takeaways["best_feature_set"].map(FEATURE_LABELS),
                    best_model=curve_takeaways["best_model"].str.title(),
                    best_valid_r2=curve_takeaways["best_valid_r2"].map(lambda v: f"{v:.3f}"),
                    tail_gain_80_to_100=curve_takeaways["tail_gain_80_to_100"].map(lambda v: f"{v:.3f}"),
                    best_cv_r2_std=curve_takeaways["best_cv_r2_std"].map(lambda v: f"{v:.3f}"),
                ),
                pdf,
                footnote="Assessment is based on the validation learning curve tail from 80% to 100% of the current training set and the repeated grouped-CV variance of the best-performing model/feature combination.",
            )
            draw_image_grid_page(
                "Learning Curves",
                [output_path(f"learning_curve_{target}.png") for target in TARGET_COLS[:2]] + [output_path("learning_curve_S1_VV_div_VH.png")],
                pdf,
            )
            draw_image_grid_page(
                "Learning Curve RMSE",
                [output_path(f"learning_curve_rmse_{target}.png") for target in TARGET_COLS[:2]] + [output_path("learning_curve_rmse_S1_VV_div_VH.png")],
                pdf,
            )
            draw_dataframe_page(
                "Repeated Grouped-CV Stability",
                cv_summary.assign(
                    feature_set=cv_summary["feature_set"].map(FEATURE_LABELS),
                    model_name=cv_summary["model_name"].str.title(),
                    r2_mean=cv_summary["r2_mean"].map(lambda v: f"{v:.3f}"),
                    r2_std=cv_summary["r2_std"].map(lambda v: f"{v:.3f}"),
                    rmse_mean=cv_summary["rmse_mean"].map(lambda v: f"{v:.3f}"),
                    rmse_std=cv_summary["rmse_std"].map(lambda v: f"{v:.3f}"),
                    mae_mean=cv_summary["mae_mean"].map(lambda v: f"{v:.3f}"),
                    mae_std=cv_summary["mae_std"].map(lambda v: f"{v:.3f}"),
                    pearson_r_mean=cv_summary["pearson_r_mean"].map(lambda v: f"{v:.3f}"),
                ),
                pdf,
                footnote="Each row aggregates 12 validation folds per setup: 3 random seeds x 4 grouped folds.",
            )
            draw_image_grid_page("Cross-Validation Stability", [output_path("cv_stability_r2.png")], pdf)
            draw_dataframe_page("Coverage By Region", by_region, pdf)
            draw_dataframe_page("Coverage By Dynamic World Class", by_class, pdf)
            draw_dataframe_page("Coverage By Region x Class", by_region_class, pdf)
            draw_image_grid_page(
                "Coverage Visuals",
                [
                    output_path("coverage_by_region.png"),
                    output_path("coverage_by_land_cover.png"),
                    output_path("coverage_heatmap_region_land_cover.png"),
                ],
                pdf,
            )
            draw_dataframe_page(
                "Best Model Per Target",
                best_models.assign(
                    feature_set=best_models["feature_set"].map(FEATURE_LABELS),
                    model_name=best_models["model_name"].str.title(),
                    r2_mean=best_models["r2_mean"].map(lambda v: f"{v:.3f}"),
                    r2_std=best_models["r2_std"].map(lambda v: f"{v:.3f}"),
                    rmse_mean=best_models["rmse_mean"].map(lambda v: f"{v:.3f}"),
                    mae_mean=best_models["mae_mean"].map(lambda v: f"{v:.3f}"),
                ),
                pdf,
            )
            draw_dataframe_page(
                "Best Model Region Performance",
                region_perf.assign(
                    feature_set=region_perf["feature_set"].map(FEATURE_LABELS),
                    model_name=region_perf["model_name"].str.title(),
                    r2=region_perf["r2"].map(lambda v: f"{v:.3f}"),
                    rmse=region_perf["rmse"].map(lambda v: f"{v:.3f}"),
                    mae=region_perf["mae"].map(lambda v: f"{v:.3f}"),
                    pearson_r=region_perf["pearson_r"].map(lambda v: f"{v:.3f}"),
                ),
                pdf,
            )
            draw_dataframe_page(
                "Best Model Land-Cover Performance",
                class_perf.assign(
                    feature_set=class_perf["feature_set"].map(FEATURE_LABELS),
                    model_name=class_perf["model_name"].str.title(),
                    r2=class_perf["r2"].map(lambda v: f"{v:.3f}"),
                    rmse=class_perf["rmse"].map(lambda v: f"{v:.3f}"),
                    mae=class_perf["mae"].map(lambda v: f"{v:.3f}"),
                    pearson_r=class_perf["pearson_r"].map(lambda v: f"{v:.3f}"),
                ),
                pdf,
            )
            draw_image_grid_page(
                "Subgroup Stability",
                [
                    output_path("subgroup_region_r2_S1_VV.png"),
                    output_path("subgroup_region_r2_S1_VH.png"),
                    output_path("subgroup_region_r2_S1_VV_div_VH.png"),
                    output_path("subgroup_land_cover_r2_S1_VV.png"),
                ],
                pdf,
            )
            draw_image_grid_page(
                "Subgroup Stability (cont.)",
                [
                    output_path("subgroup_land_cover_r2_S1_VH.png"),
                    output_path("subgroup_land_cover_r2_S1_VV_div_VH.png"),
                    output_path("redundancy_spatial_block_histogram.png"),
                    output_path("redundancy_nearest_neighbor_cosine.png"),
                ],
                pdf,
            )
            draw_text_page(
                "Redundancy Diagnostics",
                [
                    f"- Spatial blocks: {redundancy['n_spatial_blocks']} total, mean {redundancy['mean_rows_per_spatial_block']:.1f} rows, max {redundancy['max_rows_per_spatial_block']} rows.",
                    f"- Spatial-block effective sample size: {redundancy['spatial_block_effective_n']:.1f}.",
                    f"- Region/class effective sample size: {redundancy['region_class_effective_n']:.1f}.",
                    f"- Mean nearest-neighbor cosine: {redundancy['mean_nearest_neighbor_cosine']:.3f}.",
                    f"- 95th percentile nearest-neighbor cosine: {redundancy['p95_nearest_neighbor_cosine']:.3f}.",
                    f"- Exact duplicate embedding rows: {redundancy['exact_duplicate_embedding_rows']}.",
                    "",
                ]
                + build_final_recommendation(curve_takeaways, region_perf, class_perf, redundancy),
                pdf,
            )
    REPORTS_DIR.mkdir(exist_ok=True)
    shutil.copyfile(report_path, report_copy_path("summary_report.pdf"))
    return report_path


def main() -> None:
    ensure_output_dirs()
    full_df = assign_spatial_blocks(load_dataset())
    feature_frames = build_feature_frames(full_df)

    curve_summary = run_learning_curves(full_df, feature_frames)
    save_learning_curve_plots(curve_summary)

    cv_summary = run_repeated_grouped_cv(full_df, feature_frames)
    plot_cv_stability(cv_summary)
    best_models = select_best_models(cv_summary)

    by_region, by_class, by_region_class = build_coverage_tables(full_df)
    plot_coverage(by_region, by_class, by_region_class)

    region_perf, class_perf = evaluate_best_models_by_subgroup(full_df, feature_frames, best_models)
    plot_subgroup_performance(region_perf, class_perf)

    redundancy = compute_redundancy_diagnostics(full_df, feature_frames)
    plot_redundancy(full_df, redundancy)

    takeaways = build_learning_curve_takeaways(curve_summary, cv_summary)
    report_path = build_pdf_report(
        takeaways,
        cv_summary,
        by_region,
        by_class,
        by_region_class,
        best_models,
        region_perf,
        class_perf,
        redundancy,
    )

    print("Saved learning curve summary to:", output_path("learning_curve_summary.csv"))
    print("Saved CV stability summary to:", output_path("cv_stability_summary.csv"))
    print("Saved subgroup performance to:", output_path("best_model_region_performance.csv"))
    print("Saved redundancy summary to:", output_path("redundancy_summary.json"))
    print("Saved PDF report to:", report_path)
    print("Copied PDF report to:", report_copy_path("summary_report.pdf"))


if __name__ == "__main__":
    main()
