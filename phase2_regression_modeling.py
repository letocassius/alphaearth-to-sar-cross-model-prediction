#!/usr/bin/env python3
"""
Phase 2 regression modeling for AlphaEarth-to-SAR cross-modal prediction.

Workflow:
1. Load the combined all-regions CSV.
2. Create a smaller balanced subset across region x Dynamic World label.
3. Train and evaluate ridge and boosted-tree regressors for SAR targets.
4. Save metrics, plots, and feature-importance summaries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv")
SUBSET_PATH = Path("DataSources/alphaearth_s1_dw_samples_balanced_subset_2024.csv")
OUTPUT_DIR = Path("phase2_outputs")
RANDOM_STATE = 42
SAMPLES_PER_REGION_LABEL = 20
EMBEDDING_COLS = [f"A{i:02d}" for i in range(64)]
TARGET_COLS = ["S1_VV", "S1_VH", "S1_VV_div_VH"]
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


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["dw_label"] = df["dw_label"].astype(int)
    df["dw_label_name"] = df["dw_label"].map(DW_LABEL_NAMES)
    df["region_dw_label"] = df["region"] + "__" + df["dw_label"].astype(str)
    return df


def create_balanced_subset(df: pd.DataFrame, samples_per_group: int) -> pd.DataFrame:
    counts = df.groupby(["region", "dw_label"]).size()
    if (counts < samples_per_group).any():
        raise ValueError("At least one region x dw_label cell is smaller than the requested sample size.")

    subset = (
        df.groupby(["region", "dw_label"], group_keys=False)
        .sample(n=samples_per_group, random_state=RANDOM_STATE)
        .sort_values(["region", "dw_label", "system:index"])
        .reset_index(drop=True)
    )
    subset["region_dw_label"] = subset["region"] + "__" + subset["dw_label"].astype(str)
    subset.to_csv(SUBSET_PATH, index=False)
    return subset


def make_ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_hgbr_pipeline(
    learning_rate: float,
    max_depth: int | None,
    min_samples_leaf: int,
    l2_regularization: float,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=HistGradientBoostingRegressor(
                        random_state=RANDOM_STATE,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        max_iter=200,
                        min_samples_leaf=min_samples_leaf,
                        l2_regularization=l2_regularization,
                    ),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )


def safe_pearsonr(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(pearsonr(y_true, y_pred).statistic)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": safe_pearsonr(y_true, y_pred),
    }


def save_predicted_vs_actual_plot(
    target: str,
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.65, edgecolor="none")
    data_min = min(float(np.min(y_true)), float(np.min(y_pred)))
    data_max = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([data_min, data_max], [data_min, data_max], linestyle="--", color="black", linewidth=1)
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"{model_name}: Predicted vs Actual {target}")
    plt.text(
        0.03,
        0.97,
        f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nMAE = {metrics['mae']:.3f}\nr = {metrics['pearson_r']:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"predicted_vs_actual_{target}_{model_name.lower()}.png", dpi=200)
    plt.close()


def save_ridge_importance(target: str, best_estimator: Pipeline) -> pd.DataFrame:
    ridge = best_estimator.named_steps["model"]
    coefficients = pd.DataFrame(
        {
            "feature": EMBEDDING_COLS,
            "coefficient": ridge.coef_,
            "abs_coefficient": np.abs(ridge.coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coefficients.to_csv(OUTPUT_DIR / f"feature_importance_{target}_ridge.csv", index=False)
    return coefficients


def save_hgbr_importance(
    target: str,
    best_estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    result = permutation_importance(
        best_estimator,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
        scoring="r2",
    )
    importance = pd.DataFrame(
        {
            "feature": EMBEDDING_COLS,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(OUTPUT_DIR / f"feature_importance_{target}_histgradientboosting.csv", index=False)
    return importance


def mean_cv_r2(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: KFold) -> float:
    scores: List[float] = []
    for train_fold, valid_fold in cv.split(X):
        fitted = clone(model)
        fitted.fit(X.iloc[train_fold], y.iloc[train_fold])
        pred = fitted.predict(X.iloc[valid_fold])
        scores.append(r2_score(y.iloc[valid_fold], pred))
    return float(np.mean(scores))


def fit_best_ridge(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, float, Dict[str, float]]:
    params = {"alpha": 1.0}
    model = make_ridge_pipeline(**params)
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_score = mean_cv_r2(model, X_train, y_train, cv)
    model.fit(X_train, y_train)
    return model, cv_score, params


def fit_best_hgbr(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, float, Dict[str, float]]:
    params = {
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    }
    model = make_hgbr_pipeline(**params)
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_score = mean_cv_r2(model, X_train, y_train, cv)
    model.fit(X_train, y_train)
    return model, cv_score, params


def build_summary_markdown(
    full_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    regional_df: pd.DataFrame,
) -> str:
    vv_metrics = metrics_df[metrics_df["target"] == "S1_VV"].sort_values("r2", ascending=False)
    best_vv = vv_metrics.iloc[0]
    lines = [
        "# Phase 2 Regression Modeling Summary",
        "",
        "## Dataset Revision",
        f"- Full combined dataset: {len(full_df)} rows.",
        f"- Regions: {full_df['region'].nunique()} ({', '.join(sorted(full_df['region'].unique()))}).",
        f"- Dynamic World labels: {full_df['dw_label'].nunique()} with 320 samples per label overall.",
        f"- Balanced subset: {len(subset_df)} rows built from {SAMPLES_PER_REGION_LABEL} samples per region x dw_label cell.",
        "",
        "## Overall Test Performance",
        metrics_df.round(4).to_string(index=False),
        "",
        "## Regional S1_VV Performance",
        regional_df.round(4).to_string(index=False),
        "",
        "## Main Takeaway",
        (
            f"- Best S1_VV model on the held-out test set: {best_vv['model']} "
            f"(R²={best_vv['r2']:.3f}, RMSE={best_vv['rmse']:.3f}, MAE={best_vv['mae']:.3f})."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_output_dirs()

    full_df = load_dataset()
    subset_df = create_balanced_subset(full_df, samples_per_group=SAMPLES_PER_REGION_LABEL)

    train_idx, test_idx = train_test_split(
        subset_df.index,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=subset_df["region_dw_label"],
    )
    train_df = subset_df.loc[train_idx].reset_index(drop=True)
    test_df = subset_df.loc[test_idx].reset_index(drop=True)
    X_train = train_df[EMBEDDING_COLS]
    X_test = test_df[EMBEDDING_COLS]

    metrics_rows: List[Dict[str, object]] = []
    regional_rows: List[Dict[str, object]] = []

    for target in TARGET_COLS:
        y_train = train_df[target]
        y_test = test_df[target]

        model_fits = [
            ("ridge", fit_best_ridge(X_train, y_train)),
            ("histgradientboosting", fit_best_hgbr(X_train, y_train)),
        ]

        for model_name, (best_model, best_cv_r2, best_params) in model_fits:
            y_pred = best_model.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred)
            metrics_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "best_cv_r2": float(best_cv_r2),
                    "best_params": str(best_params),
                    **metrics,
                }
            )
            save_predicted_vs_actual_plot(target, model_name, y_test, y_pred, metrics)

            if model_name == "ridge":
                save_ridge_importance(target, best_model)
            if model_name == "histgradientboosting":
                save_hgbr_importance(target, best_model, X_test, y_test)

            if target == "S1_VV":
                regional_frame = pd.DataFrame(
                    {
                        "region": test_df["region"],
                        "y_true": y_test,
                        "y_pred": y_pred,
                    }
                )
                for region, part in regional_frame.groupby("region"):
                    region_metrics = evaluate_predictions(part["y_true"], part["y_pred"])
                    regional_rows.append(
                        {
                            "target": target,
                            "model": model_name,
                            "region": region,
                            **region_metrics,
                            "n_test": int(len(part)),
                        }
                    )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target", "r2"], ascending=[True, False])
    regional_df = pd.DataFrame(regional_rows).sort_values(["model", "region"])
    metrics_df.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)
    regional_df.to_csv(OUTPUT_DIR / "regional_metrics_S1_VV.csv", index=False)

    summary = build_summary_markdown(full_df, subset_df, metrics_df, regional_df)
    (OUTPUT_DIR / "phase2_summary.md").write_text(summary)

    print("Saved balanced subset to:", SUBSET_PATH)
    print("Saved regression metrics to:", OUTPUT_DIR / "regression_metrics.csv")
    print("Saved regional metrics to:", OUTPUT_DIR / "regional_metrics_S1_VV.csv")
    print("Saved summary to:", OUTPUT_DIR / "phase2_summary.md")


if __name__ == "__main__":
    main()
