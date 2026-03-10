#!/usr/bin/env python3
"""
Phase 2 regression modeling for AlphaEarth-to-SAR cross-modal prediction.

Workflow:
1. Load the combined all-regions CSV.
2. Create a smaller balanced subset across region x Dynamic World label.
3. Audit the modeling data and define spatial blocks for grouped validation.
4. Train and evaluate ridge and tuned LightGBM regressors for SAR targets.
5. Save metrics, plots, feature-importance summaries, and optimization artifacts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv")
SUBSET_PATH = Path("DataSources/alphaearth_s1_dw_samples_balanced_subset_2024.csv")
OUTPUT_DIR = Path("phase2_outputs")
RANDOM_STATE = 42
SAMPLES_PER_REGION_LABEL = 20
SPATIAL_BINS_PER_AXIS = 3
EMBEDDING_COLS = [f"A{i:02d}" for i in range(64)]
FEATURE_COLS = EMBEDDING_COLS
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
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
OPTUNA_TRIALS = 30


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["dw_label"] = df["dw_label"].astype(int)
    df["dw_label_name"] = df["dw_label"].map(DW_LABEL_NAMES)
    df["region_dw_label"] = df["region"] + "__" + df["dw_label"].astype(str)
    return df


def assign_spatial_blocks(df: pd.DataFrame) -> pd.DataFrame:
    enriched_parts: List[pd.DataFrame] = []
    for region, region_df in df.groupby("region", sort=False):
        part = region_df.copy()
        part["lat_bin"] = pd.qcut(
            part["latitude"].rank(method="first"),
            q=SPATIAL_BINS_PER_AXIS,
            labels=False,
            duplicates="drop",
        ).astype(int)
        part["lon_bin"] = pd.qcut(
            part["longitude"].rank(method="first"),
            q=SPATIAL_BINS_PER_AXIS,
            labels=False,
            duplicates="drop",
        ).astype(int)
        part["spatial_block"] = (
            region
            + "__"
            + part["lat_bin"].astype(str)
            + "_"
            + part["lon_bin"].astype(str)
        )
        enriched_parts.append(part)
    return pd.concat(enriched_parts, ignore_index=True)


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
    subset["dw_label_name"] = subset["dw_label"].map(DW_LABEL_NAMES)
    subset["region_dw_label"] = subset["region"] + "__" + subset["dw_label"].astype(str)
    subset = assign_spatial_blocks(subset)
    subset.to_csv(SUBSET_PATH, index=False)
    return subset


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLS].copy()


def make_ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def safe_pearsonr(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    if np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0:
        return np.nan
    return float(pearsonr(y_true_arr, y_pred_arr).statistic)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": safe_pearsonr(y_true, y_pred),
    }


def summarize_cv_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    frame = pd.DataFrame(fold_metrics)
    return {
        "cv_r2_mean": float(frame["r2"].mean()),
        "cv_r2_std": float(frame["r2"].std(ddof=0)),
        "cv_rmse_mean": float(frame["rmse"].mean()),
        "cv_rmse_std": float(frame["rmse"].std(ddof=0)),
        "cv_mae_mean": float(frame["mae"].mean()),
        "cv_mae_std": float(frame["mae"].std(ddof=0)),
        "cv_best_iteration_mean": float(frame["best_iteration"].mean()) if "best_iteration" in frame else np.nan,
        "cv_best_iteration_std": float(frame["best_iteration"].std(ddof=0)) if "best_iteration" in frame else np.nan,
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
        f"R2 = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nMAE = {metrics['mae']:.3f}\nr = {metrics['pearson_r']:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"predicted_vs_actual_{target}_{model_name.lower()}.png", dpi=200)
    plt.close()


def save_residual_histogram(
    target: str,
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> None:
    residuals = y_pred - y_true.to_numpy()
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=24, color="#93c5fd", edgecolor="#1d4ed8")
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Residual (predicted - actual)")
    plt.ylabel("Count")
    plt.title(f"{model_name}: Residual Histogram for {target}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"residual_histogram_{target}_{model_name.lower()}.png", dpi=200)
    plt.close()


def save_ridge_importance(target: str, best_estimator: Pipeline) -> pd.DataFrame:
    ridge = best_estimator.named_steps["model"]
    coefficients = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coefficient": ridge.coef_,
            "abs_coefficient": np.abs(ridge.coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coefficients.to_csv(OUTPUT_DIR / f"feature_importance_{target}_ridge.csv", index=False)
    return coefficients


def save_lightgbm_importance(
    target: str,
    model: LGBMRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    gain_importance = model.booster_.feature_importance(importance_type="gain")
    split_importance = model.booster_.feature_importance(importance_type="split")
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
        scoring="r2",
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "gain_importance": gain_importance,
            "split_importance": split_importance,
            "permutation_mean": result.importances_mean,
            "permutation_std": result.importances_std,
        }
    ).sort_values("permutation_mean", ascending=False)
    importance.to_csv(OUTPUT_DIR / f"feature_importance_{target}_lightgbm.csv", index=False)
    return importance


def audit_dataset(full_df: pd.DataFrame, subset_df: pd.DataFrame) -> Dict[str, Any]:
    missing_ratio = float(subset_df[FEATURE_COLS + TARGET_COLS].isna().mean().mean())
    duplicated_rows = int(subset_df.duplicated(subset=FEATURE_COLS + ["region", "dw_label"]).sum())
    feature_variances = subset_df[FEATURE_COLS].var()
    corr = subset_df[FEATURE_COLS].corr().abs()
    high_corr_pairs = int(np.triu((corr > 0.95).to_numpy(), k=1).sum())

    audit: Dict[str, Any] = {
        "dataset_version": str(DATA_PATH),
        "n_full_rows": int(len(full_df)),
        "n_subset_rows": int(len(subset_df)),
        "n_features": int(len(FEATURE_COLS)),
        "n_regions": int(subset_df["region"].nunique()),
        "n_land_use_labels": int(subset_df["dw_label"].nunique()),
        "missing_ratio": missing_ratio,
        "duplicated_subset_rows": duplicated_rows,
        "constant_embedding_dims": int((feature_variances == 0).sum()),
        "near_constant_embedding_dims": int((feature_variances < 1e-4).sum()),
        "high_corr_feature_pairs_abs_gt_0_95": high_corr_pairs,
        "spatial_blocks": int(subset_df["spatial_block"].nunique()),
        "split_method": "70/30 stratified holdout by region_dw_label; CV uses StratifiedGroupKFold grouped by spatial_block",
        "leakage_note": "Held-out split preserves region x label balance but still mixes regions; grouped CV reduces within-region spatial leakage during tuning.",
        "target_summary": {},
    }
    for target in TARGET_COLS:
        target_series = subset_df[target]
        audit["target_summary"][target] = {
            "mean": float(target_series.mean()),
            "std": float(target_series.std()),
            "min": float(target_series.min()),
            "max": float(target_series.max()),
            "skew": float(target_series.skew()),
            "outlier_count_abs_z_gt_3": int((((target_series - target_series.mean()) / target_series.std()).abs() > 3).sum()),
        }

    (OUTPUT_DIR / "dataset_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def save_test_predictions(
    test_df: pd.DataFrame,
    target: str,
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> None:
    out = test_df[["system:index", "region", "dw_label", "dw_label_name", "latitude", "longitude", "spatial_block"]].copy()
    out["target"] = target
    out["model"] = model_name
    out["actual"] = y_true.to_numpy()
    out["predicted"] = y_pred
    out["residual"] = out["predicted"] - out["actual"]
    out.to_csv(OUTPUT_DIR / f"test_predictions_{target}_{model_name}.csv", index=False)


def subgroup_metrics(
    test_df: pd.DataFrame,
    target: str,
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "region": test_df["region"],
            "dw_label_name": test_df["dw_label_name"],
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    region_rows: List[Dict[str, object]] = []
    land_use_rows: List[Dict[str, object]] = []

    for region, part in frame.groupby("region"):
        region_rows.append(
            {
                "target": target,
                "model": model_name,
                "region": region,
                **evaluate_predictions(part["y_true"], part["y_pred"]),
                "n_test": int(len(part)),
            }
        )
    for label, part in frame.groupby("dw_label_name"):
        land_use_rows.append(
            {
                "target": target,
                "model": model_name,
                "dw_label_name": label,
                **evaluate_predictions(part["y_true"], part["y_pred"]),
                "n_test": int(len(part)),
            }
        )
    return pd.DataFrame(region_rows), pd.DataFrame(land_use_rows)


def fit_best_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedGroupKFold,
    strat_labels: pd.Series,
    groups: pd.Series,
) -> Tuple[Pipeline, Dict[str, float], Dict[str, float]]:
    trial_rows: List[Dict[str, float]] = []
    best_alpha = RIDGE_ALPHAS[0]
    best_rmse = float("inf")

    for alpha in RIDGE_ALPHAS:
        fold_metrics: List[Dict[str, float]] = []
        for train_fold, valid_fold in cv.split(X_train, strat_labels, groups):
            model = make_ridge_pipeline(alpha)
            model.fit(X_train.iloc[train_fold], y_train.iloc[train_fold])
            pred = model.predict(X_train.iloc[valid_fold])
            metrics = evaluate_predictions(y_train.iloc[valid_fold], pred)
            fold_metrics.append({**metrics, "best_iteration": np.nan})
        summary = summarize_cv_metrics(fold_metrics)
        trial_rows.append({"alpha": alpha, **summary})
        if summary["cv_rmse_mean"] < best_rmse:
            best_rmse = summary["cv_rmse_mean"]
            best_alpha = alpha

    best_model = make_ridge_pipeline(best_alpha)
    best_model.fit(X_train, y_train)
    best_summary = next(row for row in trial_rows if row["alpha"] == best_alpha)
    return best_model, best_summary, {"alpha": best_alpha}


def build_lightgbm_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 4000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 96),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "verbosity": -1,
    }


def fit_best_lightgbm(
    target: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedGroupKFold,
    strat_labels: pd.Series,
    groups: pd.Series,
) -> Tuple[LGBMRegressor, Dict[str, float], Dict[str, Any]]:
    trial_history: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = build_lightgbm_params(trial)
        fold_metrics: List[Dict[str, float]] = []

        for train_fold, valid_fold in cv.split(X_train, strat_labels, groups):
            model = LGBMRegressor(**params)
            model.fit(
                X_train.iloc[train_fold],
                y_train.iloc[train_fold],
                eval_set=[(X_train.iloc[valid_fold], y_train.iloc[valid_fold])],
                eval_metric="rmse",
                callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=0)],
            )
            pred = model.predict(X_train.iloc[valid_fold], num_iteration=model.best_iteration_)
            metrics = evaluate_predictions(y_train.iloc[valid_fold], pred)
            fold_metrics.append({**metrics, "best_iteration": float(model.best_iteration_ or params["n_estimators"])})

        summary = summarize_cv_metrics(fold_metrics)
        trial_history.append({"trial": trial.number, **params, **summary})
        trial.set_user_attr("cv_summary", summary)
        return summary["cv_rmse_mean"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    cv_results = pd.DataFrame(trial_history).sort_values("cv_rmse_mean")
    cv_results.to_csv(OUTPUT_DIR / f"cv_results_{target}_lightgbm.csv", index=False)

    best_params = dict(study.best_params)
    best_summary = dict(study.best_trial.user_attrs["cv_summary"])
    best_n_estimators = max(50, int(round(best_summary["cv_best_iteration_mean"])))
    best_params.update(
        {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": best_n_estimators,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": -1,
        }
    )
    (OUTPUT_DIR / f"best_params_{target}_lightgbm.json").write_text(json.dumps(best_params, indent=2))

    final_model = LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)
    return final_model, best_summary, best_params


def build_summary_markdown(
    full_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    audit: Dict[str, Any],
    metrics_df: pd.DataFrame,
    regional_df: pd.DataFrame,
    land_use_df: pd.DataFrame,
) -> str:
    best_vv = metrics_df[metrics_df["target"] == "S1_VV"].sort_values("r2", ascending=False).iloc[0]
    best_ratio = metrics_df[metrics_df["target"] == "S1_VV_div_VH"].sort_values("r2", ascending=False).iloc[0]
    lines = [
        "# Phase 2 Regression Modeling Summary",
        "",
        "## Dataset Revision",
        f"- Full combined dataset: {len(full_df)} rows.",
        f"- Regions: {full_df['region'].nunique()} ({', '.join(sorted(full_df['region'].unique()))}).",
        f"- Dynamic World labels: {full_df['dw_label'].nunique()} with 320 samples per label overall.",
        f"- Balanced subset: {len(subset_df)} rows built from {SAMPLES_PER_REGION_LABEL} samples per region x dw_label cell.",
        f"- Spatial validation blocks: {audit['spatial_blocks']} blocks built from {SPATIAL_BINS_PER_AXIS} x {SPATIAL_BINS_PER_AXIS} quantile bins within each region.",
        "",
        "## Pipeline Audit",
        f"- Missing ratio across features and targets: {audit['missing_ratio']:.4f}.",
        f"- Duplicate subset rows on feature + region + label keys: {audit['duplicated_subset_rows']}.",
        f"- Near-constant embedding dimensions: {audit['near_constant_embedding_dims']}.",
        f"- Highly correlated embedding pairs (abs corr > 0.95): {audit['high_corr_feature_pairs_abs_gt_0_95']}.",
        f"- Leakage note: {audit['leakage_note']}",
        "",
        "## Overall Test Performance",
        metrics_df.round(4).to_string(index=False),
        "",
        "## Regional S1_VV Performance",
        regional_df[regional_df["target"] == "S1_VV"].round(4).to_string(index=False),
        "",
        "## Land-Use S1_VV Performance",
        land_use_df[land_use_df["target"] == "S1_VV"].round(4).to_string(index=False),
        "",
        "## Main Takeaways",
        (
            f"- Best S1_VV model on the held-out test set: {best_vv['model']} "
            f"(R2={best_vv['r2']:.3f}, RMSE={best_vv['rmse']:.3f}, MAE={best_vv['mae']:.3f})."
        ),
        (
            f"- Best VV/VH-ratio model on the held-out test set: {best_ratio['model']} "
            f"(R2={best_ratio['r2']:.3f}, RMSE={best_ratio['rmse']:.3f}, MAE={best_ratio['mae']:.3f})."
        ),
        "- Tuned LightGBM now uses Bayesian optimization, grouped CV over spatial blocks, and early-stopping-informed tree counts.",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_output_dirs()

    full_df = load_dataset()
    subset_df = create_balanced_subset(full_df, samples_per_group=SAMPLES_PER_REGION_LABEL)
    audit = audit_dataset(full_df, subset_df)

    train_idx, test_idx = train_test_split(
        subset_df.index,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=subset_df["region_dw_label"],
    )
    train_df = subset_df.loc[train_idx].reset_index(drop=True)
    test_df = subset_df.loc[test_idx].reset_index(drop=True)
    X_train = build_feature_frame(train_df)
    X_test = build_feature_frame(test_df)
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    strat_labels = train_df["region_dw_label"]
    groups = train_df["spatial_block"]

    metrics_rows: List[Dict[str, object]] = []
    regional_frames: List[pd.DataFrame] = []
    land_use_frames: List[pd.DataFrame] = []

    experiment_start = time.time()

    for target in TARGET_COLS:
        y_train = train_df[target]
        y_test = test_df[target]

        model_fits = [
            ("ridge", fit_best_ridge(X_train, y_train, cv, strat_labels, groups)),
            ("lightgbm", fit_best_lightgbm(target, X_train, y_train, cv, strat_labels, groups)),
        ]

        for model_name, (best_model, cv_summary, best_params) in model_fits:
            fit_start = time.time()
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            fit_seconds = time.time() - fit_start

            train_metrics = evaluate_predictions(y_train, y_train_pred)
            test_metrics = evaluate_predictions(y_test, y_test_pred)
            metrics_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "feature_set": "embedding_only",
                    "train_r2": train_metrics["r2"],
                    "train_rmse": train_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "cv_r2_mean": cv_summary["cv_r2_mean"],
                    "cv_r2_std": cv_summary["cv_r2_std"],
                    "cv_rmse_mean": cv_summary["cv_rmse_mean"],
                    "cv_rmse_std": cv_summary["cv_rmse_std"],
                    "cv_mae_mean": cv_summary["cv_mae_mean"],
                    "cv_mae_std": cv_summary["cv_mae_std"],
                    "cv_best_iteration_mean": cv_summary["cv_best_iteration_mean"],
                    "cv_best_iteration_std": cv_summary["cv_best_iteration_std"],
                    "best_params": json.dumps(best_params, sort_keys=True),
                    "r2": test_metrics["r2"],
                    "rmse": test_metrics["rmse"],
                    "mae": test_metrics["mae"],
                    "pearson_r": test_metrics["pearson_r"],
                    "fit_seconds": fit_seconds,
                }
            )
            save_predicted_vs_actual_plot(target, model_name, y_test, y_test_pred, test_metrics)
            save_residual_histogram(target, model_name, y_test, y_test_pred)
            save_test_predictions(test_df, target, model_name, y_test, y_test_pred)

            if model_name == "ridge":
                save_ridge_importance(target, best_model)
            elif model_name == "lightgbm":
                save_lightgbm_importance(target, best_model, X_test, y_test)

            region_df, land_use_df = subgroup_metrics(test_df, target, model_name, y_test, y_test_pred)
            regional_frames.append(region_df)
            land_use_frames.append(land_use_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target", "r2"], ascending=[True, False])
    regional_df = pd.concat(regional_frames, ignore_index=True).sort_values(["target", "model", "region"])
    land_use_df = pd.concat(land_use_frames, ignore_index=True).sort_values(["target", "model", "dw_label_name"])
    metrics_df.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)
    regional_df.to_csv(OUTPUT_DIR / "regional_metrics.csv", index=False)
    land_use_df.to_csv(OUTPUT_DIR / "land_use_metrics.csv", index=False)

    optimization_report = {
        "dataset_version": str(DATA_PATH),
        "random_seed": RANDOM_STATE,
        "feature_set": "embedding_only",
        "validation_strategy": "StratifiedGroupKFold(n_splits=4, grouped by spatial_block, stratified by region_dw_label)",
        "holdout_strategy": "train_test_split(test_size=0.30, stratify=region_dw_label)",
        "runtime_seconds": time.time() - experiment_start,
    }
    (OUTPUT_DIR / "optimization_report.json").write_text(json.dumps(optimization_report, indent=2))

    summary = build_summary_markdown(full_df, subset_df, audit, metrics_df, regional_df, land_use_df)
    (OUTPUT_DIR / "phase2_summary.md").write_text(summary)

    print("Saved balanced subset to:", SUBSET_PATH)
    print("Saved dataset audit to:", OUTPUT_DIR / "dataset_audit.json")
    print("Saved regression metrics to:", OUTPUT_DIR / "regression_metrics.csv")
    print("Saved regional metrics to:", OUTPUT_DIR / "regional_metrics.csv")
    print("Saved land-use metrics to:", OUTPUT_DIR / "land_use_metrics.csv")
    print("Saved summary to:", OUTPUT_DIR / "phase2_summary.md")


if __name__ == "__main__":
    main()
