#!/usr/bin/env python3
"""
Full-dataset LightGBM experiments that preserve the existing Phase 2 outputs.

This script uses all 2,880 rows and compares:
1. embedding_only
2. embedding_plus_context (region + Dynamic World probabilities)

It performs Bayesian hyperparameter tuning, repeated grouped-CV stability checks,
held-out evaluation, subgroup diagnostics, and saves a separate PDF-ready output set.
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "DataSources" / "alphaearth_s1_dw_samples_all_regions_2024.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "full_dataset"
RANDOM_STATE = 42
SPATIAL_BINS_PER_AXIS = 4
OPTUNA_TRIALS = 20
STABILITY_SEEDS = [42, 52, 62]
EMBEDDING_COLS = [f"A{i:02d}" for i in range(64)]
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


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["dw_label"] = df["dw_label"].astype(int)
    df["dw_label_name"] = df["dw_label"].map(DW_LABEL_NAMES)
    df["region_dw_label"] = df["region"] + "__" + df["dw_label"].astype(str)
    return df


def assign_spatial_blocks(df: pd.DataFrame) -> pd.DataFrame:
    out_parts: List[pd.DataFrame] = []
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
        part["spatial_block"] = region + "__" + part["lat_bin"].astype(str) + "_" + part["lon_bin"].astype(str)
        out_parts.append(part)
    return pd.concat(out_parts, ignore_index=True)


def build_feature_frames(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    region_dummies = pd.get_dummies(df["region"], prefix="region", dtype=float)
    feature_frames = {
        "embedding_only": df[EMBEDDING_COLS].copy(),
        "embedding_plus_context": pd.concat([df[EMBEDDING_COLS], df[DW_PROB_COLS], region_dummies], axis=1),
    }
    return feature_frames


def build_lightgbm_params(trial: optuna.Trial, seed: int) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": -1,
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
        "cv_best_iteration_mean": float(frame["best_iteration"].mean()),
        "cv_best_iteration_std": float(frame["best_iteration"].std(ddof=0)),
    }


def tune_lightgbm(
    feature_set: str,
    target: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strat_labels: pd.Series,
    groups: pd.Series,
) -> Tuple[LGBMRegressor, Dict[str, Any], Dict[str, float]]:
    trial_history: List[Dict[str, Any]] = []
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = build_lightgbm_params(trial, seed=RANDOM_STATE)
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
    cv_results.to_csv(OUTPUT_DIR / f"cv_results_{feature_set}_{target}_lightgbm.csv", index=False)

    best_summary = dict(study.best_trial.user_attrs["cv_summary"])
    best_params = dict(study.best_params)
    best_params.update(
        {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": max(100, int(round(best_summary["cv_best_iteration_mean"]))),
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbosity": -1,
        }
    )
    (OUTPUT_DIR / f"best_params_{feature_set}_{target}_lightgbm.json").write_text(json.dumps(best_params, indent=2))

    final_model = LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)
    return final_model, best_params, best_summary


def run_stability_check(
    feature_set: str,
    target: str,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strat_labels: pd.Series,
    groups: pd.Series,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in STABILITY_SEEDS:
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        for fold_idx, (train_fold, valid_fold) in enumerate(cv.split(X_train, strat_labels, groups), start=1):
            fold_params = dict(params)
            fold_params["random_state"] = seed
            model = LGBMRegressor(**fold_params)
            model.fit(
                X_train.iloc[train_fold],
                y_train.iloc[train_fold],
                eval_set=[(X_train.iloc[valid_fold], y_train.iloc[valid_fold])],
                eval_metric="rmse",
                callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=0)],
            )
            pred = model.predict(X_train.iloc[valid_fold], num_iteration=model.best_iteration_)
            metrics = evaluate_predictions(y_train.iloc[valid_fold], pred)
            rows.append(
                {
                    "feature_set": feature_set,
                    "target": target,
                    "seed": seed,
                    "fold": fold_idx,
                    **metrics,
                    "best_iteration": float(model.best_iteration_ or params["n_estimators"]),
                }
            )
    return pd.DataFrame(rows)


def save_feature_importance(
    feature_set: str,
    target: str,
    model: LGBMRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    gain_importance = model.booster_.feature_importance(importance_type="gain")
    split_importance = model.booster_.feature_importance(importance_type="split")
    perm = permutation_importance(
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
            "permutation_mean": perm.importances_mean,
            "permutation_std": perm.importances_std,
        }
    ).sort_values("permutation_mean", ascending=False)
    importance.to_csv(OUTPUT_DIR / f"feature_importance_{feature_set}_{target}_lightgbm.csv", index=False)


def save_predicted_vs_actual_plot(
    feature_set: str,
    target: str,
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
    plt.title(f"{feature_set}: Predicted vs Actual {target}")
    plt.text(
        0.03,
        0.97,
        f"R2 = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nMAE = {metrics['mae']:.3f}\nr = {metrics['pearson_r']:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"predicted_vs_actual_{feature_set}_{target}.png", dpi=200)
    plt.close()


def save_residual_histogram(feature_set: str, target: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    residuals = y_pred - y_true.to_numpy()
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=28, color="#bfdbfe", edgecolor="#1d4ed8")
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Residual (predicted - actual)")
    plt.ylabel("Count")
    plt.title(f"{feature_set}: Residual Histogram for {target}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"residual_histogram_{feature_set}_{target}.png", dpi=200)
    plt.close()


def save_test_predictions(
    test_df: pd.DataFrame,
    feature_set: str,
    target: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> None:
    out = test_df[["system:index", "region", "dw_label_name", "latitude", "longitude", "spatial_block"]].copy()
    out["feature_set"] = feature_set
    out["target"] = target
    out["actual"] = y_true.to_numpy()
    out["predicted"] = y_pred
    out["residual"] = out["predicted"] - out["actual"]
    out.to_csv(OUTPUT_DIR / f"test_predictions_{feature_set}_{target}.csv", index=False)


def subgroup_metrics(
    test_df: pd.DataFrame,
    feature_set: str,
    target: str,
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
    region_rows: List[Dict[str, Any]] = []
    land_use_rows: List[Dict[str, Any]] = []
    for region, part in frame.groupby("region"):
        region_rows.append(
            {
                "feature_set": feature_set,
                "target": target,
                "region": region,
                **evaluate_predictions(part["y_true"], part["y_pred"]),
                "n_test": int(len(part)),
            }
        )
    for label, part in frame.groupby("dw_label_name"):
        land_use_rows.append(
            {
                "feature_set": feature_set,
                "target": target,
                "dw_label_name": label,
                **evaluate_predictions(part["y_true"], part["y_pred"]),
                "n_test": int(len(part)),
            }
        )
    return pd.DataFrame(region_rows), pd.DataFrame(land_use_rows)


def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    feature_variances = df[EMBEDDING_COLS].var()
    corr = df[EMBEDDING_COLS].corr().abs()
    audit = {
        "dataset_path": str(DATA_PATH),
        "n_rows": int(len(df)),
        "n_embedding_features": len(EMBEDDING_COLS),
        "n_regions": int(df["region"].nunique()),
        "n_land_use_labels": int(df["dw_label"].nunique()),
        "spatial_blocks": int(df["spatial_block"].nunique()),
        "missing_ratio": float(df[EMBEDDING_COLS + DW_PROB_COLS + TARGET_COLS].isna().mean().mean()),
        "duplicate_rows_embedding_region_label": int(df.duplicated(subset=EMBEDDING_COLS + ["region", "dw_label"]).sum()),
        "near_constant_embedding_dims": int((feature_variances < 1e-4).sum()),
        "high_corr_feature_pairs_abs_gt_0_95": int(np.triu((corr > 0.95).to_numpy(), k=1).sum()),
        "targets": {},
    }
    for target in TARGET_COLS:
        series = df[target]
        audit["targets"][target] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "skew": float(series.skew()),
        }
    (OUTPUT_DIR / "dataset_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    ensure_output_dirs()
    full_df = assign_spatial_blocks(load_dataset())
    audit_dataset(full_df)

    feature_frames = build_feature_frames(full_df)
    train_idx, test_idx = train_test_split(
        full_df.index,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=full_df["region_dw_label"],
    )
    train_df = full_df.loc[train_idx].reset_index(drop=True)
    test_df = full_df.loc[test_idx].reset_index(drop=True)
    strat_labels = train_df["region_dw_label"]
    groups = train_df["spatial_block"]

    metrics_rows: List[Dict[str, Any]] = []
    stability_frames: List[pd.DataFrame] = []
    regional_frames: List[pd.DataFrame] = []
    land_use_frames: List[pd.DataFrame] = []

    experiment_start = time.time()

    for feature_set, X_full in feature_frames.items():
        X_train = X_full.loc[train_idx].reset_index(drop=True)
        X_test = X_full.loc[test_idx].reset_index(drop=True)

        for target in TARGET_COLS:
            y_train = train_df[target]
            y_test = test_df[target]

            model, best_params, cv_summary = tune_lightgbm(feature_set, target, X_train, y_train, strat_labels, groups)
            stability_df = run_stability_check(feature_set, target, best_params, X_train, y_train, strat_labels, groups)
            stability_frames.append(stability_df)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_metrics = evaluate_predictions(y_train, y_train_pred)
            test_metrics = evaluate_predictions(y_test, y_test_pred)
            stability_summary = stability_df.agg({"r2": ["mean", "std"], "rmse": ["mean", "std"], "mae": ["mean", "std"]})

            metrics_rows.append(
                {
                    "feature_set": feature_set,
                    "target": target,
                    "train_r2": train_metrics["r2"],
                    "train_rmse": train_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "cv_r2_mean": cv_summary["cv_r2_mean"],
                    "cv_r2_std": cv_summary["cv_r2_std"],
                    "cv_rmse_mean": cv_summary["cv_rmse_mean"],
                    "cv_rmse_std": cv_summary["cv_rmse_std"],
                    "cv_mae_mean": cv_summary["cv_mae_mean"],
                    "cv_mae_std": cv_summary["cv_mae_std"],
                    "stability_r2_mean": float(stability_summary.loc["mean", "r2"]),
                    "stability_r2_std": float(stability_summary.loc["std", "r2"]),
                    "stability_rmse_mean": float(stability_summary.loc["mean", "rmse"]),
                    "stability_rmse_std": float(stability_summary.loc["std", "rmse"]),
                    "best_params": json.dumps(best_params, sort_keys=True),
                    "r2": test_metrics["r2"],
                    "rmse": test_metrics["rmse"],
                    "mae": test_metrics["mae"],
                    "pearson_r": test_metrics["pearson_r"],
                }
            )

            save_feature_importance(feature_set, target, model, X_test, y_test)
            save_predicted_vs_actual_plot(feature_set, target, y_test, y_test_pred, test_metrics)
            save_residual_histogram(feature_set, target, y_test, y_test_pred)
            save_test_predictions(test_df, feature_set, target, y_test, y_test_pred)

            region_df, land_use_df = subgroup_metrics(test_df, feature_set, target, y_test, y_test_pred)
            regional_frames.append(region_df)
            land_use_frames.append(land_use_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target", "r2"], ascending=[True, False])
    stability_df = pd.concat(stability_frames, ignore_index=True).sort_values(["target", "feature_set", "seed", "fold"])
    regional_df = pd.concat(regional_frames, ignore_index=True).sort_values(["target", "feature_set", "region"])
    land_use_df = pd.concat(land_use_frames, ignore_index=True).sort_values(["target", "feature_set", "dw_label_name"])

    metrics_df.to_csv(OUTPUT_DIR / "full_dataset_lightgbm_metrics.csv", index=False)
    stability_df.to_csv(OUTPUT_DIR / "full_dataset_lightgbm_stability.csv", index=False)
    regional_df.to_csv(OUTPUT_DIR / "full_dataset_lightgbm_regional_metrics.csv", index=False)
    land_use_df.to_csv(OUTPUT_DIR / "full_dataset_lightgbm_land_use_metrics.csv", index=False)
    (OUTPUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "dataset_path": str(DATA_PATH),
                "rows_used": int(len(full_df)),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "feature_sets": list(feature_frames.keys()),
                "targets": TARGET_COLS,
                "spatial_bins_per_axis": SPATIAL_BINS_PER_AXIS,
                "optuna_trials": OPTUNA_TRIALS,
                "stability_seeds": STABILITY_SEEDS,
                "runtime_seconds": time.time() - experiment_start,
            },
            indent=2,
        )
    )

    print("Saved metrics to:", OUTPUT_DIR / "full_dataset_lightgbm_metrics.csv")
    print("Saved stability results to:", OUTPUT_DIR / "full_dataset_lightgbm_stability.csv")
    print("Saved regional metrics to:", OUTPUT_DIR / "full_dataset_lightgbm_regional_metrics.csv")
    print("Saved land-use metrics to:", OUTPUT_DIR / "full_dataset_lightgbm_land_use_metrics.csv")


if __name__ == "__main__":
    main()
