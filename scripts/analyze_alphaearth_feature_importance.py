#!/usr/bin/env python3
"""Rank AlphaEarth embedding dimensions for SAR prediction.

The analysis uses two complementary signals:

1. SHAP mean absolute contribution from the fitted LightGBM models.
2. Absolute Pearson correlation between each embedding dimension and each SAR target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import EMBEDDING_BANDS, RANDOM_STATE, SAR_BANDS
from src.modeling import build_model


DEFAULT_DATASET = Path("outputs/single_image_sar_reconstruction_sf_downtown_golden_gate/sampled_alphaearth_to_sar_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/feature_importance_alphaearth_to_sar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AlphaEarth feature importance for SAR prediction.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-shap-rows", type=int, default=1000)
    parser.add_argument("--top-n-plot", type=int, default=20)
    return parser.parse_args()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(y_true: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for target_index, target in enumerate(SAR_BANDS):
        truth = y_true[target].to_numpy()
        pred = y_pred[:, target_index]
        rows.append(
            {
                "target": target,
                "r2": r2_score(truth, pred),
                "rmse": _rmse(truth, pred),
                "mae": mean_absolute_error(truth, pred),
                "pearson_r_observed_vs_predicted": pearsonr(truth, pred).statistic,
            }
        )
    return pd.DataFrame(rows)


def compute_pearson_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in EMBEDDING_BANDS:
        for target in SAR_BANDS:
            result = pearsonr(data[feature], data[target])
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "pearson_r": result.statistic,
                    "pearson_abs_r": abs(result.statistic),
                    "pearson_p_value": result.pvalue,
                }
            )
    return pd.DataFrame(rows)


def compute_shap_table(model, X_explain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_index, target in enumerate(SAR_BANDS):
        estimator = model.estimators_[target_index]
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_explain)
        shap_values = np.asarray(shap_values)
        mean_abs = np.abs(shap_values).mean(axis=0)
        signed_mean = shap_values.mean(axis=0)
        for feature_index, feature in enumerate(EMBEDDING_BANDS):
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "mean_abs_shap": mean_abs[feature_index],
                    "mean_signed_shap": signed_mean[feature_index],
                }
            )
    shap_table = pd.DataFrame(rows)
    shap_table["shap_share_within_target"] = shap_table["mean_abs_shap"] / shap_table.groupby("target")[
        "mean_abs_shap"
    ].transform("sum")
    return shap_table


def build_rankings(shap_table: pd.DataFrame, pearson_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_target = shap_table.merge(pearson_table, on=["feature", "target"], how="inner")
    per_target["shap_rank_within_target"] = per_target.groupby("target")["mean_abs_shap"].rank(
        ascending=False, method="min"
    )
    per_target["pearson_rank_within_target"] = per_target.groupby("target")["pearson_abs_r"].rank(
        ascending=False, method="min"
    )
    per_target["combined_rank_within_target"] = (
        per_target["shap_rank_within_target"] + per_target["pearson_rank_within_target"]
    ) / 2.0
    per_target = per_target.sort_values(["target", "combined_rank_within_target", "shap_rank_within_target"])

    overall = (
        per_target.groupby("feature", as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            mean_shap_share=("shap_share_within_target", "mean"),
            mean_abs_pearson_r=("pearson_abs_r", "mean"),
            max_abs_pearson_r=("pearson_abs_r", "max"),
            best_single_target_shap=("mean_abs_shap", "max"),
            best_single_target_pearson_abs_r=("pearson_abs_r", "max"),
        )
        .reset_index(drop=True)
    )
    overall["shap_rank"] = overall["mean_shap_share"].rank(ascending=False, method="min")
    overall["pearson_rank"] = overall["mean_abs_pearson_r"].rank(ascending=False, method="min")
    overall["combined_rank_score"] = (overall["shap_rank"] + overall["pearson_rank"]) / 2.0
    overall = overall.sort_values(["combined_rank_score", "shap_rank", "pearson_rank"]).reset_index(drop=True)
    overall.insert(0, "overall_rank", np.arange(1, len(overall) + 1))
    return overall, per_target


def plot_overall(overall: pd.DataFrame, output_path: Path, top_n: int) -> None:
    plot_data = overall.head(top_n).iloc[::-1]
    fig, ax1 = plt.subplots(figsize=(10, 8))
    y = np.arange(len(plot_data))
    ax1.barh(y - 0.18, plot_data["mean_shap_share"], height=0.36, color="#276fbf", label="Mean SHAP share")
    ax1.set_xlabel("Mean SHAP share across SAR targets")
    ax1.set_yticks(y)
    ax1.set_yticklabels(plot_data["feature"])
    ax2 = ax1.twiny()
    ax2.barh(y + 0.18, plot_data["mean_abs_pearson_r"], height=0.36, color="#c44536", label="Mean |Pearson r|")
    ax2.set_xlabel("Mean absolute Pearson r")
    ax1.set_title(f"Top {top_n} AlphaEarth embedding dimensions for SAR prediction")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_target_heatmap(per_target: pd.DataFrame, output_path: Path, metric: str, title: str) -> None:
    matrix = per_target.pivot(index="feature", columns="target", values=metric).loc[EMBEDDING_BANDS, SAR_BANDS]
    fig, ax = plt.subplots(figsize=(7, 13))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(SAR_BANDS)))
    ax.set_xticklabels(SAR_BANDS, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(EMBEDDING_BANDS)))
    ax.set_yticklabels(EMBEDDING_BANDS)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def markdown_table(data: pd.DataFrame, columns: list[str], rows: int) -> str:
    table = data.loc[:, columns].head(rows).copy()
    headers = [str(column) for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.5f}")
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    return markdown_table(data, list(data.columns), len(data))


def write_report(
    output_path: Path,
    dataset_path: Path,
    model_metrics: pd.DataFrame,
    overall: pd.DataFrame,
    per_target: pd.DataFrame,
    sample_count: int,
    train_count: int,
    test_count: int,
    top_three_shap_share: float,
) -> None:
    top_by_target = []
    for target in SAR_BANDS:
        target_rows = per_target[per_target["target"] == target].sort_values(
            ["combined_rank_within_target", "shap_rank_within_target"]
        )
        top_by_target.append(f"### {target}\n\n" + markdown_table(
            target_rows,
            [
                "feature",
                "mean_abs_shap",
                "shap_share_within_target",
                "pearson_r",
                "pearson_abs_r",
                "combined_rank_within_target",
            ],
            10,
        ))

    report = f"""# AlphaEarth Embedding Feature Importance for SAR Prediction

## Purpose

This report ranks the 64 Google AlphaEarth embedding dimensions (`A00`-`A63`) by how predictive they are for Sentinel-1 SAR values in the saved San Francisco downtown / Golden Gate sampled dataset.

## Data and Method

- Dataset: `{dataset_path}`
- Sampled pixels: {sample_count:,}
- Train / held-out split from existing `split` column: {train_count:,} / {test_count:,}
- Model: same LightGBM `MultiOutputRegressor` configuration used by the reconstruction pipeline.
- SHAP signal: mean absolute TreeSHAP value on held-out pixels, computed separately for `S1_VV`, `S1_VH`, and `S1_VV_div_VH`.
- Pearson signal: absolute Pearson correlation coefficient between each embedding dimension and each SAR target over all sampled pixels.
- Overall rank: average of the SHAP rank and Pearson rank, where SHAP uses mean within-target contribution share and Pearson uses mean absolute correlation across targets.

## Held-Out Model Check

{dataframe_to_markdown(model_metrics)}

## Overall Feature Ranking

`A27`, `A63`, and `A25` together account for {top_three_shap_share:.1%} of average SHAP importance across the three SAR targets.

{markdown_table(overall, [
    "overall_rank",
    "feature",
    "mean_shap_share",
    "mean_abs_pearson_r",
    "max_abs_pearson_r",
    "shap_rank",
    "pearson_rank",
    "combined_rank_score",
], 64)}

## Top Features by SAR Target

{chr(10).join(top_by_target)}

## Interpretation Notes

- Features near the top are consistently important by both nonlinear model attribution and linear association.
- A high SHAP rank with a weaker Pearson rank indicates nonlinear or interaction-driven predictive value.
- A high Pearson rank with weaker SHAP rank indicates a strong marginal relationship that the fitted model may partly replace with correlated embedding dimensions.
- These rankings are scene-specific: they describe this single exported AOI and time period, not a global AlphaEarth feature ordering.

## Generated Artifacts

- `overall_feature_ranking.csv`: complete 64-feature ranking.
- `feature_importance_by_target.csv`: per-target SHAP and Pearson table.
- `pearson_correlations_by_target.csv`: raw Pearson coefficients and p-values.
- `shap_importance_by_target.csv`: raw SHAP contribution summaries.
- `top20_overall_feature_importance.png`: top-feature comparison plot.
- `shap_share_heatmap.png`: SHAP contribution heatmap by target.
- `pearson_abs_heatmap.png`: absolute Pearson heatmap by target.
"""
    output_path.write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.dataset)
    missing = sorted(set(EMBEDDING_BANDS + SAR_BANDS + ["split"]) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    train = data[data["split"] == "train"].copy()
    test = data[data["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Dataset must contain non-empty train and test splits.")

    X_train = train[EMBEDDING_BANDS]
    y_train = train[SAR_BANDS]
    X_test = test[EMBEDDING_BANDS]
    y_test = test[SAR_BANDS]

    model = build_model(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    model_metrics = evaluate_model(y_test, y_pred)
    model_metrics.to_csv(args.output_dir / "heldout_metrics_for_feature_importance_model.csv", index=False)

    if len(X_test) > args.max_shap_rows:
        X_explain = X_test.sample(args.max_shap_rows, random_state=RANDOM_STATE)
    else:
        X_explain = X_test

    pearson_table = compute_pearson_table(data)
    shap_table = compute_shap_table(model, X_explain)
    overall, per_target = build_rankings(shap_table, pearson_table)

    pearson_table.to_csv(args.output_dir / "pearson_correlations_by_target.csv", index=False)
    shap_table.to_csv(args.output_dir / "shap_importance_by_target.csv", index=False)
    per_target.to_csv(args.output_dir / "feature_importance_by_target.csv", index=False)
    overall.to_csv(args.output_dir / "overall_feature_ranking.csv", index=False)

    plot_overall(overall, args.output_dir / "top20_overall_feature_importance.png", args.top_n_plot)
    plot_target_heatmap(
        per_target,
        args.output_dir / "shap_share_heatmap.png",
        "shap_share_within_target",
        "AlphaEarth SHAP contribution share by SAR target",
    )
    plot_target_heatmap(
        per_target,
        args.output_dir / "pearson_abs_heatmap.png",
        "pearson_abs_r",
        "Absolute Pearson correlation by SAR target",
    )

    write_report(
        args.output_dir / "alphaearth_feature_importance_report.md",
        args.dataset,
        model_metrics,
        overall,
        per_target,
        sample_count=len(data),
        train_count=len(train),
        test_count=len(test),
        top_three_shap_share=float(overall[overall["feature"].isin(["A27", "A63", "A25"])]["mean_shap_share"].sum()),
    )

    metadata = {
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "sample_count": int(len(data)),
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_shap_rows": args.max_shap_rows,
        "top_overall_features": overall.head(10)["feature"].tolist(),
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Wrote feature-importance report to {args.output_dir / 'alphaearth_feature_importance_report.md'}")
    print(overall.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
