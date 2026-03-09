#!/usr/bin/env python3
"""
Build a PDF report summarizing Phase 2 model performance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


OUTPUT_DIR = Path("phase2_outputs")
REPORT_PATH = OUTPUT_DIR / "phase2_model_performance_report.pdf"
METRICS_PATH = OUTPUT_DIR / "regression_metrics.csv"
REGIONAL_PATH = OUTPUT_DIR / "regional_metrics_S1_VV.csv"
SUBSET_PATH = Path("DataSources/alphaearth_s1_dw_samples_balanced_subset_2024.csv")
FULL_DATA_PATH = Path("DataSources/alphaearth_s1_dw_samples_all_regions_2024.csv")
TARGETS = ["S1_VV", "S1_VH", "S1_VV_div_VH"]
MODEL_LABELS = {
    "ridge": "Ridge regression",
    "histgradientboosting": "Histogram gradient boosting",
}


def draw_text_page(title: str, lines: list[str], pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
    ax.axis("off")

    ax.text(0.0, 0.98, title, fontsize=22, fontweight="bold", va="top")
    y = 0.90
    for line in lines:
        if line == "":
            y -= 0.03
            continue
        ax.text(0.0, y, line, fontsize=11, va="top", family="monospace" if line.startswith("  ") else None)
        y -= 0.035

    pdf.savefig(fig)
    plt.close(fig)


def draw_dataframe_page(title: str, df: pd.DataFrame, pdf: PdfPages, footnote: str | None = None) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.82])
    ax.axis("off")

    ax.text(0.0, 1.04, title, fontsize=20, fontweight="bold", va="bottom", transform=ax.transAxes)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.04, 1, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dbeafe")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")

    if footnote:
        ax.text(0.0, 0.0, footnote, fontsize=9, color="#444444", transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close(fig)


def draw_image_grid_page(title: str, image_paths: list[Path], pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.subplots_adjust(top=0.90, left=0.05, right=0.95, bottom=0.06, hspace=0.12, wspace=0.08)
    fig.suptitle(title, fontsize=20, fontweight="bold")

    flat_axes = axes.flatten()
    for ax, image_path in zip(flat_axes, image_paths):
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(image_path.stem.replace("_", " "), fontsize=10)
        ax.axis("off")

    for ax in flat_axes[len(image_paths):]:
        ax.axis("off")

    pdf.savefig(fig)
    plt.close(fig)


def format_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    table = metrics_df.copy()
    table["model"] = table["model"].map(MODEL_LABELS)
    table["best_cv_r2"] = table["best_cv_r2"].map(lambda x: f"{x:.3f}")
    table["r2"] = table["r2"].map(lambda x: f"{x:.3f}")
    table["rmse"] = table["rmse"].map(lambda x: f"{x:.3f}")
    table["mae"] = table["mae"].map(lambda x: f"{x:.3f}")
    table["pearson_r"] = table["pearson_r"].map(lambda x: f"{x:.3f}")
    table["best_params"] = table["best_params"].map(lambda s: s if len(s) <= 56 else s[:53] + "...")
    return table[["target", "model", "best_cv_r2", "r2", "rmse", "mae", "pearson_r", "best_params"]]


def format_regional_table(regional_df: pd.DataFrame) -> pd.DataFrame:
    table = regional_df.copy()
    table["model"] = table["model"].map(MODEL_LABELS)
    for col in ["r2", "rmse", "mae", "pearson_r"]:
        table[col] = table[col].map(lambda x: f"{x:.3f}")
    return table[["region", "model", "r2", "rmse", "mae", "pearson_r", "n_test"]]


def build_feature_lines() -> list[str]:
    lines = ["Top 5 embedding dimensions by target and model", ""]
    for target in TARGETS:
        ridge = pd.read_csv(OUTPUT_DIR / f"feature_importance_{target}_ridge.csv").head(5)
        hgbr = pd.read_csv(OUTPUT_DIR / f"feature_importance_{target}_histgradientboosting.csv").head(5)

        lines.append(f"{target}")
        lines.append("  Ridge: " + ", ".join(f"{row.feature} ({row.abs_coefficient:.3f})" for row in ridge.itertuples()))
        lines.append(
            "  HistGB: "
            + ", ".join(f"{row.feature} ({row.importance_mean:.3f})" for row in hgbr.itertuples())
        )
        lines.append("")
    return lines


def main() -> None:
    metrics_df = pd.read_csv(METRICS_PATH).sort_values(["target", "r2"], ascending=[True, False]).reset_index(drop=True)
    regional_df = pd.read_csv(REGIONAL_PATH).sort_values(["region", "model"]).reset_index(drop=True)
    subset_df = pd.read_csv(SUBSET_PATH)
    full_df = pd.read_csv(FULL_DATA_PATH)

    best_rows = metrics_df.loc[metrics_df.groupby("target")["r2"].idxmax()].sort_values("target")
    summary_lines = [
        "AlphaEarth to SAR Cross-Modal Prediction",
        "",
        f"Full dataset size: {len(full_df)} rows",
        f"Balanced modeling subset: {len(subset_df)} rows",
        f"Regions: {', '.join(sorted(subset_df['region'].unique()))}",
        f"Dynamic World classes: {subset_df['dw_label'].nunique()}",
        "",
        "Held-out test split: 70/30 stratified by region x dw_label",
        "Models compared: ridge regression vs histogram gradient boosting",
        "",
        "Best held-out model by target:",
    ]
    for row in best_rows.itertuples():
        summary_lines.append(
            f"- {row.target}: {MODEL_LABELS[row.model]} with R2={row.r2:.3f}, RMSE={row.rmse:.3f}, MAE={row.mae:.3f}, r={row.pearson_r:.3f}"
        )
    summary_lines.extend(
        [
            "",
            "Key interpretation:",
            "- Ridge regression outperformed the boosted trees on all three SAR targets in this 720-row balanced subset.",
            "- S1_VH was the easiest target to predict; the VV/VH ratio remained the hardest.",
            "- For S1_VV, amazon_forest had the strongest regional fit and sf_bay_urban had the weakest for both models.",
        ]
    )

    with PdfPages(REPORT_PATH) as pdf:
        draw_text_page("Phase 2 Model Performance Report", summary_lines, pdf)
        draw_dataframe_page(
            "Overall Held-out Performance",
            format_metrics_table(metrics_df),
            pdf,
            footnote="Metrics are computed on the 30 percent held-out test set. best_cv_r2 is the mean 3-fold CV score on the training split.",
        )
        draw_dataframe_page(
            "Regional S1_VV Performance",
            format_regional_table(regional_df),
            pdf,
            footnote="Each region contributes 54 held-out samples in the balanced test split.",
        )
        draw_text_page("Feature Importance Summary", build_feature_lines(), pdf)
        draw_image_grid_page(
            "Predicted vs Actual: Ridge Regression",
            [OUTPUT_DIR / f"predicted_vs_actual_{target}_ridge.png" for target in TARGETS],
            pdf,
        )
        draw_image_grid_page(
            "Predicted vs Actual: Histogram Gradient Boosting",
            [OUTPUT_DIR / f"predicted_vs_actual_{target}_histgradientboosting.png" for target in TARGETS],
            pdf,
        )

    print(f"Saved PDF report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
