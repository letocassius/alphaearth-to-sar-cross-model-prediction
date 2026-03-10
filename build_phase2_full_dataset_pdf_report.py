#!/usr/bin/env python3
"""
Build a PDF report for the full-dataset LightGBM ablation experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


OUTPUT_DIR = Path("phase2_full_dataset_outputs")
REPORT_PATH = OUTPUT_DIR / "phase2_full_dataset_lightgbm_report.pdf"
TOP_LEVEL_COPY = Path("phase2_full_dataset_lightgbm_report.pdf")
METRICS_PATH = OUTPUT_DIR / "full_dataset_lightgbm_metrics.csv"
STABILITY_PATH = OUTPUT_DIR / "full_dataset_lightgbm_stability.csv"
REGIONAL_PATH = OUTPUT_DIR / "full_dataset_lightgbm_regional_metrics.csv"
LAND_USE_PATH = OUTPUT_DIR / "full_dataset_lightgbm_land_use_metrics.csv"
TARGETS = ["S1_VV", "S1_VH", "S1_VV_div_VH"]
FEATURE_LABELS = {
    "embedding_only": "Embeddings only",
    "embedding_plus_context": "Embeddings + region + Dynamic World",
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
    table["feature_set"] = table["feature_set"].map(FEATURE_LABELS)
    for col in ["cv_r2_mean", "cv_rmse_mean", "stability_r2_mean", "stability_rmse_mean", "r2", "rmse", "mae", "pearson_r"]:
        table[col] = table[col].map(lambda x: f"{x:.3f}")
    return table[["target", "feature_set", "cv_r2_mean", "cv_rmse_mean", "stability_r2_mean", "stability_rmse_mean", "r2", "rmse", "mae", "pearson_r"]]


def build_feature_lines(targets: list[str]) -> list[str]:
    lines = ["Top 5 features by permutation importance", ""]
    for feature_set in FEATURE_LABELS:
        lines.append(FEATURE_LABELS[feature_set])
        for target in targets:
            importance = pd.read_csv(OUTPUT_DIR / f"feature_importance_{feature_set}_{target}_lightgbm.csv").head(5)
            lines.append(
                f"  {target}: " + ", ".join(f"{row.feature} ({row.permutation_mean:.3f})" for row in importance.itertuples())
            )
        lines.append("")
    return lines


def main() -> None:
    metrics_df = pd.read_csv(METRICS_PATH).sort_values(["target", "r2"], ascending=[True, False]).reset_index(drop=True)
    stability_df = pd.read_csv(STABILITY_PATH)
    regional_df = pd.read_csv(REGIONAL_PATH)
    land_use_df = pd.read_csv(LAND_USE_PATH)

    best_rows = metrics_df.loc[metrics_df.groupby("target")["r2"].idxmax()].sort_values("target")
    summary_lines = [
        "Full-dataset LightGBM ablation",
        "",
        "Dataset: all 2,880 rows",
        "Validation: 70/30 stratified holdout + repeated grouped CV on training split",
        "Feature sets compared: embeddings only vs embeddings plus region and Dynamic World context",
        "",
        "Best held-out feature set by target:",
    ]
    for row in best_rows.itertuples():
        summary_lines.append(
            f"- {row.target}: {FEATURE_LABELS[row.feature_set]} with R2={row.r2:.3f}, RMSE={row.rmse:.3f}, MAE={row.mae:.3f}, r={row.pearson_r:.3f}"
        )

    vv = metrics_df[metrics_df["target"] == "S1_VV"].set_index("feature_set")
    vh = metrics_df[metrics_df["target"] == "S1_VH"].set_index("feature_set")
    ratio = metrics_df[metrics_df["target"] == "S1_VV_div_VH"].set_index("feature_set")
    summary_lines.extend(
        [
            "",
            "Interpretation:",
            f"- S1_VV: {FEATURE_LABELS[vv['r2'].idxmax()]} leads ({vv['r2'].max():.3f} vs {vv['r2'].min():.3f}).",
            f"- S1_VH: {FEATURE_LABELS[vh['r2'].idxmax()]} leads ({vh['r2'].max():.3f} vs {vh['r2'].min():.3f}).",
            f"- VV/VH ratio: {FEATURE_LABELS[ratio['r2'].idxmax()]} leads ({ratio['r2'].max():.3f} vs {ratio['r2'].min():.3f}).",
            "- Stability metrics summarize repeated grouped-CV over three random seeds.",
        ]
    )

    stability_summary = (
        stability_df.groupby(["target", "feature_set"])[["r2", "rmse", "mae"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    stability_summary.columns = [
        "target",
        "feature_set",
        "r2_mean",
        "r2_std",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
    ]
    stability_summary["feature_set"] = stability_summary["feature_set"].map(FEATURE_LABELS)
    for col in ["r2_mean", "r2_std", "rmse_mean", "rmse_std", "mae_mean", "mae_std"]:
        stability_summary[col] = stability_summary[col].map(lambda x: f"{x:.3f}")

    regional_vv = regional_df[regional_df["target"] == "S1_VV"].copy()
    regional_vv["feature_set"] = regional_vv["feature_set"].map(FEATURE_LABELS)
    for col in ["r2", "rmse", "mae", "pearson_r"]:
        regional_vv[col] = regional_vv[col].map(lambda x: f"{x:.3f}")

    land_use_vv = land_use_df[land_use_df["target"] == "S1_VV"].copy()
    land_use_vv["feature_set"] = land_use_vv["feature_set"].map(FEATURE_LABELS)
    for col in ["r2", "rmse", "mae", "pearson_r"]:
        land_use_vv[col] = land_use_vv[col].map(lambda x: f"{x:.3f}")

    with PdfPages(REPORT_PATH) as pdf:
        draw_text_page("Full-dataset LightGBM Report", summary_lines, pdf)
        draw_dataframe_page(
            "Held-out Performance by Feature Set",
            format_metrics_table(metrics_df),
            pdf,
            footnote="The holdout test set is shared across the two feature-set ablations.",
        )
        draw_dataframe_page(
            "Repeated Grouped-CV Stability",
            stability_summary,
            pdf,
            footnote="Each row summarizes 12 grouped validation folds: 4 folds across 3 random seeds.",
        )
        draw_dataframe_page(
            "Regional S1_VV Diagnostics",
            regional_vv[["region", "feature_set", "r2", "rmse", "mae", "pearson_r", "n_test"]].reset_index(drop=True),
            pdf,
            footnote="Region-level diagnostics on the held-out test set.",
        )
        draw_dataframe_page(
            "Land-Use S1_VV Diagnostics",
            land_use_vv[["dw_label_name", "feature_set", "r2", "rmse", "mae", "pearson_r", "n_test"]].reset_index(drop=True),
            pdf,
            footnote="Land-use diagnostics on the held-out test set.",
        )
        draw_text_page("Feature Importance Summary", build_feature_lines(TARGETS), pdf)
        draw_image_grid_page(
            "Predicted vs Actual: Embeddings Only",
            [OUTPUT_DIR / f"predicted_vs_actual_embedding_only_{target}.png" for target in TARGETS],
            pdf,
        )
        draw_image_grid_page(
            "Predicted vs Actual: Embeddings + Context",
            [OUTPUT_DIR / f"predicted_vs_actual_embedding_plus_context_{target}.png" for target in TARGETS],
            pdf,
        )
        draw_image_grid_page(
            "Residual Histograms: Embeddings + Context",
            [OUTPUT_DIR / f"residual_histogram_embedding_plus_context_{target}.png" for target in TARGETS],
            pdf,
        )

    TOP_LEVEL_COPY.write_bytes(REPORT_PATH.read_bytes())
    print(f"Saved PDF report to: {REPORT_PATH}")
    print(f"Copied PDF report to: {TOP_LEVEL_COPY}")


if __name__ == "__main__":
    main()
