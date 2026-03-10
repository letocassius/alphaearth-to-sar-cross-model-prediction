#!/usr/bin/env python3
"""
Build a consolidated PDF summary across the subset Phase 2 and full-dataset experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PHASE2_METRICS = Path("phase2_outputs/regression_metrics.csv")
FULL_METRICS = Path("phase2_full_dataset_outputs/full_dataset_lightgbm_metrics.csv")
PHASE2_LAND_USE = Path("phase2_outputs/land_use_metrics.csv")
FULL_LAND_USE = Path("phase2_full_dataset_outputs/full_dataset_lightgbm_land_use_metrics.csv")
REPORT_PATH = Path("alphaearth_to_sar_consolidated_summary_report.pdf")


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
    table.scale(1, 1.25)
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


def main() -> None:
    phase2 = pd.read_csv(PHASE2_METRICS).copy()
    full = pd.read_csv(FULL_METRICS).copy()
    phase2_land = pd.read_csv(PHASE2_LAND_USE)
    full_land = pd.read_csv(FULL_LAND_USE)

    phase2["experiment"] = "720-row subset"
    phase2["variant"] = phase2["model"].replace({"ridge": "ridge", "lightgbm": "tuned_lightgbm"})
    phase2_summary = phase2[["experiment", "target", "variant", "r2", "rmse", "mae", "pearson_r"]].copy()

    full["experiment"] = "2,880-row full dataset"
    full["variant"] = full["feature_set"]
    full_summary = full[["experiment", "target", "variant", "r2", "rmse", "mae", "pearson_r"]].copy()

    comparison = pd.concat([phase2_summary, full_summary], ignore_index=True)
    for col in ["r2", "rmse", "mae", "pearson_r"]:
        comparison[col] = comparison[col].map(lambda x: f"{x:.3f}")

    phase2_vv = phase2[phase2["target"] == "S1_VV"].sort_values("r2", ascending=False).iloc[0]
    phase2_vh = phase2[phase2["target"] == "S1_VH"].sort_values("r2", ascending=False).iloc[0]
    phase2_ratio = phase2[phase2["target"] == "S1_VV_div_VH"].sort_values("r2", ascending=False).iloc[0]
    full_vv = full[full["target"] == "S1_VV"].sort_values("r2", ascending=False).iloc[0]
    full_vh = full[full["target"] == "S1_VH"].sort_values("r2", ascending=False).iloc[0]
    full_ratio = full[full["target"] == "S1_VV_div_VH"].sort_values("r2", ascending=False).iloc[0]

    summary_lines = [
        "AlphaEarth to SAR consolidated summary",
        "",
        "Experiments covered:",
        "- Phase 2 subset benchmark: 720 balanced rows, ridge vs tuned LightGBM",
        "- Full-dataset ablation: 2,880 rows, embeddings only vs embeddings plus context",
        "",
        "Best result by target in each experiment:",
        f"- Subset S1_VV: {phase2_vv['variant']} with R2={phase2_vv['r2']:.3f}",
        f"- Subset S1_VH: {phase2_vh['variant']} with R2={phase2_vh['r2']:.3f}",
        f"- Subset VV/VH ratio: {phase2_ratio['variant']} with R2={phase2_ratio['r2']:.3f}",
        f"- Full S1_VV: {full_vv['variant']} with R2={full_vv['r2']:.3f}",
        f"- Full S1_VH: {full_vh['variant']} with R2={full_vh['r2']:.3f}",
        f"- Full VV/VH ratio: {full_ratio['variant']} with R2={full_ratio['r2']:.3f}",
        "",
        "Main conclusion:",
        "- The AlphaEarth embeddings carry most of the predictive signal for S1_VV and S1_VH.",
        "- Nonlinear LightGBM helps most on the harder VV/VH ratio target, but it does not displace the simpler baseline on the main backscatter targets.",
        "- Adding region and Dynamic World context on the full dataset produces little or no overall gain, so the embeddings themselves appear to be doing most of the work.",
    ]

    land_use_compare = pd.concat(
        [
            phase2_land.assign(experiment="720-row subset", variant=lambda df: df["model"]),
            full_land.assign(experiment="2,880-row full dataset", variant=lambda df: df["feature_set"]),
        ],
        ignore_index=True,
    )
    land_use_vv = land_use_compare[land_use_compare["target"] == "S1_VV"][
        ["experiment", "variant", "dw_label_name", "r2", "rmse", "mae"]
    ].copy()
    for col in ["r2", "rmse", "mae"]:
        land_use_vv[col] = land_use_vv[col].map(lambda x: f"{x:.3f}")

    with PdfPages(REPORT_PATH) as pdf:
        draw_text_page("Consolidated Summary Report", summary_lines, pdf)
        draw_dataframe_page(
            "Model Comparison Across Experiments",
            comparison.reset_index(drop=True),
            pdf,
            footnote="This table summarizes the held-out metrics from the subset benchmark and the full-dataset ablation.",
        )
        draw_dataframe_page(
            "S1_VV Land-use Diagnostics Across Experiments",
            land_use_vv.reset_index(drop=True),
            pdf,
            footnote="Land-use breakdowns help show where nonlinear models or added context matter even when overall R2 changes only slightly.",
        )

    print(f"Saved consolidated report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
