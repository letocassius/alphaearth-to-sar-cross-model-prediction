#!/usr/bin/env python3
"""
Build a PDF summary report for the single-image pixel-fraction experiment.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs" / "single_image_pixel_fraction"
REPORTS_DIR = ROOT_DIR / "reports"

METRICS_PATH = OUTPUT_DIR / "sentinel2_alphaearth_pixel_pairs_2024_n5000_B4_fraction_metrics.csv"
METADATA_PATH = OUTPUT_DIR / "sentinel2_alphaearth_pixel_pairs_2024_n5000_B4_run_metadata.json"
OVERLAY_PATH = OUTPUT_DIR / "sentinel2_alphaearth_pixel_pairs_2024_n5000_B4_test_predictions_fraction_0p1_overlay.png"
REPORT_PATH = OUTPUT_DIR / "single_image_pixel_fraction_report.pdf"
REPORT_COPY_PATH = REPORTS_DIR / "single_image_pixel_fraction_report.pdf"


def draw_text_page(title: str, lines: list[str], pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
    ax.axis("off")
    ax.text(0.0, 0.98, title, fontsize=22, fontweight="bold", va="top")
    y = 0.91
    for line in lines:
        ax.text(0.0, y, line, fontsize=11.5, va="top")
        y -= 0.045 if line else 0.028
    pdf.savefig(fig)
    plt.close(fig)


def draw_dataframe_page(title: str, df: pd.DataFrame, pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.84])
    ax.axis("off")
    ax.text(0.0, 1.03, title, fontsize=20, fontweight="bold", va="bottom", transform=ax.transAxes)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.05, 0.98, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dbeafe")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
        cell.PAD = 0.02
    pdf.savefig(fig)
    plt.close(fig)


def draw_image_page(title: str, image_path: Path, pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis("off")
    ax.text(0.0, 1.02, title, fontsize=20, fontweight="bold", va="bottom", transform=ax.transAxes)
    if image_path.exists():
        ax.imshow(mpimg.imread(image_path))
    else:
        ax.text(0.5, 0.5, f"Image unavailable\n{image_path.name}", ha="center", va="center", fontsize=15)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")

    REPORTS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    metrics_df = pd.read_csv(METRICS_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        metadata = pd.read_json(METADATA_PATH, typ="series").to_dict()

    best_row = metrics_df.sort_values("r2", ascending=False).iloc[0]
    summary_lines = [
        "Single-image AlphaEarth-to-pixel-value feasibility experiment.",
        "",
        f"Input CSV: {metadata.get('csv_path', METRICS_PATH.name)}",
        f"Model: {metadata.get('model', 'StandardScaler + Ridge(alpha=1.0)')}",
        f"Target band: {metadata.get('target_col', 'B4')}",
        f"Train rows: {metadata.get('train_rows_total', 'unknown')}",
        f"Test rows: {metadata.get('test_rows_total', 'unknown')}",
        "",
        "Best result on this split:",
        f"Fraction = {best_row['fraction']:.2f}",
        f"R^2 = {best_row['r2']:.3f}",
        f"RMSE = {best_row['rmse']:.4f}",
        f"MAE = {best_row['mae']:.4f}",
        f"Pearson r = {best_row['pearson_r']:.3f}",
        "",
        "Observed pattern:",
        "Performance improves rapidly from 1% to 25% of training rows,",
        "then largely plateaus. On this split, 50% gives the best R^2.",
    ]

    metrics_table = metrics_df.copy()
    metrics_table["fraction"] = metrics_table["fraction"].map(lambda v: f"{v:.2f}")
    metrics_table["r2"] = metrics_table["r2"].map(lambda v: f"{v:.3f}")
    metrics_table["rmse"] = metrics_table["rmse"].map(lambda v: f"{v:.4f}")
    metrics_table["mae"] = metrics_table["mae"].map(lambda v: f"{v:.4f}")
    metrics_table["pearson_r"] = metrics_table["pearson_r"].map(lambda v: f"{v:.3f}")

    with PdfPages(REPORT_PATH) as pdf:
        draw_text_page("Single-Image Pixel Fraction Report", summary_lines, pdf)
        draw_dataframe_page("Fraction Sweep Metrics", metrics_table, pdf)
        draw_image_page("Held-Out Overlay At Fraction 0.10", OVERLAY_PATH, pdf)

    shutil.copyfile(REPORT_PATH, REPORT_COPY_PATH)
    print(f"Wrote PDF report: {REPORT_PATH}")
    print(f"Copied PDF report to: {REPORT_COPY_PATH}")


if __name__ == "__main__":
    main()
