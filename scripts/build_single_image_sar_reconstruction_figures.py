#!/usr/bin/env python3
"""Build the kept figure set for the single-image SAR reconstruction report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.decomposition import PCA


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "single_image_sar_reconstruction_sf_downtown_golden_gate"
DEFAULT_PRED_PATH = (
    DEFAULT_OUTPUT_DIR
    / "predicted_tiles"
    / "predicted_alphaearth_from_sar_small_stack_sf_downtown_golden_gate_2024.tif"
)
DEFAULT_TRUTH_PATH = (
    ROOT_DIR
    / "DataSources"
    / "single_image_sar_reconstruction"
    / "sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif"
)
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "full_image_metrics_by_band.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", type=Path, default=DEFAULT_PRED_PATH)
    parser.add_argument("--truth-path", type=Path, default=DEFAULT_TRUTH_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _band_limits(arr: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(arr)
    vals = arr[mask]
    if vals.size == 0:
        return -1.0, 1.0
    return float(np.quantile(vals, 0.02)), float(np.quantile(vals, 0.98))


def build_residual_summary(metrics_df: pd.DataFrame, out_path: Path) -> None:
    df = metrics_df.sort_values("r2", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)

    axes[0].bar(x, df["r2"], color="#2563eb")
    axes[0].set_title("Full-Image R^2 By Embedding Band")
    axes[0].set_ylabel("R^2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["band"], rotation=90, fontsize=8)

    axes[1].bar(x, df["rmse"], color="#dc2626", label="RMSE")
    axes[1].plot(x, df["mae"], color="#111827", linewidth=1.5, label="MAE")
    axes[1].set_title("Residual Magnitude By Embedding Band")
    axes[1].set_ylabel("Error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["band"], rotation=90, fontsize=8)
    axes[1].legend()

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_all_band_large_views(
    pred_path: Path,
    truth_path: Path,
    output_dir: Path,
) -> None:
    with rasterio.open(pred_path) as pred_src, rasterio.open(truth_path) as truth_src:
        pred = pred_src.read().astype(np.float32)
        truth = truth_src.read(list(range(4, 68))).astype(np.float32)

    mask = np.all(np.isfinite(pred), axis=0) & np.all(np.isfinite(truth), axis=0)
    h, w = mask.shape
    pred_px = pred[:, mask].T
    truth_px = truth[:, mask].T

    pca = PCA(n_components=3, random_state=42)
    truth_rgb_px = pca.fit_transform(truth_px)
    pred_rgb_px = pca.transform(pred_px)
    all_rgb = np.vstack([truth_rgb_px, pred_rgb_px])
    lo = np.quantile(all_rgb, 0.02, axis=0)
    hi = np.quantile(all_rgb, 0.98, axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    truth_rgb_px = np.clip((truth_rgb_px - lo) / scale, 0, 1)
    pred_rgb_px = np.clip((pred_rgb_px - lo) / scale, 0, 1)

    truth_rgb = np.zeros((h, w, 3), dtype=np.float32)
    pred_rgb = np.zeros((h, w, 3), dtype=np.float32)
    truth_rgb[mask] = truth_rgb_px
    pred_rgb[mask] = pred_rgb_px

    rmse_map = np.full((h, w), np.nan, dtype=np.float32)
    rmse_map[mask] = np.sqrt(np.mean((truth_px - pred_px) ** 2, axis=1))
    rmse_lim = float(np.quantile(rmse_map[np.isfinite(rmse_map)], 0.98))

    fig_specs = [
        ("true_embedding_all_band_large.png", truth_rgb, "True Embedding Image (PCA RGB from all 64 bands)", None),
        (
            "reproduced_embedding_all_band_large.png",
            pred_rgb,
            "Reproduced Image (PCA RGB from all 64 bands)",
            None,
        ),
        (
            "residual_heatmap_all_band_large.png",
            rmse_map,
            "Residual Heatmap: Per-pixel RMSE across 64 bands",
            ("magma", 0, rmse_lim),
        ),
    ]

    for name, arr, title, heat in fig_specs:
        fig, ax = plt.subplots(figsize=(14, 7.2), constrained_layout=True)
        if heat is None:
            im = ax.imshow(arr)
        else:
            cmap, vmin, vmax = heat
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.85, label="RMSE across 64 bands")
        ax.set_title(title, fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(output_dir / name, dpi=260, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    residual_summary_path = args.output_dir / "residual_summary_by_band.png"

    metrics_df = pd.read_csv(args.metrics_path)
    build_all_band_large_views(args.pred_path, args.truth_path, args.output_dir)
    build_residual_summary(metrics_df, residual_summary_path)
    print(residual_summary_path)


if __name__ == "__main__":
    main()
