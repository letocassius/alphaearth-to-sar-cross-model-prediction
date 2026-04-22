#!/usr/bin/env python3
"""Build the kept figure set for the single-image AlphaEarth-to-SAR report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "single_image_sar_reconstruction_sf_downtown_golden_gate"
DEFAULT_PRED_PATH = (
    DEFAULT_OUTPUT_DIR
    / "sar_hat_from_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif"
)
DEFAULT_TRUTH_PATH = (
    ROOT_DIR
    / "DataSources"
    / "single_image_sar_reconstruction"
    / "sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif"
)
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "gap_fill_metrics_by_band.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", type=Path, default=DEFAULT_PRED_PATH)
    parser.add_argument("--truth-path", type=Path, default=DEFAULT_TRUTH_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_channels(truth: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth_rgb = np.moveaxis(truth, 0, -1).astype(np.float32)
    pred_rgb = np.moveaxis(pred, 0, -1).astype(np.float32)
    out_truth = np.zeros_like(truth_rgb, dtype=np.float32)
    out_pred = np.zeros_like(pred_rgb, dtype=np.float32)

    for band_idx in range(truth_rgb.shape[-1]):
        combined = np.concatenate(
            [
                truth_rgb[..., band_idx][np.isfinite(truth_rgb[..., band_idx])],
                pred_rgb[..., band_idx][np.isfinite(pred_rgb[..., band_idx])],
            ]
        )
        if combined.size == 0:
            lo, hi = -1.0, 1.0
        else:
            lo, hi = np.quantile(combined, [0.02, 0.98])
        scale = hi - lo if hi > lo else 1.0
        out_truth[..., band_idx] = np.clip((truth_rgb[..., band_idx] - lo) / scale, 0, 1)
        out_pred[..., band_idx] = np.clip((pred_rgb[..., band_idx] - lo) / scale, 0, 1)

    return out_truth, out_pred


def build_residual_summary(metrics_df: pd.DataFrame, out_path: Path) -> None:
    df = metrics_df.sort_values("r2", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)

    axes[0].bar(x, df["r2"], color="#2563eb")
    axes[0].set_title("Gap-Fill R^2 By SAR Band")
    axes[0].set_ylabel("R^2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["band"], fontsize=10)

    axes[1].bar(x, df["rmse"], color="#dc2626", label="RMSE")
    axes[1].plot(x, df["mae"], color="#111827", linewidth=1.5, marker="o", label="MAE")
    axes[1].set_title("Gap-Fill Residual Magnitude By SAR Band")
    axes[1].set_ylabel("Error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["band"], fontsize=10)
    axes[1].legend()

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_sar_large_views(
    pred_path: Path,
    truth_path: Path,
    output_dir: Path,
) -> None:
    with rasterio.open(pred_path) as pred_src, rasterio.open(truth_path) as truth_src:
        pred = pred_src.read([1, 2, 3]).astype(np.float32)
        truth = truth_src.read([1, 2, 3]).astype(np.float32)

    mask = np.all(np.isfinite(pred), axis=0) & np.all(np.isfinite(truth), axis=0)
    truth_rgb, pred_rgb = normalize_channels(truth, pred)
    truth_rgb[~mask] = 0
    pred_rgb[~mask] = 0

    rmse_map = np.full(mask.shape, np.nan, dtype=np.float32)
    residuals = truth[:, mask].T - pred[:, mask].T
    rmse_map[mask] = np.sqrt(np.mean(residuals**2, axis=1))
    rmse_lim = float(np.quantile(rmse_map[np.isfinite(rmse_map)], 0.98))

    fig_specs = [
        ("true_sar_all_band_large.png", truth_rgb, "Actual Sentinel-1 SAR (VV, VH, VV - VH as RGB)", None),
        ("sarhat_all_band_large.png", pred_rgb, "SARhat From AlphaEarth Embeddings", None),
        (
            "residual_heatmap_sar_large.png",
            rmse_map,
            "Residual Heatmap: Per-pixel RMSE across SAR bands",
            ("magma", 0, rmse_lim),
        ),
    ]

    for name, arr, title, heat in fig_specs:
        fig, ax = plt.subplots(figsize=(14, 7.2), constrained_layout=True)
        if heat is None:
            ax.imshow(arr)
        else:
            cmap, vmin, vmax = heat
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.85, label="RMSE across SAR bands")
        ax.set_title(title, fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(output_dir / name, dpi=260, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    residual_summary_path = args.output_dir / "residual_summary_by_sar_band.png"

    metrics_df = pd.read_csv(args.metrics_path)
    build_sar_large_views(args.pred_path, args.truth_path, args.output_dir)
    build_residual_summary(metrics_df, residual_summary_path)
    print(residual_summary_path)


if __name__ == "__main__":
    main()
