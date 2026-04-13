#!/usr/bin/env python3
"""
Build a side-by-side figure comparing Sentinel-2 natural color with AlphaEarth
PCA RGB for the same AOI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.decomposition import PCA


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "DataSources" / "single_image_sar_reconstruction"
DEFAULT_S2_PATH = DATA_DIR / "sentinel2_natural_color_sf_downtown_golden_gate_2024.tif"
DEFAULT_TRUTH_PATH = DATA_DIR / "sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif"
DEFAULT_OUT_PATH = (
    ROOT_DIR
    / "outputs"
    / "single_image_sar_reconstruction_sf_downtown_golden_gate"
    / "sentinel2_vs_alphaearth_pca_rgb.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentinel2-path", type=Path, default=DEFAULT_S2_PATH)
    parser.add_argument("--truth-path", type=Path, default=DEFAULT_TRUTH_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.sentinel2_path) as s2_src:
        s2 = s2_src.read().astype(np.float32)
    s2 = np.moveaxis(s2, 0, -1)
    s2 = (s2 - 0.02) / (0.30 - 0.02)
    s2 = np.clip(s2, 0, 1)
    s2 = np.power(s2, 1 / 1.15)

    with rasterio.open(args.truth_path) as truth_src:
        truth = truth_src.read(list(range(4, 68))).astype(np.float32)

    mask = np.all(np.isfinite(truth), axis=0)
    h, w = mask.shape
    truth_px = truth[:, mask].T

    pca = PCA(n_components=3, random_state=42)
    rgb_px = pca.fit_transform(truth_px)
    lo = np.quantile(rgb_px, 0.02, axis=0)
    hi = np.quantile(rgb_px, 0.98, axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    rgb_px = np.clip((rgb_px - lo) / scale, 0, 1)

    alpha_rgb = np.zeros((h, w, 3), dtype=np.float32)
    alpha_rgb[mask] = rgb_px

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].imshow(s2)
    axes[0].set_title("Sentinel-2 Natural Color", fontsize=18)
    axes[1].imshow(alpha_rgb)
    axes[1].set_title("AlphaEarth PCA RGB", fontsize=18)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(args.output_path, dpi=240, bbox_inches="tight")
    print(args.output_path)


if __name__ == "__main__":
    main()
