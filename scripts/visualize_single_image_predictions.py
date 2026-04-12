#!/usr/bin/env python3
"""
Visualize single-image prediction results on top of the exported Sentinel-2 GeoTIFF.

Inputs:
- RGB GeoTIFF exported from Earth Engine
- Predictions CSV written by run_single_image_pixel_fraction_experiment.py

For one chosen training fraction, this script maps each held-out point back into
image coordinates and writes a figure with:
1. RGB image
2. true target values at sampled test pixels
3. predicted target values at sampled test pixels
4. residuals at sampled test pixels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rio_transform


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "single_image_pixel_fraction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif-path", type=Path, required=True, help="Path to the Sentinel-2 RGB GeoTIFF.")
    parser.add_argument(
        "--predictions-path",
        type=Path,
        required=True,
        help="Path to the *_test_predictions.csv output.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Fraction value to visualize, for example 0.1.",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Optional target column name for the title. If omitted, infer from predictions metadata when possible.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=18.0,
        help="Scatter marker size in pixels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures.",
    )
    return parser.parse_args()


def ensure_lon_lat(df: pd.DataFrame) -> pd.DataFrame:
    if "longitude" in df.columns and "latitude" in df.columns:
        return df
    if ".geo" not in df.columns:
        raise ValueError("Predictions CSV must contain longitude/latitude or .geo columns.")

    def parse_geojson_point(value: str) -> tuple[float, float]:
        obj = json.loads(value)
        coords = obj["coordinates"]
        return float(coords[0]), float(coords[1])

    coords = df[".geo"].apply(parse_geojson_point)
    out = df.copy()
    out["longitude"] = coords.apply(lambda xy: xy[0])
    out["latitude"] = coords.apply(lambda xy: xy[1])
    return out


def load_rgb(tif_path: Path) -> tuple[np.ndarray, rasterio.Affine, object]:
    with rasterio.open(tif_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
        transform = src.transform
        crs = src.crs

    rgb = np.moveaxis(rgb, 0, -1)
    rgb = np.clip(rgb, 0, 1)
    return rgb, transform, crs


def attach_image_coords(
    df: pd.DataFrame,
    transform: rasterio.Affine,
    raster_crs: object,
    image_shape: tuple[int, int, int],
) -> pd.DataFrame:
    height, width = image_shape[0], image_shape[1]
    out = df.copy()
    xs = out["longitude"].astype(float).to_list()
    ys = out["latitude"].astype(float).to_list()

    if raster_crs and str(raster_crs).upper() != "EPSG:4326":
        xs, ys = rio_transform("EPSG:4326", raster_crs, xs, ys)

    rows = []
    cols = []
    keep_mask = []
    inverse = ~transform

    for x_coord, y_coord in zip(xs, ys):
        col_f, row_f = inverse * (float(x_coord), float(y_coord))
        row = int(np.floor(row_f))
        col = int(np.floor(col_f))
        inside = 0 <= row < height and 0 <= col < width
        rows.append(row)
        cols.append(col)
        keep_mask.append(inside)

    out["row"] = rows
    out["col"] = cols
    out = out[np.asarray(keep_mask, dtype=bool)].reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions_path)
    predictions = ensure_lon_lat(predictions)

    fraction_df = predictions[np.isclose(predictions["fraction"], args.fraction)].copy()
    if fraction_df.empty:
        available = sorted(predictions["fraction"].unique().tolist())
        raise ValueError(f"No rows found for fraction={args.fraction}. Available fractions: {available}")

    rgb, transform, raster_crs = load_rgb(args.tif_path)
    fraction_df = attach_image_coords(fraction_df, transform, raster_crs, rgb.shape)

    if fraction_df.empty:
        raise ValueError("No prediction points fell within the TIFF extent after coordinate transform.")

    target_label = args.target_col or "pixel_value"
    true_vals = fraction_df["y_true"].to_numpy()
    pred_vals = fraction_df["y_pred"].to_numpy()
    resid_vals = fraction_df["residual"].to_numpy()
    rows = fraction_df["row"].to_numpy()
    cols = fraction_df["col"].to_numpy()

    vmin = float(min(true_vals.min(), pred_vals.min()))
    vmax = float(max(true_vals.max(), pred_vals.max()))
    resid_abs = float(np.max(np.abs(resid_vals)))

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=True)

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")

    axes[1].imshow(rgb, alpha=0.35)
    sc_true = axes[1].scatter(cols, rows, c=true_vals, s=args.point_size, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"True {target_label}")

    axes[2].imshow(rgb, alpha=0.35)
    sc_pred = axes[2].scatter(cols, rows, c=pred_vals, s=args.point_size, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Predicted {target_label}")

    axes[3].imshow(rgb, alpha=0.35)
    sc_resid = axes[3].scatter(
        cols,
        rows,
        c=resid_vals,
        s=args.point_size,
        cmap="coolwarm",
        vmin=-resid_abs,
        vmax=resid_abs,
    )
    axes[3].set_title("Residual")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_true = fig.colorbar(sc_true, ax=[axes[1], axes[2]], shrink=0.8)
    cbar_true.set_label(target_label)
    cbar_resid = fig.colorbar(sc_resid, ax=axes[3], shrink=0.8)
    cbar_resid.set_label("y_true - y_pred")

    fig.suptitle(
        f"Single-image AlphaEarth prediction overlay | fraction={args.fraction} | n={len(fraction_df)} test points",
        fontsize=13,
    )

    stem = f"{args.predictions_path.stem}_fraction_{str(args.fraction).replace('.', 'p')}"
    out_path = args.output_dir / f"{stem}_overlay.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figure: {out_path}")


if __name__ == "__main__":
    main()
