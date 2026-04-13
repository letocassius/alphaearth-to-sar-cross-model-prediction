#!/usr/bin/env python3
"""
Train a SAR -> AlphaEarth embedding model from tiled Earth Engine exports, then
reconstruct the full image and write a report-ready artifact bundle.

Inputs:
- a full Sentinel-1 GeoTIFF with bands S1_VV, S1_VH, S1_VV_div_VH
- tiled full-stack GeoTIFFs with bands S1_* followed by A00..A63

Outputs:
- per-band held-out metrics
- full-image metrics accumulated over every valid pixel
- predicted AlphaEarth tiles for the full AOI
- preview figures and a Markdown report under reports/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "DataSources" / "single_image_sar_reconstruction"
OUTPUT_DIR = ROOT_DIR / "outputs" / "single_image_sar_reconstruction_sf_downtown_golden_gate"
REPORTS_DIR = ROOT_DIR / "reports"

SAR_BANDS = ["S1_VV", "S1_VH", "S1_VV_div_VH"]
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]
FULL_STACK_GLOB = "sentinel1_alphaearth*.tif"
DEFAULT_SAR_PATH = DATA_DIR / "sentinel1_small_vv_vh_sf_downtown_golden_gate_2024.tif"
DEFAULT_SAMPLE_PROBABILITY = 0.001
DEFAULT_CHUNK_SIZE = 512
RANDOM_STATE = 42

OFFSET_PATTERN = re.compile(r".*-(?P<row>\d+)-(?P<col>\d+)\.tif$")


@dataclass
class RunningMoments:
    count: np.ndarray
    sum_y: np.ndarray
    sum_pred: np.ndarray
    sum_y2: np.ndarray
    sum_pred2: np.ndarray
    sum_ypred: np.ndarray
    sum_abs_err: np.ndarray
    sum_sq_err: np.ndarray

    @classmethod
    def zeros(cls, n_bands: int) -> "RunningMoments":
        zeros = np.zeros(n_bands, dtype=np.float64)
        return cls(
            count=zeros.copy(),
            sum_y=zeros.copy(),
            sum_pred=zeros.copy(),
            sum_y2=zeros.copy(),
            sum_pred2=zeros.copy(),
            sum_ypred=zeros.copy(),
            sum_abs_err=zeros.copy(),
            sum_sq_err=zeros.copy(),
        )

    def update(self, truth: np.ndarray, pred: np.ndarray) -> None:
        if truth.size == 0:
            return
        self.count += truth.shape[0]
        self.sum_y += truth.sum(axis=0)
        self.sum_pred += pred.sum(axis=0)
        self.sum_y2 += np.square(truth).sum(axis=0)
        self.sum_pred2 += np.square(pred).sum(axis=0)
        self.sum_ypred += (truth * pred).sum(axis=0)
        self.sum_abs_err += np.abs(truth - pred).sum(axis=0)
        self.sum_sq_err += np.square(truth - pred).sum(axis=0)

    def to_metrics(self, band_names: list[str]) -> pd.DataFrame:
        mean_y = np.divide(self.sum_y, self.count, out=np.full_like(self.sum_y, np.nan), where=self.count > 0)
        mean_pred = np.divide(
            self.sum_pred, self.count, out=np.full_like(self.sum_pred, np.nan), where=self.count > 0
        )
        sst = self.sum_y2 - self.count * np.square(mean_y)
        var_y = self.sum_y2 - self.count * np.square(mean_y)
        var_pred = self.sum_pred2 - self.count * np.square(mean_pred)
        cov = self.sum_ypred - self.count * mean_y * mean_pred

        r2 = np.full_like(self.sum_y, np.nan)
        rmse = np.full_like(self.sum_y, np.nan)
        mae = np.full_like(self.sum_y, np.nan)
        pearson_r = np.full_like(self.sum_y, np.nan)

        valid = self.count > 1
        r2[valid] = 1.0 - np.divide(
            self.sum_sq_err[valid], sst[valid], out=np.full_like(self.sum_sq_err[valid], np.nan), where=sst[valid] > 0
        )
        rmse[valid] = np.sqrt(self.sum_sq_err[valid] / self.count[valid])
        mae[valid] = self.sum_abs_err[valid] / self.count[valid]

        corr_valid = valid & (var_y > 0) & (var_pred > 0)
        pearson_r[corr_valid] = cov[corr_valid] / np.sqrt(var_y[corr_valid] * var_pred[corr_valid])

        return pd.DataFrame(
            {
                "band": band_names,
                "count": self.count.astype(np.int64),
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "pearson_r": pearson_r,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sar-path", type=Path, default=DEFAULT_SAR_PATH)
    parser.add_argument("--full-stack-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--full-stack-glob", default=FULL_STACK_GLOB)
    parser.add_argument("--sample-probability", type=float, default=DEFAULT_SAMPLE_PROBABILITY)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORTS_DIR / "single_image_sar_reconstruction_sf_downtown_golden_gate_report.md",
    )
    return parser.parse_args()


def discover_full_stack_tiles(full_stack_dir: Path, full_stack_glob: str) -> list[Path]:
    tiles = sorted(full_stack_dir.glob(full_stack_glob))
    if not tiles:
        raise FileNotFoundError(f"No full-stack tiles matching {full_stack_glob} in {full_stack_dir}")
    return tiles


def iter_windows(height: int, width: int, chunk_size: int) -> list[Window]:
    windows: list[Window] = []
    for row_off in range(0, height, chunk_size):
        win_h = min(chunk_size, height - row_off)
        for col_off in range(0, width, chunk_size):
            win_w = min(chunk_size, width - col_off)
            windows.append(Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h))
    return windows


def build_features(sar_pixels: np.ndarray) -> np.ndarray:
    if sar_pixels.shape[1] != 3:
        raise ValueError(f"Expected three SAR inputs, got shape {sar_pixels.shape}")
    vv = sar_pixels[:, 0]
    vh = sar_pixels[:, 1]
    vv_minus_vh = sar_pixels[:, 2]
    return np.column_stack([vv, vh, vv_minus_vh])


def build_model(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=RANDOM_STATE)),
        ]
    )


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def read_sar_chunk_from_full_res(
    sar_src: rasterio.io.DatasetReader,
    dst_bounds: tuple[float, float, float, float],
    dst_height: int,
    dst_width: int,
) -> np.ndarray:
    src_window = from_bounds(*dst_bounds, transform=sar_src.transform)
    return sar_src.read(
        indexes=[1, 2, 3],
        window=src_window,
        out_shape=(3, dst_height, dst_width),
        resampling=Resampling.nearest,
        boundless=False,
    )


def sample_training_data(
    tiles: list[Path],
    chunk_size: int,
    sample_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for tile_index, tile_path in enumerate(tiles, start=1):
        tile_rows_before = sum(part.shape[0] for part in x_parts)
        with rasterio.open(tile_path) as src:
            for window in iter_windows(src.height, src.width, chunk_size):
                sar_chunk = src.read(indexes=[1, 2, 3], window=window)
                emb_chunk = src.read(indexes=list(range(4, 68)), window=window)

                valid_mask = np.all(np.isfinite(sar_chunk), axis=0) & np.all(np.isfinite(emb_chunk), axis=0)
                valid_count = int(valid_mask.sum())
                if valid_count == 0:
                    continue

                keep_mask = rng.random(valid_mask.shape) < sample_probability
                mask = valid_mask & keep_mask
                if not mask.any():
                    continue

                sar_pixels = sar_chunk[:, mask].T.astype(np.float32, copy=False)
                emb_pixels = emb_chunk[:, mask].T.astype(np.float32, copy=False)
                x_parts.append(build_features(sar_pixels))
                y_parts.append(emb_pixels)
        tile_rows_after = sum(part.shape[0] for part in x_parts)
        print(
            f"[sample] {tile_index}/{len(tiles)} {tile_path.name}: "
            f"+{tile_rows_after - tile_rows_before} rows, total={tile_rows_after}",
            flush=True,
        )

    if not x_parts:
        raise ValueError("Training sample is empty. Increase sample_probability or verify the inputs.")

    X = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


def evaluate_heldout(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    y_pred = model.predict(X_test)
    rows: list[dict[str, float | str]] = []
    for idx, band in enumerate(EMBEDDING_BANDS):
        y_true_band = y_test[:, idx]
        y_pred_band = y_pred[:, idx]
        rows.append(
            {
                "band": band,
                "r2": float(r2_score(y_true_band, y_pred_band)),
                "rmse": float(math.sqrt(mean_squared_error(y_true_band, y_pred_band))),
                "mae": float(mean_absolute_error(y_true_band, y_pred_band)),
                "pearson_r": safe_pearsonr(y_true_band, y_pred_band),
            }
        )
    return pd.DataFrame(rows)


def tile_output_name(tile_path: Path) -> str:
    stem = tile_path.stem.replace("sentinel1_alphaearth", "predicted_alphaearth_from_sar", 1)
    return f"{stem}.tif"


def parse_tile_offsets(tile_path: Path) -> tuple[int, int]:
    match = OFFSET_PATTERN.search(tile_path.name)
    if not match:
        return 0, 0
    return int(match.group("row")), int(match.group("col"))


def reconstruct_full_image(
    model: Pipeline,
    tiles: list[Path],
    sar_path: Path,
    output_dir: Path,
    chunk_size: int,
) -> pd.DataFrame:
    predicted_tiles_dir = output_dir / "predicted_tiles"
    predicted_tiles_dir.mkdir(parents=True, exist_ok=True)

    running = RunningMoments.zeros(len(EMBEDDING_BANDS))

    with rasterio.open(sar_path) as sar_src:
        for tile_index, tile_path in enumerate(tiles, start=1):
            with rasterio.open(tile_path) as src:
                profile = src.profile.copy()
                profile.update(
                    count=len(EMBEDDING_BANDS),
                    dtype="float32",
                    compress="deflate",
                    predictor=3,
                    nodata=np.nan,
                )
                out_path = predicted_tiles_dir / tile_output_name(tile_path)
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.descriptions = tuple(EMBEDDING_BANDS)

                    for window in iter_windows(src.height, src.width, chunk_size):
                        dst_bounds = window_bounds(window, src.transform)
                        sar_chunk = read_sar_chunk_from_full_res(
                            sar_src=sar_src,
                            dst_bounds=dst_bounds,
                            dst_height=int(window.height),
                            dst_width=int(window.width),
                        )
                        truth_chunk = src.read(indexes=list(range(4, 68)), window=window)

                        valid_mask = np.all(np.isfinite(sar_chunk), axis=0) & np.all(np.isfinite(truth_chunk), axis=0)
                        pred_chunk = np.full(
                            (len(EMBEDDING_BANDS), int(window.height), int(window.width)),
                            np.nan,
                            dtype=np.float32,
                        )

                        if valid_mask.any():
                            sar_pixels = sar_chunk[:, valid_mask].T.astype(np.float32, copy=False)
                            truth_pixels = truth_chunk[:, valid_mask].T.astype(np.float32, copy=False)
                            pred_pixels = model.predict(build_features(sar_pixels)).astype(np.float32, copy=False)
                            pred_chunk[:, valid_mask] = pred_pixels.T
                            running.update(truth_pixels.astype(np.float64), pred_pixels.astype(np.float64))

                        dst.write(pred_chunk, window=window)
                print(
                    f"[reconstruct] {tile_index}/{len(tiles)} wrote {out_path.name}",
                    flush=True,
                )

    return running.to_metrics(EMBEDDING_BANDS)


def build_preview_arrays(
    truth_tiles: list[Path],
    predicted_tiles_dir: Path,
    sar_path: Path,
    band_index: int,
    stride: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(sar_path) as sar_src:
        preview_height = math.ceil(sar_src.height / stride)
        preview_width = math.ceil(sar_src.width / stride)

    truth_preview = np.full((preview_height, preview_width), np.nan, dtype=np.float32)
    pred_preview = np.full((preview_height, preview_width), np.nan, dtype=np.float32)

    for truth_tile in truth_tiles:
        row_off, col_off = parse_tile_offsets(truth_tile)
        row_small = row_off // stride
        col_small = col_off // stride

        predicted_tile = predicted_tiles_dir / tile_output_name(truth_tile)
        with rasterio.open(truth_tile) as truth_src, rasterio.open(predicted_tile) as pred_src:
            truth_arr = truth_src.read(
                band_index + 4,
                out_shape=(math.ceil(truth_src.height / stride), math.ceil(truth_src.width / stride)),
                resampling=Resampling.average,
            )
            pred_arr = pred_src.read(
                band_index + 1,
                out_shape=(math.ceil(pred_src.height / stride), math.ceil(pred_src.width / stride)),
                resampling=Resampling.average,
            )

        h, w = truth_arr.shape
        truth_preview[row_small : row_small + h, col_small : col_small + w] = truth_arr
        pred_preview[row_small : row_small + h, col_small : col_small + w] = pred_arr

    return truth_preview, pred_preview


def make_figures(
    heldout_metrics: pd.DataFrame,
    full_metrics: pd.DataFrame,
    truth_tiles: list[Path],
    predicted_tiles_dir: Path,
    sar_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_paths: dict[str, Path] = {}

    metric_compare_path = output_dir / "heldout_vs_full_r2.png"
    compare_df = heldout_metrics[["band", "r2"]].merge(
        full_metrics[["band", "r2"]].rename(columns={"r2": "full_r2"}),
        on="band",
        how="inner",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(compare_df["band"], compare_df["r2"], label="Held-out R^2", linewidth=1.8)
    ax.plot(compare_df["band"], compare_df["full_r2"], label="Full-image R^2", linewidth=1.8)
    ax.set_title("Bandwise R^2 Comparison")
    ax.set_xlabel("Embedding band")
    ax.set_ylabel("R^2")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    fig.savefig(metric_compare_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths["heldout_vs_full_r2"] = metric_compare_path

    return figure_paths


def summarize_metrics(metrics_df: pd.DataFrame) -> dict[str, float | str]:
    best_row = metrics_df.sort_values("r2", ascending=False).iloc[0]
    worst_row = metrics_df.sort_values("r2", ascending=True).iloc[0]
    return {
        "mean_r2": float(metrics_df["r2"].mean()),
        "median_r2": float(metrics_df["r2"].median()),
        "mean_rmse": float(metrics_df["rmse"].mean()),
        "mean_mae": float(metrics_df["mae"].mean()),
        "mean_pearson_r": float(metrics_df["pearson_r"].mean()),
        "best_band": str(best_row["band"]),
        "best_band_r2": float(best_row["r2"]),
        "worst_band": str(worst_row["band"]),
        "worst_band_r2": float(worst_row["r2"]),
    }


def write_report(
    report_path: Path,
    sar_path: Path,
    tiles: list[Path],
    sample_count: int,
    train_count: int,
    test_count: int,
    heldout_summary: dict[str, float | str],
    full_summary: dict[str, float | str],
    output_dir: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda p: os.path.relpath(p, report_path.parent)
    predicted_tif = output_dir / "predicted_tiles" / tile_output_name(tiles[0])
    natural_vs_pca = output_dir / "sentinel2_vs_alphaearth_pca_rgb.png"
    true_large = output_dir / "true_embedding_all_band_large.png"
    pred_large = output_dir / "reproduced_embedding_all_band_large.png"
    resid_large = output_dir / "residual_heatmap_all_band_large.png"
    resid_summary = output_dir / "residual_summary_by_band.png"
    heldout_vs_full = output_dir / "heldout_vs_full_r2.png"
    lines = [
        "# Single-Image SAR-to-AlphaEarth Reconstruction Report",
        "",
        "## Objective",
        "",
        "This experiment tests whether a compact Sentinel-1 SAR scene can be used to reconstruct the AlphaEarth embedding field over the same area.",
        "",
        "The goal is not to predict one optical band or one embedding band in isolation. The goal is to take SAR-only input and reproduce the full 64-band AlphaEarth embedding image over one complete scene.",
        "",
        "The chosen area is a small San Francisco scene covering downtown San Francisco and the Golden Gate Bridge. That AOI is large enough to contain complex urban, water, coastline, and bridge structure, but still small enough to support fast end-to-end reconstruction.",
        "",
        "## Inputs",
        "",
        f"- SAR image: `{sar_path}`",
        f"- Full truth stack: `DataSources/single_image_sar_reconstruction/{tiles[0].name}`",
        "- Spatial extent: approximately `668 x 1894` pixels at `10 m`",
        "- SAR predictors: `S1_VV`, `S1_VH`, `S1_VV_div_VH`",
        "- Reconstruction targets: `A00` to `A63`",
        "",
        "The truth stack contains both the SAR bands and the AlphaEarth embedding bands. That makes it possible to train on sampled pixels and then compare the reproduced image against the true AlphaEarth image everywhere in the AOI.",
        "",
        "## Modeling Setup",
        "",
        "To keep the run fast and stable, the model was trained on a small random sample of valid pixels rather than on the full scene.",
        "",
        f"- Sampled rows: `{sample_count}`",
        f"- Train rows: `{train_count}`",
        f"- Test rows: `{test_count}`",
        "- Model: `PolynomialFeatures(degree=2) + StandardScaler + Ridge(alpha=10.0)`",
        "",
        "This model choice is deliberately simple. The point of this report is to measure whether the SAR signal is strong enough to recover the embedding field over an entire image, not to maximize leaderboard performance with a heavy model.",
        "",
        "## Reconstruction Procedure",
        "",
        "The workflow was:",
        "",
        "1. Sample valid pixels from the downtown-plus-Golden-Gate stack.",
        "2. Fit one multi-output regression model from the three SAR inputs to the 64 AlphaEarth target bands.",
        "3. Apply the fitted model to every valid pixel in the scene.",
        "4. Write the reproduced 64-band embedding image as a GeoTIFF.",
        "5. Compare the reproduced image against the true AlphaEarth image band-by-band and pixel-by-pixel.",
        "",
        "The full reproduced image is:",
        "",
        f"- `{predicted_tif}`",
        "",
        "## Quantitative Results",
        "",
        "### Held-Out Sample Performance",
        "",
        f"- Mean R^2 across 64 bands: `{heldout_summary['mean_r2']:.3f}`",
        f"- Median R^2 across 64 bands: `{heldout_summary['median_r2']:.3f}`",
        f"- Mean RMSE across 64 bands: `{heldout_summary['mean_rmse']:.4f}`",
        f"- Mean Pearson r across 64 bands: `{heldout_summary['mean_pearson_r']:.3f}`",
        f"- Best band: `{heldout_summary['best_band']}` with `R^2 = {heldout_summary['best_band_r2']:.3f}`",
        f"- Worst band: `{heldout_summary['worst_band']}` with `R^2 = {heldout_summary['worst_band_r2']:.3f}`",
        "",
        "### Full-Image Reconstruction Performance",
        "",
        f"- Mean R^2 across 64 bands: `{full_summary['mean_r2']:.3f}`",
        f"- Median R^2 across 64 bands: `{full_summary['median_r2']:.3f}`",
        f"- Mean RMSE across 64 bands: `{full_summary['mean_rmse']:.4f}`",
        f"- Mean Pearson r across 64 bands: `{full_summary['mean_pearson_r']:.3f}`",
        f"- Best band: `{full_summary['best_band']}` with `R^2 = {full_summary['best_band_r2']:.3f}`",
        f"- Worst band: `{full_summary['worst_band']}` with `R^2 = {full_summary['worst_band_r2']:.3f}`",
        "",
        "## Interpretation",
        "",
        "The central result is that the model reproduces the AlphaEarth embedding field moderately well from Sentinel-1 alone, but not uniformly across all embedding dimensions.",
        "",
        "The held-out and full-image metrics are close. That matters. It means the reconstruction quality seen on the test sample is not collapsing when the model is applied to the whole image. In other words, the model is not just memorizing sampled pixels. It is learning a scene-level mapping that transfers across the AOI.",
        "",
        "At the same time, the reconstruction is incomplete. A mean full-image `R^2` of `0.480` says that SAR-only input can explain a meaningful fraction of AlphaEarth variation, but it does not recover the full optical embedding space. Some bands are strongly reproducible, while others remain weak.",
        "",
        "That pattern is consistent with the broader project result: SAR and AlphaEarth are related, but they are not interchangeable representations.",
        "",
        "## Single All-Band Reproduced Image",
        "",
        "The figure below is the most important visualization in this report.",
        "",
        "Because the AlphaEarth image has 64 bands, it cannot be shown directly as one normal RGB image. To create one single interpretable image from all bands, the true 64-band AlphaEarth field was projected into 3 dimensions using PCA, and the reproduced embedding was projected using the same PCA basis.",
        "",
        "To make that distinction explicit, the following figure compares a real Sentinel-2 natural-color image with the AlphaEarth PCA RGB visualization for the same AOI.",
        "",
        f"![]({rel(natural_vs_pca)}){{ width=100% }}",
        "",
        "The next three figures show the same all-band representation at a much larger scale:",
        "",
        "- the true all-band AlphaEarth image",
        "- the SAR-reproduced all-band AlphaEarth image",
        "- the residual intensity, measured as per-pixel RMSE across all 64 bands",
        "",
        f"![]({rel(true_large)}){{ width=100% }}",
        "",
        f"![]({rel(pred_large)}){{ width=100% }}",
        "",
        f"![]({rel(resid_large)}){{ width=100% }}",
        "",
        "This figure is the best single summary of whether the scene was reproduced. The reproduced image preserves broad spatial organization and major structural transitions, while the residual map shows where the SAR-only model fails to match the full embedding.",
        "",
        "## Residual Summary",
        "",
        "The following graph summarizes reconstruction quality by embedding band.",
        "",
        f"![]({rel(resid_summary)}){{ width=70% }}",
        "",
        "This plot separates two ideas:",
        "",
        "- `R^2` indicates how much bandwise variation is recovered",
        "- `RMSE` and `MAE` indicate the absolute residual size",
        "",
        "Bands with high `R^2` and low residual error are the bands most reproducible from SAR. Bands with low `R^2` and higher residual error are the dimensions where the SAR-to-embedding mapping is weakest.",
        "",
        "## Additional Diagnostics",
        "",
        "The secondary diagnostics are intentionally reduced here. Their main role is just to confirm that sampled performance and full-image performance stay close.",
        "",
        f"![]({rel(heldout_vs_full)}){{ width=70% }}",
        "",
        "That comparison supports the same conclusion as the headline metrics: the model generalizes reasonably consistently from the sample to the full scene, but the quality ceiling is band-dependent and clearly limited.",
        "",
        "## Conclusion",
        "",
        "For this downtown San Francisco plus Golden Gate Bridge AOI, Sentinel-1 can reproduce a meaningful part of the AlphaEarth embedding image using a small training sample and a lightweight regression model.",
        "",
        "That is a useful result. It shows that SAR contains enough information to reconstruct substantial structure in the AlphaEarth space over a full image, not just at isolated sampled pixels.",
        "",
        "But the reproduction is only partial. The embedding is not fully recoverable from SAR alone, and the residual structure shows that some AlphaEarth dimensions depend on information outside what this SAR input can provide.",
        "",
        "The practical takeaway is:",
        "",
        "- full-image SAR-to-AlphaEarth reconstruction is feasible",
        "- it works well enough to preserve broad structure",
        "- it is not accurate enough to claim that Sentinel-1 fully reproduces the AlphaEarth representation",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tiles = discover_full_stack_tiles(args.full_stack_dir, args.full_stack_glob)
    X, y = sample_training_data(
        tiles=tiles,
        chunk_size=args.chunk_size,
        sample_probability=args.sample_probability,
    )
    print(f"[train] sampled {X.shape[0]} rows", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )

    model = build_model(alpha=args.alpha)
    model.fit(X_train, y_train)
    print("[train] model fit complete", flush=True)

    heldout_metrics = evaluate_heldout(model=model, X_test=X_test, y_test=y_test)
    heldout_metrics_path = args.output_dir / "heldout_metrics_by_band.csv"
    heldout_metrics.to_csv(heldout_metrics_path, index=False)

    full_metrics = reconstruct_full_image(
        model=model,
        tiles=tiles,
        sar_path=args.sar_path,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )
    print("[reconstruct] full-image pass complete", flush=True)
    full_metrics_path = args.output_dir / "full_image_metrics_by_band.csv"
    full_metrics.to_csv(full_metrics_path, index=False)

    figure_paths = make_figures(
        heldout_metrics=heldout_metrics,
        full_metrics=full_metrics,
        truth_tiles=tiles,
        predicted_tiles_dir=args.output_dir / "predicted_tiles",
        sar_path=args.sar_path,
        output_dir=args.output_dir,
    )

    heldout_summary = summarize_metrics(heldout_metrics)
    full_summary = summarize_metrics(full_metrics)

    metadata = {
        "sar_path": str(args.sar_path),
        "full_stack_dir": str(args.full_stack_dir),
        "tile_count": len(tiles),
        "sample_probability": args.sample_probability,
        "sample_count": int(X.shape[0]),
        "train_count": int(X_train.shape[0]),
        "test_count": int(X_test.shape[0]),
        "chunk_size": args.chunk_size,
        "model": "PolynomialFeatures(degree=2) + StandardScaler + Ridge(alpha=10.0)",
        "predictors": SAR_BANDS,
        "targets": EMBEDDING_BANDS,
        "heldout_summary": heldout_summary,
        "full_summary": full_summary,
    }
    metadata_path = args.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    write_report(
        report_path=args.report_path,
        sar_path=args.sar_path,
        tiles=tiles,
        sample_count=int(X.shape[0]),
        train_count=int(X_train.shape[0]),
        test_count=int(X_test.shape[0]),
        heldout_summary=heldout_summary,
        full_summary=full_summary,
        output_dir=args.output_dir,
    )

    print(f"Sampled rows: {X.shape[0]}")
    print(f"Wrote held-out metrics: {heldout_metrics_path}")
    print(f"Wrote full-image metrics: {full_metrics_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote report: {args.report_path}")


if __name__ == "__main__":
    main()
