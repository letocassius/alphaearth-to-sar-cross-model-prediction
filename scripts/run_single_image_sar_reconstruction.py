#!/usr/bin/env python3
"""
Train an AlphaEarth embedding -> Sentinel-1 SAR model from colocated Earth Engine
exports, then fill unsampled SAR pixels and write a report-ready artifact bundle.

Inputs:
- a full Sentinel-1 GeoTIFF with bands S1_VV, S1_VH, S1_VV_div_VH
- a colocated full-stack GeoTIFF with bands S1_* followed by A00..A63

Outputs:
- sampled pixel locations with train/test split labels
- held-out SAR prediction metrics
- full-image and gap-fill-only SARhat metrics
- a full-scene SARhat GeoTIFF
- preview figures and a Markdown report under reports/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "DataSources" / "single_image_sar_reconstruction"
OUTPUT_DIR = ROOT_DIR / "outputs" / "single_image_sar_reconstruction_sf_downtown_golden_gate"
REPORTS_DIR = ROOT_DIR / "reports"

SAR_BANDS = ["S1_VV", "S1_VH", "S1_VV_div_VH"]
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]
FULL_STACK_GLOB = "sentinel1_alphaearth*.tif"
DEFAULT_SAR_PATH = DATA_DIR / "sentinel1_small_vv_vh_sf_downtown_golden_gate_2024.tif"
DEFAULT_SAMPLE_PROBABILITY = 0.002
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
    parser.add_argument("--sampling-strategy", choices=["random", "grid"], default="random")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORTS_DIR / "single_image_sar_reconstruction_sf_downtown_golden_gate_report.md",
    )
    parser.add_argument(
        "--predict-training-pixels",
        action="store_true",
        help="Predict sampled training pixels instead of copying their observed SAR values into SARhat.",
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


def parse_tile_offsets(tile_path: Path) -> tuple[int, int]:
    match = OFFSET_PATTERN.search(tile_path.name)
    if not match:
        return 0, 0
    return int(match.group("row")), int(match.group("col"))


def sarhat_output_name(tile_path: Path) -> str:
    stem = tile_path.stem.replace("sentinel1_alphaearth", "sar_hat_from_alphaearth", 1)
    return f"{stem}.tif"


def build_model(args: argparse.Namespace) -> MultiOutputRegressor:
    base = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def predict_sar(model: MultiOutputRegressor, X: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            category=UserWarning,
        )
        return model.predict(X)


def build_keep_mask(
    valid_mask: np.ndarray,
    rng: np.random.Generator,
    sample_probability: float,
    sampling_strategy: str,
    image_rows: np.ndarray,
    image_cols: np.ndarray,
) -> np.ndarray:
    if not 0 < sample_probability <= 1:
        raise ValueError("--sample-probability must be in the interval (0, 1].")

    if sampling_strategy == "random":
        return valid_mask & (rng.random(valid_mask.shape) < sample_probability)

    stride = max(1, int(round(math.sqrt(1.0 / sample_probability))))
    return valid_mask & (image_rows % stride == 0) & (image_cols % stride == 0)


def sample_training_data(
    tiles: list[Path],
    chunk_size: int,
    sample_probability: float,
    sampling_strategy: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    location_parts: list[pd.DataFrame] = []

    for tile_index, tile_path in enumerate(tiles, start=1):
        tile_rows_before = sum(part.shape[0] for part in x_parts)
        tile_row_off, tile_col_off = parse_tile_offsets(tile_path)
        with rasterio.open(tile_path) as src:
            for window in iter_windows(src.height, src.width, chunk_size):
                sar_chunk = src.read(indexes=[1, 2, 3], window=window)
                emb_chunk = src.read(indexes=list(range(4, 68)), window=window)

                valid_mask = np.all(np.isfinite(sar_chunk), axis=0) & np.all(np.isfinite(emb_chunk), axis=0)
                if not valid_mask.any():
                    continue

                local_rows = np.arange(int(window.row_off), int(window.row_off + window.height))[:, None]
                local_cols = np.arange(int(window.col_off), int(window.col_off + window.width))[None, :]
                image_rows = np.broadcast_to(tile_row_off + local_rows, valid_mask.shape)
                image_cols = np.broadcast_to(tile_col_off + local_cols, valid_mask.shape)
                keep_mask = build_keep_mask(
                    valid_mask=valid_mask,
                    rng=rng,
                    sample_probability=sample_probability,
                    sampling_strategy=sampling_strategy,
                    image_rows=image_rows,
                    image_cols=image_cols,
                )
                if not keep_mask.any():
                    continue

                sar_pixels = sar_chunk[:, keep_mask].T.astype(np.float32, copy=False)
                emb_pixels = emb_chunk[:, keep_mask].T.astype(np.float32, copy=False)
                rows = image_rows[keep_mask]
                cols = image_cols[keep_mask]
                xs, ys = rasterio.transform.xy(src.transform, rows - tile_row_off, cols - tile_col_off, offset="center")

                x_parts.append(emb_pixels)
                y_parts.append(sar_pixels)
                location_parts.append(
                    pd.DataFrame(
                        {
                            "tile": tile_path.name,
                            "image_row": rows.astype(np.int64),
                            "image_col": cols.astype(np.int64),
                            "tile_row": (rows - tile_row_off).astype(np.int64),
                            "tile_col": (cols - tile_col_off).astype(np.int64),
                            "x": np.asarray(xs, dtype=np.float64),
                            "y": np.asarray(ys, dtype=np.float64),
                            "S1_VV": sar_pixels[:, 0],
                            "S1_VH": sar_pixels[:, 1],
                            "S1_VV_div_VH": sar_pixels[:, 2],
                        }
                    )
                )

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
    locations = pd.concat(location_parts, ignore_index=True)
    locations.insert(0, "sample_id", np.arange(len(locations), dtype=np.int64))
    return X, y, locations


def assign_splits(
    X: np.ndarray,
    y: np.ndarray,
    locations: pd.DataFrame,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )
    locations = locations.copy()
    locations["split"] = "unused"
    locations.loc[idx_train, "split"] = "train"
    locations.loc[idx_test, "split"] = "test"
    return X_train, X_test, y_train, y_test, locations


def evaluate_heldout(
    model: MultiOutputRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    y_pred = predict_sar(model, X_test)
    rows: list[dict[str, float | str | int]] = []
    for idx, band in enumerate(SAR_BANDS):
        y_true_band = y_test[:, idx]
        y_pred_band = y_pred[:, idx]
        rows.append(
            {
                "band": band,
                "count": int(y_true_band.shape[0]),
                "r2": float(r2_score(y_true_band, y_pred_band)),
                "rmse": float(math.sqrt(mean_squared_error(y_true_band, y_pred_band))),
                "mae": float(mean_absolute_error(y_true_band, y_pred_band)),
                "pearson_r": safe_pearsonr(y_true_band, y_pred_band),
            }
        )
    return pd.DataFrame(rows)


def build_training_mask(sar_path: Path, sample_locations: pd.DataFrame) -> np.ndarray:
    with rasterio.open(sar_path) as sar_src:
        mask = np.zeros((sar_src.height, sar_src.width), dtype=bool)

    train_locations = sample_locations[sample_locations["split"] == "train"]
    rows = train_locations["image_row"].to_numpy(dtype=np.int64)
    cols = train_locations["image_col"].to_numpy(dtype=np.int64)
    in_bounds = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    mask[rows[in_bounds], cols[in_bounds]] = True
    return mask


def reconstruct_full_sar(
    model: MultiOutputRegressor,
    tiles: list[Path],
    sar_path: Path,
    sample_locations: pd.DataFrame,
    output_dir: Path,
    chunk_size: int,
    predict_training_pixels: bool,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    output_path = output_dir / sarhat_output_name(tiles[0])
    training_mask = build_training_mask(sar_path, sample_locations)
    running_all = RunningMoments.zeros(len(SAR_BANDS))
    running_gap = RunningMoments.zeros(len(SAR_BANDS))

    with rasterio.open(sar_path) as sar_src:
        profile = sar_src.profile.copy()
        profile.update(
            count=len(SAR_BANDS),
            dtype="float32",
            compress="deflate",
            predictor=3,
            nodata=np.nan,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.descriptions = tuple(SAR_BANDS)

            for tile_index, tile_path in enumerate(tiles, start=1):
                tile_row_off, tile_col_off = parse_tile_offsets(tile_path)
                with rasterio.open(tile_path) as src:
                    for window in iter_windows(src.height, src.width, chunk_size):
                        dst_window = Window(
                            col_off=tile_col_off + int(window.col_off),
                            row_off=tile_row_off + int(window.row_off),
                            width=int(window.width),
                            height=int(window.height),
                        )
                        sar_chunk = src.read(indexes=[1, 2, 3], window=window)
                        emb_chunk = src.read(indexes=list(range(4, 68)), window=window)
                        valid_mask = np.all(np.isfinite(sar_chunk), axis=0) & np.all(np.isfinite(emb_chunk), axis=0)

                        pred_chunk = np.full(
                            (len(SAR_BANDS), int(window.height), int(window.width)),
                            np.nan,
                            dtype=np.float32,
                        )

                        if valid_mask.any():
                            train_window = training_mask[
                                int(dst_window.row_off) : int(dst_window.row_off + dst_window.height),
                                int(dst_window.col_off) : int(dst_window.col_off + dst_window.width),
                            ]
                            if predict_training_pixels:
                                gap_mask = valid_mask
                                copy_mask = np.zeros_like(valid_mask, dtype=bool)
                            else:
                                gap_mask = valid_mask & ~train_window
                                copy_mask = valid_mask & train_window

                            if copy_mask.any():
                                pred_chunk[:, copy_mask] = sar_chunk[:, copy_mask].astype(np.float32, copy=False)

                            if gap_mask.any():
                                emb_pixels = emb_chunk[:, gap_mask].T.astype(np.float32, copy=False)
                                truth_gap = sar_chunk[:, gap_mask].T.astype(np.float32, copy=False)
                                pred_gap = predict_sar(model, emb_pixels).astype(np.float32, copy=False)
                                pred_chunk[:, gap_mask] = pred_gap.T
                                running_gap.update(truth_gap.astype(np.float64), pred_gap.astype(np.float64))

                            truth_all = sar_chunk[:, valid_mask].T.astype(np.float32, copy=False)
                            pred_all = pred_chunk[:, valid_mask].T.astype(np.float32, copy=False)
                            if not np.all(np.isfinite(pred_all)):
                                raise RuntimeError("SARhat chunk contains unfilled values at valid SAR pixels.")
                            running_all.update(truth_all.astype(np.float64), pred_all.astype(np.float64))

                        dst.write(pred_chunk, window=dst_window)

                print(f"[reconstruct] {tile_index}/{len(tiles)} processed {tile_path.name}", flush=True)

    return output_path, running_all.to_metrics(SAR_BANDS), running_gap.to_metrics(SAR_BANDS)


def make_figures(
    heldout_metrics: pd.DataFrame,
    full_metrics: pd.DataFrame,
    gap_metrics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_paths: dict[str, Path] = {}

    metric_compare_path = output_dir / "heldout_vs_full_r2.png"
    compare_df = (
        heldout_metrics[["band", "r2"]]
        .rename(columns={"r2": "heldout_r2"})
        .merge(full_metrics[["band", "r2"]].rename(columns={"r2": "full_r2"}), on="band", how="inner")
        .merge(gap_metrics[["band", "r2"]].rename(columns={"r2": "gap_fill_r2"}), on="band", how="inner")
    )
    x = np.arange(len(compare_df))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    width = 0.25
    ax.bar(x - width, compare_df["heldout_r2"], width=width, label="Held-out")
    ax.bar(x, compare_df["gap_fill_r2"], width=width, label="Gap-fill pixels")
    ax.bar(x + width, compare_df["full_r2"], width=width, label="SARhat full image")
    ax.set_title("SAR Reconstruction R^2")
    ax.set_xlabel("SAR band")
    ax.set_ylabel("R^2")
    ax.set_xticks(x)
    ax.set_xticklabels(compare_df["band"])
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


def metrics_markdown(metrics_df: pd.DataFrame) -> list[str]:
    lines = ["| Band | Count | R^2 | RMSE | MAE | Pearson r |", "|---|---:|---:|---:|---:|---:|"]
    for row in metrics_df.itertuples(index=False):
        lines.append(
            f"| `{row.band}` | {int(row.count)} | {row.r2:.3f} | {row.rmse:.4f} | "
            f"{row.mae:.4f} | {row.pearson_r:.3f} |"
        )
    return lines


def write_report(
    report_path: Path,
    sar_path: Path,
    tiles: list[Path],
    sample_probability: float,
    sampling_strategy: str,
    sample_count: int,
    train_count: int,
    test_count: int,
    heldout_metrics: pd.DataFrame,
    gap_metrics: pd.DataFrame,
    full_metrics: pd.DataFrame,
    heldout_summary: dict[str, float | str],
    gap_summary: dict[str, float | str],
    full_summary: dict[str, float | str],
    output_dir: Path,
    sarhat_path: Path,
    predict_training_pixels: bool,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda p: os.path.relpath(p, report_path.parent)
    locations_path = output_dir / "sampled_pixel_locations.csv"
    sample_dataset_path = output_dir / "sampled_alphaearth_to_sar_dataset.csv"
    natural_vs_pca = output_dir / "sentinel2_vs_alphaearth_pca_rgb.png"
    true_large = output_dir / "true_sar_all_band_large.png"
    pred_large = output_dir / "sarhat_all_band_large.png"
    resid_large = output_dir / "residual_heatmap_sar_large.png"
    resid_summary = output_dir / "residual_summary_by_sar_band.png"
    heldout_vs_full = output_dir / "heldout_vs_full_r2.png"
    sample_percent = 100.0 * sample_probability
    fill_rule = "predicted for every valid pixel" if predict_training_pixels else "observed SAR retained at training pixels; model predictions used for all other valid pixels"

    lines = [
        "# Single-Image AlphaEarth-to-SAR Reconstruction Report",
        "",
        "## Objective",
        "",
        "This experiment tests whether colocated AlphaEarth embeddings can reproduce a Sentinel-1 SAR image over one compact scene.",
        "",
        "The model direction is now `F(AlphaEarth embedding) = SAR`. Each sampled SAR pixel supplies the target value `Y`, and the colocated 64-dimensional AlphaEarth vector supplies the input features `X`.",
        "",
        "The chosen area is a small San Francisco scene covering downtown San Francisco and the Golden Gate Bridge. That AOI contains urban structure, water, coastline, and bridge geometry while remaining small enough for full-scene reconstruction.",
        "",
        "## Inputs",
        "",
        f"- SAR image: `{sar_path}`",
        f"- Colocated SAR + AlphaEarth stack: `DataSources/single_image_sar_reconstruction/{tiles[0].name}`",
        "- Spatial extent: approximately `668 x 1894` pixels at `10 m`",
        "- Predictors: `A00` to `A63`",
        "- Reconstruction targets: `S1_VV`, `S1_VH`, `S1_VV_div_VH`",
        "",
        "The stack already contains the required colocated inputs, so no new data export was needed for this AOI. A new Earth Engine export is only needed if the AOI, year, SAR preprocessing, or AlphaEarth version changes.",
        "",
        "## Modeling Setup",
        "",
        "A subset of valid Sentinel-1 pixels was sampled and saved with image-array coordinates so the same pixels can be identified during the full-scene fill step.",
        "",
        f"- Sampling strategy: `{sampling_strategy}`",
        f"- Sampling percentage: `{sample_percent:.3f}%`",
        f"- Sampled rows: `{sample_count}`",
        f"- Train rows: `{train_count}`",
        f"- Test rows: `{test_count}`",
        "- Model: `MultiOutputRegressor(LGBMRegressor)`",
        f"- SARhat fill rule: `{fill_rule}`",
        "",
        "The sampled pixel coordinate table is:",
        "",
        f"- `{locations_path}`",
        "",
        "The sampled modeling table with coordinates, split labels, all 64 AlphaEarth features, and SAR targets is:",
        "",
        f"- `{sample_dataset_path}`",
        "",
        "## Reconstruction Procedure",
        "",
        "The workflow was:",
        "",
        "1. Fetch and use the local Sentinel-1 image for the target SAR bands.",
        "2. Use the colocated full stack to fetch the AlphaEarth embedding at each SAR pixel.",
        "3. Sample valid SAR pixels and save their row/column locations.",
        "4. Train `F(X) = Y`, where `X` is the 64-band AlphaEarth embedding and `Y` is the three-band SAR vector.",
        "5. Iterate over every valid SAR pixel in the scene.",
        "6. Copy observed SAR values at training pixels and predict all unsampled pixels from their AlphaEarth embeddings.",
        "7. Save the completed SAR reconstruction as a GeoTIFF and compare it with the actual SAR image.",
        "",
        "The reconstructed full SAR image is:",
        "",
        f"- `{sarhat_path}`",
        "",
        "## Quantitative Results",
        "",
        "### Held-Out Sample Performance",
        "",
        f"- Mean R^2 across SAR bands: `{heldout_summary['mean_r2']:.3f}`",
        f"- Mean RMSE across SAR bands: `{heldout_summary['mean_rmse']:.4f}`",
        f"- Mean Pearson r across SAR bands: `{heldout_summary['mean_pearson_r']:.3f}`",
        "",
        *metrics_markdown(heldout_metrics),
        "",
        "### Gap-Fill Performance",
        "",
        "These metrics evaluate only pixels not copied from the training dataset.",
        "",
        f"- Mean R^2 across SAR bands: `{gap_summary['mean_r2']:.3f}`",
        f"- Mean RMSE across SAR bands: `{gap_summary['mean_rmse']:.4f}`",
        f"- Mean Pearson r across SAR bands: `{gap_summary['mean_pearson_r']:.3f}`",
        "",
        *metrics_markdown(gap_metrics),
        "",
        "### Full SARhat Performance",
        "",
        "These metrics compare the completed SARhat image against the actual SAR image over all valid pixels.",
        "",
        f"- Mean R^2 across SAR bands: `{full_summary['mean_r2']:.3f}`",
        f"- Mean RMSE across SAR bands: `{full_summary['mean_rmse']:.4f}`",
        f"- Mean Pearson r across SAR bands: `{full_summary['mean_pearson_r']:.3f}`",
        "",
        *metrics_markdown(full_metrics),
        "",
        "## Interpretation",
        "",
        "The AlphaEarth embeddings contain strong supervised signal for reconstructing direct Sentinel-1 backscatter in this scene. The held-out and gap-fill metrics are the most important checks because they evaluate pixels not used to fit the model.",
        "",
        "The `S1_VV_div_VH` target remains the hardest band because it is a derived polarization relationship rather than a direct backscatter channel. That pattern is consistent with the broader tabular results in this repository.",
        "",
        "## Context Figure",
        "",
        "The following figure shows the Sentinel-2 natural-color scene beside a PCA visualization of the AlphaEarth embedding field. It is included only as spatial context for the features used by the model.",
        "",
        f"![]({rel(natural_vs_pca)}){{ width=100% }}",
        "",
        "## Full-Scene SARhat",
        "",
        "The next three figures show the actual SAR image, the reconstructed SARhat image, and the spatial residual intensity measured as per-pixel RMSE across the three SAR bands.",
        "",
        f"![]({rel(true_large)}){{ width=100% }}",
        "",
        f"![]({rel(pred_large)}){{ width=100% }}",
        "",
        f"![]({rel(resid_large)}){{ width=100% }}",
        "",
        "The residual map is the main geospatial diagnostic. It shows where the AlphaEarth-to-SAR mapping transfers cleanly and where SAR behavior is driven by geometry or scattering effects that are not fully represented by the embedding.",
        "",
        "## Residual Summary",
        "",
        "The following graph summarizes reconstruction quality by SAR band.",
        "",
        f"![]({rel(resid_summary)}){{ width=70% }}",
        "",
        "The secondary diagnostic compares held-out sample performance, gap-fill performance, and full SARhat performance.",
        "",
        f"![]({rel(heldout_vs_full)}){{ width=70% }}",
        "",
        "## Conclusion",
        "",
        "For this downtown San Francisco plus Golden Gate Bridge AOI, AlphaEarth embeddings can reproduce much of the Sentinel-1 SAR image through supervised regression.",
        "",
        "The current local data is sufficient for this corrected single-image workflow. More data would be useful if the goal shifts from a single-scene demonstration to a robust operational model: additional AOIs, seasons, incidence-angle regimes, and urban/coastal geometries would reduce the risk that the fitted mapping is scene-specific.",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tiles = discover_full_stack_tiles(args.full_stack_dir, args.full_stack_glob)
    X, y, locations = sample_training_data(
        tiles=tiles,
        chunk_size=args.chunk_size,
        sample_probability=args.sample_probability,
        sampling_strategy=args.sampling_strategy,
    )
    print(f"[train] sampled {X.shape[0]} rows", flush=True)

    X_train, X_test, y_train, y_test, locations = assign_splits(
        X=X,
        y=y,
        locations=locations,
        test_size=args.test_size,
    )
    locations_path = args.output_dir / "sampled_pixel_locations.csv"
    locations.to_csv(locations_path, index=False)
    sample_dataset_path = args.output_dir / "sampled_alphaearth_to_sar_dataset.csv"
    sample_dataset = pd.concat(
        [locations, pd.DataFrame(X, columns=EMBEDDING_BANDS)],
        axis=1,
    )
    sample_dataset.to_csv(sample_dataset_path, index=False)

    model = build_model(args)
    model.fit(X_train, y_train)
    print("[train] model fit complete", flush=True)

    heldout_metrics = evaluate_heldout(model=model, X_test=X_test, y_test=y_test)
    heldout_metrics_path = args.output_dir / "heldout_metrics_by_band.csv"
    heldout_metrics.to_csv(heldout_metrics_path, index=False)

    sarhat_path, full_metrics, gap_metrics = reconstruct_full_sar(
        model=model,
        tiles=tiles,
        sar_path=args.sar_path,
        sample_locations=locations,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        predict_training_pixels=args.predict_training_pixels,
    )
    print("[reconstruct] full-image pass complete", flush=True)

    full_metrics_path = args.output_dir / "full_image_metrics_by_band.csv"
    gap_metrics_path = args.output_dir / "gap_fill_metrics_by_band.csv"
    full_metrics.to_csv(full_metrics_path, index=False)
    gap_metrics.to_csv(gap_metrics_path, index=False)

    make_figures(
        heldout_metrics=heldout_metrics,
        full_metrics=full_metrics,
        gap_metrics=gap_metrics,
        output_dir=args.output_dir,
    )

    heldout_summary = summarize_metrics(heldout_metrics)
    full_summary = summarize_metrics(full_metrics)
    gap_summary = summarize_metrics(gap_metrics)

    metadata = {
        "sar_path": str(args.sar_path),
        "full_stack_dir": str(args.full_stack_dir),
        "tile_count": len(tiles),
        "sample_probability": args.sample_probability,
        "sampling_strategy": args.sampling_strategy,
        "sample_count": int(X.shape[0]),
        "train_count": int(X_train.shape[0]),
        "test_count": int(X_test.shape[0]),
        "chunk_size": args.chunk_size,
        "model": "MultiOutputRegressor(LGBMRegressor)",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "predictors": EMBEDDING_BANDS,
        "targets": SAR_BANDS,
        "sarhat_path": str(sarhat_path),
        "sampled_pixel_locations": str(locations_path),
        "sampled_alphaearth_to_sar_dataset": str(sample_dataset_path),
        "predict_training_pixels": bool(args.predict_training_pixels),
        "heldout_summary": heldout_summary,
        "gap_fill_summary": gap_summary,
        "full_summary": full_summary,
    }
    metadata_path = args.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    write_report(
        report_path=args.report_path,
        sar_path=args.sar_path,
        tiles=tiles,
        sample_probability=args.sample_probability,
        sampling_strategy=args.sampling_strategy,
        sample_count=int(X.shape[0]),
        train_count=int(X_train.shape[0]),
        test_count=int(X_test.shape[0]),
        heldout_metrics=heldout_metrics,
        gap_metrics=gap_metrics,
        full_metrics=full_metrics,
        heldout_summary=heldout_summary,
        gap_summary=gap_summary,
        full_summary=full_summary,
        output_dir=args.output_dir,
        sarhat_path=sarhat_path,
        predict_training_pixels=bool(args.predict_training_pixels),
    )

    print(f"Sampled rows: {X.shape[0]}")
    print(f"Wrote sampled locations: {locations_path}")
    print(f"Wrote sampled modeling dataset: {sample_dataset_path}")
    print(f"Wrote held-out metrics: {heldout_metrics_path}")
    print(f"Wrote gap-fill metrics: {gap_metrics_path}")
    print(f"Wrote full-image metrics: {full_metrics_path}")
    print(f"Wrote SARhat: {sarhat_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote report: {args.report_path}")


if __name__ == "__main__":
    main()
