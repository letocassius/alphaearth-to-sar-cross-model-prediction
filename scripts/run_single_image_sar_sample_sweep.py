#!/usr/bin/env python3
"""Run a training-sample sweep for AlphaEarth -> SAR on one image stack.

This is a metrics-only companion to run_single_image_sar_reconstruction.py.
It does not write reconstructed images. Instead, it holds out a fixed
validation set and measures how SAR prediction metrics change as the number of
sampled training pixels increases.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "DataSources" / "single_image_sar_reconstruction"
DEFAULT_STACK_PATH = DATA_DIR / "sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "single_image_sar_sample_sweep_sf_downtown_golden_gate"
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "single_image_sar_sample_sweep_sf_downtown_golden_gate_report.md"

SAR_BANDS = ["S1_VV", "S1_VH", "S1_VV_div_VH"]
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]
DEFAULT_SAMPLE_PERCENTS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.2, 1.0]
RANDOM_STATE = 42


def format_percent(value: float) -> str:
    if value == 0:
        return "0"
    if value < 0.01:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if value < 0.1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.1f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-path", type=Path, default=DEFAULT_STACK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--sample-percents", type=float, nargs="+", default=DEFAULT_SAMPLE_PERCENTS)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    return parser.parse_args()


def iter_windows(height: int, width: int, chunk_size: int) -> list[Window]:
    windows: list[Window] = []
    for row_off in range(0, height, chunk_size):
        win_h = min(chunk_size, height - row_off)
        for col_off in range(0, width, chunk_size):
            win_w = min(chunk_size, width - col_off)
            windows.append(Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h))
    return windows


def load_valid_pixels(stack_path: Path, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    with rasterio.open(stack_path) as src:
        for window in iter_windows(src.height, src.width, chunk_size):
            sar_chunk = src.read(indexes=[1, 2, 3], window=window)
            emb_chunk = src.read(indexes=list(range(4, 68)), window=window)
            valid_mask = np.all(np.isfinite(sar_chunk), axis=0) & np.all(np.isfinite(emb_chunk), axis=0)
            if not valid_mask.any():
                continue
            y_parts.append(sar_chunk[:, valid_mask].T.astype(np.float32, copy=False))
            x_parts.append(emb_chunk[:, valid_mask].T.astype(np.float32, copy=False))

    if not x_parts:
        raise ValueError(f"No valid pixels found in {stack_path}")
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


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


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_percent: float,
    train_rows: int,
    method: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for band_idx, band in enumerate(SAR_BANDS):
        truth = y_true[:, band_idx]
        pred = y_pred[:, band_idx]
        rows.append(
            {
                "sample_percent": sample_percent,
                "train_rows": train_rows,
                "method": method,
                "band": band,
                "count": int(truth.shape[0]),
                "r2": float(r2_score(truth, pred)),
                "rmse": float(math.sqrt(mean_squared_error(truth, pred))),
                "mae": float(mean_absolute_error(truth, pred)),
            }
        )
    return rows


def plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    specs = [
        ("mean_r2", "Mean R^2", "#2563eb"),
        ("mean_rmse", "Mean RMSE", "#dc2626"),
        ("mean_mae", "Mean MAE", "#111827"),
    ]

    for ax, (column, title, color) in zip(axes, specs):
        ax.plot(summary_df["sample_percent"], summary_df[column], marker="o", linewidth=2, color=color)
        ax.set_title(title)
        ax.set_xlabel("Training pixels sampled (%)")
        ax.grid(True, alpha=0.25)
        if column == "mean_r2":
            ax.axhline(0, color="#6b7280", linewidth=1)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_by_band(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    specs = [("r2", "R^2"), ("rmse", "RMSE"), ("mae", "MAE")]

    for ax, (column, title) in zip(axes, specs):
        for band in SAR_BANDS:
            band_df = metrics_df[metrics_df["band"] == band].sort_values("sample_percent")
            ax.plot(band_df["sample_percent"], band_df[column], marker="o", linewidth=2, label=band)
        ax.set_title(title)
        ax.set_xlabel("Training pixels sampled (%)")
        ax.grid(True, alpha=0.25)
        if column == "r2":
            ax.axhline(0, color="#6b7280", linewidth=1)
        ax.legend()

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_low_rate_zoom(summary_df: pd.DataFrame, output_path: Path) -> None:
    zoom_df = summary_df[summary_df["sample_percent"] <= 0.2].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    specs = [
        ("mean_r2", "Mean R^2", "#2563eb"),
        ("mean_rmse", "Mean RMSE", "#dc2626"),
        ("mean_mae", "Mean MAE", "#111827"),
    ]

    for ax, (column, title, color) in zip(axes, specs):
        ax.plot(zoom_df["train_rows"], zoom_df[column], marker="o", linewidth=2, color=color)
        ax.set_title(f"{title}, 0-0.2%")
        ax.set_xlabel("Training rows")
        ax.grid(True, alpha=0.25)
        if column == "mean_r2":
            ax.axhline(0, color="#6b7280", linewidth=1)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def metrics_table_markdown(metrics_df: pd.DataFrame) -> list[str]:
    lines = [
        "| Sample % | Train rows | Band | R^2 | RMSE | MAE |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    table_df = metrics_df.copy()
    table_df["band"] = pd.Categorical(table_df["band"], categories=SAR_BANDS, ordered=True)
    for row in table_df.sort_values(["sample_percent", "band"]).itertuples(index=False):
        lines.append(
            f"| {format_percent(float(row.sample_percent))} | {int(row.train_rows)} | `{row.band}` | "
            f"{row.r2:.3f} | {row.rmse:.4f} | {row.mae:.4f} |"
        )
    return lines


def summary_table_markdown(summary_df: pd.DataFrame) -> list[str]:
    lines = [
        "| Sample % | Train rows | Mean R^2 | Mean RMSE | Mean MAE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary_df.sort_values("sample_percent").itertuples(index=False):
        lines.append(
            f"| {format_percent(float(row.sample_percent))} | {int(row.train_rows)} | "
            f"{row.mean_r2:.3f} | {row.mean_rmse:.4f} | {row.mean_mae:.4f} |"
        )
    return lines


def largest_r2_jump(summary_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    ordered = summary_df.sort_values("sample_percent").reset_index(drop=True)
    deltas = ordered["mean_r2"].diff()
    jump_idx = int(deltas.iloc[1:].idxmax())
    return ordered.iloc[jump_idx - 1], ordered.iloc[jump_idx], float(deltas.iloc[jump_idx])


def largest_learned_r2_jump(summary_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    learned = summary_df[summary_df["train_rows"] > 0].sort_values("sample_percent").reset_index(drop=True)
    deltas = learned["mean_r2"].diff()
    jump_idx = int(deltas.iloc[1:].idxmax())
    return learned.iloc[jump_idx - 1], learned.iloc[jump_idx], float(deltas.iloc[jump_idx])


def write_report(
    args: argparse.Namespace,
    valid_pixel_count: int,
    validation_rows: int,
    train_pool_rows: int,
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    summary_plot_path: Path,
    band_plot_path: Path,
    zoom_plot_path: Path,
) -> None:
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda p: Path("..") / p.relative_to(ROOT_DIR)
    before_jump, after_jump, jump_delta = largest_r2_jump(summary_df)
    before_learned_jump, after_learned_jump, learned_jump_delta = largest_learned_r2_jump(summary_df)
    lines = [
        "# Single-Image SAR Training Sample Sweep",
        "",
        "## Objective",
        "",
        "This experiment tests how AlphaEarth-to-Sentinel-1 SAR prediction quality changes as more valid image pixels are used for supervised training.",
        "",
        "Unlike the reconstruction report, this run does not write a reconstructed SAR image. It only trains models and evaluates R^2, RMSE, and MAE on one fixed validation set.",
        "",
        "## Setup",
        "",
        f"- Input stack: `{args.stack_path}`",
        f"- Valid pixels: `{valid_pixel_count}`",
        f"- Validation fraction: `{args.validation_fraction:.3f}`",
        f"- Validation rows: `{validation_rows}`",
        f"- Candidate training pool rows: `{train_pool_rows}`",
        "- Predictors: `A00` to `A63`",
        "- Targets: `S1_VV`, `S1_VH`, `S1_VV_div_VH`",
        "- Model for nonzero sample rates: `MultiOutputRegressor(LGBMRegressor)`",
        "",
        "The `0.0%` row is a no-training random baseline. It predicts each validation pixel by drawing random SAR values from the non-validation pool, so it measures chance-level performance rather than a learned AlphaEarth-to-SAR mapping.",
        "",
        "## Mean Metrics Across SAR Bands",
        "",
        *summary_table_markdown(summary_df),
        "",
        f"![]({rel(summary_plot_path)}){{ width=100% }}",
        "",
        "## Low-Rate Detail",
        "",
        "The largest mean R^2 jump in this run occurs between "
        f"`{format_percent(float(before_jump.sample_percent))}%` "
        f"({int(before_jump.train_rows)} rows) and "
        f"`{format_percent(float(after_jump.sample_percent))}%` "
        f"({int(after_jump.train_rows)} rows): "
        f"mean R^2 increases by `{jump_delta:.3f}`.",
        "",
        "Among learned models only, the sharpest increase occurs between "
        f"`{format_percent(float(before_learned_jump.sample_percent))}%` "
        f"({int(before_learned_jump.train_rows)} rows) and "
        f"`{format_percent(float(after_learned_jump.sample_percent))}%` "
        f"({int(after_learned_jump.train_rows)} rows): "
        f"mean R^2 increases by `{learned_jump_delta:.3f}`.",
        "",
        f"![]({rel(zoom_plot_path)}){{ width=100% }}",
        "",
        "## Metrics By SAR Band",
        "",
        *metrics_table_markdown(metrics_df),
        "",
        f"![]({rel(band_plot_path)}){{ width=90% }}",
        "",
        "## Interpretation",
        "",
        "R^2 should increase and RMSE/MAE should decrease as training pixels are added if the embeddings contain predictive SAR information. The random baseline is expected to have strongly negative R^2 because it draws unrelated SAR values from the scene distribution instead of using AlphaEarth features.",
        "",
    ]
    args.report_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_valid_pixels(args.stack_path, args.chunk_size)
    n_pixels = X.shape[0]
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = rng.permutation(n_pixels)
    validation_rows = int(round(args.validation_fraction * n_pixels))
    val_idx = shuffled[:validation_rows]
    train_pool_idx = shuffled[validation_rows:]

    X_val = X[val_idx]
    y_val = y[val_idx]
    X_train_pool = X[train_pool_idx]
    y_train_pool = y[train_pool_idx]

    rows: list[dict[str, float | int | str]] = []
    for sample_percent in sorted(args.sample_percents):
        if sample_percent < 0:
            raise ValueError("--sample-percents cannot contain negative values")

        requested_rows = int(round((sample_percent / 100.0) * n_pixels))
        train_rows = min(requested_rows, X_train_pool.shape[0])

        if train_rows == 0:
            random_pick = rng.integers(0, y_train_pool.shape[0], size=y_val.shape[0])
            y_pred = y_train_pool[random_pick]
            rows.extend(
                evaluate_predictions(
                    y_true=y_val,
                    y_pred=y_pred,
                    sample_percent=sample_percent,
                    train_rows=0,
                    method="random_sar_draw",
                )
            )
            print(f"[sweep] {format_percent(sample_percent)}%: random baseline", flush=True)
            continue

        train_idx = rng.choice(X_train_pool.shape[0], size=train_rows, replace=False)
        model = build_model(args)
        model.fit(X_train_pool[train_idx], y_train_pool[train_idx])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                category=UserWarning,
            )
            y_pred = model.predict(X_val)
        rows.extend(
            evaluate_predictions(
                y_true=y_val,
                y_pred=y_pred,
                sample_percent=sample_percent,
                train_rows=train_rows,
                method="lightgbm",
            )
        )
        print(f"[sweep] {format_percent(sample_percent)}%: trained on {train_rows} rows", flush=True)

    metrics_df = pd.DataFrame(rows).sort_values(["sample_percent", "band"]).reset_index(drop=True)
    summary_df = (
        metrics_df.groupby(["sample_percent", "train_rows"], as_index=False)
        .agg(mean_r2=("r2", "mean"), mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"))
        .sort_values("sample_percent")
        .reset_index(drop=True)
    )

    metrics_path = args.output_dir / "metrics_by_sample_percent_and_band.csv"
    summary_path = args.output_dir / "summary_by_sample_percent.csv"
    summary_plot_path = args.output_dir / "sample_sweep_mean_metrics.png"
    band_plot_path = args.output_dir / "sample_sweep_metrics_by_band.png"
    zoom_plot_path = args.output_dir / "sample_sweep_low_rate_zoom.png"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_summary(summary_df, summary_plot_path)
    plot_by_band(metrics_df, band_plot_path)
    plot_low_rate_zoom(summary_df, zoom_plot_path)
    write_report(
        args=args,
        valid_pixel_count=n_pixels,
        validation_rows=validation_rows,
        train_pool_rows=X_train_pool.shape[0],
        metrics_df=metrics_df,
        summary_df=summary_df,
        summary_plot_path=summary_plot_path,
        band_plot_path=band_plot_path,
        zoom_plot_path=zoom_plot_path,
    )
    print(args.report_path)


if __name__ == "__main__":
    main()
