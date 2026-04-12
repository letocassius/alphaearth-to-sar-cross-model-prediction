#!/usr/bin/env python3
"""
Run a single-image AlphaEarth-to-pixel-value experiment over one exported GEE CSV.

Expected input schema:
- Sentinel pixel bands such as B2, B3, B4, B8, B11, B12
- AlphaEarth embedding bands A00 ... A63

The script:
1. loads one Earth Engine export CSV
2. picks one Sentinel band as the regression target
3. splits train/test once
4. varies the fraction of training rows used
5. trains a standardized Ridge regressor from embedding -> target pixel value
6. writes metrics and predictions for each fraction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "single_image_pixel_fraction"
EMBEDDING_COLS = [f"A{i:02d}" for i in range(64)]
DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
DEFAULT_TARGET = "B4"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Path to the Earth Engine CSV. If omitted, the script searches DataSources/.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET,
        help="Sentinel band to predict from the embedding, for example B4 or B8.",
    )
    parser.add_argument(
        "--fractions",
        default=",".join(str(v) for v in DEFAULT_FRACTIONS),
        help="Comma-separated training fractions, for example 0.01,0.05,0.1,0.25,0.5,1.0",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out test fraction.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=50,
        help="Minimum rows required for a fraction-specific training subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for outputs.",
    )
    return parser.parse_args()


def parse_fraction_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one fraction is required.")
    for value in values:
        if value <= 0 or value > 1:
            raise ValueError(f"Fraction must be in (0, 1], got {value}.")
    return sorted(set(values))


def find_default_csv() -> Path:
    candidates = sorted((ROOT_DIR / "DataSources").rglob("sentinel2_alphaearth_pixel_pairs_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No Earth Engine export CSV matching 'sentinel2_alphaearth_pixel_pairs_*.csv' "
            "was found under DataSources/."
        )
    return candidates[-1]


def safe_pearsonr(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    if y_true_arr.size < 2 or np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0:
        return float("nan")
    return float(pearsonr(y_true_arr, y_pred_arr).statistic)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def load_dataset(csv_path: Path, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_embedding = [col for col in EMBEDDING_COLS if col not in df.columns]
    if missing_embedding:
        raise ValueError(f"Missing embedding columns: {missing_embedding}")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

    keep_cols = EMBEDDING_COLS + [target_col]
    optional_cols = [col for col in ("longitude", "latitude", ".geo", "system:index") if col in df.columns]
    out = df[keep_cols + optional_cols].copy()
    out = out.dropna(subset=EMBEDDING_COLS + [target_col]).reset_index(drop=True)
    return out


def evaluate_fraction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    fraction: float,
    min_train_rows: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    subset_n = max(min_train_rows, int(round(len(train_df) * fraction)))
    subset_n = min(subset_n, len(train_df))
    subset_df = train_df.sample(n=subset_n, random_state=RANDOM_STATE).copy()

    X_train = subset_df[EMBEDDING_COLS]
    y_train = subset_df[target_col]
    X_test = test_df[EMBEDDING_COLS]
    y_test = test_df[target_col]

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "fraction": fraction,
        "train_rows": int(subset_n),
        "test_rows": int(len(test_df)),
        "target_col": target_col,
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": rmse(y_test, y_pred),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "pearson_r": safe_pearsonr(y_test, y_pred),
    }

    predictions = test_df.copy()
    predictions["fraction"] = fraction
    predictions["y_true"] = y_test.to_numpy()
    predictions["y_pred"] = y_pred
    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]
    return metrics, predictions


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path if args.csv_path else find_default_csv()
    fractions = parse_fraction_list(args.fractions)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path, args.target_col)
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=RANDOM_STATE)

    metrics_rows: list[dict[str, float]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fraction in fractions:
        metrics, predictions = evaluate_fraction(
            train_df=train_df,
            test_df=test_df,
            target_col=args.target_col,
            fraction=fraction,
            min_train_rows=args.min_train_rows,
        )
        metrics_rows.append(metrics)
        prediction_frames.append(predictions)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("fraction").reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    stem = f"{csv_path.stem}_{args.target_col}"
    metrics_path = output_dir / f"{stem}_fraction_metrics.csv"
    predictions_path = output_dir / f"{stem}_test_predictions.csv"
    metadata_path = output_dir / f"{stem}_run_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    metadata = {
        "csv_path": str(csv_path),
        "target_col": args.target_col,
        "fractions": fractions,
        "train_rows_total": int(len(train_df)),
        "test_rows_total": int(len(test_df)),
        "embedding_cols": EMBEDDING_COLS,
        "model": "StandardScaler + Ridge(alpha=1.0)",
        "random_state": RANDOM_STATE,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Loaded dataset: {csv_path}")
    print(f"Rows after NA drop: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print()
    print(metrics_df.to_string(index=False))
    print()
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
