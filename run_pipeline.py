#!/usr/bin/env python3
"""Run the full AlphaEarth-to-SAR reconstruction pipeline.

This is the only command most users need:

    python run_pipeline.py

Expected input:
    data/raw/sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif

The full-stack GeoTIFF should contain:
    bands 1-3: Sentinel-1 SAR targets
    bands 4-67: AlphaEarth embedding features A00-A63
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FULL_STACK_PATH,
    DEFAULT_SAMPLE_PROBABILITY,
    DEFAULT_SAMPLING_STRATEGY,
    DEFAULT_TEST_SIZE,
    EMBEDDING_BANDS,
    FIGURES_DIR,
    MODELS_DIR,
    SAR_BANDS,
)
from src.evaluation import evaluate_predictions, summarize_metrics
from src.modeling import build_model, predict_sar, save_model
from src.plotting import build_residual_summary, build_sar_large_views
from src.reconstruction import reconstruct_full_sar
from src.sampling import assign_splits, sample_training_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AlphaEarth embedding -> Sentinel-1 SAR reconstruction.")
    parser.add_argument("--full-stack-path", type=Path, default=DEFAULT_FULL_STACK_PATH)
    parser.add_argument(
        "--sar-path",
        type=Path,
        default=None,
        help="Optional separate SAR GeoTIFF. If omitted, the first 3 bands of --full-stack-path are used.",
    )
    parser.add_argument("--sample-probability", type=float, default=DEFAULT_SAMPLE_PROBABILITY)
    parser.add_argument("--sampling-strategy", choices=["random", "grid"], default=DEFAULT_SAMPLING_STRATEGY)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument(
        "--predict-training-pixels",
        action="store_true",
        help="Predict every valid pixel instead of copying observed SAR values at training pixels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_stack_path = args.full_stack_path
    sar_path = args.sar_path or full_stack_path

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not full_stack_path.exists():
        raise FileNotFoundError(
            f"Could not find {full_stack_path}. Put the colocated SAR + AlphaEarth GeoTIFF in data/raw/ "
            "or pass --full-stack-path."
        )
    if not sar_path.exists():
        raise FileNotFoundError(f"Could not find SAR path: {sar_path}")

    tiles = [full_stack_path]

    print("[1/6] Sampling training pixels")
    X, y, locations = sample_training_data(
        tiles=tiles,
        chunk_size=args.chunk_size,
        sample_probability=args.sample_probability,
        sampling_strategy=args.sampling_strategy,
    )
    X_train, X_test, y_train, y_test, locations = assign_splits(
        X=X,
        y=y,
        locations=locations,
        test_size=args.test_size,
    )

    locations_path = DATA_PROCESSED_DIR / "sampled_pixel_locations.csv"
    dataset_path = DATA_PROCESSED_DIR / "sampled_alphaearth_to_sar_dataset.csv"
    locations.to_csv(locations_path, index=False)
    pd.concat([locations, pd.DataFrame(X, columns=EMBEDDING_BANDS)], axis=1).to_csv(dataset_path, index=False)

    print("[2/6] Training LightGBM model")
    model = build_model(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
    )
    model.fit(X_train, y_train)
    model_path = MODELS_DIR / "alphaearth_to_sar_lightgbm.joblib"
    save_model(model, model_path)

    print("[3/6] Evaluating held-out sampled pixels")
    y_test_pred = predict_sar(model, X_test)
    heldout_metrics = evaluate_predictions(y_test, y_test_pred, SAR_BANDS)
    heldout_metrics_path = DATA_PROCESSED_DIR / "heldout_metrics_by_band.csv"
    heldout_metrics.to_csv(heldout_metrics_path, index=False)

    print("[4/6] Reconstructing full SARhat GeoTIFF")
    sarhat_path, full_metrics, gap_metrics = reconstruct_full_sar(
        model=model,
        tiles=tiles,
        sar_path=sar_path,
        sample_locations=locations,
        output_dir=DATA_PROCESSED_DIR,
        chunk_size=args.chunk_size,
        predict_training_pixels=args.predict_training_pixels,
    )
    full_metrics_path = DATA_PROCESSED_DIR / "full_image_metrics_by_band.csv"
    gap_metrics_path = DATA_PROCESSED_DIR / "gap_fill_metrics_by_band.csv"
    full_metrics.to_csv(full_metrics_path, index=False)
    gap_metrics.to_csv(gap_metrics_path, index=False)

    print("[5/6] Creating figures")
    build_sar_large_views(pred_path=sarhat_path, truth_path=sar_path, output_dir=FIGURES_DIR)
    build_residual_summary(gap_metrics, FIGURES_DIR / "residual_summary_by_sar_band.png")

    print("[6/6] Saving run summary")
    summary = {
        "full_stack_path": str(full_stack_path),
        "sar_path": str(sar_path),
        "sample_probability": args.sample_probability,
        "sampling_strategy": args.sampling_strategy,
        "sample_count": int(X.shape[0]),
        "train_count": int(X_train.shape[0]),
        "test_count": int(X_test.shape[0]),
        "model_path": str(model_path),
        "sarhat_path": str(sarhat_path),
        "sampled_pixel_locations": str(locations_path),
        "sampled_dataset": str(dataset_path),
        "heldout_metrics": str(heldout_metrics_path),
        "gap_fill_metrics": str(gap_metrics_path),
        "full_image_metrics": str(full_metrics_path),
        "heldout_summary": summarize_metrics(heldout_metrics),
        "gap_fill_summary": summarize_metrics(gap_metrics),
        "full_image_summary": summarize_metrics(full_metrics),
    }
    summary_path = DATA_PROCESSED_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nDone. Key outputs:")
    print(f"  SARhat GeoTIFF: {sarhat_path}")
    print(f"  Metrics: {gap_metrics_path}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
