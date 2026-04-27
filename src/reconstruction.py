"""Full-scene SARhat reconstruction."""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.multioutput import MultiOutputRegressor

from .config import SAR_BANDS
from .data_io import iter_windows, parse_tile_offsets, sarhat_output_name
from .evaluation import RunningMoments
from .modeling import predict_sar


def build_training_mask(sar_path: Path, sample_locations: pd.DataFrame) -> np.ndarray:
    """Return a mask of pixels that were used for model training."""
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
    predict_training_pixels: bool = False,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Predict all non-training valid pixels and write full SARhat GeoTIFF."""
    output_path = output_dir / sarhat_output_name(tiles[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_mask = build_training_mask(sar_path, sample_locations)
    running_all = RunningMoments.zeros(len(SAR_BANDS))
    running_gap = RunningMoments.zeros(len(SAR_BANDS))

    with rasterio.open(sar_path) as sar_src:
        profile = sar_src.profile.copy()
        profile.update(count=len(SAR_BANDS), dtype="float32", compress="deflate", predictor=3, nodata=np.nan)

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

                        pred_chunk = np.full((len(SAR_BANDS), int(window.height), int(window.width)), np.nan, dtype=np.float32)

                        if valid_mask.any():
                            train_window = training_mask[
                                int(dst_window.row_off): int(dst_window.row_off + dst_window.height),
                                int(dst_window.col_off): int(dst_window.col_off + dst_window.width),
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
