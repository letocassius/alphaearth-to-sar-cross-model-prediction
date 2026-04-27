"""Training sample construction for AlphaEarth-to-SAR modeling."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE
from .data_io import iter_windows, parse_tile_offsets


def build_keep_mask(
    valid_mask: np.ndarray,
    rng: np.random.Generator,
    sample_probability: float,
    sampling_strategy: str,
    image_rows: np.ndarray,
    image_cols: np.ndarray,
) -> np.ndarray:
    """Choose sampled pixels using either random or grid sampling."""
    if not 0 < sample_probability <= 1:
        raise ValueError("--sample-probability must be in the interval (0, 1].")

    if sampling_strategy == "random":
        return valid_mask & (rng.random(valid_mask.shape) < sample_probability)

    if sampling_strategy == "grid":
        stride = max(1, int(round(math.sqrt(1.0 / sample_probability))))
        return valid_mask & (image_rows % stride == 0) & (image_cols % stride == 0)

    raise ValueError("sampling_strategy must be either 'random' or 'grid'.")


def sample_training_data(
    tiles: list[Path],
    chunk_size: int,
    sample_probability: float,
    sampling_strategy: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Sample SAR targets and colocated 64-d AlphaEarth embeddings from full-stack tiles."""
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

                keep_mask = build_keep_mask(valid_mask, rng, sample_probability, sampling_strategy, image_rows, image_cols)
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
        print(f"[sample] {tile_index}/{len(tiles)} {tile_path.name}: +{tile_rows_after - tile_rows_before} rows, total={tile_rows_after}", flush=True)

    if not x_parts:
        raise ValueError("Training sample is empty. Increase sample_probability or verify inputs.")

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
    """Assign train/test labels while preserving sampled location metadata."""
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=test_size, random_state=RANDOM_STATE
    )
    locations = locations.copy()
    locations["split"] = "unused"
    locations.loc[idx_train, "split"] = "train"
    locations.loc[idx_test, "split"] = "test"
    return X_train, X_test, y_train, y_test, locations
