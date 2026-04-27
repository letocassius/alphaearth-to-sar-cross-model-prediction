"""Raster and path utilities for the AlphaEarth-to-SAR pipeline."""

from pathlib import Path
import re

import rasterio
from rasterio.windows import Window

OFFSET_PATTERN = re.compile(r".*-(?P<row>\d+)-(?P<col>\d+)\.tif$")


def discover_full_stack_tiles(full_stack_dir: Path, full_stack_glob: str) -> list[Path]:
    """Return colocated SAR + AlphaEarth GeoTIFF tiles matching the glob."""
    tiles = sorted(full_stack_dir.glob(full_stack_glob))
    if not tiles:
        raise FileNotFoundError(f"No full-stack tiles matching {full_stack_glob} in {full_stack_dir}")
    return tiles


def iter_windows(height: int, width: int, chunk_size: int) -> list[Window]:
    """Create rasterio windows that cover an image in chunks."""
    windows: list[Window] = []
    for row_off in range(0, height, chunk_size):
        win_h = min(chunk_size, height - row_off)
        for col_off in range(0, width, chunk_size):
            win_w = min(chunk_size, width - col_off)
            windows.append(Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h))
    return windows


def parse_tile_offsets(tile_path: Path) -> tuple[int, int]:
    """Parse row/column offsets from a tiled filename, if present."""
    match = OFFSET_PATTERN.search(tile_path.name)
    if not match:
        return 0, 0
    return int(match.group("row")), int(match.group("col"))


def sarhat_output_name(tile_path: Path) -> str:
    """Build the predicted SAR GeoTIFF filename from the full-stack tile name."""
    stem = tile_path.stem.replace("sentinel1_alphaearth", "sar_hat_from_alphaearth", 1)
    return f"{stem}.tif"


def read_three_band_raster(path: Path) -> tuple:
    """Read the first three bands from a raster as float32 and return data plus profile."""
    with rasterio.open(path) as src:
        data = src.read([1, 2, 3]).astype("float32")
        profile = src.profile.copy()
    return data, profile
