#!/usr/bin/env python3
"""
Offline spatial join + sanity EDA for AlphaEarth embeddings and Sentinel-1 SAR.

Implements exactly two steps:
1) Offline spatial join using nearest-neighbor KD-tree with 10m threshold.
2) Sanity EDA (missingness, SAR histograms, embedding PCA diagnostics).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


MAX_MATCH_DISTANCE_M = 10.0
MERGED_OUTPUT = Path("alphaearth_sentinel1_merged.csv")
EDA_DIR = Path("eda_outputs")


def normalize_colname(col: str) -> str:
    """Normalize column names for robust matching."""
    return re.sub(r"[^a-z0-9]+", "", col.lower())


def find_input_csvs(base_dir: Path) -> Tuple[Path, Path]:
    """
    Locate AlphaEarth and Sentinel-1 CSVs automatically.
    AlphaEarth is inferred via many embedding-like columns (A00, A01, ...).
    SAR is inferred via presence of VV/VH-like column names.
    """
    csv_paths = sorted(
        p
        for p in base_dir.rglob("*.csv")
        if p.name != MERGED_OUTPUT.name and not p.name.startswith(".")
    )
    if not csv_paths:
        raise FileNotFoundError("No CSV files found in the working directory tree.")

    alpha_candidates: List[Tuple[int, Path]] = []
    sar_candidates: List[Tuple[int, Path]] = []

    for path in csv_paths:
        cols = list(pd.read_csv(path, nrows=1).columns)
        normalized_cols = [normalize_colname(c) for c in cols]

        embedding_count = sum(bool(re.fullmatch(r"A\d{1,3}", c)) for c in cols)
        if embedding_count >= 10:
            alpha_candidates.append((embedding_count, path))

        sar_score = 0
        for nc in normalized_cols:
            if "vv" in nc:
                sar_score += 1
            if "vh" in nc:
                sar_score += 1
            if "sar" in nc:
                sar_score += 1
        if sar_score > 0:
            sar_candidates.append((sar_score, path))

    if not alpha_candidates:
        raise ValueError("Could not identify AlphaEarth embedding CSV automatically.")
    if not sar_candidates:
        raise ValueError("Could not identify Sentinel-1 SAR CSV automatically.")

    alpha_path = sorted(alpha_candidates, key=lambda x: x[0], reverse=True)[0][1]
    sar_path = sorted(sar_candidates, key=lambda x: x[0], reverse=True)[0][1]

    if alpha_path == sar_path:
        # Try next-best SAR candidate if top choice collides.
        for _, candidate in sorted(sar_candidates, key=lambda x: x[0], reverse=True):
            if candidate != alpha_path:
                sar_path = candidate
                break
        else:
            raise ValueError("AlphaEarth and SAR CSV resolution collided on the same file.")

    return alpha_path, sar_path


def _parse_geo_point(value: object) -> Tuple[float, float]:
    """
    Parse a GeoJSON Point from a cell string/object.
    Returns (latitude, longitude).
    """
    if pd.isna(value):
        return np.nan, np.nan

    if isinstance(value, str):
        obj = json.loads(value)
    elif isinstance(value, dict):
        obj = value
    else:
        return np.nan, np.nan

    coords = obj.get("coordinates", None)
    if not isinstance(coords, list) or len(coords) < 2:
        return np.nan, np.nan

    lon, lat = coords[0], coords[1]
    return float(lat), float(lon)


def detect_lat_lon_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Automatically detect latitude/longitude columns.
    Falls back to parsing GeoJSON from known geometry columns if needed.
    """
    cols = list(df.columns)
    norm_to_col = {normalize_colname(c): c for c in cols}

    lat_exact = ["lat", "latitude", "y"]
    lon_exact = ["lon", "lng", "long", "longitude", "x"]

    lat_col: Optional[str] = None
    lon_col: Optional[str] = None

    for name in lat_exact:
        if name in norm_to_col:
            lat_col = norm_to_col[name]
            break
    for name in lon_exact:
        if name in norm_to_col:
            lon_col = norm_to_col[name]
            break

    if lat_col is None:
        for c in cols:
            if "lat" in normalize_colname(c):
                lat_col = c
                break
    if lon_col is None:
        for c in cols:
            nc = normalize_colname(c)
            if "lon" in nc or "lng" in nc or "long" in nc:
                lon_col = c
                break

    if lat_col is not None and lon_col is not None:
        return df, lat_col, lon_col

    geo_candidates = [c for c in cols if normalize_colname(c) in {".geo", "geo", "geometry", "geom"} or c in {".geo"}]
    if not geo_candidates and ".geo" in cols:
        geo_candidates = [".geo"]

    for geo_col in geo_candidates:
        parsed = df[geo_col].map(_parse_geo_point)
        parsed_df = pd.DataFrame(parsed.tolist(), columns=["__lat_auto__", "__lon_auto__"], index=df.index)
        if parsed_df["__lat_auto__"].notna().sum() > 0 and parsed_df["__lon_auto__"].notna().sum() > 0:
            out = df.copy()
            out["__lat_auto__"] = parsed_df["__lat_auto__"]
            out["__lon_auto__"] = parsed_df["__lon_auto__"]
            return out, "__lat_auto__", "__lon_auto__"

    raise ValueError("Could not detect latitude/longitude columns or parse geometry column.")


def detect_embedding_columns(df: pd.DataFrame) -> List[str]:
    """Identify embedding columns automatically."""
    cols = list(df.columns)
    embedding_cols = [c for c in cols if re.fullmatch(r"A\d{1,3}", c)]
    if embedding_cols:
        return sorted(embedding_cols)

    # Fallback: columns explicitly containing "embedding"
    embedding_cols = [c for c in cols if "embedding" in c.lower()]
    if embedding_cols:
        return embedding_cols

    raise ValueError("Could not identify embedding columns automatically.")


def compute_planar_xy_m(lat: np.ndarray, lon: np.ndarray, lat0_rad: float) -> np.ndarray:
    """
    Approximate local planar projection in meters for KD-tree matching.
    Uses equirectangular projection around shared reference latitude.
    """
    earth_r = 6_371_000.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = earth_r * lon_rad * np.cos(lat0_rad)
    y = earth_r * lat_rad
    return np.column_stack([x, y])


def make_spatial_join(alpha_df: pd.DataFrame, sar_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Perform nearest-neighbor spatial join and filter matches > 10m."""
    alpha_df, a_lat_col, a_lon_col = detect_lat_lon_columns(alpha_df)
    sar_df, s_lat_col, s_lon_col = detect_lat_lon_columns(sar_df)

    alpha_lat = pd.to_numeric(alpha_df[a_lat_col], errors="coerce")
    alpha_lon = pd.to_numeric(alpha_df[a_lon_col], errors="coerce")
    sar_lat = pd.to_numeric(sar_df[s_lat_col], errors="coerce")
    sar_lon = pd.to_numeric(sar_df[s_lon_col], errors="coerce")

    alpha_valid = alpha_df.loc[alpha_lat.notna() & alpha_lon.notna()].copy()
    sar_valid = sar_df.loc[sar_lat.notna() & sar_lon.notna()].copy()

    alpha_lat_valid = pd.to_numeric(alpha_valid[a_lat_col], errors="coerce").to_numpy()
    alpha_lon_valid = pd.to_numeric(alpha_valid[a_lon_col], errors="coerce").to_numpy()
    sar_lat_valid = pd.to_numeric(sar_valid[s_lat_col], errors="coerce").to_numpy()
    sar_lon_valid = pd.to_numeric(sar_valid[s_lon_col], errors="coerce").to_numpy()

    lat0 = np.deg2rad(np.nanmean(np.concatenate([alpha_lat_valid, sar_lat_valid])))
    alpha_xy = compute_planar_xy_m(alpha_lat_valid, alpha_lon_valid, lat0)
    sar_xy = compute_planar_xy_m(sar_lat_valid, sar_lon_valid, lat0)

    tree = cKDTree(sar_xy)
    match_dist_m, sar_idx = tree.query(alpha_xy, k=1)
    keep_mask = match_dist_m <= MAX_MATCH_DISTANCE_M

    matched_alpha = alpha_valid.iloc[np.where(keep_mask)[0]].reset_index(drop=True).copy()
    matched_sar = sar_valid.iloc[sar_idx[keep_mask]].reset_index(drop=True).copy()
    matched_dist = match_dist_m[keep_mask]

    # Preserve both coordinate sets explicitly.
    matched_alpha = matched_alpha.rename(columns={a_lat_col: "alpha_latitude", a_lon_col: "alpha_longitude"})
    matched_sar = matched_sar.rename(columns={s_lat_col: "sar_latitude", s_lon_col: "sar_longitude"})

    # Keep AlphaEarth embeddings (and alpha coords), plus SAR variables (and SAR coords).
    embedding_cols = detect_embedding_columns(matched_alpha)
    alpha_keep_cols = ["alpha_latitude", "alpha_longitude"] + embedding_cols
    alpha_subset = matched_alpha[alpha_keep_cols].copy()

    # Remove geometry-like columns from SAR table and prevent name collisions.
    sar_drop_cols = [c for c in matched_sar.columns if normalize_colname(c) in {"geo", "geometry", "geom"} or c == ".geo"]
    sar_subset = matched_sar.drop(columns=sar_drop_cols, errors="ignore").copy()

    if "sar_latitude" not in sar_subset.columns or "sar_longitude" not in sar_subset.columns:
        raise ValueError("SAR coordinates missing after preprocessing.")

    # Rename overlapping SAR columns except coordinates already renamed.
    overlap = set(alpha_subset.columns).intersection(set(sar_subset.columns))
    for col in overlap:
        if col not in {"sar_latitude", "sar_longitude"}:
            sar_subset = sar_subset.rename(columns={col: f"sar_{col}"})

    merged = pd.concat([alpha_subset, sar_subset.reset_index(drop=True)], axis=1)
    merged["match_distance_m"] = matched_dist

    report = {
        "n_alpha_points": float(len(alpha_df)),
        "n_sar_points": float(len(sar_df)),
        "n_matched_pairs": float(keep_mask.sum()),
        "distance_min_m": float(np.min(matched_dist)) if len(matched_dist) else np.nan,
        "distance_median_m": float(np.median(matched_dist)) if len(matched_dist) else np.nan,
        "distance_max_m": float(np.max(matched_dist)) if len(matched_dist) else np.nan,
    }
    return merged, report


def select_sar_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Identify VV, VH, and VV/VH ratio columns automatically (if available)."""
    cols = list(df.columns)
    normalized = {c: normalize_colname(c) for c in cols}

    vv_col = None
    vh_col = None
    ratio_col = None

    for c, nc in normalized.items():
        if "vv" in nc and "vh" not in nc and "div" not in nc and "ratio" not in nc:
            vv_col = c
            break
    for c, nc in normalized.items():
        if "vh" in nc and "vv" not in nc and "div" not in nc and "ratio" not in nc:
            vh_col = c
            break
    for c, nc in normalized.items():
        if ("vv" in nc and "vh" in nc) or "ratio" in nc:
            ratio_col = c
            break

    return vv_col, vh_col, ratio_col


def save_histogram(series: pd.Series, title: str, output_path: Path) -> None:
    """Save a histogram PNG for a numeric series."""
    data = pd.to_numeric(series, errors="coerce").dropna()
    plt.figure(figsize=(7, 4.5))
    plt.hist(data, bins=40, edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_sanity_eda(merged: pd.DataFrame) -> Dict[str, object]:
    """Run only the requested sanity EDA tasks."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Missing value checks
    missing_counts = merged.isna().sum()
    rows_with_any_missing = int(merged.isna().any(axis=1).sum())

    # 2) SAR variable histograms
    vv_col, vh_col, ratio_col = select_sar_columns(merged)
    saved_histograms: List[str] = []

    if vv_col is not None:
        p = EDA_DIR / "hist_vv.png"
        save_histogram(merged[vv_col], f"Histogram: {vv_col}", p)
        saved_histograms.append(str(p))
    if vh_col is not None:
        p = EDA_DIR / "hist_vh.png"
        save_histogram(merged[vh_col], f"Histogram: {vh_col}", p)
        saved_histograms.append(str(p))
    if ratio_col is not None:
        p = EDA_DIR / "hist_vv_vh_ratio.png"
        save_histogram(merged[ratio_col], f"Histogram: {ratio_col}", p)
        saved_histograms.append(str(p))

    # 3) Embedding diagnostics via PCA
    embedding_cols = detect_embedding_columns(merged)
    emb = merged[embedding_cols].apply(pd.to_numeric, errors="coerce")
    emb = emb.dropna(axis=0, how="any")
    if emb.empty:
        raise ValueError("No complete embedding rows available for PCA after dropping NaNs.")

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)
    pca = PCA()
    pca.fit(emb_scaled)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    comps = np.arange(1, len(cumulative) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(comps, cumulative, marker="o", markersize=3)
    plt.title("PCA Cumulative Explained Variance (Embeddings)")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.ylim(0, 1.01)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    pca_plot_path = EDA_DIR / "pca_cumulative_explained_variance.png"
    plt.savefig(pca_plot_path, dpi=150)
    plt.close()

    def variance_at(k: int) -> float:
        if len(cumulative) == 0:
            return np.nan
        idx = min(k, len(cumulative)) - 1
        return float(cumulative[idx])

    report = {
        "missing_counts": missing_counts,
        "rows_with_any_missing": rows_with_any_missing,
        "vv_col": vv_col,
        "vh_col": vh_col,
        "ratio_col": ratio_col,
        "saved_histograms": saved_histograms,
        "pca_plot": str(pca_plot_path),
        "var_explained_first_5": variance_at(5),
        "var_explained_first_10": variance_at(10),
        "var_explained_first_20": variance_at(20),
        "n_embedding_cols": len(embedding_cols),
        "n_rows_for_pca": int(len(emb)),
    }
    return report


def print_summary(alpha_path: Path, sar_path: Path, join_report: Dict[str, float], eda_report: Dict[str, object]) -> None:
    """Print concise textual summary to console."""
    print("\n=== Offline Spatial Join Summary ===")
    print(f"AlphaEarth file: {alpha_path}")
    print(f"Sentinel-1 file: {sar_path}")
    print(f"AlphaEarth points: {int(join_report['n_alpha_points'])}")
    print(f"SAR points: {int(join_report['n_sar_points'])}")
    print(f"Matched pairs (<= {MAX_MATCH_DISTANCE_M:.0f}m): {int(join_report['n_matched_pairs'])}")
    print(
        "Match distance (m) [min / median / max]: "
        f"{join_report['distance_min_m']:.4f} / {join_report['distance_median_m']:.4f} / {join_report['distance_max_m']:.4f}"
    )
    print(f"Merged output: {MERGED_OUTPUT}")

    print("\n=== Sanity EDA Summary ===")
    print("Missing values per column:")
    print(eda_report["missing_counts"].to_string())
    print(f"\nRows with any missing values: {eda_report['rows_with_any_missing']}")

    print("\nSAR histogram columns used:")
    print(f"VV: {eda_report['vv_col']}")
    print(f"VH: {eda_report['vh_col']}")
    print(f"VV/VH ratio: {eda_report['ratio_col']}")
    print("Saved histogram files:")
    for p in eda_report["saved_histograms"]:
        print(f" - {p}")

    print("\nEmbedding PCA diagnostics:")
    print(f"Embedding columns detected: {eda_report['n_embedding_cols']}")
    print(f"Rows used for PCA: {eda_report['n_rows_for_pca']}")
    print(f"Cumulative variance (first 5 components): {eda_report['var_explained_first_5']:.6f}")
    print(f"Cumulative variance (first 10 components): {eda_report['var_explained_first_10']:.6f}")
    print(f"Cumulative variance (first 20 components): {eda_report['var_explained_first_20']:.6f}")
    print(f"PCA plot: {eda_report['pca_plot']}")


def main() -> None:
    alpha_path, sar_path = find_input_csvs(Path("."))

    alpha_df = pd.read_csv(alpha_path)
    sar_df = pd.read_csv(sar_path)

    merged, join_report = make_spatial_join(alpha_df, sar_df)
    merged.to_csv(MERGED_OUTPUT, index=False)

    eda_report = run_sanity_eda(merged)
    print_summary(alpha_path, sar_path, join_report, eda_report)


if __name__ == "__main__":
    main()
