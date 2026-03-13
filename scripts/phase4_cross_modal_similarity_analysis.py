#!/usr/bin/env python3
"""
Phase 4 cross-modal similarity analysis for AlphaEarth-to-SAR.

This script quantifies whether nearest neighbors in embedding space align with
nearest neighbors in SAR signature space and exports summary tables, plots, and
representative query diagnostics for reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pearsonr, spearmanr
from rasterio.windows import Window
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "DataSources" / "alphaearth_s1_dw_samples_all_regions_2024.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "full_dataset"
SENTINEL2_DIR = ROOT_DIR / "DataSources" / "sentinel2_context"
EMBEDDING_COLS = [f"A{i:02d}" for i in range(64)]
SAR_COLS = ["S1_VV", "S1_VH", "S1_VV_div_VH"]
K_VALUES = [5, 10, 20]
MAX_K = max(K_VALUES)
RANDOM_SEED = 42
OVERALL_PAIR_SAMPLE = 200_000
DW_LABEL_NAMES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["dw_label"] = df["dw_label"].astype(int)
    df["dw_label_name"] = df["dw_label"].map(DW_LABEL_NAMES)
    return df


def compute_neighbors(distance_matrix: np.ndarray, max_k: int) -> np.ndarray:
    order = np.argsort(distance_matrix, axis=1)
    return order[:, 1 : max_k + 1]


def compute_overlap_scores(embedding_neighbors: np.ndarray, sar_neighbors: np.ndarray, k: int) -> np.ndarray:
    overlaps = np.empty(embedding_neighbors.shape[0], dtype=float)
    for idx in range(embedding_neighbors.shape[0]):
        overlaps[idx] = (
            len(set(embedding_neighbors[idx, :k]).intersection(set(sar_neighbors[idx, :k]))) / float(k)
        )
    return overlaps


def summarize_overlap(frame: pd.DataFrame, group_cols: list[str], overlap_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in frame.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "n_queries": int(len(part)),
                "mean_overlap": float(part[overlap_col].mean()),
                "median_overlap": float(part[overlap_col].median()),
                "p10_overlap": float(part[overlap_col].quantile(0.10)),
                "p90_overlap": float(part[overlap_col].quantile(0.90)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def distance_correlation_summary(
    embedding_dist: np.ndarray,
    sar_dist: np.ndarray,
    indices: np.ndarray | None = None,
    sample_size: int | None = None,
) -> dict[str, float]:
    if indices is None:
        sub_embedding = embedding_dist
        sub_sar = sar_dist
    else:
        sub_embedding = embedding_dist[np.ix_(indices, indices)]
        sub_sar = sar_dist[np.ix_(indices, indices)]

    upper = np.triu_indices(sub_embedding.shape[0], k=1)
    emb_values = sub_embedding[upper]
    sar_values = sub_sar[upper]
    if sample_size is not None and len(emb_values) > sample_size:
        rng = np.random.default_rng(RANDOM_SEED)
        keep = rng.choice(len(emb_values), size=sample_size, replace=False)
        emb_values = emb_values[keep]
        sar_values = sar_values[keep]

    pearson_stats = pearsonr(emb_values, sar_values)
    spearman_stats = spearmanr(emb_values, sar_values)
    return {
        "n_pairs": int(len(emb_values)),
        "pearson_r": float(pearson_stats.statistic),
        "pearson_p_value": float(pearson_stats.pvalue),
        "spearman_rho": float(spearman_stats.statistic),
        "spearman_p_value": float(spearman_stats.pvalue),
    }


def save_overlap_curve(overall_overlap: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(overall_overlap["k"], overall_overlap["mean_overlap"], marker="o", linewidth=2, color="#1d4ed8")
    for row in overall_overlap.itertuples():
        plt.text(row.k, row.mean_overlap + 0.01, f"{row.mean_overlap:.3f}", ha="center", fontsize=9)
    plt.ylim(0.0, 1.0)
    plt.xlabel("k")
    plt.ylabel("Mean overlap")
    plt.title("Phase 4: Embedding vs SAR k-NN Overlap")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase4_knn_overlap_by_k.png", dpi=220)
    plt.close()


def save_land_use_overlap_bar(land_use_overlap: pd.DataFrame) -> None:
    plot_df = land_use_overlap[land_use_overlap["k"] == 10].sort_values("mean_overlap", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["dw_label_name"], plot_df["mean_overlap"], color="#0891b2")
    plt.xlabel("Mean overlap at k=10")
    plt.ylabel("Land use")
    plt.title("Phase 4: Cross-modal Similarity by Land Use")
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase4_knn_overlap_by_land_use_k10.png", dpi=220)
    plt.close()


def save_distance_scatter(embedding_dist: np.ndarray, sar_dist: np.ndarray, summary: dict[str, float]) -> None:
    upper = np.triu_indices(embedding_dist.shape[0], k=1)
    emb_values = embedding_dist[upper]
    sar_values = sar_dist[upper]
    rng = np.random.default_rng(RANDOM_SEED)
    keep = rng.choice(len(emb_values), size=min(25_000, len(emb_values)), replace=False)
    emb_values = emb_values[keep]
    sar_values = sar_values[keep]

    plt.figure(figsize=(7, 6))
    plt.scatter(emb_values, sar_values, s=5, alpha=0.20, color="#1d4ed8", edgecolor="none")
    plt.xlabel("Embedding cosine distance")
    plt.ylabel("SAR Euclidean distance (scaled features)")
    plt.title("Phase 4: Embedding Distance vs SAR Distance")
    plt.text(
        0.03,
        0.97,
        f"Pearson r = {summary['pearson_r']:.3f}\nSpearman rho = {summary['spearman_rho']:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase4_embedding_vs_sar_distance.png", dpi=220)
    plt.close()


def format_neighbor_labels(query_df: pd.DataFrame, neighbor_indices: np.ndarray) -> str:
    labels = []
    for idx in neighbor_indices:
        row = query_df.iloc[int(idx)]
        labels.append(f"{row['dw_label_name']}@{row['region']}")
    return ", ".join(labels)


def build_representative_queries(
    df: pd.DataFrame,
    overlap_at_10: np.ndarray,
    embedding_neighbors: np.ndarray,
    sar_neighbors: np.ndarray,
) -> pd.DataFrame:
    frame = df.copy()
    frame["overlap_at_10"] = overlap_at_10

    selections = [
        ("highest_overlap", frame["overlap_at_10"].idxmax()),
        ("lowest_overlap", frame["overlap_at_10"].idxmin()),
    ]
    for label in ["water", "crops"]:
        part = frame[frame["dw_label_name"] == label]
        if part.empty:
            continue
        median_rank = part["overlap_at_10"].sub(part["overlap_at_10"].median()).abs().idxmin()
        selections.append((f"{label}_representative", median_rank))

    rows: list[dict[str, Any]] = []
    seen = set()
    for reason, idx in selections:
        if idx in seen:
            continue
        seen.add(idx)
        query = frame.loc[idx]
        rows.append(
            {
                "reason": reason,
                "system:index": query["system:index"],
                "region": query["region"],
                "dw_label_name": query["dw_label_name"],
                "latitude": float(query["latitude"]),
                "longitude": float(query["longitude"]),
                "overlap_at_10": float(query["overlap_at_10"]),
                "query_index": int(idx),
                "embedding_top1_index": int(embedding_neighbors[idx, 0]),
                "sar_top1_index": int(sar_neighbors[idx, 0]),
                "embedding_neighbor_labels": format_neighbor_labels(frame, embedding_neighbors[idx, :5]),
                "sar_neighbor_labels": format_neighbor_labels(frame, sar_neighbors[idx, :5]),
            }
        )
    return pd.DataFrame(rows)


def save_representative_query_figure(rep_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(top=0.90, left=0.05, right=0.95, bottom=0.06, hspace=0.18, wspace=0.12)
    fig.suptitle("Phase 4: Representative Query Diagnostics", fontsize=18, fontweight="bold")
    for ax, row in zip(axes.flatten(), rep_df.itertuples()):
        ax.axis("off")
        lines = [
            f"Reason: {row.reason}",
            f"Query: {row.dw_label_name} @ {row.region}",
            f"Overlap@10: {row.overlap_at_10:.3f}",
            "",
            "Embedding NN labels:",
            row.embedding_neighbor_labels,
            "",
            "SAR NN labels:",
            row.sar_neighbor_labels,
        ]
        y = 0.95
        for line in lines:
            ax.text(0.0, y, line, va="top", fontsize=10, wrap=True)
            y -= 0.10 if line == "" else 0.09
    for ax in axes.flatten()[len(rep_df) :]:
        ax.axis("off")
    plt.savefig(OUTPUT_DIR / "phase4_representative_queries.png", dpi=220)
    plt.close(fig)


def load_sentinel2_chip(region: str, latitude: float, longitude: float, chip_size: int = 96) -> np.ndarray:
    tif_path = SENTINEL2_DIR / f"sentinel2_context_{region}_2024.tif"
    with rasterio.open(tif_path) as src:
        row, col = src.index(longitude, latitude)
        half = chip_size // 2
        window = Window(col - half, row - half, chip_size, chip_size)
        chip = src.read([1, 2, 3], window=window, boundless=True, fill_value=np.nan)

    chip = np.moveaxis(chip, 0, -1)
    finite = chip[np.isfinite(chip)]
    if finite.size == 0:
        return np.zeros((chip_size, chip_size, 3), dtype=float)

    lo, hi = np.nanpercentile(chip, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(chip)), float(np.nanmax(chip))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros((chip_size, chip_size, 3), dtype=float)

    chip = np.nan_to_num((chip - lo) / (hi - lo), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(chip, 0.0, 1.0)


def save_sentinel2_neighbor_chips(rep_df: pd.DataFrame, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(rep_df), 3, figsize=(10, 3.2 * len(rep_df)))
    fig.subplots_adjust(top=0.94, left=0.05, right=0.98, bottom=0.04, hspace=0.18, wspace=0.08)
    fig.suptitle("Phase 4: Sentinel-2 Query and Neighbor Chips", fontsize=18, fontweight="bold")

    if len(rep_df) == 1:
        axes = np.asarray([axes])

    for row_idx, rep in enumerate(rep_df.itertuples()):
        query = df.iloc[rep.query_index]
        emb_nn = df.iloc[rep.embedding_top1_index]
        sar_nn = df.iloc[rep.sar_top1_index]
        tiles = [
            ("Query", query, rep.overlap_at_10),
            ("Top embedding NN", emb_nn, rep.overlap_at_10),
            ("Top SAR NN", sar_nn, rep.overlap_at_10),
        ]
        for col_idx, (title, item, overlap_value) in enumerate(tiles):
            ax = axes[row_idx, col_idx]
            chip = load_sentinel2_chip(item["region"], float(item["latitude"]), float(item["longitude"]))
            ax.imshow(chip)
            subtitle = f"{item['dw_label_name']} @ {item['region']}"
            if col_idx == 0:
                subtitle += f"\nOverlap@10={overlap_value:.3f}"
            ax.set_title(f"{title}\n{subtitle}", fontsize=9)
            ax.axis("off")

    plt.savefig(OUTPUT_DIR / "phase4_sentinel2_neighbor_chips.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    embedding = df[EMBEDDING_COLS].to_numpy(dtype=float)
    sar = StandardScaler().fit_transform(df[SAR_COLS].to_numpy(dtype=float))

    embedding_dist = pairwise_distances(embedding, metric="cosine")
    sar_dist = pairwise_distances(sar, metric="euclidean")
    np.fill_diagonal(embedding_dist, np.inf)
    np.fill_diagonal(sar_dist, np.inf)

    embedding_neighbors = compute_neighbors(embedding_dist, MAX_K)
    sar_neighbors = compute_neighbors(sar_dist, MAX_K)

    row_overlap = pd.DataFrame(
        {
            "system:index": df["system:index"],
            "region": df["region"],
            "dw_label_name": df["dw_label_name"],
            "latitude": df["latitude"],
            "longitude": df["longitude"],
        }
    )

    overall_rows: list[dict[str, Any]] = []
    land_use_frames: list[pd.DataFrame] = []
    region_frames: list[pd.DataFrame] = []
    for k in K_VALUES:
        overlap_scores = compute_overlap_scores(embedding_neighbors, sar_neighbors, k)
        overlap_col = f"overlap_at_{k}"
        row_overlap[overlap_col] = overlap_scores
        overall_rows.append(
            {
                "k": k,
                "n_queries": int(len(overlap_scores)),
                "mean_overlap": float(np.mean(overlap_scores)),
                "median_overlap": float(np.median(overlap_scores)),
                "p10_overlap": float(np.quantile(overlap_scores, 0.10)),
                "p90_overlap": float(np.quantile(overlap_scores, 0.90)),
            }
        )
        land_use_summary = summarize_overlap(row_overlap, ["dw_label_name"], overlap_col)
        land_use_summary.insert(0, "k", k)
        land_use_frames.append(land_use_summary)

        region_summary = summarize_overlap(row_overlap, ["region"], overlap_col)
        region_summary.insert(0, "k", k)
        region_frames.append(region_summary)

    overall_overlap = pd.DataFrame(overall_rows)
    land_use_overlap = pd.concat(land_use_frames, ignore_index=True).sort_values(["k", "mean_overlap"], ascending=[True, False])
    region_overlap = pd.concat(region_frames, ignore_index=True).sort_values(["k", "mean_overlap"], ascending=[True, False])

    overall_corr = distance_correlation_summary(embedding_dist, sar_dist, sample_size=OVERALL_PAIR_SAMPLE)
    overall_corr.update({"scope": "overall", "group": "all"})
    corr_rows = [overall_corr]

    for group_col in ["region", "dw_label_name"]:
        for group_name, part in df.groupby(group_col, sort=False):
            indices = part.index.to_numpy()
            stats = distance_correlation_summary(embedding_dist, sar_dist, indices=indices)
            stats.update({"scope": group_col, "group": group_name})
            corr_rows.append(stats)

    distance_correlation = pd.DataFrame(corr_rows)

    rep_queries = build_representative_queries(df, row_overlap["overlap_at_10"].to_numpy(), embedding_neighbors, sar_neighbors)
    rep_queries.to_csv(OUTPUT_DIR / "phase4_representative_queries.csv", index=False)

    overall_overlap.to_csv(OUTPUT_DIR / "phase4_knn_overlap_overall.csv", index=False)
    land_use_overlap.to_csv(OUTPUT_DIR / "phase4_knn_overlap_by_land_use.csv", index=False)
    region_overlap.to_csv(OUTPUT_DIR / "phase4_knn_overlap_by_region.csv", index=False)
    distance_correlation.to_csv(OUTPUT_DIR / "phase4_distance_correlation.csv", index=False)
    row_overlap.to_csv(OUTPUT_DIR / "phase4_query_overlap_scores.csv", index=False)

    save_overlap_curve(overall_overlap)
    save_land_use_overlap_bar(land_use_overlap)
    save_distance_scatter(embedding_dist, sar_dist, overall_corr)
    save_representative_query_figure(rep_queries)
    save_sentinel2_neighbor_chips(rep_queries, df)

    summary_payload = {
        "sentinel2_context_present": bool(SENTINEL2_DIR.exists()),
        "sentinel2_tif_count": len(list(SENTINEL2_DIR.glob("*.tif"))),
        "sentinel2_chip_figure_generated": True,
        "embedding_neighbors_k_values": K_VALUES,
        "overall_overlap": json.loads(overall_overlap.to_json(orient="records")),
        "best_land_use_at_k10": json.loads(
            land_use_overlap[land_use_overlap["k"] == 10].sort_values("mean_overlap", ascending=False).head(3).to_json(orient="records")
        ),
        "worst_land_use_at_k10": json.loads(
            land_use_overlap[land_use_overlap["k"] == 10].sort_values("mean_overlap", ascending=True).head(3).to_json(orient="records")
        ),
        "overall_distance_correlation": overall_corr,
        "qualitative_note": "Sentinel-2 context rasters were used to extract representative RGB chips for the query pixel, the top embedding-space neighbor, and the top SAR-space neighbor.",
    }
    (OUTPUT_DIR / "phase4_summary.json").write_text(json.dumps(summary_payload, indent=2))

    print("Saved Phase 4 outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
