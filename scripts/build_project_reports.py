#!/usr/bin/env python3
"""
Build phase-specific summary PDFs and a cumulative project summary PDF.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs" / "full_dataset"
REPORTS_DIR = ROOT_DIR / "reports"
PHASE2_REPORT_PATH = OUTPUT_DIR / "phase2_modeling_summary_report.pdf"
PHASE3_REPORT_PATH = OUTPUT_DIR / "phase3_failure_analysis_summary_report.pdf"
PHASE4_REPORT_PATH = OUTPUT_DIR / "phase4_cross_modal_similarity_summary_report.pdf"
PROJECT_REPORT_PATH = OUTPUT_DIR / "project_summary_report.pdf"
TOP_LEVEL_COPIES = {
    PHASE2_REPORT_PATH: REPORTS_DIR / "phase2_modeling_summary_report.pdf",
    PHASE3_REPORT_PATH: REPORTS_DIR / "phase3_failure_analysis_summary_report.pdf",
    PHASE4_REPORT_PATH: REPORTS_DIR / "phase4_cross_modal_similarity_summary_report.pdf",
    PROJECT_REPORT_PATH: REPORTS_DIR / "project_summary_report.pdf",
}
LEGACY_COMBINED_COPY = REPORTS_DIR / "phase2_full_dataset_lightgbm_report.pdf"
LEGACY_COMBINED_OUTPUT_COPY = OUTPUT_DIR / "phase2_full_dataset_lightgbm_report.pdf"

METRICS_PATH = OUTPUT_DIR / "full_dataset_lightgbm_metrics.csv"
STABILITY_PATH = OUTPUT_DIR / "full_dataset_lightgbm_stability.csv"
REGIONAL_PATH = OUTPUT_DIR / "full_dataset_lightgbm_regional_metrics.csv"
LAND_USE_PATH = OUTPUT_DIR / "full_dataset_lightgbm_land_use_metrics.csv"
PHASE3_SELECTION_PATH = OUTPUT_DIR / "phase3_best_model_selection.csv"
PHASE3_LAND_USE_PATH = OUTPUT_DIR / "phase3_land_use_error.csv"
PHASE3_SPATIAL_PATH = OUTPUT_DIR / "phase3_spatial_autocorrelation.csv"
PHASE3_OUTLIERS_PATH = OUTPUT_DIR / "phase3_outliers_S1_VV.csv"
PHASE4_OVERALL_PATH = OUTPUT_DIR / "phase4_knn_overlap_overall.csv"
PHASE4_LAND_USE_PATH = OUTPUT_DIR / "phase4_knn_overlap_by_land_use.csv"
PHASE4_REGION_PATH = OUTPUT_DIR / "phase4_knn_overlap_by_region.csv"
PHASE4_CORR_PATH = OUTPUT_DIR / "phase4_distance_correlation.csv"
PHASE4_QUERIES_PATH = OUTPUT_DIR / "phase4_representative_queries.csv"
TARGETS = ["S1_VV", "S1_VH", "S1_VV_div_VH"]
FEATURE_LABELS = {
    "embedding_only": "Embeddings only",
    "embedding_plus_context": "Embeddings + region + Dynamic World",
}


def draw_text_page(title: str, lines: list[str], pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
    ax.axis("off")
    ax.text(0.0, 0.98, title, fontsize=22, fontweight="bold", va="top")
    y = 0.90
    for line in lines:
        if line == "":
            y -= 0.03
            continue
        wrapped = wrap_line(line, width=82 if line.startswith("  ") else 92, preserve_indent=line.startswith("  "))
        ax.text(0.0, y, wrapped, fontsize=11, va="top", family="monospace" if line.startswith("  ") else None)
        y -= 0.035 * (wrapped.count("\n") + 1)
    pdf.savefig(fig)
    plt.close(fig)


def draw_dataframe_page(title: str, df: pd.DataFrame, pdf: PdfPages, footnote: str | None = None) -> None:
    prepared = prepare_table_dataframe(df)
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.82])
    ax.axis("off")
    ax.text(0.0, 1.04, title, fontsize=20, fontweight="bold", va="bottom", transform=ax.transAxes)
    table = ax.table(
        cellText=prepared["df"].values,
        colLabels=prepared["columns"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.06, 0.98, 0.88],
        colWidths=prepared["col_widths"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(prepared["font_size"])
    table.scale(1, prepared["row_scale"])
    for (row, col), cell in table.get_celld().items():
        cell.get_text().set_wrap(True)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dbeafe")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
        cell.PAD = 0.02
    if footnote:
        ax.text(0.0, 0.0, wrap_line(footnote, width=150), fontsize=8.5, color="#444444", transform=ax.transAxes, va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def draw_image_grid_page(title: str, image_paths: list[Path], pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.subplots_adjust(top=0.90, left=0.05, right=0.95, bottom=0.06, hspace=0.12, wspace=0.08)
    fig.suptitle(title, fontsize=20, fontweight="bold")
    flat_axes = axes.flatten()
    for ax, image_path in zip(flat_axes, image_paths):
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(image_path.stem.replace("_", " "), fontsize=10)
        ax.axis("off")
    for ax in flat_axes[len(image_paths):]:
        ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def format_float_columns(df: pd.DataFrame, columns: list[str], decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].map(lambda value: f"{value:.{decimals}f}")
    return out


def format_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    table = metrics_df.copy()
    table["feature_set"] = table["feature_set"].map(FEATURE_LABELS)
    table = format_float_columns(
        table,
        [
            "cv_r2_mean",
            "cv_rmse_mean",
            "stability_r2_mean",
            "stability_rmse_mean",
            "r2",
            "rmse",
            "mae",
            "pearson_r",
        ],
    )
    return table[
        ["target", "feature_set", "cv_r2_mean", "cv_rmse_mean", "stability_r2_mean", "stability_rmse_mean", "r2", "rmse", "mae", "pearson_r"]
    ]


def wrap_line(text: object, width: int, preserve_indent: bool = False) -> str:
    raw = str(text)
    if raw == "":
        return raw
    indent = ""
    content = raw
    if preserve_indent:
        indent = raw[: len(raw) - len(raw.lstrip(" "))]
        content = raw.lstrip(" ")
    wrapped_lines = textwrap.wrap(content, width=max(width, 8), break_long_words=False, break_on_hyphens=False) or [content]
    if preserve_indent:
        return "\n".join(indent + line if idx == 0 else " " * len(indent) + line for idx, line in enumerate(wrapped_lines))
    return "\n".join(wrapped_lines)


def prepare_table_dataframe(df: pd.DataFrame) -> dict[str, object]:
    ncols = len(df.columns)
    header_budget = max(10, min(24, int(90 / max(ncols, 1))))
    cell_budget = max(10, min(28, int(100 / max(ncols, 1))))

    wrapped_cols = [wrap_line(col, width=header_budget) for col in df.columns]
    wrapped_df = df.copy()
    for col in wrapped_df.columns:
        series_as_text = wrapped_df[col].astype(str)
        max_len = int(series_as_text.map(len).max()) if len(series_as_text) else len(str(col))
        width = max(8, min(32, max(cell_budget, int(max_len ** 0.5 * 4))))
        if max_len > width:
            wrapped_df[col] = series_as_text.map(lambda value, current_width=width: wrap_line(value, width=current_width))
        else:
            wrapped_df[col] = series_as_text

    raw_lengths = []
    for col in df.columns:
        raw_lengths.append(max(len(str(col)), int(df[col].astype(str).map(len).quantile(0.85)) if len(df[col]) else len(str(col))))
    total = float(sum(raw_lengths)) if raw_lengths else 1.0
    col_widths = [0.98 * (length / total) for length in raw_lengths]

    if ncols >= 9:
        font_size = 7.2
        row_scale = 1.8
    elif ncols >= 7:
        font_size = 8.0
        row_scale = 1.6
    else:
        font_size = 8.6
        row_scale = 1.45

    if any("\n" in value for value in wrapped_df.astype(str).to_numpy().flatten()):
        row_scale += 0.25

    return {
        "df": wrapped_df,
        "columns": wrapped_cols,
        "col_widths": col_widths,
        "font_size": font_size,
        "row_scale": row_scale,
    }


def build_feature_lines() -> list[str]:
    lines = ["Top 5 features by permutation importance", ""]
    for feature_set in FEATURE_LABELS:
        lines.append(FEATURE_LABELS[feature_set])
        for target in TARGETS:
            importance = pd.read_csv(OUTPUT_DIR / f"feature_importance_{feature_set}_{target}_lightgbm.csv").head(5)
            lines.append(
                f"  {target}: " + ", ".join(f"{row.feature} ({row.permutation_mean:.3f})" for row in importance.itertuples())
            )
        lines.append("")
    return lines


def knn_overlap_baseline(n_queries: int, k: int) -> float:
    if n_queries <= 1:
        return 0.0
    return float(k / (n_queries - 1))


def build_phase2_summary_lines(metrics_df: pd.DataFrame, stability_df: pd.DataFrame) -> list[str]:
    best_rows = metrics_df.loc[metrics_df.groupby("target")["r2"].idxmax()].sort_values("target")
    stability_summary = (
        stability_df.groupby(["target", "feature_set"])[["r2", "rmse"]]
        .mean()
        .reset_index()
        .sort_values(["target", "r2"], ascending=[True, False])
    )
    lines = [
        "Phase 2: Full-dataset modeling summary",
        "",
        "Dataset: all 2,880 rows",
        "Validation: 70/30 stratified holdout + repeated grouped CV on training split",
        "Feature sets compared: embeddings only vs embeddings plus region and Dynamic World context",
        "Current implementation differs from the original proposal: it uses the full dataset and a LightGBM ablation rather than the earlier 720-row subset plus ridge baseline plan.",
        "",
        "Best held-out feature set by target:",
    ]
    for row in best_rows.itertuples():
        lines.append(
            f"- {row.target}: {FEATURE_LABELS[row.feature_set]} with R2={row.r2:.3f}, RMSE={row.rmse:.3f}, MAE={row.mae:.3f}, r={row.pearson_r:.3f}"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            f"- S1_VV is best with {FEATURE_LABELS[best_rows.loc[best_rows['target'] == 'S1_VV', 'feature_set'].iloc[0]]}.",
            f"- S1_VH is best with {FEATURE_LABELS[best_rows.loc[best_rows['target'] == 'S1_VH', 'feature_set'].iloc[0]]}.",
            f"- VV/VH ratio is best with {FEATURE_LABELS[best_rows.loc[best_rows['target'] == 'S1_VV_div_VH', 'feature_set'].iloc[0]]}.",
            f"- Mean grouped-CV R2 across the selected models ranges from {stability_summary['r2'].min():.3f} to {stability_summary['r2'].max():.3f}.",
            "- Important caveat: the outer holdout is stratified random within sampled regions, not spatially disjoint. Treat these as within-region generalization estimates rather than strict geographic transfer performance.",
        ]
    )
    return lines


def build_phase3_summary_lines(
    phase3_selection: pd.DataFrame,
    phase3_land_use: pd.DataFrame,
    phase3_spatial: pd.DataFrame,
) -> list[str]:
    lines = [
        "Phase 3: Failure-mode analysis summary",
        "",
        "Best held-out model used per target:",
    ]
    for row in phase3_selection.sort_values("target").itertuples():
        lines.append(
            f"- {row.target}: {row.model_label} (R2={row.r2:.3f}, RMSE={row.rmse:.3f}, MAE={row.mae:.3f})"
        )

    primary_land = phase3_land_use[phase3_land_use["target"] == "S1_VV"].sort_values("mae", ascending=False).head(3)
    primary_spatial = phase3_spatial[phase3_spatial["target"] == "S1_VV"].copy()
    exploratory_hits = primary_spatial[primary_spatial["p_value"] < 0.05].sort_values("p_value")

    lines.extend(
        [
            "",
            "Primary-target findings for S1_VV:",
            "Top three hardest land-use classes by MAE:",
        ]
    )
    for row in primary_land.itertuples():
        lines.append(f"  {row.dw_label_name}: MAE={row.mae:.3f}, bias={row.bias:.3f}, p90 abs error={row.p90_abs_error:.3f}")
    lines.append("")
    if not exploratory_hits.empty:
        top_hit = exploratory_hits.iloc[0]
        lines.append(
            "Exploratory spatial result:"
        )
        lines.append(
            f"  {top_hit['region']}: Moran's I={top_hit['morans_i']:.3f}, p={top_hit['p_value']:.3f}"
        )
        lines.append(
            "  Treat this as weak evidence only: the effect is small and it is the only uncorrected p < 0.05 result across the region-level Phase 3 checks."
        )
    else:
        lines.append("No S1_VV region produced even exploratory p < 0.05 Moran's I evidence.")
    return lines


def build_project_summary_lines(
    phase2_metrics: pd.DataFrame,
    phase3_land_use: pd.DataFrame,
    phase3_spatial: pd.DataFrame,
    phase4_overall: pd.DataFrame,
    phase4_land_use: pd.DataFrame,
    phase4_corr: pd.DataFrame,
) -> list[str]:
    best_rows = phase2_metrics.loc[phase2_metrics.groupby("target")["r2"].idxmax()].sort_values("target")
    primary_land = phase3_land_use[phase3_land_use["target"] == "S1_VV"].sort_values("mae", ascending=False).head(3)
    primary_spatial = phase3_spatial[phase3_spatial["target"] == "S1_VV"].copy()
    exploratory_hits = primary_spatial[primary_spatial["p_value"] < 0.05].sort_values("p_value")
    k10_row = phase4_overall.loc[phase4_overall["k"] == 10].iloc[0]
    overlap_k10 = k10_row["mean_overlap"]
    baseline_k10 = knn_overlap_baseline(int(k10_row["n_queries"]), 10)
    best_land_use_k10 = phase4_land_use[phase4_land_use["k"] == 10].sort_values("mean_overlap", ascending=False).iloc[0]
    overall_corr = phase4_corr[phase4_corr["scope"] == "overall"].iloc[0]
    region_corr = phase4_corr[phase4_corr["scope"] == "region"].sort_values("pearson_r", ascending=False)
    strongest_region = region_corr.iloc[0]
    lines = [
        "Project summary report",
        "",
        "This report combines the current Phase 2, Phase 3, and Phase 4 results.",
        "Method note: the current completed workflow uses all 2,880 rows and a LightGBM ablation; it should be described as an updated implementation of the original proposal, not an exact execution of it.",
        "",
        "Phase 2 headline results:",
    ]
    for row in best_rows.itertuples():
        lines.append(
            f"- {row.target}: best model is {FEATURE_LABELS[row.feature_set]} with R2={row.r2:.3f}, RMSE={row.rmse:.3f}, MAE={row.mae:.3f}"
        )
    lines.extend(
        [
            "",
            "Phase 3 headline results for S1_VV:",
            "Hardest land-use classes:",
        ]
    )
    for row in primary_land.itertuples():
        lines.append(f"  {row.dw_label_name}: MAE={row.mae:.3f}, bias={row.bias:.3f}")
    lines.append("")
    if not exploratory_hits.empty:
        top_hit = exploratory_hits.iloc[0]
        lines.append(
            f"Exploratory spatial signal only: {top_hit['region']} has Moran's I={top_hit['morans_i']:.3f} with p={top_hit['p_value']:.3f}; this should not be presented as a strong confirmed clustering result."
        )
    else:
        lines.append("No region showed statistically significant S1_VV residual clustering.")
    lines.extend(
        [
            "",
            "Phase 4 headline results:",
            f"- Overall cross-modal neighbor overlap at k=10 is {overlap_k10:.3f} versus a random-match baseline of about {baseline_k10:.3f}.",
            f"- Best land-use overlap at k=10 occurs for {best_land_use_k10['dw_label_name']} ({best_land_use_k10['mean_overlap']:.3f}).",
            f"- Overall embedding-to-SAR distance correlation is positive but modest: Pearson r={overall_corr['pearson_r']:.3f}, Spearman rho={overall_corr['spearman_rho']:.3f}.",
            f"- Within-region correlation is much stronger, peaking in {strongest_region['group']} at Pearson r={strongest_region['pearson_r']:.3f}.",
        ]
    )
    return lines


def build_phase4_summary_lines(
    phase4_overall: pd.DataFrame,
    phase4_land_use: pd.DataFrame,
    phase4_corr: pd.DataFrame,
) -> list[str]:
    k10_row = phase4_overall.loc[phase4_overall["k"] == 10].iloc[0]
    baseline_k10 = knn_overlap_baseline(int(k10_row["n_queries"]), 10)
    best_land_use = phase4_land_use[phase4_land_use["k"] == 10].sort_values("mean_overlap", ascending=False).head(3)
    worst_land_use = phase4_land_use[phase4_land_use["k"] == 10].sort_values("mean_overlap", ascending=True).head(3)
    overall_corr = phase4_corr[phase4_corr["scope"] == "overall"].iloc[0]
    region_corr = phase4_corr[phase4_corr["scope"] == "region"].sort_values("pearson_r", ascending=False)
    strongest_region = region_corr.iloc[0]
    weakest_region = region_corr.iloc[-1]
    lines = [
        "Phase 4: Cross-modal similarity summary",
        "",
        "Question:",
        "Do embedding-space neighbors match SAR-space neighbors?",
        "",
        "Overall k-NN overlap:",
    ]
    for row in phase4_overall.itertuples():
        lines.append(f"- k={row.k}: mean overlap={row.mean_overlap:.3f}, median overlap={row.median_overlap:.3f}")
    lines.extend(
        [
            "",
            f"At k=10, the observed overlap ({k10_row['mean_overlap']:.3f}) is above the random-match baseline (~{baseline_k10:.3f}), but the median query still has zero overlap.",
            f"Overall pooled distance correlation: Pearson r={overall_corr['pearson_r']:.3f}, Spearman rho={overall_corr['spearman_rho']:.3f}",
            f"Within-region Pearson correlations range from {weakest_region['pearson_r']:.3f} ({weakest_region['group']}) to {strongest_region['pearson_r']:.3f} ({strongest_region['group']}).",
            "This suggests pooled cross-region comparisons dilute a stronger within-region relationship.",
            "",
            "Highest land-use overlap at k=10:",
        ]
    )
    for row in best_land_use.itertuples():
        lines.append(f"  {row.dw_label_name}: mean overlap={row.mean_overlap:.3f}")
    lines.append("")
    lines.append("Lowest land-use overlap at k=10:")
    for row in worst_land_use.itertuples():
        lines.append(f"  {row.dw_label_name}: mean overlap={row.mean_overlap:.3f}")
    lines.append("")
    lines.append("Representative Sentinel-2 RGB chips are included for the query pixel, the top embedding-space neighbor, and the top SAR-space neighbor.")
    return lines


def load_report_inputs() -> dict[str, pd.DataFrame]:
    inputs = {
        "metrics": pd.read_csv(METRICS_PATH).sort_values(["target", "r2"], ascending=[True, False]).reset_index(drop=True),
        "stability": pd.read_csv(STABILITY_PATH),
        "regional": pd.read_csv(REGIONAL_PATH),
        "land_use": pd.read_csv(LAND_USE_PATH),
        "phase3_selection": pd.read_csv(PHASE3_SELECTION_PATH),
        "phase3_land_use": pd.read_csv(PHASE3_LAND_USE_PATH),
        "phase3_spatial": pd.read_csv(PHASE3_SPATIAL_PATH),
        "phase3_outliers": pd.read_csv(PHASE3_OUTLIERS_PATH),
        "phase4_overall": pd.read_csv(PHASE4_OVERALL_PATH),
        "phase4_land_use": pd.read_csv(PHASE4_LAND_USE_PATH),
        "phase4_region": pd.read_csv(PHASE4_REGION_PATH),
        "phase4_corr": pd.read_csv(PHASE4_CORR_PATH),
        "phase4_queries": pd.read_csv(PHASE4_QUERIES_PATH),
    }
    return inputs


def build_phase2_report(inputs: dict[str, pd.DataFrame]) -> None:
    metrics_df = inputs["metrics"]
    stability_df = inputs["stability"]
    regional_df = inputs["regional"]
    land_use_df = inputs["land_use"]

    stability_summary = (
        stability_df.groupby(["target", "feature_set"])[["r2", "rmse", "mae"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    stability_summary.columns = ["target", "feature_set", "r2_mean", "r2_std", "rmse_mean", "rmse_std", "mae_mean", "mae_std"]
    stability_summary["feature_set"] = stability_summary["feature_set"].map(FEATURE_LABELS)
    stability_summary = format_float_columns(stability_summary, ["r2_mean", "r2_std", "rmse_mean", "rmse_std", "mae_mean", "mae_std"])

    regional_vv = regional_df[regional_df["target"] == "S1_VV"].copy()
    regional_vv["feature_set"] = regional_vv["feature_set"].map(FEATURE_LABELS)
    regional_vv = format_float_columns(regional_vv, ["r2", "rmse", "mae", "pearson_r"])

    land_use_vv = land_use_df[land_use_df["target"] == "S1_VV"].copy()
    land_use_vv["feature_set"] = land_use_vv["feature_set"].map(FEATURE_LABELS)
    land_use_vv = format_float_columns(land_use_vv, ["r2", "rmse", "mae", "pearson_r"])

    with PdfPages(PHASE2_REPORT_PATH) as pdf:
        draw_text_page("Phase 2 Modeling Summary", build_phase2_summary_lines(metrics_df, stability_df), pdf)
        draw_dataframe_page(
            "Held-out Performance by Feature Set",
            format_metrics_table(metrics_df),
            pdf,
            footnote="The holdout test set is shared across the two feature-set ablations. It is a stratified random holdout within sampled regions rather than a spatially disjoint holdout, so these estimates may be optimistic relative to stricter geographic transfer tests.",
        )
        draw_dataframe_page(
            "Repeated Grouped-CV Stability",
            stability_summary,
            pdf,
            footnote="Each row summarizes 12 grouped validation folds: 4 folds across 3 random seeds.",
        )
        draw_dataframe_page(
            "Regional S1_VV Diagnostics",
            regional_vv[["region", "feature_set", "r2", "rmse", "mae", "pearson_r", "n_test"]].reset_index(drop=True),
            pdf,
            footnote="Region-level diagnostics on the held-out test set.",
        )
        draw_dataframe_page(
            "Land-use S1_VV Diagnostics",
            land_use_vv[["dw_label_name", "feature_set", "r2", "rmse", "mae", "pearson_r", "n_test"]].reset_index(drop=True),
            pdf,
            footnote="Land-use diagnostics on the held-out test set.",
        )
        draw_text_page("Feature Importance Summary", build_feature_lines(), pdf)
        draw_image_grid_page(
            "Predicted vs Actual",
            [
                OUTPUT_DIR / "predicted_vs_actual_embedding_only_S1_VV.png",
                OUTPUT_DIR / "predicted_vs_actual_embedding_plus_context_S1_VV.png",
                OUTPUT_DIR / "predicted_vs_actual_embedding_only_S1_VH.png",
                OUTPUT_DIR / "predicted_vs_actual_embedding_plus_context_S1_VH.png",
            ],
            pdf,
        )
        draw_image_grid_page(
            "Residual Histograms",
            [
                OUTPUT_DIR / "residual_histogram_embedding_only_S1_VV.png",
                OUTPUT_DIR / "residual_histogram_embedding_plus_context_S1_VV.png",
                OUTPUT_DIR / "residual_histogram_embedding_only_S1_VV_div_VH.png",
                OUTPUT_DIR / "residual_histogram_embedding_plus_context_S1_VV_div_VH.png",
            ],
            pdf,
        )


def build_phase3_report(inputs: dict[str, pd.DataFrame]) -> None:
    phase3_selection = inputs["phase3_selection"]
    phase3_land_use = inputs["phase3_land_use"]
    phase3_spatial = inputs["phase3_spatial"]
    phase3_outliers = inputs["phase3_outliers"]

    phase3_selection_display = format_float_columns(phase3_selection.copy(), ["r2", "rmse", "mae", "pearson_r"])
    phase3_land_use_display = format_float_columns(
        phase3_land_use[phase3_land_use["target"] == "S1_VV"].sort_values("mae", ascending=False).head(9).copy(),
        ["mae", "rmse", "bias", "median_abs_error", "p90_abs_error", "max_abs_error"],
    )
    phase3_spatial_display = format_float_columns(phase3_spatial.copy(), ["morans_i", "p_value"])
    phase3_outliers_display = format_float_columns(
        phase3_outliers.head(10).copy(),
        ["label_confidence", "dominant_class_prob", "latitude", "longitude", "actual", "predicted", "residual", "abs_residual"],
    )

    with PdfPages(PHASE3_REPORT_PATH) as pdf:
        draw_text_page("Phase 3 Failure Analysis Summary", build_phase3_summary_lines(phase3_selection, phase3_land_use, phase3_spatial), pdf)
        draw_dataframe_page(
            "Phase 3 Best-model Selection",
            phase3_selection_display[["target", "feature_set", "model_label", "r2", "rmse", "mae", "pearson_r"]],
            pdf,
            footnote="Each target uses the best held-out Phase 2 model for downstream failure analysis.",
        )
        draw_dataframe_page(
            "Phase 3 S1_VV Land-use Failures",
            phase3_land_use_display[
                ["dw_label_name", "mae", "rmse", "bias", "median_abs_error", "p90_abs_error", "max_abs_error", "n"]
            ].reset_index(drop=True),
            pdf,
            footnote="These are the primary-target failure metrics for the selected S1_VV model.",
        )
        draw_dataframe_page(
            "Phase 3 Spatial Autocorrelation",
            phase3_spatial_display[["target", "region", "morans_i", "p_value", "significant_at_0_05", "n"]].reset_index(drop=True),
            pdf,
            footnote="Moran's I is computed within each region using row-standardized inverse-distance k-nearest-neighbor weights. Treat isolated p < 0.05 results as exploratory because only one weak hit appears and no multiple-testing correction is applied.",
        )
        draw_dataframe_page(
            "Phase 3 S1_VV Outlier Catalog",
            phase3_outliers_display[
                ["outlier_rank", "region", "dw_label_name", "label_confidence", "dominant_class", "actual", "predicted", "residual", "abs_residual"]
            ].reset_index(drop=True),
            pdf,
            footnote="The outlier catalog highlights the largest held-out S1_VV misses for manual review.",
        )
        draw_image_grid_page(
            "Phase 3 Failure-mode Visuals",
            [
                OUTPUT_DIR / "phase3_residual_maps_S1_VV.png",
                OUTPUT_DIR / "phase3_land_use_diagnostics_S1_VV.png",
                OUTPUT_DIR / "phase3_region_land_use_heatmap_S1_VV.png",
                OUTPUT_DIR / "phase3_residual_boxplot_S1_VV.png",
            ],
            pdf,
        )


def build_phase4_report(inputs: dict[str, pd.DataFrame]) -> None:
    phase4_overall = inputs["phase4_overall"]
    phase4_land_use = inputs["phase4_land_use"]
    phase4_region = inputs["phase4_region"]
    phase4_corr = inputs["phase4_corr"]
    phase4_queries = inputs["phase4_queries"]

    phase4_overall_display = format_float_columns(phase4_overall.copy(), ["mean_overlap", "median_overlap", "p10_overlap", "p90_overlap"])
    phase4_land_use_display = format_float_columns(
        phase4_land_use[phase4_land_use["k"] == 10].sort_values("mean_overlap", ascending=False).copy(),
        ["mean_overlap", "median_overlap", "p10_overlap", "p90_overlap"],
    )
    phase4_region_display = format_float_columns(
        phase4_region[phase4_region["k"] == 10].sort_values("mean_overlap", ascending=False).copy(),
        ["mean_overlap", "median_overlap", "p10_overlap", "p90_overlap"],
    )
    phase4_corr_display = format_float_columns(
        phase4_corr.copy(),
        ["pearson_r", "pearson_p_value", "spearman_rho", "spearman_p_value"],
    )
    phase4_queries_display = format_float_columns(phase4_queries.copy(), ["latitude", "longitude", "overlap_at_10"])

    with PdfPages(PHASE4_REPORT_PATH) as pdf:
        draw_text_page("Phase 4 Cross-modal Similarity Summary", build_phase4_summary_lines(phase4_overall, phase4_land_use, phase4_corr), pdf)
        draw_dataframe_page(
            "Phase 4 Overall k-NN Overlap",
            phase4_overall_display[["k", "n_queries", "mean_overlap", "median_overlap", "p10_overlap", "p90_overlap"]],
            pdf,
            footnote="Overlap is the fraction of embedding-space k-nearest neighbors that also appear in SAR-space k-nearest neighbors.",
        )
        draw_dataframe_page(
            "Phase 4 Land-use Overlap at k=10",
            phase4_land_use_display[["dw_label_name", "n_queries", "mean_overlap", "median_overlap", "p10_overlap", "p90_overlap"]],
            pdf,
            footnote="The land-use table shows where cross-modal neighbor agreement is strongest or weakest.",
        )
        draw_dataframe_page(
            "Phase 4 Region Overlap at k=10",
            phase4_region_display[["region", "n_queries", "mean_overlap", "median_overlap", "p10_overlap", "p90_overlap"]],
            pdf,
            footnote="Regional overlap helps assess whether similarity alignment is geography-dependent.",
        )
        draw_dataframe_page(
            "Phase 4 Distance Correlation",
            phase4_corr_display[["scope", "group", "n_pairs", "pearson_r", "spearman_rho"]],
            pdf,
            footnote="Pairwise distance correlation compares embedding cosine distance to SAR Euclidean distance.",
        )
        draw_dataframe_page(
            "Phase 4 Representative Queries",
            phase4_queries_display[["reason", "region", "dw_label_name", "overlap_at_10", "embedding_neighbor_labels", "sar_neighbor_labels"]],
            pdf,
            footnote="Representative queries provide a qualitative check on where neighbor sets align or diverge.",
        )
        draw_image_grid_page(
            "Phase 4 Visual Diagnostics",
            [
                OUTPUT_DIR / "phase4_knn_overlap_by_k.png",
                OUTPUT_DIR / "phase4_knn_overlap_by_land_use_k10.png",
                OUTPUT_DIR / "phase4_embedding_vs_sar_distance.png",
                OUTPUT_DIR / "phase4_sentinel2_neighbor_chips.png",
            ],
            pdf,
        )


def build_project_report(inputs: dict[str, pd.DataFrame]) -> None:
    metrics_df = inputs["metrics"]
    phase3_land_use = inputs["phase3_land_use"]
    phase3_spatial = inputs["phase3_spatial"]
    phase4_overall = inputs["phase4_overall"]
    phase4_land_use = inputs["phase4_land_use"]
    phase4_corr = inputs["phase4_corr"]

    selected_models = metrics_df.loc[metrics_df.groupby("target")["r2"].idxmax()].sort_values("target").copy()
    selected_models["feature_set"] = selected_models["feature_set"].map(FEATURE_LABELS)
    selected_models = format_float_columns(selected_models, ["r2", "rmse", "mae", "pearson_r"])

    project_land_use = format_float_columns(
        phase3_land_use[phase3_land_use["target"] == "S1_VV"].sort_values("mae", ascending=False).head(5).copy(),
        ["mae", "rmse", "bias", "p90_abs_error"],
    )
    project_spatial = format_float_columns(
        phase3_spatial[phase3_spatial["target"] == "S1_VV"].copy(),
        ["morans_i", "p_value"],
    )
    project_phase4 = format_float_columns(
        phase4_overall.copy(),
        ["mean_overlap", "median_overlap", "p10_overlap", "p90_overlap"],
    )
    project_phase4_land = format_float_columns(
        phase4_land_use[phase4_land_use["k"] == 10].sort_values("mean_overlap", ascending=False).head(5).copy(),
        ["mean_overlap", "median_overlap", "p10_overlap", "p90_overlap"],
    )

    with PdfPages(PROJECT_REPORT_PATH) as pdf:
        draw_text_page(
            "Project Summary Report",
            build_project_summary_lines(metrics_df, phase3_land_use, phase3_spatial, phase4_overall, phase4_land_use, phase4_corr),
            pdf,
        )
        draw_dataframe_page(
            "Best Phase 2 Models by Target",
            selected_models[["target", "feature_set", "r2", "rmse", "mae", "pearson_r"]].reset_index(drop=True),
            pdf,
            footnote="These are the best held-out full-dataset models selected for the project summary.",
        )
        draw_dataframe_page(
            "Top S1_VV Failure Modes",
            project_land_use[["dw_label_name", "mae", "rmse", "bias", "p90_abs_error", "n"]].reset_index(drop=True),
            pdf,
            footnote="The cumulative report focuses on the highest-error S1_VV land-use classes.",
        )
        draw_dataframe_page(
            "S1_VV Spatial Residual Structure",
            project_spatial[["region", "morans_i", "p_value", "significant_at_0_05", "n"]].reset_index(drop=True),
            pdf,
            footnote="Significant spatial clustering indicates geographically structured failure modes.",
        )
        draw_dataframe_page(
            "Phase 4 Cross-modal Similarity",
            project_phase4[["k", "n_queries", "mean_overlap", "median_overlap", "p10_overlap", "p90_overlap"]].reset_index(drop=True),
            pdf,
            footnote="Cross-modal similarity is summarized by k-NN overlap between embedding space and SAR space.",
        )
        draw_dataframe_page(
            "Top Phase 4 Land-use Alignment at k=10",
            project_phase4_land[["dw_label_name", "mean_overlap", "median_overlap", "p10_overlap", "p90_overlap", "n_queries"]].reset_index(drop=True),
            pdf,
            footnote="This table highlights which land-use classes show the strongest cross-modal neighbor agreement.",
        )
        draw_image_grid_page(
            "Project Visual Summary",
            [
                OUTPUT_DIR / "predicted_vs_actual_embedding_only_S1_VV.png",
                OUTPUT_DIR / "phase3_residual_maps_S1_VV.png",
                OUTPUT_DIR / "phase4_knn_overlap_by_land_use_k10.png",
                OUTPUT_DIR / "phase4_sentinel2_neighbor_chips.png",
            ],
            pdf,
        )


def copy_reports() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    for source, target in TOP_LEVEL_COPIES.items():
        target.write_bytes(source.read_bytes())
    LEGACY_COMBINED_COPY.write_bytes(PROJECT_REPORT_PATH.read_bytes())
    LEGACY_COMBINED_OUTPUT_COPY.write_bytes(PROJECT_REPORT_PATH.read_bytes())


def main() -> None:
    inputs = load_report_inputs()
    build_phase2_report(inputs)
    build_phase3_report(inputs)
    build_phase4_report(inputs)
    build_project_report(inputs)
    copy_reports()
    print(f"Saved PDF report to: {PHASE2_REPORT_PATH}")
    print(f"Saved PDF report to: {PHASE3_REPORT_PATH}")
    print(f"Saved PDF report to: {PHASE4_REPORT_PATH}")
    print(f"Saved PDF report to: {PROJECT_REPORT_PATH}")
    print(f"Copied project summary to legacy path: {LEGACY_COMBINED_COPY}")


if __name__ == "__main__":
    main()
