from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "full_dataset"
FEATURE_SETS = ("embedding_only", "embedding_plus_context")
KEY_COLUMNS = ("system:index", "region", "dw_label_name", "latitude", "longitude", "spatial_block")


@dataclass
class Metrics:
    r2: float
    rmse: float
    mae: float


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in KEY_COLUMNS)


def compute_metrics(y_true: list[float], y_pred: list[float]) -> Metrics:
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot compute metrics with zero rows.")
    residuals = [pred - true for true, pred in zip(y_true, y_pred)]
    mae = sum(abs(res) for res in residuals) / n
    rmse = math.sqrt(sum(res * res for res in residuals) / n)
    mean_true = sum(y_true) / n
    ss_res = sum(res * res for res in residuals)
    ss_tot = sum((true - mean_true) ** 2 for true in y_true)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0
    return Metrics(r2=r2, rmse=rmse, mae=mae)


def nice_limits(values: list[float], padding_ratio: float = 0.05) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        pad = max(1.0, abs(min_value) * padding_ratio)
        return min_value - pad, max_value + pad
    pad = (max_value - min_value) * padding_ratio
    return min_value - pad, max_value + pad


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if math.isclose(src_min, src_max):
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def render_scatter_svg(
    path: Path,
    y_true: list[float],
    y_pred: list[float],
    title: str,
    subtitle: str,
) -> None:
    width = 860
    height = 760
    left = 95
    right = 40
    top = 90
    bottom = 90
    plot_width = width - left - right
    plot_height = height - top - bottom

    values = y_true + y_pred
    x_min, x_max = nice_limits(values)
    y_min, y_max = nice_limits(values)

    ticks = 6
    tick_values = [x_min + (x_max - x_min) * i / ticks for i in range(ticks + 1)]

    points = []
    for true_value, pred_value in zip(y_true, y_pred):
        x = scale(true_value, x_min, x_max, left, left + plot_width)
        y = scale(pred_value, y_min, y_max, top + plot_height, top)
        points.append((x, y))

    diagonal_x1 = scale(x_min, x_min, x_max, left, left + plot_width)
    diagonal_y1 = scale(x_min, y_min, y_max, top + plot_height, top)
    diagonal_x2 = scale(x_max, x_min, x_max, left, left + plot_width)
    diagonal_y2 = scale(x_max, y_min, y_max, top + plot_height, top)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{left}" y="42" font-family="Helvetica, Arial, sans-serif" font-size="26" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="{left}" y="68" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#4b5563">{subtitle}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#f9fafb" stroke="#d1d5db" />',
        f'<line x1="{diagonal_x1:.2f}" y1="{diagonal_y1:.2f}" x2="{diagonal_x2:.2f}" y2="{diagonal_y2:.2f}" stroke="#ef4444" stroke-width="2" stroke-dasharray="7 6" />',
    ]

    for tick in tick_values:
        x = scale(tick, x_min, x_max, left, left + plot_width)
        y = scale(tick, y_min, y_max, top + plot_height, top)
        label = f"{tick:.2f}"
        svg_lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e5e7eb" stroke-width="1" />')
        svg_lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />')
        svg_lines.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">{label}</text>')
        svg_lines.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#374151">{label}</text>')

    svg_lines.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5" />',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5" />',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 28}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="17" fill="#111827">True S1_VV_div_VH (dB)</text>',
            f'<text x="28" y="{top + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 28 {top + plot_height / 2:.2f})" font-family="Helvetica, Arial, sans-serif" font-size="17" fill="#111827">Predicted S1_VV_div_VH (dB)</text>',
        ]
    )

    for x, y in points:
        svg_lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.4" fill="#2563eb" fill-opacity="0.42" />')

    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines), encoding="utf-8")


def compare_feature_set(feature_set: str) -> list[dict[str, object]]:
    vv_path = OUTPUT_DIR / f"test_predictions_{feature_set}_S1_VV.csv"
    vh_path = OUTPUT_DIR / f"test_predictions_{feature_set}_S1_VH.csv"
    direct_path = OUTPUT_DIR / f"test_predictions_{feature_set}_S1_VV_div_VH.csv"

    vv_rows = load_prediction_rows(vv_path)
    vh_rows = load_prediction_rows(vh_path)
    direct_rows = load_prediction_rows(direct_path)

    vv_map = {row_key(row): row for row in vv_rows}
    vh_map = {row_key(row): row for row in vh_rows}
    direct_map = {row_key(row): row for row in direct_rows}

    common_keys = sorted(set(vv_map) & set(vh_map) & set(direct_map))
    if not common_keys:
        raise ValueError(f"No overlapping prediction rows found for {feature_set}.")

    true_values: list[float] = []
    direct_predictions: list[float] = []
    structural_predictions: list[float] = []

    structural_rows: list[dict[str, object]] = []
    for key in common_keys:
        direct_row = direct_map[key]
        vv_row = vv_map[key]
        vh_row = vh_map[key]

        true_value = float(direct_row["actual"])
        structural_prediction = float(vv_row["predicted"]) - float(vh_row["predicted"])
        direct_prediction = float(direct_row["predicted"])

        true_values.append(true_value)
        direct_predictions.append(direct_prediction)
        structural_predictions.append(structural_prediction)

        structural_rows.append(
            {
                **{column: direct_row[column] for column in direct_row.keys()},
                "predicted": structural_prediction,
                "residual": structural_prediction - true_value,
                "source_vv_file": vv_path.name,
                "source_vh_file": vh_path.name,
            }
        )

    direct_metrics = compute_metrics(true_values, direct_predictions)
    structural_metrics = compute_metrics(true_values, structural_predictions)

    structural_out_path = OUTPUT_DIR / f"test_predictions_{feature_set}_S1_VV_div_VH_structural_from_vv_minus_vh.csv"
    with structural_out_path.open("w", newline="") as handle:
        fieldnames = list(structural_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(structural_rows)

    direct_plot_path = OUTPUT_DIR / f"comparison_true_vs_pred_{feature_set}_direct_s1_vv_minus_vh.svg"
    render_scatter_svg(
        direct_plot_path,
        y_true=true_values,
        y_pred=direct_predictions,
        title=f"{feature_set}: Direct prediction",
        subtitle="True S1_VV_div_VH vs direct model prediction",
    )

    structural_plot_path = OUTPUT_DIR / f"comparison_true_vs_pred_{feature_set}_structural_s1_vv_minus_vh.svg"
    render_scatter_svg(
        structural_plot_path,
        y_true=true_values,
        y_pred=structural_predictions,
        title=f"{feature_set}: Structural baseline",
        subtitle="True S1_VV_div_VH vs VV_hat - VH_hat",
    )

    return [
        {
            "feature_set": feature_set,
            "approach": "direct_prediction",
            "prediction_file": direct_path.name,
            "r2": direct_metrics.r2,
            "rmse": direct_metrics.rmse,
            "mae": direct_metrics.mae,
            "plot_file": direct_plot_path.name,
        },
        {
            "feature_set": feature_set,
            "approach": "structural_prediction",
            "prediction_file": structural_out_path.name,
            "r2": structural_metrics.r2,
            "rmse": structural_metrics.rmse,
            "mae": structural_metrics.mae,
            "plot_file": structural_plot_path.name,
        },
    ]


def main() -> None:
    all_rows: list[dict[str, object]] = []
    for feature_set in FEATURE_SETS:
        all_rows.extend(compare_feature_set(feature_set))

    summary_path = OUTPUT_DIR / "polarization_difference_strategy_comparison.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_set", "approach", "prediction_file", "r2", "rmse", "mae", "plot_file"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved summary metrics to {summary_path}")
    for row in all_rows:
        print(
            f"{row['feature_set']} | {row['approach']} | "
            f"R2={row['r2']:.6f} RMSE={row['rmse']:.6f} MAE={row['mae']:.6f}"
        )


if __name__ == "__main__":
    main()
