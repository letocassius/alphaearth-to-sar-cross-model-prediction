"""Metrics utilities for SAR reconstruction evaluation."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import SAR_BANDS


@dataclass
class RunningMoments:
    """Streaming metric accumulator for large raster predictions."""

    count: np.ndarray
    sum_y: np.ndarray
    sum_pred: np.ndarray
    sum_y2: np.ndarray
    sum_pred2: np.ndarray
    sum_ypred: np.ndarray
    sum_abs_err: np.ndarray
    sum_sq_err: np.ndarray

    @classmethod
    def zeros(cls, n_bands: int) -> "RunningMoments":
        zeros = np.zeros(n_bands, dtype=np.float64)
        return cls(
            count=zeros.copy(),
            sum_y=zeros.copy(),
            sum_pred=zeros.copy(),
            sum_y2=zeros.copy(),
            sum_pred2=zeros.copy(),
            sum_ypred=zeros.copy(),
            sum_abs_err=zeros.copy(),
            sum_sq_err=zeros.copy(),
        )

    def update(self, truth: np.ndarray, pred: np.ndarray) -> None:
        if truth.size == 0:
            return
        self.count += truth.shape[0]
        self.sum_y += truth.sum(axis=0)
        self.sum_pred += pred.sum(axis=0)
        self.sum_y2 += np.square(truth).sum(axis=0)
        self.sum_pred2 += np.square(pred).sum(axis=0)
        self.sum_ypred += (truth * pred).sum(axis=0)
        self.sum_abs_err += np.abs(truth - pred).sum(axis=0)
        self.sum_sq_err += np.square(truth - pred).sum(axis=0)

    def to_metrics(self, band_names: list[str]) -> pd.DataFrame:
        mean_y = np.divide(self.sum_y, self.count, out=np.full_like(self.sum_y, np.nan), where=self.count > 0)
        mean_pred = np.divide(self.sum_pred, self.count, out=np.full_like(self.sum_pred, np.nan), where=self.count > 0)
        sst = self.sum_y2 - self.count * np.square(mean_y)
        var_y = self.sum_y2 - self.count * np.square(mean_y)
        var_pred = self.sum_pred2 - self.count * np.square(mean_pred)
        cov = self.sum_ypred - self.count * mean_y * mean_pred

        r2 = np.full_like(self.sum_y, np.nan)
        rmse = np.full_like(self.sum_y, np.nan)
        mae = np.full_like(self.sum_y, np.nan)
        pearson_r = np.full_like(self.sum_y, np.nan)

        valid = self.count > 1
        r2[valid] = 1.0 - np.divide(
            self.sum_sq_err[valid],
            sst[valid],
            out=np.full_like(self.sum_sq_err[valid], np.nan),
            where=sst[valid] > 0,
        )
        rmse[valid] = np.sqrt(self.sum_sq_err[valid] / self.count[valid])
        mae[valid] = self.sum_abs_err[valid] / self.count[valid]

        corr_valid = valid & (var_y > 0) & (var_pred > 0)
        pearson_r[corr_valid] = cov[corr_valid] / np.sqrt(var_y[corr_valid] * var_pred[corr_valid])

        return pd.DataFrame({"band": band_names, "count": self.count.astype(np.int64), "r2": r2, "rmse": rmse, "mae": mae, "pearson_r": pearson_r})


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, band_names: list[str] = SAR_BANDS) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for idx, band in enumerate(band_names):
        true_band = y_true[:, idx]
        pred_band = y_pred[:, idx]
        rows.append(
            {
                "band": band,
                "count": int(true_band.shape[0]),
                "r2": float(r2_score(true_band, pred_band)),
                "rmse": float(math.sqrt(mean_squared_error(true_band, pred_band))),
                "mae": float(mean_absolute_error(true_band, pred_band)),
                "pearson_r": safe_pearsonr(true_band, pred_band),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(metrics_df: pd.DataFrame) -> dict[str, float | str]:
    best_row = metrics_df.sort_values("r2", ascending=False).iloc[0]
    worst_row = metrics_df.sort_values("r2", ascending=True).iloc[0]
    return {
        "mean_r2": float(metrics_df["r2"].mean()),
        "median_r2": float(metrics_df["r2"].median()),
        "mean_rmse": float(metrics_df["rmse"].mean()),
        "mean_mae": float(metrics_df["mae"].mean()),
        "mean_pearson_r": float(metrics_df["pearson_r"].mean()),
        "best_band": str(best_row["band"]),
        "best_band_r2": float(best_row["r2"]),
        "worst_band": str(worst_row["band"]),
        "worst_band_r2": float(worst_row["r2"]),
    }
