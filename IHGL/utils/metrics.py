"""Metrics for values hidden by the evaluation mask."""

from __future__ import annotations

import numpy as np


def _values(target: np.ndarray, forecast: np.ndarray, eval_points: np.ndarray):
    selected = np.asarray(eval_points) == 1
    if not np.any(selected):
        raise ValueError("No values are available for evaluation")
    return np.asarray(target)[selected], np.asarray(forecast)[selected]


def calculate_metrics(
    target: np.ndarray, forecast: np.ndarray, eval_points: np.ndarray
) -> dict[str, float]:
    target_values, forecast_values = _values(target, forecast, eval_points)
    error = target_values - forecast_values
    nonzero = np.abs(target_values) > np.finfo(np.float32).eps
    mape = np.mean(np.abs(error[nonzero] / target_values[nonzero])) if np.any(nonzero) else 0.0
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "MAPE": float(mape),
    }


# Backwards-compatible names used by the original scripts.
def calc_RMSE(target, forecast, eval_points):
    return calculate_metrics(target, forecast, eval_points)["RMSE"]


def calc_MAE(target, forecast, eval_points):
    return calculate_metrics(target, forecast, eval_points)["MAE"]


def calc_MAPE(target, forecast, eval_points):
    return calculate_metrics(target, forecast, eval_points)["MAPE"]
