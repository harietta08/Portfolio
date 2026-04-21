"""
forecasting/evaluate_forecast.py

Forecast evaluation — MAE, MAPE, RMSE on held-out test set.

Rule: Never report a forecast without documenting accuracy.
This module enforces that rule. Every Prophet model must pass through
here before its output is shown in Tableau or cited in the README.

Evaluation strategy:
  - 4-week (28-day) held-out test set: last 28 days of available data
  - MAE  : average absolute ridership miss (in riders/day)
  - MAPE : percentage miss — comparable across routes of different sizes
  - RMSE : penalises large misses more heavily (useful for service planning)
  - Weekly re-evaluation: metrics recomputed every Monday in forecast_flow

These metrics are:
  1. Logged per route/bucket in every Prefect flow run
  2. Written to BigQuery for trend tracking over time
  3. Displayed in Tableau as a forecast accuracy scorecard

MAPE thresholds:
  ok       : MAPE ≤ 20%  — acceptable for operational planning
  warning  : MAPE 20–35% — model may need retraining
  error    : MAPE > 35%  — model needs immediate retraining, do not use

Interview answer on evaluation:
  "I evaluated the Prophet models on a 4-week held-out test set using
  MAE and MAPE. MAPE is the primary metric because it is scale-independent
  — a 500-rider miss means something different on a 2,000-rider route
  versus a 20,000-rider route. I track MAPE weekly and trigger retraining
  when it exceeds 35%."
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from forecasting.train_prophet import ProphetForecaster

logger = logging.getLogger(__name__)

MAPE_WARNING_THRESHOLD = 20.0
MAPE_ERROR_THRESHOLD = 35.0


class ForecastEvaluator:
    """
    Evaluates a trained ProphetForecaster against historical actuals.

    Parameters
    ----------
    forecaster : ProphetForecaster
        A fitted ProphetForecaster instance.
    """

    def __init__(self, forecaster: ProphetForecaster):
        self.forecaster = forecaster

    # ─── Public interface ─────────────────────────────────────────────────────

    def evaluate(
        self,
        actuals_df: pd.DataFrame,
        eval_window_days: int = 28,
    ) -> dict:
        """
        Evaluate forecast accuracy over the most recent N days.

        Parameters
        ----------
        actuals_df : pd.DataFrame
            Columns: service_date, rides. Must cover the eval window.
        eval_window_days : int
            Number of days in the held-out evaluation window.

        Returns
        -------
        dict
            {
              "route": str,
              "time_bucket": str,
              "mae": float,
              "mape": float,
              "rmse": float,
              "n_eval_days": int,
              "eval_start": str,
              "eval_end": str,
              "mape_status": "ok" | "warning" | "error" | "insufficient_data"
            }
        """
        if actuals_df.empty or len(actuals_df) < eval_window_days:
            logger.warning(
                "Insufficient data for evaluation: %d rows (need %d).",
                len(actuals_df), eval_window_days,
            )
            return self._empty_metrics()

        actuals = actuals_df.copy()
        actuals["ds"] = pd.to_datetime(actuals["service_date"])
        actuals["rides"] = pd.to_numeric(actuals["rides"], errors="coerce")
        actuals = actuals.dropna(subset=["ds", "rides"])
        actuals = actuals.sort_values("ds").tail(eval_window_days)

        eval_start = actuals["ds"].min().date().isoformat()
        eval_end = actuals["ds"].max().date().isoformat()

        # Generate forecast covering the evaluation period
        try:
            forecast_df = self.forecaster.predict(horizon_days=eval_window_days)
        except Exception as exc:
            logger.error("Forecast failed during evaluation: %s", exc)
            return self._empty_metrics()

        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

        # Inner join: only dates where both actuals and forecast exist
        merged = actuals[["ds", "rides"]].merge(
            forecast_df[["ds", "yhat"]], on="ds", how="inner"
        )

        if len(merged) < 7:
            logger.warning(
                "Only %d overlapping dates — metrics may be unreliable.",
                len(merged),
            )
            if merged.empty:
                return self._empty_metrics()

        y_true = merged["rides"].values.astype(float)
        y_pred = merged["yhat"].values.astype(float)

        mae  = self._mae(y_true, y_pred)
        mape = self._mape(y_true, y_pred)
        rmse = self._rmse(y_true, y_pred)

        mape_status = self._mape_status(mape)
        self._log_mape_status(mape, mape_status)

        return {
            "route":        self.forecaster.route,
            "time_bucket":  self.forecaster.time_bucket,
            "mae":          round(float(mae), 2),
            "mape":         round(float(mape), 2),
            "rmse":         round(float(rmse), 2),
            "n_eval_days":  int(len(merged)),
            "eval_start":   eval_start,
            "eval_end":     eval_end,
            "mape_status":  mape_status,
            "evaluated_at": pd.Timestamp.utcnow().isoformat(),
        }

    def forecast_vs_actual_df(
        self,
        actuals_df: pd.DataFrame,
        horizon_days: int = 28,
    ) -> pd.DataFrame:
        """
        Produce a comparison DataFrame for Tableau visualisation.

        Combines actuals (where available) with forecast + CI, and flags
        anomalies where actual deviates from forecast by > 2 rolling std.

        Returns
        -------
        pd.DataFrame
            Columns: ds, actual_rides (nullable), yhat, yhat_lower,
                     yhat_upper, route, time_bucket, is_actual,
                     residual, is_anomaly.
        """
        forecast_df = self.forecaster.predict(horizon_days=horizon_days)
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

        actuals = actuals_df.copy()
        actuals["ds"] = pd.to_datetime(actuals["service_date"])
        actuals["actual_rides"] = pd.to_numeric(
            actuals["rides"], errors="coerce"
        )

        merged = forecast_df.merge(
            actuals[["ds", "actual_rides"]], on="ds", how="left"
        )
        merged["is_actual"] = merged["actual_rides"].notna()
        merged["residual"] = merged["actual_rides"] - merged["yhat"]

        # Flag anomalies using rolling std of residuals
        residuals = merged.loc[merged["is_actual"], "residual"]
        if len(residuals) >= 14:
            rolling_std = (
                residuals
                .rolling(window=14, min_periods=7)
                .std()
            )
            threshold = 2.0 * rolling_std.fillna(rolling_std.mean())
            merged.loc[merged["is_actual"], "is_anomaly"] = (
                merged.loc[merged["is_actual"], "residual"]
                .abs()
                .gt(threshold.values)
            )
        else:
            merged["is_anomaly"] = False

        merged["is_anomaly"] = merged["is_anomaly"].fillna(False)
        return merged

    # ─── Metric functions ─────────────────────────────────────────────────────

    @staticmethod
    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAPE — safe against zero actuals."""
        nonzero = y_true != 0
        if not nonzero.any():
            return float("inf")
        return float(
            np.mean(
                np.abs(
                    (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
                )
            ) * 100
        )

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _mape_status(self, mape: float) -> str:
        if mape > MAPE_ERROR_THRESHOLD:
            return "error"
        if mape > MAPE_WARNING_THRESHOLD:
            return "warning"
        return "ok"

    def _log_mape_status(self, mape: float, status: str) -> None:
        msg = (
            "MAPE=%.1f%% for route=%s bucket=%s — status=%s"
        )
        args = (
            mape,
            self.forecaster.route,
            self.forecaster.time_bucket,
            status,
        )
        if status == "error":
            logger.error(msg + ". Model retraining required.", *args)
        elif status == "warning":
            logger.warning(msg + ". Consider retraining.", *args)
        else:
            logger.info(msg, *args)

    def _empty_metrics(self) -> dict:
        return {
            "route":        self.forecaster.route,
            "time_bucket":  self.forecaster.time_bucket,
            "mae":          None,
            "mape":         None,
            "rmse":         None,
            "n_eval_days":  0,
            "eval_start":   None,
            "eval_end":     None,
            "mape_status":  "insufficient_data",
            "evaluated_at": pd.Timestamp.utcnow().isoformat(),
        }
