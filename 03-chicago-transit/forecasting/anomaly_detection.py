"""
forecasting/anomaly_detection.py

Service disruption detection — rolling Z-score + IQR anomaly flagging.

Method:
  For each route on each day:
    1. Compute rolling 14-day mean and std of residuals (actual - forecast)
    2. Z-score = residual / rolling_std
    3. IQR fence: flag if residual < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
    4. Anomaly requires BOTH methods to agree (reduces false positives)

Why rolling Z-score over a global threshold:
  Transit ridership is seasonal. A "low" ridership reading in January
  may be normal winter behaviour. The same absolute number in July
  could indicate a genuine service disruption. Rolling stats adapt to
  local seasonal patterns automatically.

Why dual-method (Z-score AND IQR):
  Z-score is sensitive to outliers in the rolling window.
  IQR is more robust but less sensitive to gradual drift.
  Requiring both to agree cuts false positives significantly.

Anomaly types surfaced in Tableau:
  LOW_RIDERSHIP  : actual << forecast. Possible causes: service disruption,
                   severe weather, major event diverting riders, data gap.
  HIGH_RIDERSHIP : actual >> forecast. Possible causes: special event
                   (game, festival, parade), service rerouting concentrating
                   riders onto fewer routes, data anomaly.

Severity levels:
  minor    : |z| 2.0–2.5
  moderate : |z| 2.5–3.0
  critical : |z| > 3.0

Interview answer:
  "I used a rolling Z-score combined with IQR fencing. The rolling
  window adapts to seasonal patterns — a low ridership day in January
  is evaluated against January norms, not annual averages. Requiring
  both methods to agree reduces false positives. Anomalies are surfaced
  as red markers in Tableau with severity labels so planners can
  prioritise investigation."
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Rolling Z-score + IQR dual-method anomaly detector for CTA ridership.

    Parameters
    ----------
    z_threshold : float
        Standard deviations beyond which a point is flagged.
        Default 2.0 — flags roughly ~5% of days under normality.
    window : int
        Rolling window in days for Z-score computation.
    require_both_methods : bool
        If True, both Z-score AND IQR must flag a point.
        If False, either method triggers a flag (higher recall, more noise).
    """

    def __init__(
        self,
        z_threshold: float = 2.0,
        window: int = 14,
        require_both_methods: bool = True,
    ):
        self.z_threshold = z_threshold
        self.window = window
        self.require_both_methods = require_both_methods

    # ─── Public interface ─────────────────────────────────────────────────────

    def detect(
        self,
        actuals_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect anomalies by comparing actuals to Prophet forecast.

        Parameters
        ----------
        actuals_df : pd.DataFrame
            Columns: service_date (or ds), rides, route.
        forecast_df : pd.DataFrame
            Columns: ds, yhat, route.
            Output of ProphetForecaster.predict().

        Returns
        -------
        pd.DataFrame
            Anomaly records with columns:
              route, service_date, actual_rides, forecast_rides,
              residual, z_score, anomaly_type, severity, flagged_at.
            Empty DataFrame if no anomalies found.
        """
        if actuals_df.empty or forecast_df.empty:
            return pd.DataFrame()

        # ── Normalise actuals ────────────────────────────────────────────────
        actuals = actuals_df.copy()
        date_col = "service_date" if "service_date" in actuals.columns else "ds"
        actuals["ds"] = pd.to_datetime(actuals[date_col])
        actuals["rides"] = pd.to_numeric(actuals["rides"], errors="coerce")
        actuals = actuals.dropna(subset=["ds", "rides"])

        # ── Normalise forecast ───────────────────────────────────────────────
        forecasts = forecast_df.copy()
        forecasts["ds"] = pd.to_datetime(forecasts["ds"])

        # ── Merge ────────────────────────────────────────────────────────────
        merged = actuals[["ds", "rides", "route"]].merge(
            forecasts[["ds", "yhat"]], on="ds", how="inner"
        )

        if len(merged) < self.window:
            logger.warning(
                "Not enough data for anomaly detection: %d rows (need %d).",
                len(merged), self.window,
            )
            return pd.DataFrame()

        merged = merged.sort_values("ds").reset_index(drop=True)
        merged["residual"] = merged["rides"] - merged["yhat"]
        merged["abs_residual"] = merged["residual"].abs()

        # ── Method 1: Rolling Z-score ────────────────────────────────────────
        min_periods = max(7, self.window // 2)
        rolling_mean = (
            merged["residual"]
            .rolling(window=self.window, min_periods=min_periods)
            .mean()
        )
        rolling_std = (
            merged["residual"]
            .rolling(window=self.window, min_periods=min_periods)
            .std()
        )

        # Guard against zero std (flat ridership periods)
        fallback_std = merged["abs_residual"].median() or 1.0
        rolling_std_safe = rolling_std.replace(0, np.nan).fillna(fallback_std)

        merged["z_score"] = (merged["residual"] - rolling_mean) / rolling_std_safe
        merged["z_flag"] = merged["z_score"].abs() > self.z_threshold

        # ── Method 2: Global IQR ─────────────────────────────────────────────
        q1 = merged["residual"].quantile(0.25)
        q3 = merged["residual"].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        merged["iqr_flag"] = (
            (merged["residual"] < lower_fence)
            | (merged["residual"] > upper_fence)
        )

        # ── Combine flags ────────────────────────────────────────────────────
        if self.require_both_methods:
            merged["is_anomaly"] = merged["z_flag"] & merged["iqr_flag"]
        else:
            merged["is_anomaly"] = merged["z_flag"] | merged["iqr_flag"]

        anomalies = merged[merged["is_anomaly"]].copy()

        if anomalies.empty:
            route_label = (
                actuals_df["route"].iloc[0]
                if "route" in actuals_df.columns
                else "unknown"
            )
            logger.info("No anomalies detected for route %s.", route_label)
            return pd.DataFrame()

        # ── Classify type and severity ───────────────────────────────────────
        anomalies["anomaly_type"] = anomalies["residual"].apply(
            lambda r: "HIGH_RIDERSHIP" if r > 0 else "LOW_RIDERSHIP"
        )
        anomalies["severity"] = anomalies["z_score"].abs().apply(
            lambda z: "critical" if z > 3.0
            else "moderate" if z > 2.5
            else "minor"
        )
        anomalies["flagged_at"] = pd.Timestamp.utcnow()

        # ── Format output ────────────────────────────────────────────────────
        result = (
            anomalies[[
                "route", "ds", "rides", "yhat",
                "residual", "z_score",
                "anomaly_type", "severity", "flagged_at",
            ]]
            .rename(columns={
                "ds":    "service_date",
                "rides": "actual_rides",
                "yhat":  "forecast_rides",
            })
            .reset_index(drop=True)
        )

        logger.info(
            "Anomaly detection: %d anomalies flagged "
            "(%d LOW_RIDERSHIP, %d HIGH_RIDERSHIP).",
            len(result),
            (result["anomaly_type"] == "LOW_RIDERSHIP").sum(),
            (result["anomaly_type"] == "HIGH_RIDERSHIP").sum(),
        )
        return result

    def detect_from_series(
        self,
        df: pd.DataFrame,
        route: Optional[str] = None,
        baseline_days: int = 30,
    ) -> pd.DataFrame:
        """
        Detect anomalies from a raw ridership series without a Prophet forecast.
        Uses rolling historical mean as the baseline instead of yhat.

        Used in:
          - notebooks/04_anomaly_detection.ipynb for EDA
          - Early pipeline stages when no trained model exists yet

        Parameters
        ----------
        df : pd.DataFrame
            Columns: service_date, rides, route.
        route : str, optional
            Filter to a single route.
        baseline_days : int
            Rolling window for historical mean baseline.

        Returns
        -------
        pd.DataFrame  — same schema as detect()
        """
        df = df.copy()
        if route:
            df = df[df["route"] == route].copy()

        df["ds"] = pd.to_datetime(df["service_date"])
        df["rides"] = pd.to_numeric(df["rides"], errors="coerce")
        df = df.dropna(subset=["ds", "rides"]).sort_values("ds").reset_index(drop=True)

        if len(df) < baseline_days:
            logger.warning(
                "Not enough data for series anomaly detection: %d rows.", len(df)
            )
            return pd.DataFrame()

        # Rolling mean baseline (shift by 1 to avoid look-ahead)
        df["baseline"] = (
            df["rides"]
            .rolling(window=baseline_days, min_periods=14)
            .mean()
            .shift(1)
        )
        df = df.dropna(subset=["baseline"])

        df["residual"] = df["rides"] - df["baseline"]
        rolling_std = (
            df["residual"]
            .rolling(window=baseline_days, min_periods=14)
            .std()
            .shift(1)
            .fillna(df["residual"].std() or 1.0)
        )

        df["z_score"] = df["residual"] / rolling_std
        df["is_anomaly"] = df["z_score"].abs() > self.z_threshold

        anomalies = df[df["is_anomaly"]].copy()
        if anomalies.empty:
            return pd.DataFrame()

        anomalies["anomaly_type"] = anomalies["residual"].apply(
            lambda r: "HIGH_RIDERSHIP" if r > 0 else "LOW_RIDERSHIP"
        )
        anomalies["severity"] = anomalies["z_score"].abs().apply(
            lambda z: "critical" if z > 3.0
            else "moderate" if z > 2.5
            else "minor"
        )
        anomalies["flagged_at"] = pd.Timestamp.utcnow()

        return anomalies[[
            "route", "ds", "rides", "baseline",
            "residual", "z_score", "anomaly_type", "severity", "flagged_at",
        ]].rename(columns={
            "ds":       "service_date",
            "rides":    "actual_rides",
            "baseline": "forecast_rides",
        }).reset_index(drop=True)
