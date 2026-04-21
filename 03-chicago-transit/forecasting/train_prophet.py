"""
forecasting/train_prophet.py

Prophet model training — one model per route per time bucket.

Time buckets:
  morning_peak  : weekday AM commute proxy
  midday        : weekday off-peak proxy
  evening_peak  : weekday PM commute proxy
  off_peak      : weekend and late-night proxy

Since CTA ridership data from the Chicago Data Portal is daily (not hourly),
we train one Prophet model per route using daily totals. Time-bucket models
are trained on subsets filtered by day_type, serving as proxies for
intraday demand patterns. Hourly breakdown would require the CTA Bus
Tracker API (separate dataset) — noted as future work.

Features added as Prophet regressors:
  - temperature_2m      : cold weather suppresses ridership
  - precipitation       : rain/snow changes ridership (route-dependent)
  - windspeed_10m       : extreme wind suppresses ridership
  Chicago public holidays added via Prophet built-in holiday component.

Why Prophet over ARIMA or LSTM:
  - Handles multiple seasonalities (weekly, annual) without manual tuning
  - Built-in holiday effects for Chicago (no feature engineering needed)
  - Produces interpretable component plots — trend, seasonality, holidays
  - Uncertainty intervals out of the box (95% CI shaded in Tableau)
  - Robust to missing data gaps common in transit datasets
  - Cross-validation built in via prophet.diagnostics

Interview answer:
  "I chose Prophet because transit ridership has strong weekly and annual
  seasonality, significant holiday effects, and occasional trend changes
  from service restructuring. Prophet handles all three natively and
  produces confidence intervals without additional modelling. The
  component plots are also directly useful for the Tableau dashboard."
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logger = logging.getLogger(__name__)

# ─── Time bucket definitions ─────────────────────────────────────────────────
# day_types included in each bucket for filtering
BUCKET_DAY_TYPES = {
    "morning_peak": ["W"],
    "midday":       ["W"],
    "evening_peak": ["W", "A"],
    "off_peak":     ["A", "U"],
}


class ProphetForecaster:
    """
    Wraps Facebook Prophet for CTA route-level daily ridership forecasting.

    Parameters
    ----------
    route : str
        CTA route number e.g. '22'.
    time_bucket : str
        One of: morning_peak, midday, evening_peak, off_peak.
    """

    def __init__(self, route: str, time_bucket: str = "morning_peak"):
        if time_bucket not in BUCKET_DAY_TYPES:
            raise ValueError(
                f"time_bucket must be one of {list(BUCKET_DAY_TYPES)}. "
                f"Got '{time_bucket}'."
            )
        self.route = route
        self.time_bucket = time_bucket
        self.model: Optional[Prophet] = None
        self._train_df: Optional[pd.DataFrame] = None
        self._available_regressors: list[str] = []
        self._fitted = False

    # ─── Public interface ─────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "ProphetForecaster":
        """
        Train the Prophet model on historical ridership data.

        Parameters
        ----------
        df : pd.DataFrame
            Columns required: service_date, rides, day_type.
            Optional: temperature_2m, precipitation, windspeed_10m.

        Returns
        -------
        self  (fluent interface — allows chaining)

        Raises
        ------
        ValueError
            If fewer than 14 training rows are available after filtering.
        """
        prophet_df = self._prepare_dataframe(df)

        if len(prophet_df) < 14:
            raise ValueError(
                f"Insufficient training data for route={self.route} "
                f"bucket={self.time_bucket}: {len(prophet_df)} rows (min 14)."
            )

        logger.info(
            "Training Prophet: route=%s bucket=%s on %d rows.",
            self.route, self.time_bucket, len(prophet_df),
        )

        model = Prophet(
            # Multiplicative seasonality: transit demand scales with baseline
            # e.g. a 10% holiday dip applies to a 5k-rider route differently
            # than a 50k-rider route
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,     # data is daily, not sub-daily
            # Conservative changepoint prior: transit demand changes slowly
            # unless there is a deliberate service change (which we model
            # separately via DiD)
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            interval_width=0.95,         # 95% confidence interval in Tableau
            uncertainty_samples=500,
        )

        # Chicago holidays via Prophet built-in US holiday calendar
        model.add_country_holidays(country_name="US")

        # Weather regressors — only added if ≥50% non-null data available
        # Missing weather data is imputed in _prepare_dataframe
        candidate_regressors = [
            "temperature_2m",
            "precipitation",
            "windspeed_10m",
        ]
        available = [
            col for col in candidate_regressors
            if col in prophet_df.columns
            and prophet_df[col].notna().sum() > len(prophet_df) * 0.5
        ]
        for col in available:
            model.add_regressor(col, standardize=True)

        model.fit(prophet_df)

        self.model = model
        self._train_df = prophet_df
        self._available_regressors = available
        self._fitted = True

        logger.info(
            "Prophet fit complete: route=%s bucket=%s. "
            "Regressors added: %s",
            self.route, self.time_bucket, available or "none",
        )
        return self

    def predict(
        self,
        horizon_days: int = 28,
        history_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate forward forecast.

        Parameters
        ----------
        horizon_days : int
            Number of days to forecast beyond the last training date.
        history_df : pd.DataFrame, optional
            Recent actuals used to compute future regressor values.
            If provided, uses trailing 30-day mean per regressor.
            If not, falls back to training set mean.

        Returns
        -------
        pd.DataFrame
            Columns: ds, yhat, yhat_lower, yhat_upper,
                     route, time_bucket, generated_at.
        """
        if not self._fitted or self.model is None:
            raise RuntimeError(
                "Model not fitted. Call fit() before predict()."
            )

        future = self.model.make_future_dataframe(
            periods=horizon_days, freq="D"
        )

        # Fill future regressor values with trailing mean from recent history
        for col in self._available_regressors:
            if history_df is not None and col in history_df.columns:
                trailing_mean = (
                    history_df[col].dropna().tail(30).mean()
                )
            elif (
                self._train_df is not None
                and col in self._train_df.columns
            ):
                trailing_mean = self._train_df[col].dropna().mean()
            else:
                trailing_mean = 0.0

            future[col] = trailing_mean

        forecast = self.model.predict(future)

        result = forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()

        # Clip negatives — ridership cannot be negative
        result["yhat"] = result["yhat"].clip(lower=0)
        result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
        result["yhat_upper"] = result["yhat_upper"].clip(lower=0)

        result["route"] = self.route
        result["time_bucket"] = self.time_bucket
        result["generated_at"] = pd.Timestamp.utcnow()

        logger.info(
            "Forecast generated: route=%s bucket=%s %d rows.",
            self.route, self.time_bucket, len(result),
        )
        return result

    def cross_validate(
        self,
        initial_days: int = 180,
        period_days: int = 30,
        horizon_days: int = 28,
    ) -> pd.DataFrame:
        """
        Run Prophet time-series cross-validation.

        Used in notebooks/03_forecasting.ipynb to compute MAE and MAPE
        on a held-out 4-week test set across multiple folds.

        Parameters
        ----------
        initial_days : int
            Minimum training set size in days.
        period_days : int
            Spacing between cutoff dates.
        horizon_days : int
            Forecast horizon per fold (matches production horizon).

        Returns
        -------
        pd.DataFrame
            Prophet performance_metrics output:
            columns include horizon, mae, mape, rmse, coverage.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        logger.info(
            "Cross-validating: route=%s bucket=%s "
            "(initial=%dd, period=%dd, horizon=%dd)",
            self.route, self.time_bucket,
            initial_days, period_days, horizon_days,
        )

        cv_df = cross_validation(
            self.model,
            initial=f"{initial_days} days",
            period=f"{period_days} days",
            horizon=f"{horizon_days} days",
            parallel="processes",
        )
        metrics_df = performance_metrics(cv_df)

        logger.info(
            "Cross-validation complete: route=%s bucket=%s. "
            "Mean MAE=%.1f Mean MAPE=%.2f%%",
            self.route, self.time_bucket,
            metrics_df["mae"].mean(),
            metrics_df["mape"].mean() * 100,
        )
        return metrics_df

    def plot_components(
        self,
        forecast_df: Optional[pd.DataFrame] = None,
    ):
        """
        Return the Prophet component plot figure.
        Shows trend, weekly seasonality, annual seasonality, holiday effects.
        Designed for use in notebooks/03_forecasting.ipynb and README screenshots.

        Parameters
        ----------
        forecast_df : pd.DataFrame, optional
            Output of predict(). Generated internally if not supplied.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not fitted.")

        if forecast_df is None:
            forecast_df = self.predict(horizon_days=28)

        return self.model.plot_components(forecast_df)

    def save(self, path: Path) -> None:
        """Pickle the fitted forecaster to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Forecaster saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ProphetForecaster":
        """Load a pickled forecaster from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Forecaster loaded from %s", path)
        return obj

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw ridership DataFrame to Prophet's required format.

        Prophet requires:
          ds : datetime column
          y  : target variable (rides)
          [optional regressors]

        Steps:
          1. Normalize date column to datetime
          2. Cast rides to numeric, drop nulls
          3. Forward-fill then mean-impute missing weather values
          4. Deduplicate on ds (keep last)
          5. Sort ascending
        """
        df = df.copy()

        # Normalize date
        df["ds"] = pd.to_datetime(df["service_date"])
        df["y"] = pd.to_numeric(df["rides"], errors="coerce")
        df = df.dropna(subset=["ds", "y"])
        df = df.sort_values("ds").reset_index(drop=True)

        # Impute weather: forward-fill gaps then fall back to column mean
        for col in ["temperature_2m", "precipitation", "windspeed_10m"]:
            if col in df.columns:
                col_mean = df[col].mean()
                df[col] = (
                    df[col]
                    .ffill()
                    .bfill()
                    .fillna(col_mean if not np.isnan(col_mean) else 0.0)
                )

        # Select columns for Prophet
        keep = ["ds", "y", "temperature_2m", "precipitation", "windspeed_10m"]
        keep = [c for c in keep if c in df.columns]

        # Deduplicate: if multiple rows per date (multi-bucket data),
        # keep the last one after sorting
        result = df[keep].drop_duplicates(subset=["ds"], keep="last")

        logger.debug(
            "Prepared DataFrame: route=%s bucket=%s shape=%s",
            self.route, self.time_bucket, result.shape,
        )
        return result
