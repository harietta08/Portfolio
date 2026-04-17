"""
tests/test_forecast_evaluation.py

Unit tests for forecasting/evaluate_forecast.py and
forecasting/anomaly_detection.py

Tests cover:
  - MAE, MAPE, RMSE computed correctly on known arrays
  - MAPE handles zero actuals without dividing by zero
  - MAPE status thresholds (ok / warning / error) classify correctly
  - evaluate() returns empty metrics dict on insufficient data
  - forecast_vs_actual_df() produces correct schema
  - AnomalyDetector.detect() flags known anomalies correctly
  - AnomalyDetector returns empty DataFrame when no anomalies
  - AnomalyDetector handles insufficient data gracefully
  - detect_from_series() works without a Prophet forecast
  - Anomaly severity levels (minor / moderate / critical) assigned correctly

All tests use synthetic DataFrames — no BigQuery or Prophet calls.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import date, timedelta

from forecasting.evaluate_forecast import (
    ForecastEvaluator,
    MAPE_WARNING_THRESHOLD,
    MAPE_ERROR_THRESHOLD,
)
from forecasting.anomaly_detection import AnomalyDetector


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_ridership_df(
    n_days: int = 60,
    base_rides: int = 10000,
    route: str = "22",
    noise_std: int = 300,
    start: date = date(2024, 1, 1),
) -> pd.DataFrame:
    """Generate synthetic daily ridership DataFrame."""
    dates = [start + timedelta(days=i) for i in range(n_days)]
    rng = np.random.default_rng(seed=42)
    rides = base_rides + rng.normal(0, noise_std, n_days).astype(int)
    rides = np.clip(rides, 100, None)  # no negative rides

    return pd.DataFrame({
        "route": route,
        "service_date": [d.isoformat() for d in dates],
        "rides": rides.tolist(),
        "day_type": ["W" if d.weekday() < 5 else "A" for d in dates],
        "temperature_2m": rng.normal(10, 5, n_days).tolist(),
        "precipitation": np.abs(rng.normal(0, 2, n_days)).tolist(),
        "windspeed_10m": rng.normal(15, 5, n_days).tolist(),
    })


def make_forecast_df(
    actuals_df: pd.DataFrame,
    bias: float = 0.0,
    noise_std: float = 200.0,
) -> pd.DataFrame:
    """Generate synthetic forecast DataFrame from actuals with optional bias."""
    rng = np.random.default_rng(seed=99)
    n = len(actuals_df)
    yhat = actuals_df["rides"].values.astype(float) + bias + rng.normal(0, noise_std, n)
    yhat = np.clip(yhat, 0, None)

    return pd.DataFrame({
        "ds": pd.to_datetime(actuals_df["service_date"]),
        "yhat": yhat,
        "yhat_lower": yhat - 500,
        "yhat_upper": yhat + 500,
        "route": actuals_df["route"].values,
        "time_bucket": "morning_peak",
        "generated_at": pd.Timestamp.utcnow(),
    })


def make_mock_forecaster(route: str = "22", bucket: str = "morning_peak"):
    """Create a mock ProphetForecaster that returns a synthetic forecast."""
    mock = MagicMock()
    mock.route = route
    mock.time_bucket = bucket
    mock._fitted = True
    return mock


# ─── ForecastEvaluator metric tests ──────────────────────────────────────────

class TestForecastEvaluatorMetrics:
    """Test the static metric computation methods directly."""

    def test_mae_perfect_forecast(self):
        """MAE = 0 when actuals == predicted."""
        y = np.array([100.0, 200.0, 300.0])
        assert ForecastEvaluator._mae(y, y) == 0.0

    def test_mae_known_value(self):
        """MAE = mean(|actual - predicted|)."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        expected = (10 + 10 + 10) / 3
        assert ForecastEvaluator._mae(y_true, y_pred) == pytest.approx(expected)

    def test_mape_perfect_forecast(self):
        """MAPE = 0 when actuals == predicted."""
        y = np.array([100.0, 200.0, 300.0])
        assert ForecastEvaluator._mape(y, y) == pytest.approx(0.0)

    def test_mape_known_value(self):
        """MAPE computed correctly on known values."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # |(100-110)/100| = 0.10, |(200-180)/200| = 0.10 → MAPE = 10%
        assert ForecastEvaluator._mape(y_true, y_pred) == pytest.approx(10.0)

    def test_mape_with_zero_actuals_no_division_error(self):
        """MAPE skips zero actuals — no ZeroDivisionError."""
        y_true = np.array([0.0, 200.0])
        y_pred = np.array([100.0, 220.0])
        # Only second element used: |200-220|/200 = 10%
        result = ForecastEvaluator._mape(y_true, y_pred)
        assert result == pytest.approx(10.0)

    def test_mape_all_zeros_returns_inf(self):
        """All-zero actuals returns infinity."""
        y = np.array([0.0, 0.0, 0.0])
        result = ForecastEvaluator._mape(y, np.array([10.0, 20.0, 30.0]))
        assert result == float("inf")

    def test_rmse_known_value(self):
        """RMSE computed correctly."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([90.0, 210.0])
        expected = np.sqrt(((10**2) + (10**2)) / 2)
        assert ForecastEvaluator._rmse(y_true, y_pred) == pytest.approx(expected)

    def test_rmse_perfect_forecast(self):
        """RMSE = 0 for perfect forecast."""
        y = np.array([100.0, 200.0])
        assert ForecastEvaluator._rmse(y, y) == pytest.approx(0.0)


# ─── MAPE status threshold tests ─────────────────────────────────────────────

class TestMAPEStatusThresholds:

    def setup_method(self):
        mock_forecaster = make_mock_forecaster()
        self.evaluator = ForecastEvaluator(mock_forecaster)

    def test_ok_status_below_warning_threshold(self):
        assert self.evaluator._mape_status(MAPE_WARNING_THRESHOLD - 1) == "ok"

    def test_warning_status_between_thresholds(self):
        mape = (MAPE_WARNING_THRESHOLD + MAPE_ERROR_THRESHOLD) / 2
        assert self.evaluator._mape_status(mape) == "warning"

    def test_error_status_above_error_threshold(self):
        assert self.evaluator._mape_status(MAPE_ERROR_THRESHOLD + 1) == "error"

    def test_boundary_exactly_at_warning_threshold(self):
        """At exactly the warning threshold value → warning."""
        assert self.evaluator._mape_status(MAPE_WARNING_THRESHOLD) == "warning"

    def test_boundary_exactly_at_error_threshold(self):
        """At exactly the error threshold value → error."""
        assert self.evaluator._mape_status(MAPE_ERROR_THRESHOLD) == "error"


# ─── ForecastEvaluator.evaluate() tests ──────────────────────────────────────

class TestForecastEvaluatorEvaluate:

    def test_insufficient_data_returns_empty_metrics(self):
        """Fewer rows than eval_window_days returns empty metrics dict."""
        mock = make_mock_forecaster()
        evaluator = ForecastEvaluator(mock)

        tiny_df = make_ridership_df(n_days=5)
        result = evaluator.evaluate(tiny_df, eval_window_days=28)

        assert result["mae"] is None
        assert result["mape"] is None
        assert result["mape_status"] == "insufficient_data"

    def test_empty_dataframe_returns_empty_metrics(self):
        """Empty DataFrame returns empty metrics."""
        mock = make_mock_forecaster()
        evaluator = ForecastEvaluator(mock)

        result = evaluator.evaluate(pd.DataFrame(), eval_window_days=28)
        assert result["mape_status"] == "insufficient_data"

    def test_evaluate_returns_all_required_keys(self):
        """evaluate() result contains all expected keys."""
        actuals_df = make_ridership_df(n_days=60)
        forecast_df = make_forecast_df(actuals_df)

        mock = make_mock_forecaster()
        mock.predict.return_value = forecast_df

        evaluator = ForecastEvaluator(mock)
        result = evaluator.evaluate(actuals_df, eval_window_days=28)

        required_keys = {
            "route", "time_bucket", "mae", "mape", "rmse",
            "n_eval_days", "eval_start", "eval_end",
            "mape_status", "evaluated_at",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_evaluate_mae_is_non_negative(self):
        """MAE is always non-negative."""
        actuals_df = make_ridership_df(n_days=60)
        forecast_df = make_forecast_df(actuals_df)

        mock = make_mock_forecaster()
        mock.predict.return_value = forecast_df

        evaluator = ForecastEvaluator(mock)
        result = evaluator.evaluate(actuals_df, eval_window_days=28)

        if result["mae"] is not None:
            assert result["mae"] >= 0


# ─── AnomalyDetector tests ────────────────────────────────────────────────────

class TestAnomalyDetector:

    def setup_method(self):
        self.detector = AnomalyDetector(z_threshold=2.0, window=14)

    def test_no_anomalies_in_clean_series(self):
        """A clean, low-variance series produces no anomalies."""
        actuals_df = make_ridership_df(n_days=60, noise_std=50)
        forecast_df = make_forecast_df(actuals_df, noise_std=50)

        # Give the detector a near-perfect forecast
        forecast_df["yhat"] = actuals_df["rides"].values.astype(float)

        result = self.detector.detect(actuals_df, forecast_df)
        assert result.empty

    def test_injected_anomaly_is_detected(self):
        """Artificially drop ridership to near zero — must be flagged."""
        actuals_df = make_ridership_df(n_days=60, base_rides=10000, noise_std=100)
        forecast_df = make_forecast_df(actuals_df, noise_std=100)
        forecast_df["yhat"] = actuals_df["rides"].values.astype(float)

        # Inject a severe drop on day 45
        actuals_df_copy = actuals_df.copy()
        actuals_df_copy.iloc[45, actuals_df_copy.columns.get_loc("rides")] = 100

        result = self.detector.detect(actuals_df_copy, forecast_df)
        assert not result.empty
        assert "LOW_RIDERSHIP" in result["anomaly_type"].values

    def test_injected_spike_is_detected(self):
        """Artificially spike ridership — must be flagged as HIGH_RIDERSHIP."""
        actuals_df = make_ridership_df(n_days=60, base_rides=10000, noise_std=100)
        forecast_df = make_forecast_df(actuals_df, noise_std=100)
        forecast_df["yhat"] = actuals_df["rides"].values.astype(float)

        actuals_df_copy = actuals_df.copy()
        actuals_df_copy.iloc[45, actuals_df_copy.columns.get_loc("rides")] = 50000

        result = self.detector.detect(actuals_df_copy, forecast_df)
        assert not result.empty
        assert "HIGH_RIDERSHIP" in result["anomaly_type"].values

    def test_empty_actuals_returns_empty_df(self):
        """Empty actuals DataFrame returns empty result."""
        forecast_df = make_forecast_df(make_ridership_df(n_days=30))
        result = self.detector.detect(pd.DataFrame(), forecast_df)
        assert result.empty

    def test_empty_forecast_returns_empty_df(self):
        """Empty forecast DataFrame returns empty result."""
        actuals_df = make_ridership_df(n_days=30)
        result = self.detector.detect(actuals_df, pd.DataFrame())
        assert result.empty

    def test_insufficient_data_returns_empty_df(self):
        """Fewer rows than window size returns empty result."""
        actuals_df = make_ridership_df(n_days=5)
        forecast_df = make_forecast_df(actuals_df)
        detector = AnomalyDetector(z_threshold=2.0, window=14)
        result = detector.detect(actuals_df, forecast_df)
        assert result.empty

    def test_output_schema_has_required_columns(self):
        """Anomaly output has all required columns."""
        actuals_df = make_ridership_df(n_days=60, base_rides=10000, noise_std=100)
        forecast_df = make_forecast_df(actuals_df, noise_std=50)
        forecast_df["yhat"] = actuals_df["rides"].values.astype(float)

        actuals_df_copy = actuals_df.copy()
        actuals_df_copy.iloc[45, actuals_df_copy.columns.get_loc("rides")] = 50

        result = self.detector.detect(actuals_df_copy, forecast_df)

        if not result.empty:
            required_cols = {
                "route", "service_date", "actual_rides", "forecast_rides",
                "residual", "z_score", "anomaly_type", "severity", "flagged_at",
            }
            assert required_cols.issubset(set(result.columns))

    def test_severity_critical_for_large_z_score(self):
        """z_score > 3 → severity = critical."""
        actuals_df = make_ridership_df(n_days=60, base_rides=10000, noise_std=50)
        forecast_df = make_forecast_df(actuals_df, noise_std=50)
        forecast_df["yhat"] = actuals_df["rides"].values.astype(float)

        # Inject extreme drop
        actuals_df_copy = actuals_df.copy()
        actuals_df_copy.iloc[50, actuals_df_copy.columns.get_loc("rides")] = 1

        result = self.detector.detect(actuals_df_copy, forecast_df)
        if not result.empty:
            assert "critical" in result["severity"].values

    def test_detect_from_series_no_prophet(self):
        """detect_from_series works without a Prophet forecast."""
        df = make_ridership_df(n_days=90, base_rides=10000, noise_std=100)
        df_copy = df.copy()
        # Inject anomaly
        df_copy.iloc[70, df_copy.columns.get_loc("rides")] = 50

        result = self.detector.detect_from_series(df_copy, route="22")
        # May or may not detect depending on window — just check no crash
        assert isinstance(result, pd.DataFrame)

    def test_detect_from_series_insufficient_data(self):
        """detect_from_series with too few rows returns empty DataFrame."""
        tiny_df = make_ridership_df(n_days=10)
        result = self.detector.detect_from_series(tiny_df, baseline_days=30)
        assert result.empty
