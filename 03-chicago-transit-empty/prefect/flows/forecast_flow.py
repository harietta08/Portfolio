"""
prefect/flows/forecast_flow.py

Prefect forecast refresh flow — weekly Prophet model scoring,
evaluation, and anomaly detection.

Schedule: Mondays at 7 AM CT (weekly refresh after weekend data lands).

Flow steps:
  1. Pull last 90 days of ridership + weather from BigQuery
  2. For each route:
     a. Load saved Prophet model or train a new one if missing / retrain=True
     b. Generate 28-day forward forecast
     c. Evaluate MAE and MAPE on last 7 days of actuals
     d. Detect anomalies where |actual - forecast| > 2 * rolling std
     e. Write forecast rows to BigQuery forecast_results table
  3. Return per-route metrics summary for Prefect Cloud dashboard

Run locally:
    python -m prefect.flows.forecast_flow
    or: make forecast
"""

import json
import logging
import os
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger

load_dotenv()

from google.cloud import bigquery
from forecasting.train_prophet import ProphetForecaster
from forecasting.evaluate_forecast import ForecastEvaluator
from forecasting.anomaly_detection import AnomalyDetector

logger = logging.getLogger(__name__)

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
BQ_DATASET = os.environ.get("BIGQUERY_DATASET", "chicago_transit")
FORECAST_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.forecast_results"

# Trained model artifacts stored locally
# In production: replace with GCS bucket read/write
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET_ROUTES = ["22", "77", "66", "151", "6", "36", "49", "82"]
TIME_BUCKETS = ["morning_peak", "midday", "evening_peak", "off_peak"]


# ─── Tasks ────────────────────────────────────────────────────────────────────

@task(retries=2, retry_delay_seconds=60, name="pull-ridership-from-bq")
def pull_ridership_from_bq(
    routes: list[str],
    lookback_days: int = 90,
) -> pd.DataFrame:
    """
    Pull ridership + weather data from BigQuery for training and evaluation.

    Returns
    -------
    pd.DataFrame
        Columns: route, service_date, day_type, rides,
                 temperature_2m, precipitation, windspeed_10m,
                 weathercode, is_precipitation.
    """
    flow_logger = get_run_logger()
    client = bigquery.Client(project=PROJECT_ID)

    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    routes_str = ", ".join(f"'{r}'" for r in routes)

    query = f"""
        SELECT
            route,
            service_date,
            day_type,
            rides,
            temperature_2m,
            precipitation,
            windspeed_10m,
            weathercode,
            is_precipitation
        FROM `{PROJECT_ID}.{BQ_DATASET}.transit_events`
        WHERE service_date >= '{cutoff}'
          AND route IN ({routes_str})
        ORDER BY route, service_date ASC
    """

    flow_logger.info(
        "Pulling ridership from BigQuery: last %d days, %d routes.",
        lookback_days, len(routes),
    )
    df = client.query(query).to_dataframe()
    flow_logger.info("Pulled %d rows from BigQuery.", len(df))
    return df


@task(name="train-or-load-model")
def train_or_load_model(
    df: pd.DataFrame,
    route: str,
    time_bucket: str,
    retrain: bool = False,
) -> Optional[ProphetForecaster]:
    """
    Load a saved Prophet model from disk, or train a new one.

    Parameters
    ----------
    df : pd.DataFrame
        Full ridership dataset from BigQuery.
    route : str
    time_bucket : str
    retrain : bool
        If True, ignore any saved model and retrain from scratch.

    Returns
    -------
    ProphetForecaster or None if insufficient data.
    """
    flow_logger = get_run_logger()
    model_path = MODEL_DIR / f"prophet_{route}_{time_bucket}.pkl"

    if model_path.exists() and not retrain:
        flow_logger.info(
            "Loading saved model: route=%s bucket=%s", route, time_bucket
        )
        with open(model_path, "rb") as f:
            return pickle.load(f)

    flow_logger.info(
        "Training new Prophet model: route=%s bucket=%s", route, time_bucket
    )
    route_df = df[df["route"] == route].copy()

    if len(route_df) < 14:
        flow_logger.warning(
            "Insufficient data for route=%s (%d rows, need ≥14). Skipping.",
            route, len(route_df),
        )
        return None

    forecaster = ProphetForecaster(route=route, time_bucket=time_bucket)
    try:
        forecaster.fit(route_df)
    except ValueError as exc:
        flow_logger.error(
            "Model training failed for route=%s bucket=%s: %s",
            route, time_bucket, exc,
        )
        return None

    with open(model_path, "wb") as f:
        pickle.dump(forecaster, f)
    flow_logger.info("Model saved to %s", model_path)
    return forecaster


@task(name="generate-forecast")
def generate_forecast(
    forecaster: Optional[ProphetForecaster],
    route: str,
    horizon_days: int = 28,
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate a 28-day forward forecast for a single route.

    Returns
    -------
    pd.DataFrame
        Columns: ds, yhat, yhat_lower, yhat_upper, route, time_bucket, generated_at.
        Empty DataFrame if forecaster is None.
    """
    flow_logger = get_run_logger()
    if forecaster is None:
        flow_logger.warning(
            "No forecaster for route=%s — skipping forecast.", route
        )
        return pd.DataFrame()

    route_df = (
        history_df[history_df["route"] == route].copy()
        if history_df is not None
        else None
    )

    forecast_df = forecaster.predict(
        horizon_days=horizon_days,
        history_df=route_df,
    )
    flow_logger.info(
        "Generated %d forecast rows for route=%s.", len(forecast_df), route
    )
    return forecast_df


@task(name="evaluate-forecast-accuracy")
def evaluate_forecast_accuracy(
    forecaster: Optional[ProphetForecaster],
    actuals_df: pd.DataFrame,
    route: str,
    eval_days: int = 7,
) -> dict:
    """
    Compute MAE and MAPE for the last eval_days of actuals.

    Returns
    -------
    dict
        {route, time_bucket, mae, mape, rmse, n_eval_days, mape_status}
    """
    flow_logger = get_run_logger()
    if forecaster is None:
        return {
            "route": route,
            "time_bucket": "unknown",
            "mae": None,
            "mape": None,
            "mape_status": "no_model",
        }

    evaluator = ForecastEvaluator(forecaster)
    route_df = actuals_df[actuals_df["route"] == route].copy()
    metrics = evaluator.evaluate(route_df, eval_window_days=eval_days)

    flow_logger.info(
        "Evaluation route=%s bucket=%s: MAE=%.1f MAPE=%.2f%% status=%s",
        route,
        metrics.get("time_bucket", "?"),
        metrics.get("mae") or 0,
        metrics.get("mape") or 0,
        metrics.get("mape_status", "?"),
    )
    return metrics


@task(name="detect-anomalies")
def detect_anomalies_task(
    actuals_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    route: str,
) -> pd.DataFrame:
    """
    Flag service disruptions: actual ridership deviates from forecast
    by more than 2 rolling standard deviations.

    Returns
    -------
    pd.DataFrame
        Anomaly records. Empty if none detected.
    """
    flow_logger = get_run_logger()
    if forecast_df.empty:
        return pd.DataFrame()

    detector = AnomalyDetector(z_threshold=2.0, window=14)
    route_actuals = actuals_df[actuals_df["route"] == route].copy()
    anomalies = detector.detect(route_actuals, forecast_df)

    if not anomalies.empty:
        flow_logger.info(
            "Route %s: %d anomalies detected (%d LOW, %d HIGH).",
            route,
            len(anomalies),
            (anomalies["anomaly_type"] == "LOW_RIDERSHIP").sum(),
            (anomalies["anomaly_type"] == "HIGH_RIDERSHIP").sum(),
        )
    else:
        flow_logger.info("Route %s: no anomalies detected.", route)

    return anomalies


@task(retries=2, retry_delay_seconds=30, name="write-forecast-to-bq")
def write_forecast_to_bq(
    forecast_df: pd.DataFrame,
    route: str,
) -> int:
    """
    Write forecast results to BigQuery forecast_results table.
    Idempotent on (route, ds, time_bucket).

    Returns
    -------
    int  — rows written
    """
    flow_logger = get_run_logger()
    if forecast_df.empty:
        return 0

    client = bigquery.Client(project=PROJECT_ID)
    rows = forecast_df.to_dict(orient="records")

    # Normalize timestamps to ISO strings for BigQuery JSON insert
    for row in rows:
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                row[k] = v.isoformat()
            elif hasattr(v, "item"):
                # Convert numpy scalars to Python native types
                row[k] = v.item()

    errors = client.insert_rows_json(FORECAST_TABLE, rows)
    if errors:
        flow_logger.error(
            "BQ forecast insert errors for route %s: %s", route, errors
        )
        return len(rows) - len(errors)

    flow_logger.info(
        "Wrote %d forecast rows for route %s to %s.",
        len(rows), route, FORECAST_TABLE,
    )
    return len(rows)


# ─── Flow ─────────────────────────────────────────────────────────────────────

@flow(
    name="cta-forecast-refresh",
    description=(
        "Weekly Prophet forecast refresh: pulls BQ data → trains/loads models "
        "→ generates forecasts → evaluates accuracy → detects anomalies → "
        "writes results to BigQuery."
    ),
    log_prints=True,
)
def forecast_flow(
    routes: Optional[list[str]] = None,
    horizon_days: int = 28,
    eval_days: int = 7,
    retrain: bool = False,
    lookback_days: int = 90,
) -> dict:
    """
    Weekly forecast refresh flow.

    Parameters
    ----------
    routes : list[str], optional
        Defaults to TARGET_ROUTES.
    horizon_days : int
        Days ahead to forecast (default 28).
    eval_days : int
        Evaluation window for MAE/MAPE (default 7 = last week).
    retrain : bool
        Force model retraining even if saved model exists.
    lookback_days : int
        Days of history to pull from BigQuery (default 90).

    Returns
    -------
    dict
        Per route-bucket metrics and anomaly counts.
        {
          "22_morning_peak": {"mae": 312.1, "mape": 4.2, "n_anomalies": 0},
          ...
        }
    """
    flow_logger = get_run_logger()
    routes = routes or TARGET_ROUTES

    flow_logger.info(
        "Forecast flow started: %d routes | horizon=%dd | eval=%dd | retrain=%s",
        len(routes), horizon_days, eval_days, retrain,
    )

    # Pull data once, share across all route/bucket iterations
    ridership_df = pull_ridership_from_bq(routes, lookback_days=lookback_days)

    if ridership_df.empty:
        flow_logger.error(
            "No data returned from BigQuery. "
            "Ensure ingest_flow has run successfully first."
        )
        return {}

    summary = {}
    total_anomalies = 0

    for route in routes:
        flow_logger.info("══ Route %s ══", route)

        for bucket in TIME_BUCKETS:
            flow_logger.info("  Bucket: %s", bucket)

            forecaster = train_or_load_model(
                ridership_df, route, bucket, retrain
            )
            forecast_df = generate_forecast(
                forecaster, route, horizon_days, ridership_df
            )
            metrics = evaluate_forecast_accuracy(
                forecaster, ridership_df, route, eval_days
            )
            anomalies = detect_anomalies_task(
                ridership_df, forecast_df, route
            )

            if not forecast_df.empty:
                write_forecast_to_bq(forecast_df, route)

            key = f"{route}_{bucket}"
            summary[key] = {
                "mae": metrics.get("mae"),
                "mape": metrics.get("mape"),
                "mape_status": metrics.get("mape_status"),
                "n_forecast_rows": len(forecast_df),
                "n_anomalies": len(anomalies),
            }
            total_anomalies += len(anomalies)

    flow_logger.info(
        "Forecast flow complete. "
        "%d route-bucket combinations | %d total anomalies flagged.",
        len(summary), total_anomalies,
    )
    return summary


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = forecast_flow()
    print(json.dumps(result, indent=2, default=str))
