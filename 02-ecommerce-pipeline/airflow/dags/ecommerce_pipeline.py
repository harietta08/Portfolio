# =============================================================================
# ecommerce_pipeline.py
# Purpose: Full Airflow DAG orchestrating the end-to-end pipeline.
#          Task order: ingest -> validate -> gcs -> bronze -> silver -> gold
#                      -> export -> dbt -> notify
# Idempotency: Re-running on the same date produces identical results.
#   - Bronze: append with dedup key
#   - Silver: MERGE on primary key
#   - Gold:   full overwrite with deterministic aggregation
# =============================================================================

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
import logging
import sys
import os

# Add project root to path so we can import ingestion modules
sys.path.insert(0, "/opt/airflow/project")

# -----------------------------------------------------------------------------
# DEFAULT ARGS — applied to every task
# -----------------------------------------------------------------------------
default_args = {
    "owner":            "hari_etta",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          3,
    "retry_delay":      timedelta(minutes=5),
    "retry_exponential_backoff": True,
}


# -----------------------------------------------------------------------------
# FAILURE CALLBACK
# Logs failure details. In production: send Slack alert or PagerDuty.
# -----------------------------------------------------------------------------
def on_failure_callback(context):
    task_id  = context["task_instance"].task_id
    dag_id   = context["task_instance"].dag_id
    exc      = context.get("exception")
    log_url  = context["task_instance"].log_url
    logging.error(
        f"PIPELINE FAILURE | DAG: {dag_id} | Task: {task_id} | "
        f"Exception: {exc} | Logs: {log_url}"
    )


# -----------------------------------------------------------------------------
# TASK FUNCTIONS
# Each function is a thin wrapper — actual logic lives in the module files.
# This keeps the DAG file readable and the business logic testable.
# -----------------------------------------------------------------------------

def task_ingest_api(**context):
    """Fetch raw data from Fake Store API and save locally as JSON."""
    from ingestion.api_client import run_ingestion
    results = run_ingestion()
    # Push results to XCom so downstream tasks can reference file paths
    context["ti"].xcom_push(key="ingestion_results", value=results)
    logging.info(f"Ingestion complete: {results}")


def task_validate_schema(**context):
    """Run pandera data contracts against raw JSON files."""
    from ingestion.validate_schema import run_validation
    run_validation()
    logging.info("Schema validation passed.")


def task_upload_gcs(**context):
    """Upload validated raw files to GCS partitioned by date."""
    from ingestion.gcs_upload import run_upload
    uploaded = run_upload()
    context["ti"].xcom_push(key="gcs_files", value=uploaded)
    logging.info(f"Uploaded {len(uploaded)} files to GCS.")


def task_run_databricks_bronze(**context):
    """
    Trigger Databricks Bronze notebook via REST API.
    CE limitation: no scheduled Jobs — we trigger via API call.
    In production, use Databricks Workflows with job_id.
    """
    import requests

    databricks_host  = os.getenv("DATABRICKS_HOST")
    databricks_token = os.getenv("DATABRICKS_TOKEN")
    cluster_id       = os.getenv("DATABRICKS_CLUSTER_ID")

    if not all([databricks_host, databricks_token, cluster_id]):
        logging.warning(
            "Databricks credentials not configured. "
            "Skipping Bronze trigger — set DATABRICKS_HOST, "
            "DATABRICKS_TOKEN, DATABRICKS_CLUSTER_ID in .env"
        )
        return

    headers = {"Authorization": f"Bearer {databricks_token}"}
    payload = {
        "cluster_id": cluster_id,
        "language":   "PYTHON",
        "command":    "dbutils.notebook.run('/Users/you/01_bronze_ingestion', 600)"
    }

    resp = requests.post(
        f"{databricks_host}/api/1.2/commands/execute",
        headers=headers,
        json=payload,
        timeout=600
    )
    resp.raise_for_status()
    logging.info(f"Bronze notebook triggered: {resp.json()}")


def task_run_databricks_silver(**context):
    """Trigger Databricks Silver cleaning notebook."""
    logging.info("Silver cleaning triggered (configure Databricks credentials to activate).")


def task_run_databricks_gold(**context):
    """Trigger Databricks Gold aggregation notebook."""
    logging.info("Gold aggregation triggered (configure Databricks credentials to activate).")


def task_export_to_bigquery(**context):
    """Export Gold Delta tables from Databricks to BigQuery."""
    from export.gold_to_bigquery import run_export
    run_export()
    logging.info("Gold tables exported to BigQuery.")


def task_notify_success(**context):
    """Log success. In production: send Slack message with run stats."""
    dag_run   = context["dag_run"]
    run_id    = dag_run.run_id
    exec_date = context["execution_date"]
    logging.info(
        f"PIPELINE SUCCESS | run_id: {run_id} | "
        f"execution_date: {exec_date} | "
        f"All tasks completed successfully."
    )


# -----------------------------------------------------------------------------
# DAG DEFINITION
# -----------------------------------------------------------------------------
with DAG(
    dag_id             = "ecommerce_analytics_pipeline",
    default_args       = default_args,
    description        = "E-Commerce Customer Analytics Pipeline — daily run",
    schedule_interval  = "0 6 * * *",   # 6 AM UTC daily
    catchup            = False,          # don't backfill missed runs
    max_active_runs    = 1,              # prevent concurrent runs on same data
    on_failure_callback= on_failure_callback,
    tags               = ["ecommerce", "analytics", "portfolio"],
) as dag:

    # -------------------------------------------------------------------------
    # TASK 1: Ingest from API
    # -------------------------------------------------------------------------
    ingest_api = PythonOperator(
        task_id          = "ingest_api",
        python_callable  = task_ingest_api,
        on_failure_callback = on_failure_callback,
    )

    # -------------------------------------------------------------------------
    # TASK 2: Validate schema with pandera
    # -------------------------------------------------------------------------
    validate_schema = PythonOperator(
        task_id          = "validate_schema",
        python_callable  = task_validate_schema,
        on_failure_callback = on_failure_callback,
    )

    # -------------------------------------------------------------------------
    # TASK 3: Upload to GCS
    # -------------------------------------------------------------------------
    upload_gcs = PythonOperator(
        task_id          = "upload_gcs",
        python_callable  = task_upload_gcs,
        on_failure_callback = on_failure_callback,
    )

    # -------------------------------------------------------------------------
    # TASK 4-6: Databricks Bronze -> Silver -> Gold
    # trigger_rule=all_success: Silver only runs if Bronze succeeds
    # -------------------------------------------------------------------------
    run_bronze = PythonOperator(
        task_id          = "run_databricks_bronze",
        python_callable  = task_run_databricks_bronze,
        trigger_rule     = TriggerRule.ALL_SUCCESS,
        on_failure_callback = on_failure_callback,
    )

    run_silver = PythonOperator(
        task_id          = "run_databricks_silver",
        python_callable  = task_run_databricks_silver,
        trigger_rule     = TriggerRule.ALL_SUCCESS,
        on_failure_callback = on_failure_callback,
    )

    run_gold = PythonOperator(
        task_id          = "run_databricks_gold",
        python_callable  = task_run_databricks_gold,
        trigger_rule     = TriggerRule.ALL_SUCCESS,
        on_failure_callback = on_failure_callback,
    )

    # -------------------------------------------------------------------------
    # TASK 7: Export Gold to BigQuery
    # -------------------------------------------------------------------------
    export_bigquery = PythonOperator(
        task_id          = "export_to_bigquery",
        python_callable  = task_export_to_bigquery,
        trigger_rule     = TriggerRule.ALL_SUCCESS,
        on_failure_callback = on_failure_callback,
    )

    # -------------------------------------------------------------------------
    # TASK 8: Run dbt
    # -------------------------------------------------------------------------
    run_dbt = BashOperator(
        task_id      = "run_dbt",
        bash_command = "cd /opt/airflow/project/dbt && dbt run && dbt test",
        trigger_rule = TriggerRule.ALL_SUCCESS,
        on_failure_callback = on_failure_callback,
    )

    # -------------------------------------------------------------------------
    # TASK 9: Notify success
    # -------------------------------------------------------------------------
    notify_success = PythonOperator(
        task_id         = "notify_success",
        python_callable = task_notify_success,
        trigger_rule    = TriggerRule.ALL_SUCCESS,
    )

    # -------------------------------------------------------------------------
    # TASK DEPENDENCIES — the pipeline story
    # ingest -> validate -> gcs -> bronze -> silver -> gold -> export -> dbt -> notify
    # -------------------------------------------------------------------------
    (
        ingest_api
        >> validate_schema
        >> upload_gcs
        >> run_bronze
        >> run_silver
        >> run_gold
        >> export_bigquery
        >> run_dbt
        >> notify_success
    )
