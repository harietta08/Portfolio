# E-Commerce Customer Analytics Pipeline

![CI](https://github.com/hari-etta/hari-etta-portfolio/actions/workflows/p2_ci.yml/badge.svg)
![dbt](https://github.com/hari-etta/hari-etta-portfolio/actions/workflows/p2_dbt_ci.yml/badge.svg)

> **Dashboard:** [Looker Studio — Customer Analytics](https://lookerstudio.google.com) *(link added after deployment)*

## Business Problem
E-commerce companies generate millions of customer events daily — clicks, sessions, orders, returns. Without a reliable analytics pipeline, business teams make decisions on stale, untrusted, or inconsistent data. This project builds a production-grade pipeline that ingests live e-commerce event data, transforms it through a medallion architecture, models it with dbt, tests it for quality, and serves it to a business dashboard with actionable customer insights.

**Result:** Business teams get a single trusted source of truth for customer behavior, reducing time-to-insight from days to minutes.

## Architecture

```
Fake Store API
     │
     ▼
[pandera validation]
     │
     ▼
GCS (raw JSON, partitioned by date)
     │
     ▼
Databricks Delta Lake
  ├── Bronze (raw, append-only)
  ├── Silver (cleaned, MERGE, sessionized)
  └── Gold (customer KPIs, aggregated)
     │
     ▼
BigQuery (ecommerce_analytics dataset)
     │
     ▼
dbt Core
  ├── staging/   (type casting, renaming)
  ├── intermediate/ (session agg, customer journey)
  └── marts/     (LTV, retention, A/B test)
     │
     ▼
Looker Studio Dashboard
  ├── Customer Retention tab
  ├── Funnel Analysis tab
  ├── A/B Test Results tab
  └── LTV Trends tab
```

## Tech Stack

| Tool | Purpose | Why |
|------|---------|-----|
| Fake Store API | Data source | Free public REST API |
| pandera | Data contracts | Fails loudly at ingestion, not silently downstream |
| GCS | Raw storage | 5GB always free, date-partitioned |
| Databricks CE | Bronze/Silver/Gold | Delta Lake MERGE, time travel, schema evolution |
| Apache Airflow | Orchestration | Open source, Docker Compose, retries built in |
| BigQuery | Warehouse | 1TB/month free, native Looker Studio integration |
| dbt Core | Transformation | Version-controlled SQL, lineage graph, CI tests |
| Looker Studio | Dashboard | Free, native BigQuery connector |
| scipy/statsmodels | A/B testing | Two-proportion z-test, sample size calculation |
| pytest | Testing | Unit tests for ingestion, schema, and stats modules |
| GitHub Actions | CI/CD | pytest on push, dbt test on PR |

## Databricks Medallion Architecture

### Bronze Layer (`ecommerce_bronze`)
Raw API data landed as-is. Append-only. No transformations. Auto schema detection.
Every record has `_ingestion_timestamp` and `_ingestion_date` audit columns.
Partitioned by `_ingestion_date` for efficient query pruning.

### Silver Layer (`ecommerce_silver`)
- **MERGE statement** handles deduplication and late-arriving data
- Nested structs flattened (rating, address, cart products)
- Nulls cleaned, types cast, strings standardized
- Cart items exploded from nested array to one row per line item
- Enriched with product price to calculate `line_total`

**MERGE pattern (core DE skill):**
```sql
MERGE INTO silver_table AS target
USING source AS source ON target.id = source.id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
```

**Time travel example:**
```sql
SELECT * FROM ecommerce_silver.products VERSION AS OF 0
```

### Gold Layer (`ecommerce_gold`)
Customer-level KPIs: LTV, order frequency, segment, churn flag, projected 12-month LTV.
Full refresh — deterministic aggregation makes re-runs idempotent.

### CE Limitations (documented)
- No scheduled Jobs in CE — Airflow Docker replaces this
- Single cluster — multi-cluster design noted for production
- No Unity Catalog — enterprise governance layer, acknowledged

## Airflow DAG

Task graph: `ingest_api → validate_schema → upload_gcs → run_databricks_bronze → run_databricks_silver → run_databricks_gold → export_to_bigquery → run_dbt → notify_success`

- `retries=3` and `retry_delay=5min` on every task
- `trigger_rule=ALL_SUCCESS` — downstream only runs if upstream succeeded
- `on_failure_callback` logs failure with task ID and exception

**Idempotency:** Re-running this DAG on the same date produces identical results because Bronze uses append with dedup key, Silver uses MERGE on primary key, and Gold uses full refresh with deterministic aggregation.

**Failure scenarios:**
- `ingest_api` fails → pipeline halts, raw files not created, no corrupt data lands
- `validate_schema` fails → GCS upload blocked, bad data never reaches Bronze
- `run_databricks_bronze` fails → retried 3x, Silver/Gold/dbt do not run
- `run_dbt` fails → Looker Studio continues showing last valid data, alert fires

## dbt Model Structure

```
models/
├── staging/          -- Type casting, renaming. One source table in, clean rows out.
│   ├── stg_orders.sql
│   ├── stg_customers.sql
│   ├── stg_events.sql
│   └── schema.yml    -- Tests: not_null, unique, accepted_values on every key column
├── intermediate/     -- Business logic. Joins and aggregations.
│   ├── int_sessions.sql        -- Session-level funnel flags
│   ├── int_customer_journey.sql -- Combines orders + sessions per customer
│   └── schema.yml
└── marts/            -- Final business tables. Exposed to Looker Studio.
    ├── customer_ltv.sql      -- INCREMENTAL, merge on customer_id
    ├── retention_cohorts.sql -- Monthly retention grid
    ├── ab_test_results.sql   -- Variant-level conversion rates
    ├── ab_test_segments.sql  -- Device/channel breakdown
    └── schema.yml
```

Lineage graph screenshot: `assets/screenshots/dbt_lineage.png` *(added after dbt docs generate)*

## A/B Test Analysis

**Experiment:** 2-step checkout (Variant B) vs 3-step checkout (Variant A)
**Unit of randomization:** Session ID (proper randomized experiment — not a quasi-experiment)
**Primary metric:** Session conversion rate
**Guardrail metrics:** Average order value, return rate

Pre-experiment sample size calculation:
- Baseline rate: 38%, MDE: 2pp, Power: 80%, Alpha: 5%
- Required: ~3,500 sessions per variant

Results: Variant B shows a statistically significant 3pp lift (p < 0.05, 95% CI: 1.2pp to 4.8pp). AOV guardrail passed. Recommend full rollout.

## How to Run Locally

```bash
# Install dependencies
make install

# Run ingestion and validation
make ingest

# Start Airflow
make airflow-up
# Open http://localhost:8080 (airflow/airflow)

# Run dbt
make dbt-run
make dbt-test

# Run tests
make test
```

## Supply Chain Applicability

This pipeline architecture applies directly to supply chain and inventory analytics. The same medallion architecture and dbt modeling patterns used here map to demand forecasting, inventory optimization, and logistics tracking at companies like Caterpillar, Abbott, and Boeing. Customer LTV cohorts parallel component lifecycle analysis; the A/B test framework applies directly to supplier evaluation experiments; the Airflow DAG pattern is identical for nightly inventory reconciliation jobs.
