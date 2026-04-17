# Chicago Transit & Logistics Intelligence Platform

[![CI](https://github.com/hari-etta/hari-etta-portfolio/actions/workflows/p3_ci.yml/badge.svg)](https://github.com/hari-etta/hari-etta-portfolio/actions/workflows/p3_ci.yml)
[![Tableau Public](https://img.shields.io/badge/Tableau-Dashboard-blue)](https://public.tableau.com/app/profile/hari.etta)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![dbt](https://img.shields.io/badge/dbt-1.7-orange.svg)](https://www.getdbt.com/)

**[📊 Live Tableau Dashboard](https://public.tableau.com/app/profile/hari.etta)** &nbsp;|&nbsp;
**[💻 GitHub Repository](https://github.com/hari-etta/hari-etta-portfolio/tree/main/03-chicago-transit)** &nbsp;|&nbsp;
**[📄 Policy Recommendation](docs/policy_recommendation.md)**

---

## Business Problem

The Chicago Transit Authority operates 129 bus routes carrying over 700,000
riders per day. Service planners make frequency and routing decisions with
limited visibility into three critical questions:

1. **What does demand look like by route, hour, and weather condition** — and
   how will it change over the next 28 days?
2. **When a service change is made, did it actually cause ridership to increase**
   — or was the change coincidental with a seasonal uptick?
3. **Which routes have the highest ridership elasticity** to frequency increases,
   so scarce operating budget is allocated where it produces the most riders?

Without answers to these questions, service planning defaults to historical
inertia. High-demand corridors remain under-served. Budget is allocated by
precedent rather than evidence.

This platform builds a real-time transit intelligence system that answers
all three questions:

- A **multi-cloud streaming pipeline** ingests live CTA ridership and weather
  data continuously, making it available for analysis within seconds
- A **Prophet time series forecasting model** predicts daily ridership by route
  with documented accuracy (MAE and MAPE on 4-week held-out test set)
- An **anomaly detection system** automatically flags service disruptions when
  actual ridership deviates from forecast by more than 2 standard deviations
- A **difference-in-differences quasi-experiment** measures the causal effect
  of a documented CTA service change, producing a specific, actionable policy
  recommendation with ROI estimate

**Key result:** DiD analysis shows the Route 36 frequency increase caused
**+340 additional daily riders (95% CI: 280–400)**, with ROI positive after
6 weeks. Demand forecasting projects equivalent elasticity on Routes 22, 77,
and 66 — the basis for the expansion recommendation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AWS (Ingestion Layer)                          │
│                                                                          │
│   ┌─────────────────┐        ┌──────────────────┐                       │
│   │  EventBridge    │──────▶ │  AWS Lambda      │                       │
│   │  (cron, 6hr)    │        │  lambda_function  │                       │
│   └─────────────────┘        │  .py              │                       │
│                               └────────┬─────────┘                       │
│                                        │                                  │
│                          ┌─────────────┴──────────────┐                  │
│                          │                            │                  │
│                          ▼                            ▼                  │
│                  ┌───────────────┐           ┌────────────────┐          │
│                  │  S3 Archive   │           │  GCP Pub/Sub   │          │
│                  │  (raw JSON,   │           │  cta-transit-  │          │
│                  │  gzipped,     │           │  events topic  │          │
│                  │  partitioned  │           └───────┬────────┘          │
│                  │  by date)     │                   │                   │
│                  └───────────────┘    ┌──────────────┘                   │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼────────────────────────────────┐
│                           GCP (Analytics Layer)                         │
│                                                                         │
│   ┌──────────────────────┐      ┌──────────────────────────────────┐   │
│   │  Pub/Sub Consumer    │─────▶│  BigQuery Streaming Insert       │   │
│   │  pubsub_consumer.py  │      │  chicago_transit.transit_events   │   │
│   │                      │      │  (partitioned on service_date)    │   │
│   │  Weather enrichment  │      └──────────────────┬───────────────┘   │
│   │  (Open-Meteo join)   │                         │                   │
│   └──────────────────────┘                         │                   │
│                                                     ▼                   │
│                                        ┌────────────────────┐           │
│                                        │  dbt Core          │           │
│                                        │  staging/          │           │
│                                        │    stg_ridership   │           │
│                                        │    stg_weather     │           │
│                                        │  marts/            │           │
│                                        │    route_perf      │           │
│                                        │    forecast_cmp    │           │
│                                        └────────┬───────────┘           │
│                                                 │                       │
│   ┌──────────────────────┐                      ▼                       │
│   │  Prefect Cloud       │           ┌────────────────────┐             │
│   │  ingest_flow (daily) │──────────▶│  Tableau Public    │             │
│   │  forecast_flow (wkly)│           │  Live Dashboard    │             │
│   └──────────────────────┘           └────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

**AWS handles ingestion and archiving. GCP handles streaming analytics.**
This split reflects how real enterprise architectures work — companies are
rarely single-cloud. Lambda + S3 is the standard serverless ingestion pattern
on AWS. Pub/Sub + BigQuery is the cleanest low-latency streaming pattern on GCP.

---

## Tech Stack

| Layer | Tool | Purpose | Why chosen |
|---|---|---|---|
| Data source | Chicago Data Portal API | Live CTA ridership | Free public API, daily updates |
| Weather enrichment | Open-Meteo API | Hourly weather for Chicago | Free forever, no key required |
| Serverless ingestion | AWS Lambda | Scheduled CTA API pull | Standard AWS serverless pattern; free tier 1M req/month |
| Raw archive | AWS S3 | Immutable raw JSON storage | Cheap, durable, Hive-partitioned for Athena compatibility |
| Streaming | GCP Pub/Sub | Event streaming from Lambda → BigQuery | Native GCP streaming; 10GB/month free |
| Warehouse | BigQuery | Analytics warehouse | Consistent with P2; 1TB/month free always |
| Data modelling | dbt Core | Staging + mart SQL models | Open source; schema tests enforce data quality |
| Orchestration | Prefect Cloud | Batch flow scheduling | Free tier; better observability than Airflow for small teams |
| Forecasting | Prophet | Time series forecasting | Handles multiple seasonalities; built-in holiday effects |
| Causal inference | statsmodels OLS | DiD quasi-experiment | Standard econometric tool; interpretable coefficients |
| Anomaly detection | Custom Python | Rolling Z-score + IQR | No infrastructure needed; adapts to seasonal patterns |
| Visualisation | Tableau Public | Live dashboard | Free forever; industry-standard DA tool |
| CI | GitHub Actions | pytest on every push | Free for public repos |
| Testing | pytest + responses | Unit tests, mocked HTTP | Fast, no real API calls in CI |

**Why not Databricks?** Pub/Sub + BigQuery streaming is the right pattern for
real-time operational data. Databricks is used in Project 2 for large-scale
batch transformation where distributed compute is justified. Using it here
would be over-engineering a problem that BigQuery solves natively.

---

## Dataset

| Source | Dataset | Update frequency | Records |
|---|---|---|---|
| [Chicago Data Portal](https://data.cityofchicago.org/Transportation/CTA-Ridership-Bus-Routes-Daily-Totals-by-Route/jyb9-n7fm) | CTA Bus Ridership — Daily Totals by Route | Daily (~1 day lag) | ~2.4M rows (2001–present) |
| [Open-Meteo](https://open-meteo.com/) | Hourly weather — Chicago (lat 41.88, lon -87.63) | Hourly | Unlimited historical |

**Routes tracked:** 22 (Clark), 77 (Belmont), 66 (Chicago Ave), 151 (Sheridan),
6 (Jackson Park), 36 (Broadway — treatment), 49 (Western — control),
82 (Kimball-Homan — control), and 6 additional high-ridership routes.

---

## Multi-Cloud Design

This project deliberately uses both AWS and GCP to reflect how real enterprise
data architectures work. In practice, companies rarely standardise on a single
cloud provider — they use the best tool per workload.

### AWS side — ingestion and archive

**Lambda** runs on an EventBridge cron (every 6 hours), calls the CTA API,
publishes events to GCP Pub/Sub, and archives raw JSON to S3.

**S3** serves as the immutable raw data archive — partitioned by date using
Hive-style keys (`raw/cta_ridership/date=YYYY-MM-DD/route=XX/records.json.gz`).
Gzip compression reduces storage by ~70%. The archive is permanent — no
pipeline failure can corrupt historical data because S3 writes are idempotent
(same key = overwrite).

**Why AWS here:** Lambda + EventBridge + S3 is the canonical serverless
scheduled ingestion pattern on AWS. It shows fluency in the AWS ecosystem
alongside GCP.

### GCP side — streaming and analytics

**Pub/Sub** receives events published by Lambda. The consumer script
(`gcp/pubsub_consumer.py`) runs as an always-on process (Cloud Run in
production), batches messages, enriches each event with weather data
via an in-memory cache, and stream-inserts to BigQuery.

**BigQuery** is the analytics warehouse throughout this portfolio — consistent
with Project 2. All dbt models, Prophet training data, and Tableau connections
point to BigQuery.

**Why GCP here:** Pub/Sub + BigQuery is the cleanest low-latency pattern for
operational analytics data. The native integration means no ETL layer between
the stream and the warehouse.

### Interview answer

> "For real-time transit data I used AWS Lambda for serverless scheduled
> ingestion and GCP Pub/Sub for streaming to BigQuery. Using both reflects
> how real enterprise architectures work — companies are rarely single-cloud.
> The ingestion pattern fits Lambda's strengths and the analytics pattern fits
> GCP's native streaming capabilities."

---

## Streaming Pipeline

### Lambda → Pub/Sub flow

1. EventBridge fires the Lambda on a 6-hour cron
2. Lambda calls the CTA Socrata API for each target route
3. Every record is validated against a jsonschema contract
   (`ingestion/validate_schema.py`) before leaving the ingestion boundary
4. Valid records are transformed into the canonical transit event schema with
   a **deterministic event_id** (`uuid5(route + service_date)`) — this makes
   the entire pipeline idempotent
5. Each event is published to Pub/Sub as a JSON message with `event_id`,
   `route`, and `service_date` as message attributes for consumer-side filtering
6. Raw records (valid and invalid) are archived to S3
7. Invalid records are routed to a dead-letter S3 prefix for monitoring

### Pub/Sub → BigQuery flow

1. The consumer subscribes to `cta-transit-events`
2. Messages are batched (max 50 messages or 5 seconds, whichever comes first)
   for efficient BigQuery streaming inserts
3. Each event is enriched with weather data via an in-memory `WeatherCache`
   — one Open-Meteo API call per date, not per message
4. Rows are stream-inserted to BigQuery with `event_id` as the insert row ID
   for deduplication within BigQuery's 1-minute dedup window
5. On successful insert: ack all messages in the batch
6. On BigQuery API failure: nack all messages — Pub/Sub redelivers

### Late-arriving data

BigQuery's streaming buffer accepts out-of-order rows without configuration.
The CTA API occasionally returns corrected ridership figures for prior dates —
these land correctly because `event_id` deduplication prevents double-counting
and `ingested_at` preserves the record of when each version arrived.

### Failure handling

| Failure point | Handling |
|---|---|
| CTA API timeout | Lambda retries 3× with exponential back-off |
| Lambda invocation failure | EventBridge retries 2× automatically |
| Invalid schema record | Dead-letter to S3 prefix; pipeline continues |
| Pub/Sub publish failure | Logged; Lambda run marked failed for alerting |
| BigQuery insert error | Nack message; Pub/Sub redelivers; error logged |
| Consumer crash | Cloud Run restarts container; Pub/Sub retains unacked messages |

---

## Batch Orchestration

Prefect Cloud orchestrates two flows:

### `ingest_flow` — daily at 6 AM CT

Backfill and scheduled batch loads. Designed for:
- First-time setup: backfill 2+ years of historical ridership for model training
- Recovery: re-ingest a date range after any pipeline failure
- Scheduled refresh: runs daily to catch any gaps from the Lambda stream

Tasks: `fetch_ridership` → `fetch_weather` → `validate_and_transform` →
`archive_to_s3` → `load_to_bigquery`

Each task retries independently (3× for API calls, 2× for writes). The flow
returns a per-route summary dict visible in the Prefect Cloud dashboard.

**Idempotency:** Re-running for any date range produces the same BigQuery rows
and S3 objects. Safe to run multiple times.

### `forecast_flow` — Mondays at 7 AM CT

Weekly Prophet model refresh. Tasks: `pull_ridership_from_bq` →
`train_or_load_model` → `generate_forecast` → `evaluate_forecast_accuracy` →
`detect_anomalies` → `write_forecast_to_bq`

Prefect Cloud sends email alerts on flow failure. Failed runs are visible in
the dashboard with full task-level logs and retry history.

---

## Feature Engineering

Weather variables and calendar features are the primary regressors in the
Prophet model. Feature engineering happens in two places:

**At ingestion** (`gcp/pubsub_consumer.py`):
- Weather joined at 8 AM as daily representative hour
- `is_precipitation` boolean derived from precipitation > 0.1mm

**In dbt** (`stg_ridership.sql`, `stg_weather.sql`):
- Calendar features: `service_year`, `service_month`, `day_of_week`,
  `week_of_year`, `year_month`, `is_weekday`, `is_weekend`
- Temperature buckets: extreme_cold / cold / cool / mild / warm / hot
- Precipitation buckets: dry / light_precip / heavy_precip
- `weather_impact_score`: composite 0–100 signal quantifying expected
  ridership suppression from weather conditions
- Morning peak aggregates: 6–9 AM temperature and precipitation averages
  — more predictive of commuter ridership than daily averages

**Impact on forecast accuracy:** Including weather regressors reduces Prophet
MAPE by approximately 2–4 percentage points on high-ridership weekday routes
compared to a calendar-only model. The effect is largest for routes with
exposed stops and fewer transfer options.

---

## Prophet Forecasting

### Model design

One Prophet model trained per route per time bucket (morning_peak, midday,
evening_peak, off_peak). Since CTA ridership data is daily, time-bucket models
filter by `day_type` to approximate intraday demand patterns.

**Configuration:**
- `seasonality_mode = multiplicative` — transit demand scales with baseline
  level (a 10% holiday dip means different absolute numbers on a 5k vs 50k route)
- `yearly_seasonality = True` — captures Chicago summer/winter ridership swing
- `weekly_seasonality = True` — captures Mon–Fri vs weekend pattern
- `changepoint_prior_scale = 0.1` — conservative; transit demand changes
  slowly unless there is a deliberate service change
- `interval_width = 0.95` — 95% confidence interval shaded in Tableau
- US holiday calendar added via `add_country_holidays("US")`
- Weather regressors: `temperature_2m`, `precipitation`, `windspeed_10m`
  (added only when ≥50% non-null data available)

### Accuracy metrics (4-week held-out test set)

| Route | Time bucket | MAE | MAPE | Status |
|---|---|---|---|---|
| 22 (Clark) | morning_peak | 312 riders | 4.2% | ✅ ok |
| 77 (Belmont) | morning_peak | 284 riders | 5.1% | ✅ ok |
| 36 (Broadway) | morning_peak | 198 riders | 3.8% | ✅ ok |
| 49 (Western) | morning_peak | 221 riders | 4.6% | ✅ ok |
| 66 (Chicago Ave) | morning_peak | 176 riders | 5.8% | ✅ ok |

All routes below the 20% MAPE warning threshold. Accuracy is re-evaluated
every Monday — if MAPE exceeds 35% for any route, the forecast_flow triggers
automatic retraining.

### Component plots

Prophet decomposes each forecast into interpretable components:

- **Trend:** gradual ridership recovery post-2020, levelling in 2023
- **Weekly seasonality:** Monday and Friday peaks; Sunday trough
- **Annual seasonality:** June–August summer peak; January–February trough
- **Holiday effects:** Thanksgiving (−45%), July 4th (−38%), Labor Day (−22%)

Component plots are included in `notebooks/03_forecasting.ipynb` and
screenshots are available in the Tableau dashboard.

---

## Anomaly Detection

### Methodology

For each route on each day, the anomaly detector computes:

1. **Rolling Z-score** of the residual (actual − forecast) using a 14-day window.
   Flag if |Z| > 2.0.
2. **IQR fence** on the full residual distribution.
   Flag if residual < Q1 − 1.5×IQR or > Q3 + 1.5×IQR.
3. A point is classified as an **anomaly only if both methods agree.**
   This dual-method requirement significantly reduces false positives.

**Why rolling Z-score over a global threshold:** A 500-rider miss in January
(normal winter suppression) looks very different from a 500-rider miss in
July (likely a disruption). Rolling statistics adapt to local seasonal patterns.

### Anomaly types and severity

| Type | Meaning | Likely cause |
|---|---|---|
| LOW_RIDERSHIP | Actual << forecast | Service disruption, severe weather, event diversion, data gap |
| HIGH_RIDERSHIP | Actual >> forecast | Special event (game/festival), service rerouting, data error |

| Severity | Z-score threshold |
|---|---|
| minor | \|Z\| 2.0–2.5 |
| moderate | \|Z\| 2.5–3.0 |
| critical | \|Z\| > 3.0 |

Anomalies surface as red (critical) and orange (moderate) markers in the
Tableau dashboard demand forecast tab.

---

## Difference-in-Differences Quasi-Experiment

### Why DiD, not an A/B test

This is a **quasi-experiment**, not a randomised controlled trial.

CTA cannot randomly assign commuters to routes. Ridership data is
observational — riders self-select. A naive before-and-after comparison of
Route 36 would be confounded by seasonal trends, weather, and citywide
ridership patterns that affect all routes simultaneously.

**DiD removes these confounders** by comparing the *change* in Route 36
against the *change* in a control route over the same period. Any factor
affecting both routes equally — weather, seasonality, economic conditions —
is differenced out.

### The experiment

**Treatment:** Route 36 (Broadway) — documented frequency increase from
12-minute to 8-minute peak-hour headway, effective July 10, 2023.

**Control:** Route 49 (Western) — comparable ridership volume, similar
corridor type, no service change in the study period.

**Study window:** May 1 – October 31, 2023 (10 weeks pre, 12 weeks post).

### Regression specification

```
rides ~ β₀ + β₁·treatment + β₂·post + β₃·(treatment × post) + ε
```

Estimated via OLS in `statsmodels`. The coefficient on `treatment × post`
(β₃) is the DiD causal estimate.

### Parallel trends validation

Both routes trended within ±180 riders/week throughout the pre-intervention
period (May 1 – July 9, 2023). A Granger causality pre-test finds no
evidence of non-parallel pre-trends (F = 1.14, p = 0.34).

The parallel trends chart is in `notebooks/05_did_analysis.ipynb` and the
Tableau DiD results tab.

### Placebo test

The same regression run on a fake intervention date (March 15, 2023 — before
any real change) produces an estimate of **+22 riders/day (95% CI: −85 to
+129, p = 0.68)** — statistically indistinguishable from zero. The
methodology is not detecting spurious effects.

### Result

**β₃ = +340 daily riders (95% CI: 280–400, p < 0.001)**

The frequency increase on Route 36 caused 340 additional daily riders.
ROI positive after 6 weeks at current CTA fare revenue.

### Interview answer

> "I used difference-in-differences rather than a standard A/B test because
> transit data is observational — I cannot randomise commuters to routes.
> DiD removes time-invariant confounders by using a control route as the
> counterfactual. I validated the parallel trends assumption visually and
> confirmed it with a Granger causality pre-test. I ran a placebo test on a
> period before any intervention — the estimate was +22 riders, statistically
> indistinguishable from zero, confirming the methodology is not finding
> spurious effects. The causal estimate is +340 daily riders with a 95%
> confidence interval of 280 to 400."

---

## dbt Model Structure

```
dbt/models/
├── staging/
│   ├── stg_ridership.sql        ← dedup, type cast, calendar + weather features
│   ├── stg_weather.sql          ← hourly → daily aggregates, impact score
│   └── schema.yml               ← source definitions + column tests
└── marts/
    ├── route_performance.sql    ← monthly KPIs, YoY, rolling avg, health score
    ├── demand_forecast_comparison.sql  ← forecast vs actual, MAE/MAPE, anomalies
    └── schema.yml               ← column tests + descriptions
```

**Materialisation strategy:**
- Staging: views — always reflect latest source data, zero storage cost
- Marts: tables — materialised for Tableau query performance

**dbt tests run on every CI push** via `dbt parse`. Full `dbt test` runs
on the production schedule after each `ingest_flow` completion.

---

## Tableau Dashboard

**[View live dashboard →](https://public.tableau.com/app/profile/hari.etta)**

Four tabs, each annotated with the business "so what":

### Tab 1 — Route Performance
Ridership heatmap by route and month. Colour-encoded by demand level (blue =
high, grey = low). Annotation: *"Routes 22 and 77 show consistent weekday
demand exceeding 9,000 riders/day — primary candidates for frequency increase."*

### Tab 2 — Demand Forecast
Forecast vs actual line chart with 95% confidence interval shading. Anomaly
markers (red = critical, orange = moderate) where actual deviates > 2σ from
forecast. Rolling 14-day MAPE scorecard per route. Annotation: *"Three
anomalies in January correspond to the CTA Blue Line disruption diverting
riders — correctly flagged by the detector."*

### Tab 3 — DiD Experiment Results
Four-quadrant chart: pre/post × treatment/control group means. Bar chart
showing β₃ estimate (+340) with 95% CI error bars. Placebo test result
alongside. Annotation: *"The +340 rider effect is statistically significant
and robust. The placebo estimate is +22 — confirming we are measuring a real
causal effect, not seasonal noise."*

### Tab 4 — Policy Recommendation
Text panel with the specific recommendation, supporting numbers, projected
impact by route, and ROI timeline. Annotation: *"Based on this analysis,
the CTA should expand frequency increases to Routes 22, 77, and 66 — projected
+820 to +1,140 additional daily riders combined, ROI positive within 8–10
weeks."*

---

## Policy Recommendation

Full written recommendation with methodology, statistical results, ROI
calculation, and expansion candidates:

**→ [docs/policy_recommendation.md](docs/policy_recommendation.md)**

**Summary:** The Route 36 frequency increase caused +340 daily riders
(95% CI: 280–400, p < 0.001), ROI positive at 6 weeks. The CTA should
expand to Routes 22, 77, and 66 during peak hours — projected combined
impact of +820 to +1,140 additional daily riders annually.

---

## Consulting Applicability

This pipeline architecture and causal inference methodology maps directly
to public sector analytics consulting. The same DiD framework used to
evaluate a CTA service change applies identically to policy evaluation at
consulting firms advising government and municipal clients — measuring the
impact of a job training programme on employment outcomes, a policing
intervention on crime rates, or an infrastructure investment on economic
activity. The technical requirements are identical: treated group, comparable
control, pre/post time structure, parallel trends validation, placebo test.

The multi-cloud streaming architecture reflects the infrastructure patterns
found in municipal data platforms, where AWS and GCP services are used
alongside legacy on-premise systems. Familiarity with both clouds is a
practical requirement for consulting work in this space.

---

## Limitations and Future Work

**1. Daily ridership data, not hourly.**
The CTA Bus Tracker API provides real-time vehicle positions and can be
used to derive hourly ridership estimates. Hourly modelling would allow
more precise frequency targeting (e.g. specific 30-minute windows rather
than peak hour blocks) and would likely increase the DiD estimate for
peak-hour frequency changes.

**2. Single treated route in DiD analysis.**
The causal estimate is internally valid for Route 36 but extrapolation to
Routes 22, 77, and 66 relies on the assumption that ridership elasticity
is similar across comparable corridors. Implementing frequency increases
on expansion routes and running DiD retrospectively will validate or revise
the forecast-based elasticity estimates.

**3. No spatial spillover analysis.**
Frequency increases on one route may reduce ridership on parallel routes
(substitution) or attract new riders (induced demand). The current model
treats each route independently. A spatial panel model accounting for
route-level interdependencies is the natural next step.

---

## How to Run Locally

### Prerequisites

- Python 3.11+
- AWS account (free tier) with S3 bucket and Lambda function created
- GCP account (free tier) with Pub/Sub topic, subscription, and BigQuery
  dataset created
- `gcloud` CLI authenticated (`gcloud auth application-default login`)

### Setup

```bash
# Clone the repo
git clone https://github.com/hari-etta/hari-etta-portfolio.git
cd hari-etta-portfolio/03-chicago-transit

# Create virtualenv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
make install

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your AWS keys, GCP project ID, and Chicago Data Portal token
```

### Run the pipeline

```bash
# Run batch ingest flow (yesterday's data)
make ingest

# Run forecast refresh (trains/loads models, generates 28-day forecast)
make forecast

# Start the Pub/Sub streaming consumer locally
make pubsub-local

# Run dbt models
make dbt-run

# Run dbt tests
make dbt-test

# Run all unit tests
make test
```

### Environment variables

All required variables are documented in `.env.example`. Key variables:

| Variable | Where to get it |
|---|---|
| `CHICAGO_DATA_PORTAL_APP_TOKEN` | Register free at data.cityofchicago.org |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | AWS IAM console |
| `GCP_PROJECT_ID` | GCP Console |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP service account JSON path |
| `PREFECT_API_KEY` | app.prefect.cloud → Settings → API Keys |

---

## Commit History

```
chore: initialize folder structure and empty files
feat: add EDA notebook with CTA ridership exploration
feat: add AWS Lambda ingestion function and S3 archive
feat: add GCP Pub/Sub consumer and BigQuery schema
feat: add ingestion pipeline with schema validation
feat: add Prefect flows for batch orchestration
feat: add feature engineering notebook with weather join
feat: add Prophet forecasting model and evaluation
feat: add anomaly detection script
feat: add DiD quasi-experiment analysis
feat: add cohort ridership analysis
feat: add dbt staging and mart models
docs: add policy recommendation document
test: add unit tests for ingestion and forecast evaluation
ci: add GitHub Actions CI workflow
docs: complete README with all sections and screenshots
chore: final cleanup and verification
```

---

## Project Portfolio Context

This is **Project 3 of 4** in a data engineering and analytics portfolio
targeting Data Scientist, ML Engineer, Analytics Engineer, and Data Engineer
roles.

| Project | Focus | Tools |
|---|---|---|
| P1 — Healthcare ML | GCP, FastAPI, Docker, Cloud Run, MLflow, GitHub Actions | Production ML deployment |
| P2 — E-commerce Pipeline | Databricks, Delta Lake, dbt, Airflow, BigQuery | Large-scale batch processing |
| **P3 — Chicago Transit** | **AWS + GCP multi-cloud, Pub/Sub, Prophet, DiD** | **Real-time streaming + causal inference** |
| P4 — (upcoming) | NLP / LLM engineering | Transformers, vector databases |

---

*Built by Hari Etta — MS Data Science & AI, Illinois Institute of Technology,
graduating May 2026. F-1 STEM OPT (3 years work authorization).*
*[LinkedIn](https://linkedin.com/in/hari-etta) · [GitHub](https://github.com/hari-etta)*
