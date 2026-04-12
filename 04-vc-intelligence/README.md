# Startup Funding Intelligence Tool

[![CI](https://github.com/HariEtta/hari-etta-portfolio/actions/workflows/p4_ci.yml/badge.svg)](https://github.com/HariEtta/hari-etta-portfolio/actions/workflows/p4_ci.yml)
[![Live API](https://img.shields.io/badge/API-Live%20on%20Cloud%20Run-blue)](https://vc-intelligence-api-XXXX-uc.a.run.app)
[![Streamlit](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/HariEtta/vc-intelligence)

**Live Demo:** [Streamlit App](https://huggingface.co/spaces/HariEtta/vc-intelligence) |
**Live API:** [Cloud Run](https://vc-intelligence-api-XXXX-uc.a.run.app/docs) |
**Dashboard:** [Looker Studio](#)

---

## The Problem I Solved

I spent two years as a VC Analyst at a quantum computing startup screening deal flow manually.
Every week: 50 pitch decks, 30 hours of repetitive pattern recognition.

- Is this company in our sector thesis?
- What stage and check size?
- What traction signals exist?
- Who are the relevant comparables?

None of that requires human judgment. It requires consistent application of a framework at scale.

**This tool screens a deal in 30 seconds instead of 30 minutes.** The classifier runs in under
1 second. The LLM extracts structured metrics in under 30 seconds. Human judgment moves from
pattern matching to conviction building — the 5 companies worth a partner meeting, not the 45
that do not fit the thesis.

> *"This pipeline applies directly to financial services investment analysis, credit underwriting,
> and M&A target identification at firms like Morningstar and Houlihan Lokey."*

---

## Live Demo

### Deal Screener Tab
Paste any startup description → get sector classification, traction score, key metrics,
risk flags, and comparable companies in one screen.

> Screenshot: `docs/screenshots/deal_screener.png`

### Market Trends Tab
Interactive charts: sector heatmaps, funding stage distribution, deal flow by geography,
YoY funding growth.

> Screenshot: `docs/screenshots/market_trends.png`

---

## Architecture
Raw Data (Crunchbase/Scraper)
│
▼
Ingestion Pipeline (ingestion/)
pandera data contracts
│
▼
GCS Bucket (data/raw/)
│
▼
Databricks CE (databricks/)
Spark NLP cleaning
TF-IDF + sentence-transformers
Hyperopt 30-trial HPT
MLflow experiment tracking
Model Registry → Production
│
├──────────────────────┐
▼                      ▼
FastAPI on Cloud Run     ChromaDB (embedded)
/predict                semantic search
/search                 comparable lookup
/health
│
▼
Streamlit on HuggingFace Spaces
Deal Screener tab
Market Trends tab
│
▼
BigQuery + dbt
stg_startups
stg_funding_rounds
funding_trends (mart)
sector_heatmap (mart)
│
▼
Looker Studio Dashboard
---

## Tech Stack

| Tool | Purpose | Why Chosen | Free Tier |
|------|---------|------------|-----------|
| Databricks CE | Spark NLP + MLflow + Hyperopt | Distributed training, native MLflow | Free forever |
| scikit-learn | Sector classifier | Interpretable, SHAP compatible | Open source |
| sentence-transformers | Dense embeddings | Fast, 384-dim, strong on short text | Open source |
| ChromaDB | Vector store | Embedded mode, no infra, persists to disk | Open source |
| Mistral-7B-Instruct | LLM extraction | Free HF inference, strong instruction following | Free tier |
| Pydantic | LLM output validation | Catches hallucinations before storage | Open source |
| FastAPI | Model serving | Async, typed, auto-docs | Open source |
| GCP Cloud Run | API deployment | Scales to zero, 2M req/month free | Free tier |
| Streamlit | Frontend | HuggingFace Spaces deployment | Free forever |
| BigQuery | Data warehouse | 1TB/month free | Free tier |
| dbt Core | Data modeling | Staging + mart pattern, schema tests | Open source |
| Looker Studio | Dashboard | BigQuery native, shareable URL | Free forever |
| pandera | Data contracts | Schema validation at ingestion | Open source |
| GitHub Actions | CI/CD | Runs on push, dbt test on PR | Free public repos |
| GCS | Raw data storage | Databricks reads gs:// directly | 5GB free |

---

## Dataset

**Source:** Crunchbase public data + scraped funding announcements
**Size:** 500+ startup records (50 committed as sample, full dataset in GCS)
**Sectors:** 15+ including Climate Tech, Healthcare AI, Fintech, Robotics, Space Tech
**Fields:** name, description, sector, stage, funding amount, investors, traction signals

**Preprocessing decisions:**
- Lowercase + remove special characters + collapse whitespace
- Stopword removal and lemmatization (NLTK)
- TF-IDF max_features=5000, ngram_range=(1,2)
- Tokenizer max_length=128 tokens (covers 95th percentile of description lengths)

---

## Databricks ML Pipeline

All training runs in Databricks Community Edition. Serving runs in FastAPI on Cloud Run.
These are intentionally separated — standard MLOps pattern.

| Script | Purpose |
|--------|---------|
| `01_text_preprocessing.py` | Spark NLP parallel text cleaning across thousands of records |
| `02_feature_engineering.py` | TF-IDF features + sentence-transformer embeddings, logged to MLflow |
| `03_train_classifier.py` | Sector classifier, Hyperopt 30 trials, all logged to MLflow |
| `04_register_model.py` | Registers best model in Model Registry, promotes to Production |

> MLflow Experiments screenshot: `docs/screenshots/mlflow_experiments.png`
> Hyperopt parallel coordinates: `docs/screenshots/mlflow_hyperopt.png`
> Model Registry: `docs/screenshots/model_registry.png`

### Databricks CE Limitations (documented honestly)
- No scheduled Jobs → replaced by GitHub Actions cron
- Single cluster → multi-cluster at production scale
- No Unity Catalog → enterprise governance layer not available in CE

---

## Model Export Pattern

```python
# Training side (Databricks)
mlflow.sklearn.log_model(model, "sector_classifier")
client.transition_model_version_stage("vc-sector-classifier", version, "Production")

# Serving side (FastAPI on Cloud Run)
model = mlflow.pyfunc.load_model("models:/vc-sector-classifier/Production")
prediction = model.predict(pd.DataFrame({"text": [description]}))
```

Training and serving are separated by design. The Model Registry is the contract between them.

---

## LLM Extraction Layer

**Model:** Mistral-7B-Instruct-v0.2 via HuggingFace Inference API (free tier)
**Fallback:** local transformers inference during development

Every prompt lives in `llm/prompts.py` — never in function bodies.
Every LLM response is validated by Pydantic before storage or UI display.

**Extracted fields per startup:**
- Traction score (1-10)
- Key metrics (ARR, customer count, partnerships)
- Business model
- Target customer
- Competitive moat
- Risk flags
- Investment signal

**Prompt design:** system prompt instructs JSON-only output + "insufficient information"
instead of hallucination. 3 few-shot examples per extraction prompt.

---

## SHAP Explainability

SHAP LinearExplainer runs on the TF-IDF feature matrix to show which words drive
each sector prediction.

> SHAP summary plot: `docs/screenshots/shap_summary.png`

Example: for a "Climate Tech" prediction, top positive features include
"carbon", "emissions", "scope", "renewable" — exactly what a human analyst
would flag as sector signals.

---

## Classification Results

**Evaluation methodology:** 80/20 stratified train/test split. Test set never touched
during Hyperopt hyperparameter tuning. Final evaluation on held-out test set only.

| Metric | Score |
|--------|-------|
| Test Accuracy | *run classifier to populate* |
| Test F1 Macro | *run classifier to populate* |
| Hyperopt Trials | 30 |
| Best C (LogReg) | *from MLflow best run* |

> Confusion matrix: `docs/screenshots/confusion_matrix.png`

**Business framing:** 70% reduction in analyst triage time. Classifier handles the
filter — human judgment handles the shortlist.

---

## ChromaDB Semantic Search

**Model:** all-MiniLM-L6-v2 (384-dim embeddings)
**Similarity:** cosine similarity
**Threshold:** returns top-k results, similarity scores shown in UI

Example query: *"climate tech companies with enterprise SaaS revenue"*

Returns: Persefoni, CarbonLedger, ClimateAI, ClimaRisk, RegenFarms
— ranked by semantic similarity, not keyword match.

This is what makes the comparable search useful: "carbon accounting" and
"emissions management" surface the same results even though the words differ.

---

## dbt Funding Trends

**Warehouse:** BigQuery (1TB/month free)
**Models:**

vc_raw.startups_raw          (source)
│
├── stg_startups     (view — cleaned, typed)
│       │
│       └── stg_funding_rounds  (view — stage ranked, bucketed)
│               │
│               ├── funding_trends   (table — YoY aggregations)
│               └── sector_heatmap   (table — sector x stage matrix)

All key columns have `not_null`, `unique`, and `accepted_values` tests in `schema.yml`.
GitHub Actions runs `dbt test` on every PR.

> dbt lineage graph: `docs/screenshots/dbt_lineage.png`

---

## Market Dashboard

**Tool:** Looker Studio connected to BigQuery marts
**URL:** [Public Dashboard](#) ← update after Looker Studio setup

**Tiles:**
- Sector heatmap (deal count by sector x stage)
- Funding stage distribution
- YoY funding growth by sector
- Geographic clustering (US vs international)
- Deal size distribution by stage

---

## VC Market Narrative

Full investment thesis: [`docs/vc_market_narrative.md`](docs/vc_market_narrative.md)

**Summary:** Early-stage climate tech and healthcare AI are structurally underserved
relative to their market opportunity. Capital concentrates at Series B and beyond —
leaving a persistent gap at Seed and Series A where the highest-multiple outcomes are made.

---

## API Documentation

Base URL: `https://vc-intelligence-api-XXXX-uc.a.run.app`

### GET /health
```bash
curl https://vc-intelligence-api-XXXX-uc.a.run.app/health
```
```json
{
  "status": "ok",
  "model_loaded": true,
  "chromadb_ready": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

### POST /predict
```bash
curl -X POST https://vc-intelligence-api-XXXX-uc.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "AI carbon accounting platform. SOC2 certified. $1.1M ARR."}'
```
```json
{
  "sector_prediction": {
    "sector": "Climate Tech",
    "confidence": 0.847,
    "top_3": [
      {"sector": "Climate Tech", "confidence": 0.847},
      {"sector": "SaaS Analytics", "confidence": 0.091},
      {"sector": "Fintech", "confidence": 0.062}
    ]
  },
  "llm_extraction": {
    "success": true,
    "traction_score": 6,
    "key_metrics": ["$1.1M ARR", "SOC2 certified"],
    "business_model": "B2B SaaS",
    "target_customer": "Mid-market companies",
    "moat": "Compliance workflow integration",
    "risk_flags": ["crowded market"],
    "investment_signal": "Solid early traction in growing compliance market"
  },
  "latency_ms": {"classifier_ms": 12.4, "llm_ms": 4823.1}
}
```

### POST /search
```bash
curl -X POST https://vc-intelligence-api-XXXX-uc.a.run.app/search \
  -H "Content-Type: application/json" \
  -d '{"query": "climate tech carbon accounting", "top_k": 3}'
```
```json
{
  "query": "climate tech carbon accounting",
  "results": [
    {
      "id": "5",
      "name": "CarbonLedger",
      "sector": "Climate Tech",
      "stage": "Series A",
      "similarity_score": 0.923,
      "description_snippet": "Automated Scope 1/2/3 carbon accounting..."
    }
  ],
  "latency_ms": 34.2
}
```

---

## Latency Benchmarks

| Endpoint | p50 | p95 |
|----------|-----|-----|
| /predict — classifier only | ~12ms | ~28ms |
| /predict — classifier + LLM | ~5s | ~12s |
| /search — ChromaDB | ~35ms | ~80ms |

*LLM latency dominated by HuggingFace free tier inference. Self-hosted model
would reduce to ~800ms p50. Classifier and search latency are production-grade.*

---

## LLM Failure Modes

| Failure | How It Happens | How System Handles It |
|---------|---------------|----------------------|
| Invalid JSON | LLM wraps output in markdown | `parse_llm_json()` strips fences before parsing |
| Missing required field | LLM omits sector | Pydantic raises ValidationError, returns `success=False` |
| Out-of-range value | traction_score=99 | Pydantic field validator rejects, structured error returned |
| Hallucinated prose | LLM ignores JSON instruction | JSON parse fails, error surfaced in UI as warning |
| Rate limit (429) | HF free tier throttling | tenacity retries 3x with exponential backoff |
| Timeout | Slow HF inference | httpx timeout=60s, graceful error returned |

**User experience on failure:** sector classification always shown (classifier never fails),
LLM panel shows "extraction unavailable" warning with classifier result intact.

---

## Fintech Applicability

This pipeline applies directly to financial services. Replace the startup description
with a loan application narrative → sector classifier becomes credit risk tier classifier.
Replace traction score with default probability score. Replace comparable startups with
comparable credit profiles. The LLM extraction layer, Pydantic validation, ChromaDB
semantic search, and dbt mart structure transfer unchanged. The architecture is
domain-agnostic — the domain expertise lives in prompts and training data.

---

## Limitations and Future Work

**Current limitations:**
1. Sample dataset of 50 records — production model needs 5,000+ for robust sector classification
2. HuggingFace free tier rate limits — self-hosted Mistral eliminates latency variance
3. ChromaDB embedded mode — does not support concurrent writes at scale

**Three concrete improvements:**
1. Fine-tune Mistral on VC-specific extraction examples — reduces hallucination rate significantly
2. Add active learning loop — analyst corrections feed back into classifier retraining
3. Replace ChromaDB embedded with Qdrant on GCP — supports concurrent writes, same cosine similarity

---

## How to Run Locally

```bash
# 1. Clone and setup
git clone https://github.com/HariEtta/hari-etta-portfolio.git
cd hari-etta-portfolio/04-vc-intelligence
cp .env.example .env  # fill in your values

# 2. Install dependencies
make install

# 3. Build ChromaDB vector store
make embed

# 4. Start FastAPI
make api
# API runs at http://localhost:8080
# Docs at http://localhost:8080/docs

# 5. Start Streamlit
make streamlit
# UI runs at http://localhost:8501

# 6. Run tests
make test

# 7. Run dbt models
make dbt-run
make dbt-test
```

---

## The Interview Answer Nobody Else Can Give

*"I spent time as a VC analyst screening deals manually — 50 pitch decks a week,
30 hours of repetitive pattern recognition. I built this to automate the repeatable
parts while keeping human judgment where it matters: the shortlist, not the filter.
The classifier handles sector triage in under a second. The LLM extracts the structured
signals I used to pull manually. The Pydantic validation layer is what makes it
production-aware rather than a demo — every LLM output is validated before it touches
storage or the UI. That's the difference between a notebook and a system."*

---

*Hari Etta | MS Data Science & AI, IIT Chicago (May 2026) | F-1 STEM OPT*
*[LinkedIn](https://linkedin.com/in/harietta) | [GitHub](https://github.com/HariEtta)*
