# IIT International Student Chatbot — Production-Ready (Week 3)

This project is a **policy-grounded chatbot** for IIT international student questions.

It is designed to be:
- **Accurate & grounded**: answers come only from your `data/` policy markdown + retrieved sources
- **Explainable**: eligibility decisions come from a deterministic YAML rule engine
- **Scalable**: policies can be updated by editing `backend/rules/policy_rules.yaml` and re-ingesting `data/`
- **Robust**: avoids guessing on ambiguous work-hour questions; avoids speculation when policy text is missing
- **Deployment-ready**: Streamlit UI + ElasticSearch (BM25 + embeddings)

---

## Architecture (high level)

1) **Ingest**
- Parse each `data/**/*.md`
- Structure-aware chunking (heading hierarchy + section labels)
- OpenAI embeddings per chunk
- Index into ElasticSearch with metadata (`policy_topic`, `doc_title`, `section_path`, `source_url`, etc.)

2) **Runtime**
- Detect: out-of-scope → refuse safely
- Detect: topic + intent (class-level)
- Guardrail: if work-hours question is ambiguous → ask clarifying question (on-campus vs CPT vs OPT)
- Route:
  - **Rules** mode for eligibility/decision questions (slot filling + YAML rule engine)
  - **Retrieval** mode for definitions, procedures, timelines, and general policy questions
- Response:
  - LLM is used only to **write** a helpful answer from retrieved policy text (strictly grounded)
  - Always includes a deduped **Sources** list

---

## Requirements

- Python 3.10+
- Docker (for ElasticSearch)
- OpenAI API key (for embeddings + grounded answer synthesis)

---

## Setup (first time)

### 1) Unzip + enter project
```bash
unzip iit_chatbot.zip
cd iit_chatbot
```

### 2) Create and activate venv

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3) Install deps
```bash
pip install -r requirements.txt
```

---

## Configure environment

Create a `.env` in project root (or edit existing):

```env
OPENAI_API_KEY=YOUR_KEY
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini

ES_URL=http://localhost:9200
ES_INDEX=iit_policy_chunks
ES_USERNAME=elastic
ES_PASSWORD=changeme

USE_LLM_ANSWER_SYNTHESIS=true
USE_LLM_SLOTS=true
USE_LLM_NARRATION=true
USE_LLM_INTENT=true
ENABLE_EVAL_HACKS=false
```

Notes:
- `ENABLE_EVAL_HACKS` should be **false** for real production behavior.
- If you want fully deterministic behavior (no LLM for intent/slots), set:
  - `USE_LLM_SLOTS=false`
  - `USE_LLM_INTENT=false`
  - (answer synthesis still uses LLM unless disabled)

---

## Start ElasticSearch

```bash
docker compose up -d
```

Verify:
```bash
curl http://localhost:9200
```

---

## Build the index + ingest data

### 1) Create index mapping
```bash
python -m ingest.es_setup
```

### 2) Ingest markdown policies
```bash
python -m ingest.ingest
```

To force a full re-ingest after changing `data/`:
- delete `ingest/manifest.json`
- run `python -m ingest.ingest` again

---

## Run the app (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Open the URL shown (usually `http://localhost:8501`).

Use the sidebar **Reset chat** button before running evaluation batches.

---

## Evaluation

If your `eval/` harness is present:
```bash
python -m eval.evaluate
```

---

## Deployment Notes

- Streamlit can be deployed on:
  - Streamlit Community Cloud (requires external ES)
  - VM/Container (recommended)
- ElasticSearch should run as a separate service (Docker/VM/managed ES).
- For production:
  - keep `.env` secrets out of git
  - lock dependencies with a requirements lock
  - set `ENABLE_EVAL_HACKS=false`
  - monitor ES health and ingestion failures

---
