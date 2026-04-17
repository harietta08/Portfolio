.PHONY: help install test lint ingest forecast dbt-run dbt-test dbt-docs pubsub-local clean

## ─── Help ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Chicago Transit Intelligence Platform"
	@echo "======================================"
	@echo "  make install        Install all Python dependencies"
	@echo "  make test           Run full pytest suite with coverage"
	@echo "  make lint           Flake8 linting check"
	@echo "  make ingest         Run Prefect batch ingest flow locally"
	@echo "  make forecast       Run Prefect forecast refresh flow locally"
	@echo "  make dbt-run        Run all dbt models (staging + marts)"
	@echo "  make dbt-test       Run dbt schema tests"
	@echo "  make dbt-docs       Generate and serve dbt docs"
	@echo "  make pubsub-local   Start Pub/Sub consumer locally"
	@echo "  make clean          Remove __pycache__ and .pytest_cache"
	@echo ""

## ─── Setup ───────────────────────────────────────────────────────────────────
install:
	pip install --upgrade pip
	pip install -r requirements.txt

## ─── Testing ─────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=ingestion --cov=forecasting --cov-report=term-missing

lint:
	flake8 ingestion/ forecasting/ aws/ gcp/ prefect/ --max-line-length=100

## ─── Prefect flows ───────────────────────────────────────────────────────────
ingest:
	python -m prefect.flows.ingest_flow

forecast:
	python -m prefect.flows.forecast_flow

## ─── dbt ─────────────────────────────────────────────────────────────────────
dbt-run:
	cd dbt && dbt run --profiles-dir .

dbt-test:
	cd dbt && dbt test --profiles-dir .

dbt-docs:
	cd dbt && dbt docs generate --profiles-dir . && dbt docs serve --profiles-dir .

## ─── Streaming consumer ──────────────────────────────────────────────────────
pubsub-local:
	python gcp/pubsub_consumer.py

## ─── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."
