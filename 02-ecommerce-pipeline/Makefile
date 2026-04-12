.PHONY: help install test lint ingest airflow-up airflow-down dbt-run dbt-test dbt-docs

help:
	@echo "E-Commerce Analytics Pipeline — Available Commands"
	@echo "=================================================="
	@echo "make install        Install Python dependencies"
	@echo "make test           Run all pytest unit tests"
	@echo "make lint           Run flake8 linter"
	@echo "make ingest         Run API ingestion + GCS upload"
	@echo "make airflow-up     Start Airflow via Docker Compose"
	@echo "make airflow-down   Stop Airflow"
	@echo "make dbt-run        Run all dbt models"
	@echo "make dbt-test       Run all dbt tests"
	@echo "make dbt-docs       Generate and serve dbt lineage docs"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=ingestion --cov=analysis --cov-report=term-missing

lint:
	flake8 ingestion/ databricks/ export/ analysis/ tests/ --max-line-length=100

ingest:
	python ingestion/api_client.py
	python ingestion/validate_schema.py
	python ingestion/gcs_upload.py

airflow-up:
	cd airflow && docker compose up -d

airflow-down:
	cd airflow && docker compose down

dbt-run:
	cd dbt && dbt run

dbt-test:
	cd dbt && dbt test

dbt-docs:
	cd dbt && dbt docs generate && dbt docs serve
