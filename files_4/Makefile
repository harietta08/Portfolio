.PHONY: help install test lint api app ingest embed dbt-run dbt-test docker-build docker-run clean

help:
	@echo "Startup Funding Intelligence Tool — available commands:"
	@echo ""
	@echo "  make install       Install all Python dependencies"
	@echo "  make test          Run all unit tests"
	@echo "  make lint          Run ruff linter"
	@echo "  make api           Start FastAPI server locally"
	@echo "  make app           Start Streamlit app locally"
	@echo "  make ingest        Run ingestion pipeline (scrape → validate → GCS)"
	@echo "  make embed         Generate embeddings and load ChromaDB"
	@echo "  make dbt-run       Run all dbt models"
	@echo "  make dbt-test      Run dbt schema tests"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container locally"
	@echo "  make clean         Remove generated artifacts"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	ruff check . --fix

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app/streamlit_app.py

ingest:
	python ingestion/scraper.py
	python ingestion/validate_schema.py
	python ingestion/gcs_upload.py

embed:
	python embeddings/embed_startups.py
	python embeddings/chromadb_store.py

dbt-run:
	cd dbt && dbt run

dbt-test:
	cd dbt && dbt test

docker-build:
	docker build -t vc-intelligence:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env vc-intelligence:latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage
