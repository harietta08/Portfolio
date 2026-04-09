# ── Startup Funding Intelligence Tool — Makefile ───────────────────────────────
# All commands run from repo root: make <target>

.PHONY: help install install-dev lint test test-cov \
        ingest embed api streamlit docker-build docker-run \
        dbt-run dbt-test clean

# ── Default ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Startup Funding Intelligence Tool"
	@echo "  ─────────────────────────────────"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install + dev/test dependencies"
	@echo "  make lint          Run flake8 + black check"
	@echo "  make test          Run pytest suite"
	@echo "  make test-cov      Run pytest with coverage report"
	@echo "  make ingest        Run ingestion pipeline (scrape → validate → GCS)"
	@echo "  make embed         Build ChromaDB vector store from processed data"
	@echo "  make api           Start FastAPI server locally (port 8080)"
	@echo "  make streamlit     Start Streamlit app locally (port 8501)"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container locally"
	@echo "  make dbt-run       Run dbt models against BigQuery"
	@echo "  make dbt-test      Run dbt schema tests"
	@echo "  make clean         Remove Python cache and build artifacts"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install black flake8

# ── Code quality ───────────────────────────────────────────────────────────────
lint:
	black --check .
	flake8 . --max-line-length=100 --exclude=venv,.venv,dbt/target

format:
	black . --line-length=100

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

# ── Pipeline steps ─────────────────────────────────────────────────────────────
ingest:
	python ingestion/scraper.py
	python ingestion/validate_schema.py
	python ingestion/gcs_upload.py

embed:
	python embeddings/embed_startups.py
	python embeddings/chromadb_store.py

# ── Serving ────────────────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

streamlit:
	streamlit run app/streamlit_app.py --server.port 8501

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t vc-intelligence:latest .

docker-run:
	docker run -p 8080:8080 --env-file .env vc-intelligence:latest

# ── dbt ────────────────────────────────────────────────────────────────────────
dbt-run:
	cd dbt && dbt run --profiles-dir .

dbt-test:
	cd dbt && dbt test --profiles-dir .

dbt-docs:
	cd dbt && dbt docs generate --profiles-dir . && dbt docs serve

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Cleaned"
