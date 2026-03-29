.PHONY: install test train api app docker-build docker-run drift-check help

help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make test          Run all unit tests"
	@echo "  make train         Train model and log to MLflow"
	@echo "  make api           Start FastAPI server locally"
	@echo "  make app           Start Streamlit app locally"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container locally"
	@echo "  make drift-check   Run weekly drift detection script"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

train:
	python src/train.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t patient-readmission:latest .

docker-run:
	docker run -p 8080:8080 --env-file .env patient-readmission:latest

drift-check:
	python scripts/check_drift.py
