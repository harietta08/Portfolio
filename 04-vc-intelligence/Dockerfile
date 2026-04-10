# ── Dockerfile ────────────────────────────────────────────────────────────────
# Containerizes FastAPI serving layer for GCP Cloud Run
# Build: docker build -t vc-intelligence .
# Run:   docker run -p 8080:8080 --env-file .env vc-intelligence

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker layer caching
# If requirements.txt unchanged, this layer is cached on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time — not at runtime
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Copy application code
COPY api/ ./api/
COPY llm/ ./llm/
COPY embeddings/ ./embeddings/
COPY data/processed/ ./data/processed/
COPY data/sample/ ./data/sample/

# Copy environment example — real .env mounted at runtime
COPY .env.example .env.example

# Cloud Run sets PORT env var — default 8080
ENV PORT=8080
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check — Cloud Run uses this
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start FastAPI with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
