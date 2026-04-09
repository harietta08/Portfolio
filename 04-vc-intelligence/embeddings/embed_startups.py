# ── embeddings/embed_startups.py ──────────────────────────────────────────────
# Purpose: Generate sentence embeddings locally from processed CSV
# Output: embeddings saved to data/processed/ for ChromaDB ingestion

from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
INPUT_PATH = Path("data/sample/startups_preprocessed.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def embed_startups(input_path: Path, output_dir: Path) -> np.ndarray:
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    # Use cleaned text if available, fall back to raw description
    text_col = "description_clean" if "description_clean" in df.columns else "description"
    texts = df[text_col].fillna("").tolist()
    logger.info(f"Embedding {len(texts)} records using {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # Save embeddings
    emb_path = output_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    logger.info(f"Embeddings saved: {emb_path} — shape {embeddings.shape}")

    # Save metadata alongside embeddings
    meta_path = output_dir / "metadata.csv"
    df[["id", "name", "sector", "stage", "funding_amount_usd", "hq_city"]].to_csv(
        meta_path, index=False
    )
    logger.info(f"Metadata saved: {meta_path}")

    return embeddings

if __name__ == "__main__":
    embeddings = embed_startups(INPUT_PATH, OUTPUT_DIR)
    print(f"Done. Shape: {embeddings.shape}")
