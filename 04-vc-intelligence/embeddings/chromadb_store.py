# ── embeddings/chromadb_store.py ──────────────────────────────────────────────
# Purpose: Build ChromaDB vector store from processed startup data
# Runs once after embed_startups.py generates embeddings
# ChromaDB persists to disk — FastAPI loads it at startup
# Interview answer: "ChromaDB runs embedded — no separate server,
#                   no infra cost, persists to disk, loads in milliseconds."

import os
import numpy as np
import pandas as pd
import chromadb
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "startups")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
METADATA_PATH = Path("data/processed/metadata.csv")
SAMPLE_PATH = Path("data/sample/startups_sample.csv")


def build_chromadb_store(
    embeddings_path: Path = EMBEDDINGS_PATH,
    metadata_path: Path = METADATA_PATH,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Build ChromaDB collection from embeddings and metadata.
    Uses cosine similarity — consistent with how embeddings were generated.
    Upserts so re-running does not create duplicates.
    """
    logger.info(f"Initializing ChromaDB at {persist_dir}")
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection for clean rebuild
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Load embeddings
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        df_meta = pd.read_csv(metadata_path)
        df_text = pd.read_csv(SAMPLE_PATH)
        logger.info(f"Loaded embeddings: {embeddings.shape}")
    else:
        # Fallback: generate embeddings from sample on the fly
        logger.warning("Embeddings not found — generating from sample data")
        from sentence_transformers import SentenceTransformer
        df_text = pd.read_csv(SAMPLE_PATH)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(df_text["description"].tolist(), show_progress_bar=True)
        df_meta = df_text[["id", "name", "sector", "stage", "funding_amount_usd", "hq_city"]]

    # Build metadata dicts for ChromaDB
    metadatas = []
    for _, row in df_meta.iterrows():
        metadatas.append({
            "name": str(row.get("name", "")),
            "sector": str(row.get("sector", "")),
            "stage": str(row.get("stage", "")),
            "funding_usd": float(row.get("funding_amount_usd", 0)),
            "city": str(row.get("hq_city", "")),
        })

    # Get documents (descriptions) for ChromaDB
    documents = df_text["description"].tolist()
    ids = [str(i) for i in range(len(documents))]

    # Add to collection in batches of 50
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_emb = embeddings[i:i + batch_size].tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=batch_emb,
        )
        logger.info(f"Added batch {i//batch_size + 1} — {len(batch_ids)} records")

    count = collection.count()
    logger.info(f"ChromaDB built — {count} documents in collection '{collection_name}'")

    # Quick sanity check
    test_results = collection.query(
        query_texts=["AI platform for carbon accounting"],
        n_results=3,
    )
    logger.info("Sanity check query results:")
    for i, doc_id in enumerate(test_results["ids"][0]):
        meta = test_results["metadatas"][0][i]
        logger.info(f"  {i+1}. {meta['name']} ({meta['sector']})")

    return collection


if __name__ == "__main__":
    collection = build_chromadb_store()
    print(f"Done — {collection.count()} startups indexed in ChromaDB")
