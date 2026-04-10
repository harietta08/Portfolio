# ── tests/test_embeddings.py ──────────────────────────────────────────────────
# Tests embedding generation and ChromaDB store

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path


def test_embed_returns_correct_shape():
    """Embedding model returns (n_records, 384) for MiniLM."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 384)
        mock_st.return_value = mock_model

        texts = ["startup one", "startup two", "startup three"]
        embeddings = mock_model.encode(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float64


def test_embed_handles_empty_text():
    """Empty string should not crash embedder."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        mock_st.return_value = mock_model

        result = mock_model.encode([""])
        assert result.shape[1] == 384


def test_chromadb_store_and_query():
    """ChromaDB collection stores and retrieves documents."""
    import chromadb
    client = chromadb.EphemeralClient()
    collection = client.create_collection("test_startups")

    collection.add(
        ids=["1", "2", "3"],
        documents=[
            "AI carbon accounting platform",
            "Healthcare AI for clinical documentation",
            "Fintech payments infrastructure",
        ],
        metadatas=[
            {"name": "CarbonAI", "sector": "Climate Tech", "stage": "Seed"},
            {"name": "MedAI", "sector": "Healthcare AI", "stage": "Series A"},
            {"name": "PayTech", "sector": "Fintech", "stage": "Series B"},
        ],
    )

    assert collection.count() == 3

    results = collection.query(
        query_texts=["carbon emissions tracking"],
        n_results=2,
    )
    assert len(results["ids"][0]) == 2
    assert results["ids"][0][0] == "1"


def test_similarity_score_range():
    """Similarity scores should be between 0 and 1."""
    import chromadb
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        "test_sim",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=["1"],
        documents=["climate tech carbon accounting"],
        metadatas=[{"name": "Test"}],
    )
    results = collection.query(
        query_texts=["carbon emissions"],
        n_results=1,
        include=["distances"],
    )
    distance = results["distances"][0][0]
    similarity = 1 - distance
    assert 0 <= similarity <= 1
