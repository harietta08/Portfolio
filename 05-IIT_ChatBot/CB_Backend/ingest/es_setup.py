from __future__ import annotations
"""Creates ElasticSearch index for policy chunks."""

from elasticsearch import Elasticsearch
from backend.config import SETTINGS

def get_es() -> Elasticsearch:
    return Elasticsearch(
        SETTINGS.es_url,
        basic_auth=(SETTINGS.es_username, SETTINGS.es_password),
        request_timeout=60
    )

def create_index() -> None:
    es = get_es()
    index = SETTINGS.es_index
    dims = 1536 if "small" in SETTINGS.openai_embed_model else 3072

    if es.indices.exists(index=index):
        print(f"Index already exists: {index}")
        return

    mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "heading": {"type": "text"},
                "doc_title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "category": {"type": "keyword"},
                "policy_topic": {"type": "keyword"},
                "section_title": {"type": "keyword"},
                "section_path": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "heading_level": {"type": "integer"},
                "source_file": {"type": "keyword"},
                "source_url": {"type": "keyword"},
                "policy_version": {"type": "keyword"},
                "indexed_at": {"type": "date"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    es.indices.create(index=index, **mapping)
    print(f"Created index: {index} (dims={dims})")

if __name__ == "__main__":
    create_index()
