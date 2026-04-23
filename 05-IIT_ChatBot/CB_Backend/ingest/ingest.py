from __future__ import annotations
"""Offline ingestion: MD → chunks → OpenAI embeddings → ElasticSearch."""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

from elasticsearch.helpers import bulk

from backend.es_client import get_es
from backend.embedding_service import embed_text
from backend.config import SETTINGS
from ingest.md_parser import read_md, parse_metadata
from ingest.chunking import make_chunks

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()

def load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}

def save_manifest(m: Dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")

def iter_md_files() -> List[Path]:
    return sorted([p for p in DATA_ROOT.rglob("*.md")])

def ingest() -> None:
    es = get_es()
    index = SETTINGS.es_index
    manifest = load_manifest()
    now = datetime.now(timezone.utc).isoformat()

    actions = []
    updated_manifest = dict(manifest)

    md_files = iter_md_files()
    if not md_files:
        raise RuntimeError(f"No .md files found under {DATA_ROOT}")

    for path in md_files:
        rel = str(path.relative_to(DATA_ROOT))
        file_hash = sha256_file(path)
        prev = manifest.get(rel, {})
        if prev.get("sha256") == file_hash:
            continue

        md_text = read_md(path)
        fallback_category = path.parent.name
        fallback_doc_id = path.stem
        meta = parse_metadata(md_text, fallback_category=fallback_category, fallback_doc_id=fallback_doc_id)

        policy_version = meta.last_checked or file_hash[:10]
        chunks = make_chunks(
            md_text=md_text,
            category=meta.category,
            policy_topic=meta.doc_id,   # use Doc_ID as the stable topic key
            doc_title=meta.title,
            source_file=str(path),
            source_url=meta.source_url,
            policy_version=policy_version
        )

        for ch in chunks:
            vec = embed_text(ch.text)
            doc = {
                "text": ch.text,
                "heading": ch.heading,
                "category": ch.category,
                "policy_topic": ch.policy_topic,
                "doc_title": ch.doc_title,
                "section_title": ch.section_title,
                "section_path": ch.section_path,
                "heading_level": ch.heading_level,
                "source_file": ch.source_file,
                "source_url": ch.source_url,
                "policy_version": ch.policy_version,
                "indexed_at": now,
                "embedding": vec
            }
            actions.append({
                "_op_type": "index",
                "_index": index,
                "_id": ch.chunk_id,
                "_source": doc
            })

        updated_manifest[rel] = {
            "sha256": file_hash,
            "indexed_at": now,
            "chunks": len(chunks),
            "embedding_model": SETTINGS.openai_embed_model,
            "doc_id": meta.doc_id,
            "source_url": meta.source_url
        }
        print(f"Prepared {len(chunks)} chunks from {rel}")

    if actions:
        bulk(es, actions, request_timeout=120)
        save_manifest(updated_manifest)
        print(f"Indexed {len(actions)} chunks into {index}")
    else:
        print("No changes detected. Nothing to index.")

if __name__ == "__main__":
    ingest()
