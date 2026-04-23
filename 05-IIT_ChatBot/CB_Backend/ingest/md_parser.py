from __future__ import annotations
"""Parses IIT policy Markdown files and extracts embedded metadata."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional, Dict

META_RE = re.compile(r"^(Category|Doc_ID|Source_URL|Last_Checked):\s*(.*)\s*$", re.M)
TITLE_RE = re.compile(r"^#\s*Title:\s*(.*)\s*$", re.M)

@dataclass
class DocMeta:
    title: str
    category: str
    doc_id: str
    source_url: Optional[str]
    last_checked: Optional[str]

def parse_metadata(md_text: str, fallback_category: str, fallback_doc_id: str) -> DocMeta:
    title_m = TITLE_RE.search(md_text)
    title = title_m.group(1).strip() if title_m else fallback_doc_id

    meta: Dict[str, str] = {m.group(1): m.group(2).strip() for m in META_RE.finditer(md_text)}
    category = meta.get("Category", fallback_category)
    doc_id = meta.get("Doc_ID", fallback_doc_id)
    source_url = meta.get("Source_URL")
    last_checked = meta.get("Last_Checked")
    return DocMeta(title=title, category=category, doc_id=doc_id, source_url=source_url, last_checked=last_checked)

def read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")
