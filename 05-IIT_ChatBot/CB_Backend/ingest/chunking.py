from __future__ import annotations
"""Structure-aware chunking by headings with controlled section labels.

This version keeps long checklists and document lists more intact by splitting
list-heavy sections on bullet boundaries instead of arbitrary token windows.
"""

from dataclasses import dataclass
import re
import hashlib
from typing import List, Optional, Tuple
import tiktoken

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[\.)])\s+")
ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str
    category: str
    policy_topic: str
    doc_title: str
    section_title: str
    section_path: str
    heading_level: int
    heading: str
    text: str
    source_file: str
    source_url: Optional[str]
    policy_version: str


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def split_by_headings(md: str) -> List[Tuple[int, str, str, str]]:
    """Return blocks as (level, heading, section_path, body).

    We preserve hierarchy using a simple stack of headings.
    This improves retrieval precision and makes citations more meaningful.
    """
    lines = md.splitlines()

    stack: List[Tuple[int, str]] = []
    cur_level = 0
    cur_heading = ""
    cur_body: List[str] = []

    def cur_path() -> str:
        parts = [h for _, h in stack if h]
        return " > ".join(parts)

    blocks: List[Tuple[int, str, str, List[str]]] = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            blocks.append((cur_level, cur_heading, cur_path(), cur_body))

            level = len(m.group(1))
            heading = m.group(2).strip()

            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))

            cur_level = level
            cur_heading = heading
            cur_body = []
        else:
            cur_body.append(line)

    blocks.append((cur_level, cur_heading, cur_path(), cur_body))

    out: List[Tuple[int, str, str, str]] = []
    for lvl, h, path, b in blocks:
        body = "\n".join(b).strip()
        if body:
            out.append((lvl, h, path, body))
    return out


def classify_section(heading: str) -> str:
    h = (heading or "").strip().lower()

    if any(k in h for k in ["portal", "portals", "isss", "sevp", "handshake", "link", "links", "url"]):
        return "PortalsLinks"
    if any(k in h for k in ["contact", "contacts", "email", "phone", "office", "hours", "walk-in", "address"]):
        return "Contacts"
    if any(k in h for k in ["form", "forms", "checklist", "document checklist"]):
        return "Forms"

    if any(k in h for k in ["eligibility", "eligible", "requirements", "rules"]):
        return "Eligibility"
    if any(k in h for k in ["steps", "how to", "process", "apply", "procedure", "request"]):
        return "Steps"
    if any(k in h for k in ["documents", "i-20", "ds-2019"]):
        return "Documents"
    if any(k in h for k in ["deadline", "dates", "timeline", "window", "processing times"]):
        return "Deadlines"
    if any(k in h for k in ["faq", "questions", "common"]):
        return "FAQ"
    if "key points" in h:
        return "KeyPoints"
    if "clean policy text" in h:
        return "PolicyText"
    return "General"


def token_len(text: str) -> int:
    return len(ENC.encode(text or ""))


def window_text(text: str, max_tokens: int, overlap_tokens: int = 40) -> List[str]:
    toks = ENC.encode(text)
    if len(toks) <= max_tokens:
        return [text.strip()]
    out: List[str] = []
    start = 0
    while start < len(toks):
        end = min(start + max_tokens, len(toks))
        out.append(ENC.decode(toks[start:end]).strip())
        if end == len(toks):
            break
        start = max(0, end - overlap_tokens)
    return out


def _split_list_aware(heading: str, body: str, max_tokens: int) -> List[str]:
    """Preserve long bullet checklists by chunking on bullet boundaries.

    This improves retrieval for "what documents / what should I submit / checklist"
    questions because adjacent list items stay together instead of being cut across
    arbitrary token windows.
    """
    lines = [ln.rstrip() for ln in (body or "").splitlines()]
    bullet_idx = [i for i, ln in enumerate(lines) if BULLET_RE.match(ln)]

    # Fall back when the section is not really a long list.
    if len(bullet_idx) < 5:
        combined = (heading.strip() + "\n" + body).strip() if heading else body.strip()
        return window_text(combined, max_tokens=max_tokens)

    first_bullet = bullet_idx[0]
    preamble_lines = lines[:first_bullet]
    preamble = "\n".join(x for x in preamble_lines if x.strip()).strip()

    items: List[str] = []
    i = first_bullet
    while i < len(lines):
        if not BULLET_RE.match(lines[i]):
            i += 1
            continue
        cur = [lines[i]]
        i += 1
        while i < len(lines) and not BULLET_RE.match(lines[i]):
            cur.append(lines[i])
            i += 1
        items.append("\n".join(x for x in cur if x.strip()).strip())

    if not items:
        combined = (heading.strip() + "\n" + body).strip() if heading else body.strip()
        return window_text(combined, max_tokens=max_tokens)

    pieces: List[str] = []
    current_lines: List[str] = []
    base_prefix = heading.strip() + "\n" if heading else ""
    if preamble:
        current_lines.append(preamble)

    for item in items:
        candidate_lines = current_lines + [item]
        candidate_text = (base_prefix + "\n".join(candidate_lines)).strip()

        if token_len(candidate_text) > max_tokens and current_lines:
            pieces.append((base_prefix + "\n".join(current_lines)).strip())
            current_lines = [item]
        else:
            current_lines = candidate_lines

    if current_lines:
        pieces.append((base_prefix + "\n".join(current_lines)).strip())

    # Safety fallback if any grouped piece is still too large.
    out: List[str] = []
    for piece in pieces:
        if token_len(piece) <= max_tokens:
            out.append(piece)
        else:
            out.extend(window_text(piece, max_tokens=max_tokens))
    return out


def make_chunks(*, md_text: str, category: str, policy_topic: str, doc_title: str, source_file: str,
                source_url: Optional[str], policy_version: str,
                max_tokens: int = 380) -> List[Chunk]:
    chunks: List[Chunk] = []
    for lvl, heading, path, body in split_by_headings(md_text):
        section_title = classify_section(heading)

        if section_title in {"Forms", "Documents", "Steps", "PolicyText"}:
            pieces = _split_list_aware(heading, body, max_tokens=max_tokens)
        else:
            combined = (heading.strip() + "\n" + body).strip() if heading else body.strip()
            pieces = window_text(combined, max_tokens=max_tokens)

        for piece in pieces:
            chunk_id = sha256_text(f"{source_file}|{heading}|{piece}|{policy_version}")[:32]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                category=category,
                policy_topic=policy_topic,
                doc_title=doc_title,
                section_title=section_title,
                section_path=path or heading or "",
                heading_level=int(lvl or 0),
                heading=heading or "",
                text=piece,
                source_file=source_file,
                source_url=source_url,
                policy_version=policy_version
            ))
    return chunks
