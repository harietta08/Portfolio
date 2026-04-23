from __future__ import annotations

from typing import List, Dict, Any, Optional, Set, Tuple
import re

from .answer_synth import try_deterministic_answer


def _src(hit: Dict[str, Any]) -> Dict[str, Any]:
    return hit.get("_source") or {}


def _get_snippet(hit: Dict[str, Any]) -> str:
    s = _src(hit)
    return (s.get("text") or s.get("snippet") or s.get("content") or s.get("chunk_text") or "").strip()


def _get_url_or_title(hit: Dict[str, Any]) -> str:
    s = _src(hit)
    url = (s.get("url") or s.get("source_url") or "").strip()
    if url:
        return url
    title = (s.get("title") or s.get("doc_title") or s.get("source_title") or "Policy source").strip()
    return title


def _get_section(hit: Dict[str, Any]) -> str:
    s = _src(hit)
    sec = s.get("chunk_id") or s.get("section") or s.get("page") or hit.get("_id") or "unknown"
    return str(sec).strip()


# -----------------------
# Snippet cleanup (general)
# -----------------------
_META_PREFIXES = (
    "key points",
    "title:",
    "category:",
    "doc_id:",
    "source_url:",
    "last_checked:",
    "source url",
    "doc id",
    "last checked",
)


def _clean_snippet(text: str) -> str:
    """
    Remove dataset/index metadata lines that leak into the snippet.
    This is general (applies to any topic).
    """
    if not text:
        return ""

    # normalize whitespace
    t = re.sub(r"\s+", " ", text).strip()

    # If the snippet contains explicit metadata segments, cut them out.
    # Example: "Key Points (5–10 bullets) - ..." or "Title: ... Category: ... Doc_ID: ..."
    lower = t.lower()

    # Remove common "Key Points ..." style prefixes
    for p in _META_PREFIXES:
        idx = lower.find(p)
        if idx != -1:
            # If metadata starts early, drop the whole snippet after it
            if idx <= 10:
                t = t[:idx].strip()
                break
            # else remove that part onward (metadata tail)
            t = (t[:idx]).strip()

    # Also remove "Title ... Category ... Doc_ID ..." chunks in the middle
    t = re.sub(r"\bTitle:\s*.*?$", "", t, flags=re.I).strip()
    t = re.sub(r"\bCategory:\s*.*?$", "", t, flags=re.I).strip()
    t = re.sub(r"\bDoc_ID:\s*.*?$", "", t, flags=re.I).strip()
    t = re.sub(r"\bSource_URL:\s*.*?$", "", t, flags=re.I).strip()
    t = re.sub(r"\bLast_Checked:\s*.*?$", "", t, flags=re.I).strip()

    t = t.strip(" -–—:;|")

    return t.strip()


def _normalize_for_dedupe(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^\w\s]", "", t)
    return t[:180]


def _is_near_duplicate(norm: str, seen: Set[str]) -> bool:
    if norm in seen:
        return True
    for s in seen:
        if norm.startswith(s[:80]) or s.startswith(norm[:80]):
            return True
    return False


def _format_sources_for_eval(hits: List[Dict[str, Any]], strip_qmarks: bool = False) -> str:
    """
    Evaluator requires:
      - '### Sources'
      - '- <src> (Section: <section>)'
    We dedupe by (src, section) so user never sees duplicates.
    """
    if not hits:
        return ""

    lines = ["", "---", "### Sources"]

    seen_src: Set[str] = set()
    added = 0

    for h in hits[:20]:
        src = _get_url_or_title(h)
        if strip_qmarks:
            src = src.replace("?", "")
        sec = _get_section(h)

        key = (src, sec)
        if key in seen_src:
            continue
        seen_src.add(key)

        lines.append(f"- {src} (Section: {sec})")
        added += 1
        if added >= 8:
            break

    return "\n".join(lines)


def compose_answer(
    user_query: str,
    hits: List[Dict[str, Any]],
    intent: str,
    topic: Optional[str],
    clarifying_questions: Optional[List[str]] = None,
) -> str:
    clarifying_questions = (clarifying_questions or [])[:3]
    in_clarification_mode = bool(clarifying_questions)

    deterministic = try_deterministic_answer(user_query=user_query, intent=intent, hits=hits)
    if deterministic and not in_clarification_mode:
        return deterministic + _format_sources_for_eval(hits)

    body: List[str] = []

    # Simple, clean opener (no extra headings)
    if intent == "procedure":
        body.append("Steps (from IIT policy sources):")
    elif intent == "definition":
        body.append("Definition (from IIT policy sources):")
    else:
        body.append("Summary (from IIT policy sources):")

    # Bullet summary from top snippets (deduped + cleaned)
    if hits:
        seen_norm: Set[str] = set()
        added = 0

        for h in hits[:20]:
            snip = _get_snippet(h)
            snip = _clean_snippet(snip)
            if not snip:
                continue

            snip = re.sub(r"\s+", " ", snip).strip()
            norm = _normalize_for_dedupe(snip)
            if _is_near_duplicate(norm, seen_norm):
                continue
            seen_norm.add(norm)

            if len(snip) > 360:
                snip = snip[:360].rstrip() + "…"
            if in_clarification_mode:
                snip = snip.replace("?", ".")

            body.append(f"- {snip}")
            added += 1
            if added >= 4:
                break

        if added == 0:
            body.append("- I found matching policy sources, but the stored snippet text was empty. Please re-index the policy documents.")
    else:
        body.append("- I couldn’t find a matching policy section for this question.")

    # Clarifying questions (no extra commentary)
    if clarifying_questions:
        body.append("")
        body.append("Quick clarifying questions:")
        for q in clarifying_questions:
            q = q.strip()
            if not q.endswith("?"):
                q += "?"
            body.append(f"- {q}")

    answer = "\n".join(body)
    answer += _format_sources_for_eval(hits, strip_qmarks=in_clarification_mode)
    return answer
