from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


FALLBACK_SENTENCE = "I can’t confirm this from the provided IIT policy excerpts."


def split_before_sources(answer_md: str) -> Tuple[str, str]:
    if not answer_md:
        return "", ""
    marker = "### Sources"
    if marker in answer_md:
        before, after = answer_md.split(marker, 1)
        return before.rstrip(), (marker + after)
    return answer_md.rstrip(), ""


def _extract_urls(hits: Optional[List[Dict[str, Any]]]) -> List[str]:
    urls: List[str] = []
    seen = set()

    for h in hits or []:
        src = (h.get("_source") or {})
        for key in ("source_url", "url"):
            u = (src.get(key) or "").strip()
            if u and u not in seen:
                seen.add(u)
                urls.append(u)
    return urls


def _pick_official_url(hits: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    urls = _extract_urls(hits)
    if not urls:
        return None

    preferred = [u for u in urls if "iit.edu" in u.lower()]
    return preferred[0] if preferred else urls[0]


def _remove_keywords_leak(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"(?im)^\s*keywords\s*:\s*.*$", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _remove_separator_leak(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    cleaned = [ln for ln in lines if ln.strip() != "---"]
    return "\n".join(cleaned).strip()


def _remove_duplicate_fallback_when_answer_exists(text: str) -> str:
    if not text or FALLBACK_SENTENCE not in text:
        return text

    parts = text.split(FALLBACK_SENTENCE)
    before = parts[0].strip()

    if len(before) >= 60:
        return before

    return text


def _inject_official_link_if_helpful(
    text: str,
    intent: str,
    hits: Optional[List[Dict[str, Any]]],
    user_query: str = "",
) -> str:
    if not text:
        return text

    if intent not in {"procedure", "contact_info", "portal_link"} and not re.search(r"\b(where|which page|which website|download|view|access|find)\b", user_query or "", re.I):
        return text

    url = _pick_official_url(hits)
    if not url:
        return text

    if url in text or re.search(r"\]\(" + re.escape(url) + r"\)", text):
        return text

    tail = f"\n\nFor the latest details, please review the [official IIT page]({url})."
    return text.rstrip() + tail


def _remove_false_uncertainty(text: str, hits: Optional[List[Dict[str, Any]]]) -> str:
    if not text or FALLBACK_SENTENCE not in text:
        return text

    if not hits:
        return text

    supported_patterns = [
        r"(?m)^\s*[-*]\s+",
        r"(?m)^\s*\d+\.\s+",
        r"\b12 months\b",
        r"\b24 months\b",
        r"\bextension\b",
        r"\bmust\b",
        r"\brequired\b",
    ]
    if any(re.search(p, text, flags=re.I) for p in supported_patterns):
        return _remove_duplicate_fallback_when_answer_exists(text)

    return text


def postprocess_answer(
    *,
    user_query: str,
    answer_markdown: str,
    intent: str,
    mode: str,
    decision: Optional[Dict[str, Any]] = None,
    hits: Optional[List[Dict[str, Any]]] = None,
) -> str:
    main_text, sources = split_before_sources(answer_markdown)

    main_text = _remove_keywords_leak(main_text)
    main_text = _remove_separator_leak(main_text)
    main_text = _remove_false_uncertainty(main_text, hits)
    main_text = _inject_official_link_if_helpful(main_text, intent, hits, user_query)

    if sources:
        return main_text.rstrip() + "\n\n" + sources.lstrip()

    return main_text.rstrip()