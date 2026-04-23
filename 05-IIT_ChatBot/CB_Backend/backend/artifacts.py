# backend/artifacts.py
from __future__ import annotations

"""
Extract and render helpful artifacts (links, portals, forms, contacts).

Requirements:
- If retrieved excerpts contain portal/form/contact links, surface them as clickable links.
- Labels must match the MD when available (e.g., [ISSS Portal](...)).
- Do NOT show vague labels like "General".
- Do NOT duplicate the same link already shown under Sources (often the page URL itself).
- Generic: no hardcoded ISSS/OGS URLs.
"""

import re
from typing import Any, Dict, List, Tuple

# Markdown links: [Label](https://...)
MD_LINK_RE = re.compile(r"\[([^\]]{2,120})\]\((https?://[^\)\s]+)\)", re.I)
# Label: https://...
LABEL_URL_RE = re.compile(r"(?m)^\s*([A-Za-z0-9][^:\n]{1,120})\s*:\s*(https?://\S+)\s*$")
# Bare URLs
URL_RE = re.compile(r"(https?://[^\s\)\]\}]+)", re.I)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"\b(?:\+?1[\s\-\.]?)?(?:\(\d{3}\)|\d{3})[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b")

SECTION_HINTS = (
    "forms", "form", "portal", "portals", "links", "link",
    "contact", "contacts", "email", "phone", "office", "hours", "walk-in"
)

GENERIC_LABELS = {"general", "overview", "home", "website", "page"}


def _src(hit: Dict[str, Any]) -> Dict[str, Any]:
    return hit.get("_source") or {}


def _hit_text(hit: Dict[str, Any]) -> str:
    s = _src(hit)
    parts = [
        s.get("section_title") or "",
        s.get("section_path") or "",
        s.get("heading") or "",
        s.get("text") or "",
        s.get("snippet") or "",
    ]
    return "\n".join([p for p in parts if p]).strip()


def _clean_label(label: str) -> str:
    label = (label or "").strip()
    label = re.sub(r"\s+", " ", label)
    label = label.strip(" -•:\t")
    return label[:120]


def _fallback_label_from_url(url: str) -> str:
    # Better fallback label than "General": use last path segment
    path = re.sub(r"^https?://", "", url)
    seg = path.split("?", 1)[0].rstrip("/").split("/")[-1]
    seg = seg.replace("-", " ").replace("_", " ").strip()
    if not seg:
        return url
    return seg[:80].title()


def _is_probably_same_as_source(url: str, hit: Dict[str, Any]) -> bool:
    s = _src(hit)
    src_url = (s.get("source_url") or s.get("url") or "").strip()
    if not src_url:
        return False

    def norm(u: str) -> str:
        return u.split("#", 1)[0].rstrip("/").strip()

    return norm(url) == norm(src_url)


def extract_supporting_artifacts(
    hits: List[Dict[str, Any]],
    *,
    max_links: int = 8,
    max_emails: int = 5,
    max_phones: int = 5,
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    seen_urls = set()
    seen_emails = set()
    seen_phones = set()

    links: List[Tuple[str, str]] = []
    emails: List[str] = []
    phones: List[str] = []

    for h in (hits or [])[:12]:
        s = _src(h)
        sec = (s.get("section_title") or s.get("section_path") or s.get("heading") or "").lower()
        txt = _hit_text(h)
        txt_l = txt.lower()

        boosted = any(k in sec for k in SECTION_HINTS) or any(k in txt_l for k in SECTION_HINTS)

        # Emails/phones always ok to extract
        for e in EMAIL_RE.findall(txt):
            e = e.lower()
            if e in seen_emails:
                continue
            seen_emails.add(e)
            emails.append(e)
            if len(emails) >= max_emails:
                break

        for p in PHONE_RE.findall(txt):
            p = (p or "").strip()
            if not p or p in seen_phones:
                continue
            seen_phones.add(p)
            phones.append(p)
            if len(phones) >= max_phones:
                break

        # Prefer labeled links from MD
        labeled_pairs: List[Tuple[str, str]] = []

        for label, url in MD_LINK_RE.findall(txt):
            label = _clean_label(label)
            url = url.rstrip(").,;]")
            if not label or label.lower() in GENERIC_LABELS:
                label = _fallback_label_from_url(url)
            labeled_pairs.append((label, url))

        for label, url in LABEL_URL_RE.findall(txt):
            label = _clean_label(label)
            url = url.rstrip(").,;]")
            if not label or label.lower() in GENERIC_LABELS:
                label = _fallback_label_from_url(url)
            labeled_pairs.append((label, url))

        # Only fall back to bare URLs if this looks like a portal/forms/contact section
        if not labeled_pairs and boosted:
            for url in URL_RE.findall(txt):
                url = url.rstrip(").,;]")
                labeled_pairs.append((_fallback_label_from_url(url), url))

        for label, url in labeled_pairs:
            if not url or url in seen_urls:
                continue
            # Don’t duplicate the Sources link (usually the page itself)
            if _is_probably_same_as_source(url, h):
                continue
            label = _clean_label(label)
            if not label or label.lower() in GENERIC_LABELS:
                continue
            seen_urls.add(url)
            links.append((label, url))
            if len(links) >= max_links:
                break

        if len(links) >= max_links and len(emails) >= max_emails and len(phones) >= max_phones:
            break

    return links, emails, phones


def render_artifacts_markdown(hits: List[Dict[str, Any]]) -> str:
    links, emails, phones = extract_supporting_artifacts(hits)

    # If we didn’t find any specific artifacts, show nothing.
    if not links and not emails and not phones:
        return ""

    out = "\n\n### Helpful links & contacts\n"

    if links:
        out += "\n**Links / Portals / Forms:**\n"
        for label, url in links:
            out += f"- [{label}]({url})\n"

    if emails:
        out += "\n**Email:**\n"
        for e in emails:
            out += f"- {e}\n"

    if phones:
        out += "\n**Phone:**\n"
        for p in phones:
            out += f"- {p}\n"

    return out.rstrip()
