from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from .llm import llm_text
from .config import SETTINGS

_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[\.)])\s+(.*\S)")
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@illinoistech\.edu\b", re.I)
_DURATION_RE = re.compile(r"\b(?:within\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty[- ]?four)\s+(?:business\s+days?|calendar\s+days?|days?|hours?|weeks?|months?)\b", re.I)
_DOC_HINT_RE = re.compile(
    r"\b(i-20|ds-2019|form|passport|i-94|sevis|receipt|fee|letter|financial|cover letter|employment letter|support letter|ead card|visa|ss-5)\b",
    re.I,
)


def _src(hit: Dict[str, Any]) -> Dict[str, Any]:
    return hit.get("_source") or {}


def _needs_location_context(user_query: str, intent: str) -> bool:
    q = (user_query or "").lower()
    return intent in {"portal_link", "contact_info"} or bool(re.search(r"\b(where|which page|which website|view|download|access|submit|find)\b", q))



def _hit_meta_text(hit: Dict[str, Any]) -> str:
    s = _src(hit)
    return " ".join([
        str(s.get("policy_topic") or ""),
        str(s.get("heading") or ""),
        str(s.get("section_title") or ""),
        str(s.get("section_path") or ""),
        str(s.get("doc_title") or ""),
    ]).lower()



def _hit_text_raw(hit: Dict[str, Any]) -> str:
    s = _src(hit)
    return str(s.get("text") or s.get("snippet") or "")



def _iter_lines(hits: List[Dict[str, Any]]):
    for h in hits:
        meta = _hit_meta_text(h)
        text = _hit_text_raw(h)
        for ln in text.splitlines():
            stripped = ln.strip()
            if stripped:
                yield meta, stripped



def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        norm = re.sub(r"\W+", " ", item.lower()).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(item.strip())
    return out



def _is_timing_query(q: str) -> bool:
    q = (q or "").lower()
    return bool(re.search(r"\b(how soon|how long|when should|within how many|earliest|latest|processing time|turnaround time|before my program deadline)\b", q))



def _is_document_query(q: str) -> bool:
    q = (q or "").lower()
    return bool(re.search(r"\b(what specific documents|what documents|must submit|required documents|what forms|need to upload|application materials|bring:)\b", q))



def _is_rcl_reasons_query(q: str) -> bool:
    q = (q or "").lower()
    return bool(re.search(r"\b(acceptable reasons|reasons for rcl|reasons for reduced course load|grounds for rcl)\b", q))



def _is_contact_query(q: str) -> bool:
    q = (q or "").lower()
    return "email" in q or "phone" in q or "contact" in q



def _is_consequence_query(q: str) -> bool:
    q = (q or "").lower()
    return bool(re.search(r"\b(consequence|consequences|what happens if|without informing|without approval|impact .*status|affect .*status|fail to)\b", q))



def _timing_answer(user_query: str, hits: List[Dict[str, Any]]) -> str:
    q = (user_query or "").lower()
    text = "\n".join(_hit_text_raw(h) for h in hits)

    if "check-in" in q or "check in" in q:
        if re.search(r"within\s+\*\*24\s*hours\*\*|within\s+24\s*hours|24\s*hours", text, re.I):
            ans = "International students should complete OGS check-in within 24 hours of arriving in the United States."
            if re.search(r"isss portal", text, re.I):
                ans += " The check-in is completed virtually through the ISSS Portal."
            return ans
        if re.search(r"two\s+business\s+days|2\s+business\s+days", text, re.I):
            return "International students should complete OGS check-in within two business days of arrival."

    if any(x in q for x in ["processing", "process", "deadline", "submit immigration-related requests"]):
        parts: List[str] = []
        if re.search(r"apply as early as possible", text, re.I):
            parts.append("Students should submit immigration-related requests as early as possible.")
        m = re.search(r"seven\s*\(?7\)?\s+business\s+days|7\s+business\s+days", text, re.I)
        if m:
            parts.append("Most immigration document requests typically take 7 business days.")
        if re.search(r"late or rushed requests (?:may not|cannot) be honored", text, re.I):
            parts.append("Late or rushed requests may not be honored.")
        return " ".join(parts).strip()

    durations = _dedupe_keep_order([m.group(0) for m in _DURATION_RE.finditer(text)])
    if durations:
        return f"The policy page indicates a time frame of {durations[0]}."
    return ""



def _document_answer(user_query: str, hits: List[Dict[str, Any]]) -> str:
    q = (user_query or "").lower()
    items: List[str] = []

    for meta, line in _iter_lines(hits):
        m = _BULLET_RE.match(line)
        if not m:
            continue
        item = m.group(1).strip()
        if item.lower().startswith(("note:", "requests listed", "request type", "student demo", "scholar demo")):
            continue
        if not _DOC_HINT_RE.search(item):
            continue

        if "change of status" in q or "change to f-1" in q:
            if any(x in item.lower() for x in ["i-20", "ds-2019", "sevis", "i-539", "g-1145", "fee", "letter", "financial", "cover letter", "passport", "i-94", "dependent status"]):
                items.append(item)
        elif "social security" in q or "ssn" in q:
            if any(x in item.lower() for x in ["employment letter", "support letter", "i-20", "passport", "i-94", "ss-5", "ead card", "ds-2019"]):
                items.append(item)
        else:
            items.append(item)

    items = _dedupe_keep_order(items)
    if not items:
        return ""

    if "change of status" in q or "change to f-1" in q:
        intro = "When applying for a Change of Status to F-1 from within the United States, students typically submit:"
    elif "social security" in q or "ssn" in q:
        intro = "For an SSN application, students typically need to bring:"
    else:
        intro = "The policy page lists the following required items:"

    lines = [intro] + [f"- {x}" for x in items[:14]]
    return "\n".join(lines)



def _rcl_reasons_answer(hits: List[Dict[str, Any]]) -> str:
    text = "\n".join(_hit_text_raw(h) for h in hits)
    reasons = []
    mapping = [
        (r"completion of program", "Completion of program"),
        (r"qualifying/comprehensive exam", "Graduate students taking the qualifying/comprehensive exam"),
        (r"academic difficulties", "Academic difficulties"),
        (r"medical condition", "Medical condition"),
    ]
    for pat, label in mapping:
        if re.search(pat, text, re.I):
            reasons.append(label)
    reasons = _dedupe_keep_order(reasons)
    if not reasons:
        return ""
    return "Acceptable reasons for a Reduced Course Load (RCL) include:\n" + "\n".join(f"- {r}" for r in reasons)



def _contact_answer(user_query: str, hits: List[Dict[str, Any]]) -> str:
    q = (user_query or "").lower()
    emails = _dedupe_keep_order([m.group(0) for h in hits for m in _EMAIL_RE.finditer(_hit_text_raw(h))])
    if not emails:
        return ""

    preferred = None
    if "ogs" in q or "global services" in q:
        for e in emails:
            if e.lower() == "global@illinoistech.edu":
                preferred = e
                break
        preferred = preferred or emails[0]
        return f"The OGS email is **{preferred}**."
    if "shwc" in q or "student health" in q:
        for e in emails:
            if "student.health" in e.lower():
                return f"The SHWC email is **{e}**."
    if len(emails) == 1:
        return f"The contact email listed in the policy excerpts is **{emails[0]}**."
    return "The policy excerpts list these contact emails:\n" + "\n".join(f"- {e}" for e in emails[:4])



def _consequence_answer(hits: List[Dict[str, Any]]) -> str:
    text = " ".join(_hit_text_raw(h) for h in hits)
    parts: List[str] = []
    if re.search(r"negatively impact immigration status", text, re.I):
        parts.append("Falling below full-time enrollment without proper approval may negatively impact immigration status.")
    if re.search(r"violation of status", text, re.I):
        parts.append("Failure to follow the reduced course load instructions may result in a violation of status.")
    if re.search(r"must submit a request.*isss portal|submit a request within the isss portal", text, re.I):
        parts.append("If a student will be below full-time and is eligible, they must submit the appropriate Reduced Course Load request in the ISSS Portal.")
    return " ".join(_dedupe_keep_order(parts)).strip()



def try_deterministic_answer(
    *,
    user_query: str,
    intent: str,
    hits: List[Dict[str, Any]],
) -> str:
    q = (user_query or "").strip()
    if not hits:
        return ""

    if _is_contact_query(q) or intent == "contact_info":
        ans = _contact_answer(q, hits)
        if ans:
            return ans

    if _is_rcl_reasons_query(q):
        ans = _rcl_reasons_answer(hits)
        if ans:
            return ans

    if _is_document_query(q):
        ans = _document_answer(q, hits)
        if ans:
            return ans

    if _is_timing_query(q) or intent == "timing":
        ans = _timing_answer(q, hits)
        if ans:
            return ans

    if _is_consequence_query(q) or intent == "consequences":
        ans = _consequence_answer(hits)
        if ans:
            return ans

    return ""



def build_context(user_query: str, intent: str, hits: List[Dict[str, Any]], max_chars: int = 7000) -> str:
    blocks: List[str] = []
    used = 0
    include_source = _needs_location_context(user_query, intent)

    for h in hits[:12]:
        s = _src(h)
        doc_title = (s.get("doc_title") or "").strip()
        section_path = (s.get("section_path") or "").strip()
        heading = (s.get("heading") or "").strip()
        section = (s.get("section_title") or "").strip()
        topic = (s.get("policy_topic") or "").strip()
        source_url = (s.get("source_url") or s.get("url") or "").strip()
        text = (s.get("text") or s.get("snippet") or "").strip()
        text = re.sub(r"\s+", " ", text)

        meta_parts = [topic, section, heading, section_path, doc_title]
        if include_source and source_url:
            meta_parts.append(source_url)
        meta = " | ".join([x for x in meta_parts if x])
        piece = f"[{meta}] {text}" if meta else text

        if len(piece) > 900:
            piece = piece[:900].rstrip() + "…"

        if used + len(piece) > max_chars:
            break

        blocks.append(piece)
        used += len(piece)

    return "\n".join(blocks).strip()



def synthesize_answer(
    *,
    user_query: str,
    intent: str,
    clarifying_questions: Optional[List[str]],
    hits: List[Dict[str, Any]],
) -> str:
    deterministic = try_deterministic_answer(user_query=user_query, intent=intent, hits=hits)
    if deterministic:
        return deterministic

    if not SETTINGS.use_llm_answer_synthesis:
        return ""
    if not SETTINGS.openai_api_key and not SETTINGS.azure_openai_api_key:
        return ""

    ctx = build_context(user_query, intent, hits)
    if not ctx:
        return ""

    has_clarifiers = bool(clarifying_questions)

    system = (
        "You are an IIT international student policy chatbot.\n"
        "STRICT RULES:\n"
        "1) Use ONLY the provided policy context.\n"
        "2) If the policy context supports an answer, answer directly and do NOT add any uncertainty disclaimer afterward.\n"
        "3) Use the fallback sentence ONLY when the context truly does not support the answer: "
        "\"I can’t confirm this from the provided IIT policy excerpts.\"\n"
        "4) Do NOT invent legal consequences, deadlines, fees, office hours, emails, or URLs.\n"
        "5) Do NOT include a Sources section.\n"
        "6) Do NOT include 'Keywords:' anywhere.\n"
        "7) Do NOT include separators like ---.\n"
        "8) For definition questions: give a short, direct explanation.\n"
        "9) For procedure questions: give only steps supported by the context.\n"
        "10) For compare questions: compare both items side by side using only supported facts. "
        "If one side has weaker evidence, do not say it is undefined unless the context truly lacks it; instead say what is supported and what is not explicitly stated.\n"
        "11) If clarifying questions are provided, summarize the relevant policy constraints briefly, then ask for the missing details. "
        "Do not make a final determination while required details are still missing.\n"
        "12) For questions asking where to view, download, access, or submit something, use the provided source page metadata or source URL when the context supports it.\n"
    )

    user = (
        f"User question: {user_query}\n"
        f"Intent: {intent}\n"
        f"Need clarification: {has_clarifiers}\n"
        f"Clarifying questions: {clarifying_questions or []}\n\n"
        f"Policy context:\n{ctx}\n\n"
        "Write the answer now."
    )

    return llm_text(system=system, user=user, temperature=0.2).strip()
