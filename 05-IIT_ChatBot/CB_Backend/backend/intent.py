from __future__ import annotations
import re


def _norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s\-\?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_SMALLTALK = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "whats up", "what's up"
}

_HELP_PAT = re.compile(
    r"\b(what can you help with|what do you do|what can you do|help me|how can you help)\b",
    re.I
)

# Keep contact-specific terms narrow so policy questions with "hours" are not misrouted.
_CONTACT_PAT = re.compile(
    r"\b(contact|contact info|contact information|office hours|walk-?in|phone number|email address|where is the office|office location|office address)\b"
    r"|\b(ogs|office of global services|shwc|student health and wellness center|seo)\b.*\bemail\b"
    r"|\bemail\b.*\b(ogs|office of global services|shwc|student health and wellness center|seo)\b",
    re.I,
)

# Trigger only when the user is actually asking for a portal / link / login.
_PORTAL_PAT = re.compile(
    r"\b(sevp portal|portal link|login portal|portal login|where do i report|which portal|what portal|link to the portal|portal should i use|where do i submit|where can i submit|where can i download|where do i download|which page|which website)\b",
    re.I,
)

_CONSEQ_PAT = re.compile(
    r"\b(consequence|consequences|what happens if|penalty|penalties|violation|unauthorized|fail to|without informing|without approval|impact my status|affect my status)\b",
    re.I,
)

_AUTH_TIMING_PAT = re.compile(
    r"\b(before|prior to)\b.*\b(approved|authorized|authorization|start date|start)\b",
    re.I
)

_COMPARE_PAT = re.compile(
    r"\b(compare|comparison|difference|different|vs\.?|versus)\b",
    re.I,
)

_TIMING_PAT = re.compile(
    r"^(how soon|how long|when should|when do|when can|within how many|earliest|latest)\b"
    r"|\b(processing time|turnaround time|typically take|processed before|before my program deadline|before .* deadline)\b",
    re.I,
)

_REASONS_PAT = re.compile(
    r"\b(acceptable reasons|what reasons|reasons for rcl|grounds for rcl|reasons for reduced course load|acceptable reasons for reduced course load)\b",
    re.I,
)


def detect_intent(user_query: str) -> str:
    q_raw = user_query or ""
    q = _norm(q_raw)

    if q in _SMALLTALK or q.startswith(("hi ", "hello ", "hey ")):
        return "smalltalk"
    if _HELP_PAT.search(q_raw):
        return "help"

    # Compare early to avoid weaker downstream matches.
    if _COMPARE_PAT.search(q_raw):
        return "compare"

    if _PORTAL_PAT.search(q_raw) or re.search(r"\bwhere\b.*\b(view|download|access|submit|find)\b", q_raw, re.I):
        return "portal_link"
    if _CONTACT_PAT.search(q_raw):
        return "contact_info"

    if _CONSEQ_PAT.search(q_raw):
        return "consequences"

    if _AUTH_TIMING_PAT.search(q_raw) and ("cpt" in q or "opt" in q or "ead" in q):
        return "authorization_timing"

    if ("work" in q or "working" in q or "employment" in q or "job" in q) and (
        "hours" in q or "hrs" in q or "full time" in q or "part time" in q or re.search(r"\b\d{1,2}\b", q)
    ):
        return "work_hours"

    if q.startswith("what is") or q.startswith("define") or "meaning of" in q or "stand for" in q or _REASONS_PAT.search(q_raw):
        return "definition"

    # Timing must be checked before procedure so questions like
    # "How long...", "How soon...", or "When should I submit..." are not misrouted.
    if _TIMING_PAT.search(q_raw):
        return "timing"

    # Procedure / advisory questions
    if any(p in q for p in [
        "how do i", "how to", "steps", "process", "apply", "application", "documents",
        "what documents", "what form", "what forms", "submit", "request", "report my employment",
        "what should i do next", "need to complete", "report this", "what do i do next",
        "schedule an appointment", "book an appointment"
    ]):
        return "procedure"

    # Eligibility / validity / requirements satisfaction
    if (
        q.startswith("can i")
        or "eligible" in q
        or "allowed" in q
        or "do i qualify" in q
        or "can they work" in q
        or "does this meet" in q
        or "meet the" in q and "requirement" in q
        or "is the application still valid" in q
        or re.match(r"^a student\b", q)
    ):
        return "eligibility"

    if any(t in q for t in ["when", "timeline", "deadline", "earliest", "latest", "how many days", "within how many days", "60 days", "how long", "turnaround time"]):
        return "timing"

    return "general"
