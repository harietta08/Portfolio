# backend/ambiguity.py
from __future__ import annotations
import re
from typing import List, Optional, Tuple


def _norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_HOURS_PAT = re.compile(r"\b(\d{1,2})\s*(hours|hrs)\b", re.I)

_BREAK_WORDS = {"summer", "winter", "break", "vacation", "holiday"}
_WORK_WORDS = {"work", "working", "employment", "job", "internship"}
_WORK_TYPE_WORDS = {"cpt", "opt", "stem opt", "on campus", "on-campus", "campus"}

_DEICTIC_REQUEST_PAT = re.compile(
    r"\b(where|which portal|which page|where can i)\b.*\b(this|that|it)\b.*\b(request|form|document)\b"
    r"|\bsubmit this request\b|\bsubmit this form\b|\bdownload this form\b|\bwhere do i submit this\b",
    re.I,
)


def work_hours_ambiguity(user_query: str, topic: Optional[str], topic_conf: float = 0.0) -> Tuple[bool, List[str]]:
    """
    Ask clarification when the question is about hours/work and doesn't specify
    which authorization type (on-campus vs CPT vs OPT).
    """
    q = _norm(user_query)

    is_workish = any(w in q for w in _WORK_WORDS)
    asks_hours = bool(_HOURS_PAT.search(user_query)) or ("full time" in q) or ("part time" in q) or ("20" in q) or ("40" in q)
    mentions_break = any(w in q for w in _BREAK_WORDS)

    if not (is_workish and (asks_hours or mentions_break)):
        return False, []

    # If user already clearly specified the type, don't ask
    if any(w in q for w in _WORK_TYPE_WORDS):
        return False, []

    # If topic is already strongly confident and specifically work-related, don't ask
    if topic in {"employment_on_campus", "employment_cpt", "employment_opt"} and topic_conf >= 0.75:
        return False, []

    qs = [
        "Do you mean **on-campus employment**, **CPT**, or **OPT**?",
        "Is this **during a semester** or during an **official break (summer/winter vacation)**?",
    ]
    return True, qs



def request_reference_ambiguity(user_query: str, intent: str) -> Tuple[bool, List[str]]:
    """Ask for clarification when a portal/request question uses an unresolved reference.

    Example: "Where do I submit this request?" should not be answered as a generic portal question
    because the correct location can vary by request type.
    """
    if intent != "portal_link":
        return False, []

    q = _norm(user_query)
    if not _DEICTIC_REQUEST_PAT.search(q):
        return False, []

    return True, [
        "Which request do you mean — CPT, OPT, STEM OPT, Reduced Course Load, Change of Status, I-20 update, or another request?"
    ]
