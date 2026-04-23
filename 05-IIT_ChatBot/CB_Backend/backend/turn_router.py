from __future__ import annotations

from typing import List
import re

from .config import SETTINGS
from .llm import llm_json


def _looks_like_followup(q: str) -> bool:
    q = (q or "").strip().lower()

    if re.search(r"\b1[\.\):\-]\s*", q):
        return True

    if re.search(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*.+$", q, flags=re.M):
        return True

    if q in {"yes", "no", "y", "n"}:
        return True

    if "yes to all" in q or "no to all" in q:
        return True

    return False


def classify_turn_type(user_query: str, pending: bool, missing: List[str]) -> str:
    q = (user_query or "").strip().lower()

    if pending and _looks_like_followup(q):
        return "followup_answer"

    if not SETTINGS.use_llm_intent:
        is_questionish = ("?" in q) or any(
            x in q for x in ["can i", "am i", "what", "how", "is it", "allowed", "eligible", "do i"]
        )
        if pending and not is_questionish:
            return "followup_answer"
        return "new_question" if is_questionish else "other"

    schema = {"turn_type": "followup_answer|new_question|other"}

    system = (
        "You are a conversation router for an IIT international student policy chatbot.\n"
        "Classify the user message as one of:\n"
        "- followup_answer\n"
        "- new_question\n"
        "- other\n\n"
        "If pending eligibility is true and the user is supplying answers to missing slots, return followup_answer.\n"
        "Return JSON only."
    )

    user = (
        f"Pending eligibility: {pending}\n"
        f"Missing slots: {missing}\n"
        f"User message: {user_query}\n"
    )

    out = llm_json(system=system, user=user, schema_hint=schema, temperature=0.0)
    tt = (out.get("turn_type") if isinstance(out, dict) else None) or ""
    tt = str(tt).strip()

    if tt in {"followup_answer", "new_question", "other"}:
        return tt
    return "other"