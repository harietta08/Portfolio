# backend/work_hours.py
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

def _all_text(hits: List[Dict[str, Any]]) -> str:
    blob = ""
    for h in hits[:8]:
        s = h.get("_source") or {}
        t = (s.get("text") or s.get("snippet") or "")
        if t:
            blob += "\n" + t
    return blob.lower()

def extract_on_campus_hours_rules(hits: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns:
      (during_term_rule, break_rule)
    """
    txt = _all_text(hits)

    # common forms
    term_rule = None
    break_rule = None

    # find 20 hours/week rule
    if re.search(r"\b20\b\s*(hours|hrs)\s*(per|/)\s*week", txt) or "20 hours per week" in txt:
        term_rule = "During the academic term, on-campus employment is typically limited to **20 hours per week**."

    # find vacation / break full-time allowance
    if ("official school vacation" in txt) or ("official vacation" in txt) or ("during breaks" in txt) or ("winter break" in txt) or ("summer break" in txt):
        if "full-time" in txt or "more than 20" in txt:
            break_rule = "During **official school breaks/vacation periods**, on-campus employment may be **full-time (more than 20 hours/week)**."

    return term_rule, break_rule

def build_on_campus_hours_answer(user_query: str, hits: List[Dict[str, Any]]) -> Optional[str]:
    term_rule, break_rule = extract_on_campus_hours_rules(hits)
    if not term_rule and not break_rule:
        return None

    ql = (user_query or "").lower()
    mentions_break = any(w in ql for w in ["summer", "winter", "break", "vacation", "holiday"])
    mentions_term = any(w in ql for w in ["semester", "term", "during classes", "fall", "spring"])

    parts = []
    if mentions_break and break_rule:
        parts.append(break_rule)
    if mentions_term and term_rule:
        parts.append(term_rule)

    # If user didn't specify, provide both succinctly
    if not parts:
        if term_rule:
            parts.append(term_rule)
        if break_rule:
            parts.append(break_rule)

    parts.append("If you want, tell me whether you’re asking about **summer** while enrolled or an **official vacation period**, and I’ll tailor it precisely.")
    return "\n\n".join(parts).strip()