from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from .config import SETTINGS
from .llm import llm_json


DEFAULT_SLOTS: List[str] = [
    "degree_level",
    "is_f1_student",
    "is_full_time_enrolled",
    "credits",
    "term",
    "number_of_online_classes",
    "final_semester",
    "completed_one_academic_year",
    "has_job_offer",
    "job_related_to_major",
    "employer_everify",
    "cpt_full_time",
    "cpt_months_full_time",
    "opt_type",
    "rcl_reason",
    "dso_approved",
]

BOOL_TRUE = {"yes", "y", "true", "1", "yeah", "yep", "correct", "affirmative"}
BOOL_FALSE = {"no", "n", "false", "0", "nope", "negative"}

ENUM_MAPS = {
    "degree_level": {
        "undergraduate": "undergraduate",
        "undergrad": "undergraduate",
        "graduate": "graduate",
        "masters": "graduate",
        "master's": "graduate",
        "ms": "graduate",
        "phd": "phd",
        "doctorate": "phd",
    },
    "term": {
        "spring": "spring",
        "summer": "summer",
        "fall": "fall",
        "autumn": "fall",
        "winter": "winter",
    },
    "rcl_reason": {
        "medical": "medical",
        "final semester": "final_semester",
        "academic difficulty": "academic_difficulty",
        "academic": "academic_difficulty",
    },
    "opt_type": {
        "pre completion": "pre_completion",
        "pre-completion": "pre_completion",
        "pre_completion": "pre_completion",
        "pre": "pre_completion",
        "post completion": "post_completion",
        "post-completion": "post_completion",
        "post_completion": "post_completion",
        "post": "post_completion",
        "stem extension": "stem_extension",
        "stem_extension": "stem_extension",
        "stem opt": "stem_extension",
        "stem": "stem_extension",
    },
}

BOOL_SLOTS = {
    "is_f1_student",
    "is_full_time_enrolled",
    "final_semester",
    "completed_one_academic_year",
    "has_job_offer",
    "job_related_to_major",
    "employer_everify",
    "cpt_full_time",
    "dso_approved",
}

INT_SLOTS = {
    "credits",
    "number_of_online_classes",
    "cpt_months_full_time",
}

YES_TO_ALL_PATTERNS = [
    r"\byes to all\b",
    r"\ball yes\b",
    r"\beverything is yes\b",
]

NO_TO_ALL_PATTERNS = [
    r"\bno to all\b",
    r"\ball no\b",
    r"\beverything is no\b",
]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _norm_token(text: str) -> str:
    return _clean(text).lower().replace("-", "_").replace(" ", "_")


def _parse_bool(text: str) -> Optional[bool]:
    t = _clean(text).lower().strip(" .,!?:;")
    if not t:
        return None

    if t in BOOL_TRUE:
        return True
    if t in BOOL_FALSE:
        return False

    if re.search(r"\b(yes|yeah|yep|true|correct|affirmative)\b", t):
        return True
    if re.search(r"\b(no|nope|false|negative|not yet)\b", t):
        return False

    return None


def _parse_int(text: str) -> Optional[int]:
    m = re.search(r"\b(\d+)\b", text or "")
    return int(m.group(1)) if m else None


def _parse_enum(slot: str, text: str) -> Optional[str]:
    mapping = ENUM_MAPS.get(slot, {})
    raw = _clean(text)
    t_norm = _norm_token(raw)

    for _, v in mapping.items():
        if _norm_token(v) == t_norm:
            return v

    for k, v in mapping.items():
        k_norm = _norm_token(k)
        if t_norm == k_norm:
            return v
        if k.lower() in raw.lower():
            return v

    return None


def _parse_single_slot(slot: str, answer: str) -> Dict[str, Any]:
    if slot in BOOL_SLOTS:
        val = _parse_bool(answer)
        return {slot: val} if val is not None else {}

    if slot in INT_SLOTS:
        val = _parse_int(answer)
        return {slot: val} if val is not None else {}

    if slot in ENUM_MAPS:
        val = _parse_enum(slot, answer)
        return {slot: val} if val is not None else {}

    return {}


def _extract_numbered_segments(text: str) -> List[tuple[int, str]]:
    text = (text or "").strip()
    if not text:
        return []

    matches = list(re.finditer(r"(\d+)[\.\):\-]\s*", text))
    if not matches:
        return []

    parts: List[tuple[int, str]] = []
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[start:end].strip(" \n\t\r-:")
        parts.append((idx, value))
    return parts


def _parse_numbered_followup(user_query: str, candidate_slots: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    segments = _extract_numbered_segments(user_query)
    if not segments:
        return out

    for idx, value in segments:
        slot_idx = idx - 1
        if 0 <= slot_idx < len(candidate_slots):
            slot = candidate_slots[slot_idx]
            out.update(_parse_single_slot(slot, value))
    return out


def _parse_slot_value_lines(user_query: str, candidate_slots: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    allowed = set(candidate_slots)

    for line in (user_query or "").splitlines():
        m = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+?)\s*$", line)
        if not m:
            continue

        slot = m.group(1).strip()
        value = m.group(2).strip()
        if slot in allowed:
            out.update(_parse_single_slot(slot, value))

    return out


def _parse_yes_no_all(user_query: str, candidate_slots: List[str]) -> Dict[str, Any]:
    q = (user_query or "").lower()

    if any(re.search(p, q) for p in YES_TO_ALL_PATTERNS):
        return {slot: True for slot in candidate_slots if slot in BOOL_SLOTS}

    if any(re.search(p, q) for p in NO_TO_ALL_PATTERNS):
        return {slot: False for slot in candidate_slots if slot in BOOL_SLOTS}

    return {}


def _deterministic_slots(user_query: str) -> Dict[str, Any]:
    q = (user_query or "").lower()
    out: Dict[str, Any] = {}

    if re.search(r"\b(first|1st)\s+(semester|term|quarter)\b", q):
        out["completed_one_academic_year"] = False

    if re.search(r"\b(completed|finished)\s+(one|1)\s+(full\s+)?academic\s+year\b", q):
        out["completed_one_academic_year"] = True

    if re.search(r"\bf[- ]?1\b", q):
        out["is_f1_student"] = True

    if ("full-time" in q or "full time" in q) and "enrolled" in q:
        out["is_full_time_enrolled"] = True

    if "e-verify" in q or "everify" in q:
        out["employer_everify"] = True

    if "related to my major" in q or "related to my program" in q:
        out["job_related_to_major"] = True

    if "not related to my major" in q or "not related to my program" in q:
        out["job_related_to_major"] = False

    return out


def extract_slots(user_query: str, candidate_slots: Optional[List[str]] = None) -> Dict[str, Any]:
    slots = list(candidate_slots or DEFAULT_SLOTS)

    slot_lines = _parse_slot_value_lines(user_query, slots)
    if slot_lines:
        return slot_lines

    numbered = _parse_numbered_followup(user_query, slots)
    if numbered:
        return numbered

    bulk = _parse_yes_no_all(user_query, slots)
    if bulk:
        return bulk

    if len(slots) == 1:
        single = _parse_single_slot(slots[0], user_query)
        if single:
            return single

    deterministic = _deterministic_slots(user_query)
    filtered = {k: v for k, v in deterministic.items() if k in slots}
    if filtered:
        return filtered

    if not SETTINGS.use_llm_slots:
        return {}

    schema_hint = {slot: "value or null" for slot in slots}
    system = (
        "Extract slot values from the user's message for an international student policy chatbot. "
        "Only extract facts explicitly stated by the user. "
        "Return null for anything not stated. "
        "Use booleans for yes/no facts, integers for counts, and lowercase enums where applicable."
    )

    out = llm_json(system=system, user=user_query, schema_hint=schema_hint, temperature=0.0)
    if not isinstance(out, dict):
        return {}

    cleaned: Dict[str, Any] = {}
    for slot in slots:
        value = out.get(slot)
        if value is not None:
            cleaned[slot] = value
    return cleaned