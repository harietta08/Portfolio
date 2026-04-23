from __future__ import annotations

"""Week 3: session memory for slot-filling.

Goals:
- Merge non-null values across turns.
- Normalize slot aliases so policies and LLM extraction stay compatible.
"""

from typing import Any, Dict


_SLOT_ALIASES = {
    # Common alias -> canonical
    "full_time_enrollment": "is_full_time_enrolled",
    "full_time": "is_full_time_enrolled",
    "is_full_time_enrolled": "is_full_time_enrolled",
}


def _normalize_slots(new_slots: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (new_slots or {}).items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        canon = _SLOT_ALIASES.get(k, k)
        out[canon] = v
    return out


def merge_slots(memory: Dict[str, Any], new_slots: Dict[str, Any]) -> Dict[str, Any]:
    """Merge new slots into memory (non-null values win) with alias normalization."""
    memory = dict(memory or {})
    normalized = _normalize_slots(new_slots)
    for k, v in normalized.items():
        memory[k] = v
    return memory


def reset_memory() -> Dict[str, Any]:
    return {}
