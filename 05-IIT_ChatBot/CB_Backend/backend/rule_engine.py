from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import yaml

from .config import SETTINGS

Decision = Dict[str, Any]


def _load_rules() -> List[Dict[str, Any]]:
    path = SETTINGS.policy_rules_path
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    pols = data.get("policies") or []
    return pols if isinstance(pols, list) else []


def get_policy_by_id(policy_id: str) -> Optional[Dict[str, Any]]:
    for pol in _load_rules():
        if pol.get("id") == policy_id:
            return pol
    return None


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _op_eval(op: str, actual: Any, expected: Any) -> bool:
    op = (op or "").lower()

    if op == "exists":
        return not _is_missing(actual)
    if op == "equals":
        return actual == expected
    if op == "in":
        return actual in (expected or [])
    if op == "not_in":
        return actual not in (expected or [])
    if op == "gte":
        try:
            return float(actual) >= float(expected)
        except Exception:
            return False
    if op == "lte":
        try:
            return float(actual) <= float(expected)
        except Exception:
            return False
    if op == "truthy":
        return bool(actual) is True
    if op == "falsy":
        return bool(actual) is False

    return False


def _policy_matches(policy: Dict[str, Any], *, topic: Optional[str], intent: str) -> bool:
    intents = policy.get("intents") or []
    topics = policy.get("topics") or []

    if intents and intent not in intents:
        return False
    if topic is not None and topics and topic not in topics:
        return False
    return True


def _resolve_clarifying_question(policy: Dict[str, Any], slot_name: str) -> Optional[str]:
    clarifying = policy.get("clarifying_questions") or {}
    if not isinstance(clarifying, dict):
        return None

    # direct slot key first
    if slot_name in clarifying:
        return str(clarifying[slot_name]).strip()

    # fallback: match condition name for that slot
    for cond in policy.get("conditions") or []:
        if cond.get("slot") == slot_name:
            cond_name = cond.get("name")
            if cond_name in clarifying:
                return str(clarifying[cond_name]).strip()

    return None


def build_clarifying_questions(policy: Dict[str, Any], missing_slots: List[str]) -> List[str]:
    out: List[str] = []
    for slot in missing_slots:
        q = _resolve_clarifying_question(policy, slot)
        if q:
            out.append(q)
    return out


def _evaluate_policy(policy: Dict[str, Any], *, topic: Optional[str], intent: str, slots: Dict[str, Any]) -> Optional[Decision]:
    if not _policy_matches(policy, topic=topic, intent=intent):
        return None

    applies = policy.get("applies_when") or []
    for c in applies:
        slot = c.get("slot")
        op = c.get("op")
        expected = c.get("value")
        actual = slots.get(slot)

        if _is_missing(actual):
            return None
        if not _op_eval(op, actual, expected):
            return None

    met: List[str] = []
    failed: List[str] = []
    missing_slots: List[str] = []

    for c in policy.get("conditions") or []:
        name = c.get("name") or c.get("slot") or "condition"
        slot = c.get("slot")
        op = c.get("op")
        expected = c.get("value")
        actual = slots.get(slot)

        if _is_missing(actual):
            missing_slots.append(slot)
            continue

        if _op_eval(op, actual, expected):
            met.append(name)
        else:
            failed.append(name)

    decision_mode = (policy.get("decision_mode") or "allow_if_all").lower()
    fixed_decision = (policy.get("decision") or "").strip().lower()

    if fixed_decision in {"allowed", "not_allowed"}:
        decision = fixed_decision
        missing_slots = []
    elif decision_mode in {"allow_if_all", "deny_on_fail"}:
        if failed:
            decision = "not_allowed"
            missing_slots = []
        elif missing_slots:
            decision = "depends"
        else:
            decision = "allowed"
    else:
        if missing_slots:
            decision = "depends"
        elif failed:
            decision = "not_allowed"
        else:
            decision = "allowed"

    pol_topics = policy.get("topics") or []
    pol_topic = pol_topics[0] if pol_topics else topic

    return {
        "decision": decision,
        "policy_id": policy.get("id"),
        "topic": pol_topic,
        "conditions_met": met,
        "conditions_failed": failed,
        "missing_slots": list(dict.fromkeys([s for s in missing_slots if s])),
        "next_steps": list(policy.get("next_steps") or []),
        "clarifying_questions": build_clarifying_questions(policy, missing_slots),
    }


def evaluate_policy_by_id(policy_id: str, *, intent: str, slots: Dict[str, Any]) -> Decision:
    policy = get_policy_by_id(policy_id)
    if not policy:
        return {
            "decision": "unknown",
            "policy_id": None,
            "topic": None,
            "conditions_met": [],
            "conditions_failed": [],
            "missing_slots": [],
            "next_steps": [],
            "clarifying_questions": [],
        }

    out = _evaluate_policy(policy, topic=None, intent=intent, slots=slots)
    if out is None:
        return {
            "decision": "unknown",
            "policy_id": policy_id,
            "topic": None,
            "conditions_met": [],
            "conditions_failed": [],
            "missing_slots": [],
            "next_steps": [],
            "clarifying_questions": [],
        }
    return out


def evaluate_best(*, intent: str, slots: Dict[str, Any], topic_hint: Optional[str] = None) -> Decision:
    policies = _load_rules()
    candidates: List[Tuple[Tuple[int, int, int, int, int], Decision]] = []

    for idx, policy in enumerate(policies):
        out = _evaluate_policy(policy, topic=topic_hint, intent=intent, slots=slots)
        if out is None:
            continue

        decision = out.get("decision")
        missing_n = len(out.get("missing_slots") or [])
        failed_n = len(out.get("conditions_failed") or [])
        met_n = len(out.get("conditions_met") or [])

        # prefer:
        # 1) final decisions over depends
        # 2) fewer missing
        # 3) more met
        # 4) fewer failed
        final_rank = 0 if decision in {"allowed", "not_allowed"} else 1
        score = (final_rank, missing_n, -met_n, failed_n, idx)
        candidates.append((score, out))

    if not candidates and topic_hint is not None:
        # fallback: if topic hint was wrong, evaluate across all policies
        for idx, policy in enumerate(policies):
            out = _evaluate_policy(policy, topic=None, intent=intent, slots=slots)
            if out is None:
                continue
            decision = out.get("decision")
            missing_n = len(out.get("missing_slots") or [])
            failed_n = len(out.get("conditions_failed") or [])
            met_n = len(out.get("conditions_met") or [])
            final_rank = 0 if decision in {"allowed", "not_allowed"} else 1
            score = (final_rank, missing_n, -met_n, failed_n, idx)
            candidates.append((score, out))

    if not candidates:
        return {
            "decision": "unknown",
            "policy_id": None,
            "topic": topic_hint,
            "conditions_met": [],
            "conditions_failed": [],
            "missing_slots": [],
            "next_steps": [],
            "clarifying_questions": [],
        }

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]