import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# --- Make project imports work no matter where we run from ---
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

PKG_ROOT = PROJECT_ROOT / "iit_chatbot"
if PKG_ROOT.exists():
    sys.path.insert(0, str(PKG_ROOT))

from backend.orchestrator import chat_turn  # noqa: E402


REPORT_DIR = PROJECT_ROOT / "eval_reports"
CONV_PATH = HERE / "conversations.yaml"


def _get_decision_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    d = result.get("decision")
    if not isinstance(d, dict):
        return {"decision": None, "policy_id": None, "missing_slots": []}
    return {
        "decision": d.get("decision"),
        "policy_id": d.get("policy_id"),
        "missing_slots": d.get("missing_slots") or [],
    }


def _assert_equal(name: str, expected: Any, actual: Any, errors: List[str]):
    if expected is not None and actual != expected:
        errors.append(f"{name} expected {expected!r}, got {actual!r}")


def _assert_not_equal(name: str, not_expected: Any, actual: Any, errors: List[str]):
    if not_expected is not None and actual == not_expected:
        errors.append(f"{name} should NOT be {not_expected!r}")


def _assert_contains_all(name: str, expected_list: List[str], actual_list: List[str], errors: List[str]):
    for x in expected_list or []:
        if x not in actual_list:
            errors.append(f"{name} should contain {x!r} (actual={actual_list})")


def _assert_contains_any(name: str, expected_list: List[str], actual_list: List[str], errors: List[str]):
    if expected_list and not any(x in actual_list for x in expected_list):
        errors.append(f"{name} should contain any of {expected_list} (actual={actual_list})")


def _assert_not_contains_any(name: str, forbidden_list: List[str], actual_list: List[str], errors: List[str]):
    for x in forbidden_list or []:
        if x in actual_list:
            errors.append(f"{name} should NOT contain {x!r} (actual={actual_list})")


def check_expect(expect: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    mode = result.get("mode")
    topic = result.get("topic")
    intent = result.get("intent")
    dec = _get_decision_fields(result)

    _assert_equal("mode", expect.get("mode"), mode, errors)
    _assert_equal("topic", expect.get("topic"), topic, errors)
    _assert_equal("intent", expect.get("intent"), intent, errors)

    _assert_equal("decision", expect.get("decision"), dec["decision"], errors)
    _assert_equal("policy_id", expect.get("policy_id"), dec["policy_id"], errors)
    _assert_not_equal("policy_id", expect.get("policy_id_not"), dec["policy_id"], errors)

    if expect.get("missing_slots_len") is not None:
        _assert_equal("missing_slots_len", expect.get("missing_slots_len"), len(dec["missing_slots"]), errors)

    _assert_contains_all("missing_slots", expect.get("missing_slots_contains_all") or [], dec["missing_slots"], errors)
    _assert_contains_any("missing_slots", expect.get("missing_slots_contains_any") or [], dec["missing_slots"], errors)
    _assert_not_contains_any("missing_slots", expect.get("missing_slots_not_contains_any") or [], dec["missing_slots"], errors)

    return errors


def run_conversation(conv: Dict[str, Any]) -> Dict[str, Any]:
    conv_id = conv.get("id", "UNKNOWN")
    turns = conv.get("turns", [])

    memory: Dict[str, Any] = {}
    turn_logs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, t in enumerate(turns, start=1):
        user_msg = t.get("user", "")
        expect = t.get("expect", {})

        result = chat_turn(user_msg, memory=memory)

        # persist memory
        if result.get("memory") is not None:
            memory = result["memory"]

        dec = _get_decision_fields(result)

        errors = check_expect(expect, result)
        turn_logs.append({
            "turn": i,
            "user": user_msg,
            "mode": result.get("mode"),
            "topic": result.get("topic"),
            "intent": result.get("intent"),
            "decision": dec["decision"],
            "policy_id": dec["policy_id"],
            "missing_slots": dec["missing_slots"],
            "errors": errors,
        })

        if errors:
            failures.append({
                "turn": i,
                "user": user_msg,
                "errors": errors,
                "observed": turn_logs[-1],
            })

    return {
        "id": conv_id,
        "passed": len(failures) == 0,
        "failures": failures,
        "turns": turn_logs,
    }


def main():
    if not CONV_PATH.exists():
        raise SystemExit(f"Missing: {CONV_PATH} (create eval/conversations.yaml)")

    data = yaml.safe_load(CONV_PATH.read_text(encoding="utf-8")) or {}
    conversations = data.get("conversations", [])

    REPORT_DIR.mkdir(exist_ok=True)

    results = [run_conversation(c) for c in conversations]
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print("\n=== Conversation Evaluation ===")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}\n")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']}")
        if not r["passed"]:
            for f in r["failures"]:
                print(f"  Turn {f['turn']}: {f['user']}")
                for e in f["errors"]:
                    print(f"    - {e}")

    out_path = REPORT_DIR / "conversations_report.json"
    out_path.write_text(json.dumps({"passed": passed, "total": total, "results": results}, indent=2), encoding="utf-8")

    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()