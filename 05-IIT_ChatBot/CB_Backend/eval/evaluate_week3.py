from __future__ import annotations
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.orchestrator import chat_turn

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from backend.orchestrator import chat_turn  # Week-3 stateful

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "data" / "eval" / "recursive_evaluation_dataset.json"
SCENARIO_SPEC_PATH = ROOT / "scenario_spec.yaml"
REPORT_DIR = ROOT / "eval_reports"

DEFAULT_FORBIDDEN = ["i am not sure", "guess", "probably"]


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = _norm(text)
    return any(_norm(k) in t for k in keywords)


def _contains_all(text: str, keywords: List[str]) -> bool:
    t = _norm(text)
    return all(_norm(k) in t for k in keywords)


def _contains_none(text: str, keywords: List[str]) -> bool:
    t = _norm(text)
    return all(_norm(k) not in t for k in keywords)


def _count_citations(answer_md: str) -> int:
    return len(re.findall(r"^\s*-\s+.+\(\s*Section:\s*.+\)\s*$", answer_md, flags=re.M))


def _count_steps(answer_md: str) -> int:
    bullets = len(re.findall(r"^\s*[-*]\s+\S+", answer_md, flags=re.M))
    numbered = len(re.findall(r"^\s*\d+\.[ \t]+\S+", answer_md, flags=re.M))
    return max(bullets, numbered)


def _num_question_marks(text: str) -> int:
    return text.count("?")


def _load_max_clarifying_questions() -> int:
    if not SCENARIO_SPEC_PATH.exists():
        return 3
    spec = yaml.safe_load(SCENARIO_SPEC_PATH.read_text(encoding="utf-8")) or {}
    gp = spec.get("global_policies", {}) or {}
    return int(gp.get("max_clarifying_questions", 3))


MAX_CLARIFY = _load_max_clarifying_questions()


@dataclass
class CaseResult:
    id: str
    passed: bool
    reason: str
    predicted_mode: str
    predicted_intent: str
    citations: int
    decision: Optional[str]
    policy_id: Optional[str]


def _expected_mode(expected_handler: str) -> str:
    h = (expected_handler or "").strip().lower()
    if h in {"rule engine", "clarification required"}:
        return "rules"
    if h in {"bm25", "embeddings", "retrieval"}:
        return "retrieval"
    # Dataset uses "Out of Scope" as category, not handler, but we keep this for future
    if h in {"refuse", "out of scope"}:
        return "refuse"
    return "retrieval"


def evaluate_case(item: Dict[str, Any]) -> CaseResult:
    qid = item.get("id", "UNKNOWN")
    question = item.get("question", "")
    expected_handler = item.get("expected_handler", "BM25")
    rules = (item.get("pass_fail_rules") or {})
    eval_type = rules.get("evaluation_type", "contains_keywords_and_no_contradiction")

    # Week-3: stateful API, but single-turn dataset => new memory each case
    out = chat_turn(question, memory={})
    ans = out.get("answer_markdown", "")
    mode = out.get("mode", "")
    intent = out.get("intent", "")

    d = out.get("decision") or {}
    decision = d.get("decision") if isinstance(d, dict) else None
    policy_id = d.get("policy_id") if isinstance(d, dict) else None

    must_any = rules.get("must_include_any", []) or []
    must_all = rules.get("must_include_all", []) or []
    must_not = rules.get("must_not_include", None)
    if must_not is None:
        must_not = DEFAULT_FORBIDDEN

    ok = True
    reasons: List[str] = []

    # Handler/mode alignment (Week-3 specific)
    exp_mode = _expected_mode(expected_handler)
    if exp_mode and mode != exp_mode:
        ok = False
        reasons.append(f"mode mismatch: expected {exp_mode}, got {mode}")

    citations = _count_citations(ans)

    # Week-3 rule-engine expectations
    if exp_mode == "rules":
        if not isinstance(d, dict) or not decision:
            ok = False
            reasons.append("missing decision JSON in rules mode")

        # Clarification Required => must be depends + should ask questions
        if (expected_handler or "").strip().lower() == "clarification required":
            if decision != "depends":
                ok = False
                reasons.append(f"expected decision 'depends' for clarification, got {decision}")
            if _num_question_marks(ans) < 1:
                ok = False
                reasons.append("expected at least 1 clarifying question")
            if _num_question_marks(ans) > MAX_CLARIFY:
                ok = False
                reasons.append(f"too many clarifying questions (> {MAX_CLARIFY})")

    # Existing deterministic checks (reused from Week-2)
    if must_any and not _contains_any(ans, must_any):
        ok = False
        reasons.append(f"missing any of: {must_any}")
    if must_all and not _contains_all(ans, must_all):
        ok = False
        reasons.append(f"missing all of: {must_all}")
    if must_not and not _contains_none(ans, must_not):
        ok = False
        reasons.append(f"contains forbidden: {must_not}")

    # Type-specific checks (same idea as Week-2)
    if eval_type == "procedure_steps":
        min_steps = int(rules.get("min_steps", 2))
        steps = _count_steps(ans)
        if steps < min_steps:
            ok = False
            reasons.append(f"missing steps (found {steps}, need {min_steps})")
        if citations < 1:
            ok = False
            reasons.append("no citations detected")

    elif eval_type == "requires_clarification":
        max_q = int(rules.get("max_questions", MAX_CLARIFY))
        qmarks = _num_question_marks(ans)
        if qmarks < 1:
            ok = False
            reasons.append("no clarifying question found")
        if qmarks > max_q:
            ok = False
            reasons.append(f"too many clarifying questions ({qmarks} > {max_q})")
        if citations < 1:
            ok = False
            reasons.append("no citations detected")

    elif eval_type == "out_of_scope_refusal":
        refusal_kw = rules.get("refusal_keywords", []) or []
        redirect_kw = rules.get("redirect_keywords", []) or []
        if refusal_kw and not _contains_any(ans, refusal_kw):
            ok = False
            reasons.append(f"missing refusal cue: {refusal_kw}")
        if redirect_kw and not _contains_any(ans, redirect_kw):
            ok = False
            reasons.append(f"missing redirect cue: {redirect_kw}")
        # citations not required

    else:
        # default: citations required
        if citations < 1 and exp_mode != "refuse":
            ok = False
            reasons.append("no citations detected")

    reason = "PASS" if ok else "; ".join(reasons) or "failed rules"
    return CaseResult(
        id=qid,
        passed=ok,
        reason=reason,
        predicted_mode=mode,
        predicted_intent=intent,
        citations=citations,
        decision=decision,
        policy_id=policy_id,
    )


def main() -> None:
    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    items = dataset.get("items", [])

    REPORT_DIR.mkdir(exist_ok=True)

    results: List[Dict[str, Any]] = []
    passed = 0

    for item in items:
        r = evaluate_case(item)
        if r.passed:
            passed += 1
        results.append({
            "id": r.id,
            "passed": r.passed,
            "reason": r.reason,
            "predicted_intent": r.predicted_intent,
            "predicted_mode": r.predicted_mode,
            "citations": r.citations,
            "decision": r.decision,
            "policy_id": r.policy_id,
            "expected_handler": item.get("expected_handler"),
            "difficulty": item.get("difficulty"),
            "category": item.get("category"),
            "question": item.get("question"),
        })

    total = len(results)
    failed = total - passed
    failures = [r for r in results if not r["passed"]][:20]

    report = {
        "dataset_name": dataset.get("dataset_name"),
        "dataset_version": dataset.get("version"),
        "total": total,
        "passed": passed,
        "failed": failed,
        "top_failures": failures,
        "results": results,
    }

    out_path = REPORT_DIR / "week3_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"✅ Passed: {passed}/{total}  |  ❌ Failed: {failed}/{total}")
    if failed:
        print("\nTop failures:")
        for f in failures[:10]:
            print(f"- {f['id']} ({f['difficulty']}): {f['reason']}")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()