from __future__ import annotations
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # .../iit_chatbot
sys.path.insert(0, str(PROJECT_ROOT))


"""
Evaluation runner for the IIT International Student Chatbot.

Run:
  python -m eval.evaluate

What it does:
- Loads data/data/eval/recursive_evaluation_dataset.json
- Runs backend.orchestrator.chat_once(question)
- Applies lightweight, deterministic rule checks (no LLM) from the dataset
- Produces a summary + a JSON report under eval_reports/
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from backend.orchestrator import chat_once

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "data" / "eval" / "recursive_evaluation_dataset.json"
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
    # citations are rendered as "- <url/file> (Section: ...)" under "### Sources"
    return len(re.findall(r"^\s*-\s+.+\(\s*Section:\s*.+\)\s*$", answer_md, flags=re.M))

def _count_steps(answer_md: str) -> int:
    # Count bullet or numbered steps.
    bullets = len(re.findall(r"^\s*[-*]\s+\S+", answer_md, flags=re.M))
    numbered = len(re.findall(r"^\s*\d+\.[ \t]+\S+", answer_md, flags=re.M))
    return max(bullets, numbered)

def _num_question_marks(text: str) -> int:
    return text.count("?")

@dataclass
class CaseResult:
    id: str
    passed: bool
    reason: str
    predicted_intent: str
    predicted_mode: str
    citations: int

def evaluate_case(item: Dict[str, Any]) -> CaseResult:
    qid = item.get("id", "UNKNOWN")
    question = item.get("question", "")
    rules = (item.get("pass_fail_rules") or {})
    eval_type = rules.get("evaluation_type", "contains_keywords_and_no_contradiction")

    out = chat_once(question)
    ans = out.get("answer_markdown", "")

    must_any = rules.get("must_include_any", []) or []
    must_all = rules.get("must_include_all", []) or []
    must_not = rules.get("must_not_include", None)
    if must_not is None:
        must_not = DEFAULT_FORBIDDEN

    ok = True
    reasons: List[str] = []

    # Common keyword checks (still used by many evaluation types)
    if must_any and not _contains_any(ans, must_any):
        ok = False; reasons.append(f"missing any of: {must_any}")
    if must_all and not _contains_all(ans, must_all):
        ok = False; reasons.append(f"missing all of: {must_all}")
    if must_not and not _contains_none(ans, must_not):
        ok = False; reasons.append(f"contains forbidden: {must_not}")

    citations = _count_citations(ans)

    # Type-specific checks
    if eval_type == "requires_clarification":
        max_q = int(rules.get("max_questions", 3))
        qmarks = _num_question_marks(ans)
        if qmarks < 1:
            ok = False; reasons.append("no clarifying question found")
        if qmarks > max_q:
            ok = False; reasons.append(f"too many clarifying questions ({qmarks} > {max_q})")
        # still require citations for policy-related clarification
        if citations < 1:
            ok = False; reasons.append("no citations detected")

    elif eval_type == "procedure_steps":
        min_steps = int(rules.get("min_steps", 2))
        steps = _count_steps(ans)
        if steps < min_steps:
            ok = False; reasons.append(f"missing steps (found {steps}, need {min_steps})")
        if citations < 1:
            ok = False; reasons.append("no citations detected")

    elif eval_type == "eligibility_conditional":
        min_cond = int(rules.get("min_conditionals", 2))
        t = _norm(ans)
        # Count simple conditional markers
        markers = [" if ", " depends ", " only if ", " when ", " provided that ", " unless "]
        cond_hits = sum(1 for m in markers if m.strip() in t)
        if cond_hits < min_cond:
            ok = False; reasons.append(f"missing conditional framing (found {cond_hits}, need {min_cond})")
        # Encourage at least one clarifying question or explicit conditions
        if _num_question_marks(ans) < 1 and not _contains_any(ans, ["depends", "only if", "must", "require"]):
            ok = False; reasons.append("eligibility answer lacks clarification/conditions")
        if citations < 1:
            ok = False; reasons.append("no citations detected")

    elif eval_type == "must_answer_all_subquestions":
        groups = rules.get("subquestions_keywords", []) or []
        for i, group in enumerate(groups, start=1):
            if not _contains_any(ans, group):
                ok = False; reasons.append(f"missing subquestion {i} keywords: {group}")
        if citations < 1:
            ok = False; reasons.append("no citations detected")

    elif eval_type == "out_of_scope_refusal":
        refusal_kw = rules.get("refusal_keywords", []) or []
        redirect_kw = rules.get("redirect_keywords", []) or []
        if refusal_kw and not _contains_any(ans, refusal_kw):
            ok = False; reasons.append(f"missing refusal cue: {refusal_kw}")
        if redirect_kw and not _contains_any(ans, redirect_kw):
            ok = False; reasons.append(f"missing redirect cue: {redirect_kw}")
        # citations NOT required here

    else:
        # Default behavior: citations required
        if citations < 1:
            ok = False; reasons.append("no citations detected")

    reason = "PASS" if ok else "; ".join(reasons) or "failed rules"
    return CaseResult(
        id=qid,
        passed=ok,
        reason=reason,
        predicted_intent=out.get("intent", ""),
        predicted_mode=out.get("mode", ""),
        citations=citations,
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
            "expected_handler": item.get("expected_handler"),
            "difficulty": item.get("difficulty"),
            "category": item.get("category"),
            "question": item.get("question"),
        })

    total = len(results)
    failed = total - passed

    # Top failures
    failures = [r for r in results if not r["passed"]]
    failures = failures[:10]

    report = {
        "dataset_name": dataset.get("dataset_name"),
        "dataset_version": dataset.get("version"),
        "total": total,
        "passed": passed,
        "failed": failed,
        "top_failures": failures,
        "results": results,
    }

    out_path = REPORT_DIR / "latest_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"✅ Passed: {passed}/{total}  |  ❌ Failed: {failed}/{total}")
    if failed:
        print("\nTop failures:")
        for f in failures:
            print(f"- {f['id']} ({f['difficulty']}): {f['reason']}")
    print(f"\nReport written to: {out_path}")

if __name__ == "__main__":
    main()
