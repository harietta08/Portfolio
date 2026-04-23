from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
import re

from .topic import detect_topic, top_topic_candidates
from .intent import detect_intent
from .ambiguity import work_hours_ambiguity, request_reference_ambiguity
from .retrieval import hybrid_retrieve
from .answer_synth import synthesize_answer
from .answer_compose import compose_answer
from .answer_postprocess import postprocess_answer
from .turn_router import classify_turn_type
from .slot_filling import extract_slots
from .session_memory import merge_slots
from .rule_engine import evaluate_best, evaluate_policy_by_id
from .work_hours import build_on_campus_hours_answer


_OOS: List[Tuple[str, str]] = [
    (r"\bsop\b|\bstatement of purpose\b", "SOP"),
    (r"\bcheat\b|\bexam\b.*\banswers\b|\bplagiar", "CHEATING"),
    (r"\bfake\b.*\b(employment|offer)\b|\bfalsif", "FRAUD"),
    (r"\bhack\b|\bexploit\b", "HACKING"),
    (r"\bbest restaurants?\b|\brestaurants?\b.*\bnear\b", "RESTAURANTS"),
    (r"\beasiest professor\b|\bwhich professor\b", "PROFESSOR"),
]


def detect_out_of_scope(q: str) -> Optional[str]:
    ql = (q or "").lower()
    for pat, tag in _OOS:
        if re.search(pat, ql):
            return tag
    return None


def build_out_of_scope_answer(tag: str) -> str:
    if tag in {"CHEATING", "FRAUD", "HACKING"}:
        return "I can’t help with that."
    return "I can’t help with that request. Please use the official IIT website or contact the relevant office."


def _sources_tail(hits: List[Dict[str, Any]], max_urls: int = 5) -> str:
    seen = set()
    urls: List[str] = []
    for h in hits or []:
        s = (h.get("_source") or {})
        url = (s.get("source_url") or s.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= max_urls:
            break
    if not urls:
        return ""
    out = "\n\n### Sources\n"
    for u in urls:
        out += f"- {u}\n"
    return out.rstrip()


def _clarify_topic_prompt(cands: List[Tuple[str, float]]) -> str:
    if not cands:
        return "I can help — what policy area is your question about (CPT, OPT, STEM OPT, on-campus work, travel, RCL, maintaining status, insurance, SSN)?"
    msg = "I can help, but I need one clarification. Which area is your question about?\n\n"
    for i, (t, _) in enumerate(cands[:4], 1):
        msg += f"{i}. {t}\n"
    return msg.strip()


def _should_filter_by_topic(topic: Optional[str], topic_conf: float) -> bool:
    return bool(topic) and float(topic_conf or 0.0) >= 0.7


def _looks_like_fresh_question(user_query: str) -> bool:
    q = (user_query or "").strip()
    ql = q.lower()

    if not q:
        return False
    if ql in {"yes", "no", "y", "n"}:
        return False
    if len(ql.split()) < 4:
        return False

    return bool(
        "?" in q
        or re.match(r"^(how|what|when|where|which|who|can|is|are|do|does|did|if|a student)\b", ql)
    )


def _adaptive_retrieval_k(user_query: str, base_k: int) -> int:
    q = (user_query or "").lower()
    if any(x in q for x in [
        "what specific documents", "what documents", "must submit", "required documents",
        "acceptable reasons", "reasons for rcl", "reasons for reduced course load",
        "how soon", "how long", "when should", "processing time", "program deadline"
    ]):
        return max(base_k, 12)
    return base_k


def _help_answer() -> str:
    return (
        "I can help with IIT international student policy questions based on the documents in my dataset, including:\n\n"
        "- CPT / OPT / STEM OPT\n"
        "- On-campus employment hours (semester vs breaks)\n"
        "- Travel signatures and re-entry documents\n"
        "- Reduced Course Load (RCL) and dropping below full-time\n"
        "- Maintaining F-1 status (SEVIS, full-time enrollment)\n"
        "- Common forms and request processes (I-20 related requests)\n\n"
        "Ask me a question like: **“How do I apply for CPT?”** or **“Can I work 40 hours in summer?”**"
    )


def _merge_hits(h1: List[Dict[str, Any]], h2: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    seen = set()
    merged: List[Dict[str, Any]] = []
    for h in (h1 or []) + (h2 or []):
        hid = h.get("_id") or id(h)
        if hid in seen:
            continue
        seen.add(hid)
        merged.append(h)
        if len(merged) >= k:
            break
    return merged


def _is_personal_eligibility_query(user_query: str) -> bool:
    q = (user_query or "").strip().lower()

    personal_patterns = [
        r"^am i\b",
        r"^can i\b",
        r"^do i qualify\b",
        r"^would i be eligible\b",
        r"^am i eligible\b",
        r"^can i apply\b",
        r"^can i drop\b",
        r"^can i work\b",
    ]

    informational_patterns = [
        r"^who is eligible\b",
        r"^what is\b",
        r"^what are\b",
        r"^what documents\b",
        r"^how do i\b",
        r"^how to\b",
        r"\bcriteria\b",
        r"\brequirements\b",
        r"\beligibility criteria\b",
    ]

    if any(re.search(p, q) for p in informational_patterns):
        return False

    return any(re.search(p, q) for p in personal_patterns)


def _slot_kind(slot: str) -> str:
    if slot in {
        "is_f1_student",
        "is_full_time_enrolled",
        "final_semester",
        "completed_one_academic_year",
        "has_job_offer",
        "job_related_to_major",
        "employer_everify",
        "cpt_full_time",
        "dso_approved",
    }:
        return "boolean"

    if slot in {"credits", "number_of_online_classes", "cpt_months_full_time"}:
        return "number"

    if slot in {"degree_level", "term", "rcl_reason", "opt_type"}:
        return "select"

    return "text"


def _slot_options(slot: str) -> Optional[List[Dict[str, str]]]:
    if slot == "degree_level":
        return [
            {"label": "Undergraduate", "value": "undergraduate"},
            {"label": "Graduate", "value": "graduate"},
            {"label": "PhD", "value": "phd"},
        ]
    if slot == "term":
        return [
            {"label": "Spring", "value": "spring"},
            {"label": "Summer", "value": "summer"},
            {"label": "Fall", "value": "fall"},
            {"label": "Winter", "value": "winter"},
        ]
    if slot == "rcl_reason":
        return [
            {"label": "Medical", "value": "medical"},
            {"label": "Final semester", "value": "final_semester"},
            {"label": "Academic difficulty", "value": "academic_difficulty"},
        ]
    if slot == "opt_type":
        return [
            {"label": "Pre-completion OPT", "value": "pre_completion"},
            {"label": "Post-completion OPT", "value": "post_completion"},
            {"label": "STEM extension", "value": "stem_extension"},
        ]
    return None


def _build_slot_form_interaction(decision: Dict[str, Any]) -> Dict[str, Any]:
    missing_slots = list(decision.get("missing_slots") or [])
    clarifying_questions = decision.get("clarifying_questions") or []

    fields: List[Dict[str, Any]] = []
    for idx, slot in enumerate(missing_slots):
        label = clarifying_questions[idx] if idx < len(clarifying_questions) else slot.replace("_", " ").capitalize()
        field: Dict[str, Any] = {
            "slot": slot,
            "label": label,
            "kind": _slot_kind(slot),
            "required": True,
            "placeholder": "",
        }
        options = _slot_options(slot)
        if options:
            field["options"] = options
        fields.append(field)

    return {
        "type": "slot_form",
        "policy_id": decision.get("policy_id"),
        "submit_label": "Continue",
        "fields": fields,
    }


def _render_clarifying_prompt() -> str:
    return "To determine this accurately, I need a few details from you."


def _render_rule_decision(decision: Dict[str, Any], policy_context: str = "") -> str:
    status = decision.get("decision")
    failed = decision.get("conditions_failed") or []

    if status == "allowed":
        text = "Based on the information you provided, you appear eligible under the policy requirements in this chatbot."
    elif status == "not_allowed":
        text = "Based on the information you provided, you do not currently appear eligible under the policy requirements in this chatbot."
    else:
        text = "I still need more information to determine this."

    if policy_context:
        text += "\n\n" + policy_context

    if failed:
        text += "\n\nReason:"
        for item in failed:
            text += f"\n- {item}"

    return text.strip()


def _policy_focus_terms(decision: Dict[str, Any], topic: Optional[str]) -> List[str]:
    terms: List[str] = []
    t = (topic or decision.get("topic") or "").lower()

    slot_map = {
        "has_job_offer": ["job offer", "internship offer"],
        "job_related_to_major": ["related to major", "field of study"],
        "completed_one_academic_year": ["one academic year", "lawful full-time"],
        "is_full_time_enrolled": ["full-time enrollment", "full-time"],
        "dso_approved": ["prior approval", "DSO approval", "before dropping"],
        "rcl_reason": ["reduced course load", "medical", "final semester", "academic difficulty"],
        "number_of_online_classes": ["online course", "distance learning", "one online class"],
        "employer_everify": ["E-Verify", "employer enrolled in E-Verify"],
        "is_f1_student": ["F-1 status", "maintaining valid F-1 status"],
    }

    for slot in (
        list(decision.get("missing_slots") or [])
        + list(decision.get("conditions_met") or [])
        + list(decision.get("conditions_failed") or [])
    ):
        vals = slot_map.get(slot, [])
        if isinstance(vals, list):
            terms.extend(vals)

    if "employment_cpt" in t:
        terms.extend(["CPT eligibility", "job offer", "related to major", "one academic year", "full-time"])
    elif "employment_opt" in t:
        terms.extend(["OPT eligibility", "F-1 status", "full-time enrollment"])
    elif "reduced_course_load" in t:
        terms.extend(["reduced course load", "prior approval", "final semester", "medical", "academic difficulty"])
    elif "enrollment_requirements" in t:
        terms.extend(["full-time enrollment", "online course limit", "distance learning"])
    elif "ssn" in t:
        terms.extend(["SSN eligibility", "employment authorization", "job offer"])
    elif "travel" in t:
        terms.extend(["travel signature", "passport", "visa", "I-20", "re-entry"])

    seen = set()
    out: List[str] = []
    for x in terms:
        k = x.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _eligibility_context_hits(
    user_query: str,
    topic: Optional[str],
    decision: Optional[Dict[str, Any]],
    *,
    k: int,
) -> List[Dict[str, Any]]:
    base_hits = _retrieve_with_topic_expansion(
        user_query,
        topic=topic if _should_filter_by_topic(topic, 0.95) else None,
        intent="eligibility",
        k=k,
    )

    focus_terms = _policy_focus_terms(decision or {}, topic)
    if not focus_terms:
        return base_hits

    expanded_query = (user_query + " " + " ".join(focus_terms)).strip()
    focus_hits = _retrieve_with_topic_expansion(
        expanded_query,
        topic=topic if _should_filter_by_topic(topic, 0.95) else None,
        intent="eligibility",
        k=k,
    )

    return _merge_hits(focus_hits, base_hits, k=k)


def _looks_like_noise(text: str) -> bool:
    tl = text.lower()
    bad_patterns = [
        "deadline",
        "last checked",
        "source url",
        "doc id",
        "request submission",
        "final date to submit",
        "hard deadline",
        "title:",
        "category:",
        "new experience",
        "final date",
    ]
    return any(x in tl for x in bad_patterns)


def _clean_policy_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = re.sub(r"(source url\s*:\s*\S+)", "", text, flags=re.I)
    text = re.sub(r"(last checked\s*:\s*[^\.]+)", "", text, flags=re.I)
    text = re.sub(r"(doc id\s*:\s*[^\.]+)", "", text, flags=re.I)
    text = re.sub(r"(category\s*:\s*[^\.]+)", "", text, flags=re.I)
    text = re.sub(r"(title\s*:\s*[^\.]+)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" -;")
    return text


def _extract_best_sentences(text: str, focus_terms: List[str], max_sentences: int = 4) -> List[str]:
    cleaned = _clean_policy_text(text)
    if not cleaned:
        return []

    pieces = re.split(r"(?<=[\.\!\?])\s+| \- ", cleaned)
    pieces = [p.strip(" -") for p in pieces if p and len(p.strip()) > 20]

    if not pieces:
        return [cleaned[:420].rstrip()]

    focus_lower = [t.lower() for t in focus_terms if t]
    scored: List[Tuple[int, str]] = []

    for p in pieces:
        pl = p.lower()
        score = 0

        for term in focus_lower:
            if term in pl:
                score += 2

        if any(x in pl for x in ["eligib", "must", "required", "allowed", "related to", "full-time", "academic year"]):
            score += 1

        if _looks_like_noise(pl):
            score -= 5

        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[str] = []
    seen = set()
    for _, sent in scored:
        key = re.sub(r"\W+", " ", sent.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(sent[:420].rstrip())
        if len(out) >= max_sentences:
            break

    return out


def _build_policy_context_snippet(
    hits: List[Dict[str, Any]],
    decision: Optional[Dict[str, Any]],
    topic: Optional[str],
    limit: int = 8,
) -> str:
    focus_terms = _policy_focus_terms(decision or {}, topic)
    bullets: List[str] = []
    seen = set()

    for h in hits:
        src = h.get("_source") or {}
        txt = (src.get("text") or src.get("snippet") or "").strip()
        if not txt:
            continue

        sentences = _extract_best_sentences(txt, focus_terms, max_sentences=3)
        for best in sentences:
            if not best:
                continue
            if _looks_like_noise(best):
                continue

            key = re.sub(r"\W+", " ", best.lower()).strip()
            if not key or key in seen:
                continue

            seen.add(key)
            bullets.append(f"- {best}")

            if len(bullets) >= limit:
                break

        if len(bullets) >= limit:
            break

    if not bullets:
        return ""

    return "Relevant policy information:\n" + "\n".join(bullets)




def _derive_retrieval_topics(user_query: str, topic: Optional[str], intent: str) -> List[Optional[str]]:
    q = (user_query or "").lower()
    topics: List[Optional[str]] = []

    def add(t: Optional[str]) -> None:
        if t not in topics:
            topics.append(t)

    add(topic)

    if topic == "employment_opt":
        reporting_cues = [
            "report", "reporting", "employer", "address", "unemployment",
            "employment requirement", "20 hours", "15 hours", "e-verify",
            "e-verified", "i-983", "validation", "ead card", "ead",
            "what should i do next", "change of employer", "material changes"
        ]
        stem_app_cues = ["stem opt", "uscis", "i-765", "60 days", "stem opt i-20", "before receiving"]
        sevp_cues = ["sevp", "portal", "what portal", "which portal"]

        if any(c in q for c in reporting_cues):
            add("employment_opt_reporting_requirements")
        if any(c in q for c in stem_app_cues):
            add("employment_stem_opt_application_procedures")
        if any(c in q for c in sevp_cues):
            add("employment_opt_sevp_portal")
        if "i-983" in q or "training plan" in q:
            add("employment_stem_opt_i983_instructions")

    elif topic == "employment_opt_reporting_requirements":
        if "portal" in q or "sevp" in q or "ead" in q:
            add("employment_opt_sevp_portal")
        if "i-983" in q or "training plan" in q:
            add("employment_stem_opt_i983_instructions")
        if "stem opt" in q and ("60 days" in q or "i-765" in q or "uscis" in q):
            add("employment_stem_opt_application_procedures")

    elif topic == "employment_stem_opt_application_procedures":
        if "i-983" in q or "training plan" in q:
            add("employment_stem_opt_i983_instructions")
        if any(c in q for c in ["employer changed", "change of employer", "report", "reporting", "e-verify", "20 hours"]):
            add("employment_opt_reporting_requirements")

    elif topic == "employment_stem_opt_i983_instructions":
        if any(c in q for c in ["employer changed", "change of employer", "report", "reporting", "material changes"]):
            add("employment_opt_reporting_requirements")

    # Health insurance / SHWC / SSN family expansion and recovery from topic collisions.
    fee_cues = [
        "health insurance fee", "student health insurance fee", "mandatory and other fees", "student accounting",
        "tuition and fees", "health insurance charge", "charged by illinois tech", "on their bill", "billed"
    ]
    ship_cues = [
        "health insurance plan", "automatically enrolled", "mandatory for international", "required to purchase",
        "waiver", "waive", "comparable coverage", "aetna", "one academic credit hour", "one billable hour",
        "health insurance mandatory", "insurance obligation", "enrolled in the illinois tech health insurance plan",
        "proof of health insurance", "off-campus doctors"
    ]
    shwc_cues = [
        "student health and wellness center", "shwc", "care hub", "academic live care", "schedule an appointment",
        "appointment", "medical assistance", "medical care", "counseling", "wellness center", "urgent emotional concerns"
    ]
    ssn_cues = [
        "ssn", "social security", "social security number", "ssa", "ss-5", "ssn support letter", "seo ssn letter"
    ]

    if any(c in q for c in ssn_cues):
        add("health_insurance_ssn")
    if any(c in q for c in fee_cues):
        add("health_insurance_student_fees")
    if any(c in q for c in ship_cues) or ("health insurance" in q and ("international student" in q or "international students" in q)):
        add("health_insurance_ship_waiver")
    if any(c in q for c in shwc_cues):
        add("health_insurance_shwc_services")

    if topic == "health_insurance_shwc_services" and any(c in q for c in ["insurance plan", "enrolled", "waiver", "mandatory", "required", "obligation"]):
        add("health_insurance_ship_waiver")
    if topic == "health_insurance_ship_waiver" and any(c in q for c in ["fee", "bill", "charge", "student accounting", "tuition and fees"]):
        add("health_insurance_student_fees")
    if topic == "health_insurance_student_fees" and any(c in q for c in ["waiver", "aetna", "insurance plan", "international", "mandatory", "automatically enrolled"]):
        add("health_insurance_ship_waiver")
    if topic == "employment_on_campus" and any(c in q for c in ssn_cues):
        add("health_insurance_ssn")
    if topic == "employment_on_campus" and any(c in q for c in shwc_cues + ["health insurance", "medical"]):
        add("health_insurance_shwc_services")
        if "health insurance" in q:
            add("health_insurance_ship_waiver")

    # Unfiltered fallback last for recall if topic-specific retrieval drifts or misses.
    add(None)
    return topics


def _retrieve_with_topic_expansion(user_query: str, *, topic: Optional[str], intent: str, k: int) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for t in _derive_retrieval_topics(user_query, topic, intent):
        hits = hybrid_retrieve(
            user_query,
            topic=t,
            intent=intent,
            k=max(k, 6),
        )
        merged = _merge_hits(merged, hits, k=max(12, k * 2))
    return merged[:k]


def _detect_compare_topics(user_query: str) -> List[str]:
    q = (user_query or "").lower()
    topics: List[str] = []

    if "cpt" in q or "curricular practical training" in q:
        topics.append("employment_cpt")
    if re.search(r"\bstem\s+opt\b", q):
        topics.append("employment_stem_opt_application_procedures")
    elif "opt" in q or "optional practical training" in q:
        topics.append("employment_opt")

    if "rcl" in q or "reduced course load" in q:
        topics.append("f1_status_reduced_course_load")
    if "ssn" in q or "social security" in q:
        topics.append("health_insurance_ssn")
    if "travel" in q or "re-entry" in q or "reentry" in q:
        topics.append("f1_status_Travel")

    seen = set()
    out = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _retrieve_compare_hits(user_query: str, topics: List[str], k: int) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    per_topic_k = max(4, k)

    topic_alias_queries = {
        "employment_cpt": "CPT curricular practical training eligibility work related to major for credit",
        "employment_opt": "OPT optional practical training eligibility employment after degree field of study",
        "employment_stem_opt_application_procedures": "STEM OPT extension eligibility E-Verify I-983",
        "f1_status_reduced_course_load": "Reduced Course Load RCL eligibility approval medical final semester academic difficulty",
        "health_insurance_ssn": "SSN social security number eligibility student employment authorization documents",
        "f1_status_Travel": "travel re-entry documents I-20 travel signature visa passport",
    }

    for t in topics:
        topic_q = f"{user_query} {topic_alias_queries.get(t, '')}".strip()
        hits = hybrid_retrieve(topic_q, topic=t, intent="compare", k=per_topic_k)
        merged = _merge_hits(merged, hits, k=max(12, len(topics) * per_topic_k))

    return merged[: max(10, len(topics) * 4)]


def chat_turn(
    user_query: str,
    *,
    memory: Optional[Dict[str, Any]] = None,
    k: int = 8,
) -> Dict[str, Any]:
    memory = dict(memory or {})

    hits: List[Dict[str, Any]] = []
    decision: Optional[Dict[str, Any]] = None
    topic: Optional[str] = None
    topic_conf_f: float = 0.0
    intent: str = detect_intent(user_query)

    oos = detect_out_of_scope(user_query)
    if oos:
        for key in [
            "_pending",
            "_pending_missing_slots",
            "_pending_policy_id",
            "_pending_topic",
            "_pending_disambiguation",
            "_pending_query",
            "_pending_intent",
        ]:
            memory.pop(key, None)
        return {
            "intent": "out_of_scope",
            "mode": "refuse",
            "topic": None,
            "topic_confidence": 0.0,
            "clarifying_questions": [],
            "answer_markdown": build_out_of_scope_answer(oos),
            "retrieved_count": 0,
            "memory": memory,
            "decision": None,
            "interaction": None,
        }

    if intent in {"smalltalk", "help"}:
        ans = "Hi! 👋\n\n" + _help_answer() if intent == "smalltalk" else _help_answer()
        return {
            "intent": intent,
            "mode": "help",
            "topic": None,
            "topic_confidence": 0.0,
            "clarifying_questions": [],
            "answer_markdown": ans,
            "retrieved_count": 0,
            "memory": memory,
            "decision": None,
            "interaction": None,
        }

    pending = bool(memory.get("_pending", False))
    pending_policy_id = memory.get("_pending_policy_id")
    pending_missing_slots = memory.get("_pending_missing_slots") or []

    topic, topic_conf = detect_topic(user_query)
    topic_conf_f = float(topic_conf or 0.0)

    pending_topic = memory.get("_pending_topic")
    if pending and _looks_like_fresh_question(user_query):
        if topic is None or topic != pending_topic:
            for key in ["_pending", "_pending_missing_slots", "_pending_policy_id", "_pending_topic"]:
                memory.pop(key, None)
            pending = False
            pending_policy_id = None
            pending_missing_slots = []

    turn_type = classify_turn_type(user_query, pending=pending, missing=list(pending_missing_slots))
    if turn_type == "new_question":
        for key in ["_pending", "_pending_missing_slots", "_pending_policy_id", "_pending_topic"]:
            memory.pop(key, None)
        pending = False
        pending_policy_id = None
        pending_missing_slots = []

    use_rule_flow = pending or (intent == "eligibility" and _is_personal_eligibility_query(user_query))

    if use_rule_flow:
        if pending and pending_policy_id:
            new_slots = extract_slots(user_query, candidate_slots=list(pending_missing_slots))
            memory = merge_slots(memory, new_slots)

            decision = evaluate_policy_by_id(
                pending_policy_id,
                intent="eligibility",
                slots=memory,
            )

            rule_topic = decision.get("topic") or topic
            rule_hits = _eligibility_context_hits(user_query, rule_topic, decision, k=k)

            if decision.get("decision") == "depends":
                memory["_pending"] = True
                memory["_pending_policy_id"] = pending_policy_id
                memory["_pending_missing_slots"] = decision.get("missing_slots") or []
                memory["_pending_topic"] = rule_topic

                return {
                    "intent": "eligibility",
                    "mode": "rules",
                    "topic": rule_topic,
                    "topic_confidence": topic_conf_f,
                    "clarifying_questions": decision.get("clarifying_questions") or [],
                    "answer_markdown": _render_clarifying_prompt(),
                    "retrieved_count": len(rule_hits),
                    "memory": memory,
                    "decision": decision,
                    "interaction": _build_slot_form_interaction(decision),
                }

            for key in ["_pending", "_pending_missing_slots", "_pending_policy_id", "_pending_topic"]:
                memory.pop(key, None)

            policy_context = _build_policy_context_snippet(
                rule_hits,
                decision,
                rule_topic,
                limit=8,
            )
            return {
                "intent": "eligibility",
                "mode": "rules",
                "topic": rule_topic,
                "topic_confidence": topic_conf_f,
                "clarifying_questions": [],
                "answer_markdown": postprocess_answer(
                    user_query=user_query,
                    answer_markdown=_render_rule_decision(decision, policy_context) + _sources_tail(rule_hits),
                    intent="eligibility",
                    mode="rules",
                    decision=decision,
                    hits=rule_hits,
                ),
                "retrieved_count": len(rule_hits),
                "memory": memory,
                "decision": decision,
                "interaction": None,
            }

        new_slots = extract_slots(user_query)
        memory = merge_slots(memory, new_slots)

        decision = evaluate_best(intent="eligibility", slots=memory, topic_hint=topic)

        if (decision.get("decision") in {"unknown", None}) and not decision.get("policy_id"):
            hits = _retrieve_with_topic_expansion(
                user_query,
                topic=topic if _should_filter_by_topic(topic, topic_conf_f) else None,
                intent="eligibility",
                k=k,
            )
            ans = synthesize_answer(
                user_query=user_query,
                intent="eligibility",
                clarifying_questions=[],
                hits=hits,
            ).strip() or compose_answer(user_query, hits, "eligibility", topic, [])

            ans = postprocess_answer(
                user_query=user_query,
                answer_markdown=ans + _sources_tail(hits),
                intent="eligibility",
                mode="retrieval",
                decision=None,
                hits=hits,
            )
            return {
                "intent": "eligibility",
                "mode": "retrieval",
                "topic": topic,
                "topic_confidence": topic_conf_f,
                "clarifying_questions": [],
                "answer_markdown": ans,
                "retrieved_count": len(hits),
                "memory": memory,
                "decision": None,
                "interaction": None,
            }

        rule_topic = decision.get("topic") or topic
        rule_hits = _eligibility_context_hits(user_query, rule_topic, decision, k=k)

        if decision.get("decision") == "depends":
            memory["_pending"] = True
            memory["_pending_policy_id"] = decision.get("policy_id")
            memory["_pending_missing_slots"] = decision.get("missing_slots") or []
            memory["_pending_topic"] = rule_topic

            return {
                "intent": "eligibility",
                "mode": "rules",
                "topic": rule_topic,
                "topic_confidence": topic_conf_f,
                "clarifying_questions": decision.get("clarifying_questions") or [],
                "answer_markdown": _render_clarifying_prompt(),
                "retrieved_count": len(rule_hits),
                "memory": memory,
                "decision": decision,
                "interaction": _build_slot_form_interaction(decision),
            }

        policy_context = _build_policy_context_snippet(
            rule_hits,
            decision,
            rule_topic,
            limit=8,
        )
        return {
            "intent": "eligibility",
            "mode": "rules",
            "topic": rule_topic,
            "topic_confidence": topic_conf_f,
            "clarifying_questions": [],
            "answer_markdown": postprocess_answer(
                user_query=user_query,
                answer_markdown=_render_rule_decision(decision, policy_context) + _sources_tail(rule_hits),
                intent="eligibility",
                mode="rules",
                decision=decision,
                hits=rule_hits,
            ),
            "retrieved_count": len(rule_hits),
            "memory": memory,
            "decision": decision,
            "interaction": None,
        }

    if intent == "compare":
        compare_topics = _detect_compare_topics(user_query)
        if len(compare_topics) >= 2:
            compare_hits = _retrieve_compare_hits(user_query, compare_topics, k=k)
            answer_md = synthesize_answer(
                user_query=user_query,
                intent="compare",
                clarifying_questions=[],
                hits=compare_hits,
            ).strip() or compose_answer(user_query, compare_hits, "general", None, [])

            answer_md = postprocess_answer(
                user_query=user_query,
                answer_markdown=answer_md + _sources_tail(compare_hits),
                intent="compare",
                mode="retrieval",
                decision=None,
                hits=compare_hits,
            )
            return {
                "intent": "compare",
                "mode": "retrieval",
                "topic": None,
                "topic_confidence": 0.0,
                "clarifying_questions": [],
                "answer_markdown": answer_md,
                "retrieved_count": len(compare_hits),
                "memory": memory,
                "decision": None,
                "interaction": None,
            }

    if not topic:
        if intent == "contact_info":
            contact_k = _adaptive_retrieval_k(user_query, k)
            hits = _retrieve_with_topic_expansion(user_query, topic=None, intent=intent, k=contact_k)
            if hits:
                answer_md = synthesize_answer(
                    user_query=user_query,
                    intent=intent,
                    clarifying_questions=[],
                    hits=hits,
                ).strip() or compose_answer(user_query, hits, intent, None, [])

                answer_md = postprocess_answer(
                    user_query=user_query,
                    answer_markdown=answer_md + _sources_tail(hits),
                    intent=intent,
                    mode="retrieval",
                    decision=None,
                    hits=hits,
                )
                return {
                    "intent": intent,
                    "mode": "retrieval",
                    "topic": None,
                    "topic_confidence": 0.0,
                    "clarifying_questions": [],
                    "answer_markdown": answer_md,
                    "retrieved_count": len(hits),
                    "memory": memory,
                    "decision": None,
                    "interaction": None,
                }

        cands = top_topic_candidates(user_query, n=4)
        return {
            "intent": intent,
            "mode": "clarify_topic",
            "topic": None,
            "topic_confidence": 0.0,
            "clarifying_questions": [],
            "answer_markdown": _clarify_topic_prompt(cands),
            "retrieved_count": 0,
            "memory": memory,
            "decision": None,
            "interaction": None,
        }

    req_amb, req_amb_qs = request_reference_ambiguity(user_query, intent)
    if req_amb:
        return {
            "intent": intent,
            "mode": "clarify",
            "topic": topic,
            "topic_confidence": topic_conf_f,
            "clarifying_questions": req_amb_qs,
            "answer_markdown": "I can help with that, but I need one clarification:\n\n- " + "\n- ".join(req_amb_qs),
            "retrieved_count": 0,
            "memory": memory,
            "decision": None,
            "interaction": None,
        }

    amb, amb_qs = work_hours_ambiguity(user_query, topic, topic_conf_f)
    if amb:
        memory["_pending_disambiguation"] = "work_hours"
        memory["_pending_query"] = user_query
        memory["_pending_intent"] = "work_hours"
        memory["_pending_topic"] = topic
        return {
            "intent": "work_hours",
            "mode": "clarify",
            "topic": topic,
            "topic_confidence": topic_conf_f,
            "clarifying_questions": amb_qs[:2],
            "answer_markdown": "I can answer that, but I need one quick clarification:\n\n"
            + "\n".join([f"{i+1}. {q}" for i, q in enumerate(amb_qs[:2])]),
            "retrieved_count": 0,
            "memory": memory,
            "decision": None,
            "interaction": None,
        }

    retrieval_k = _adaptive_retrieval_k(user_query, k)
    hits = _retrieve_with_topic_expansion(
        user_query,
        topic=topic if _should_filter_by_topic(topic, topic_conf_f) else None,
        intent=intent,
        k=retrieval_k,
    )

    if not hits:
        answer_md = (
            "I can’t confirm this from the provided IIT policy excerpts.\n\n"
            "Please check the official IIT pages for this topic or contact the relevant office for the most accurate guidance."
        )
        return {
            "intent": intent,
            "mode": "retrieval",
            "topic": topic,
            "topic_confidence": topic_conf_f,
            "clarifying_questions": [],
            "answer_markdown": answer_md,
            "retrieved_count": 0,
            "memory": memory,
            "decision": None,
            "interaction": None,
        }

    if intent == "work_hours" and topic == "employment_on_campus":
        wh = build_on_campus_hours_answer(user_query, hits)
        if wh:
            answer_md = postprocess_answer(
                user_query=user_query,
                answer_markdown=wh + _sources_tail(hits),
                intent=intent,
                mode="retrieval",
                decision=None,
                hits=hits,
            )
            return {
                "intent": intent,
                "mode": "retrieval",
                "topic": topic,
                "topic_confidence": topic_conf_f,
                "clarifying_questions": [],
                "answer_markdown": answer_md,
                "retrieved_count": len(hits),
                "memory": memory,
                "decision": None,
                "interaction": None,
            }

    answer_md = synthesize_answer(
        user_query=user_query,
        intent=intent,
        clarifying_questions=[],
        hits=hits,
    ).strip() or compose_answer(user_query, hits, intent, topic, [])

    answer_md = postprocess_answer(
        user_query=user_query,
        answer_markdown=answer_md + _sources_tail(hits),
        intent=intent,
        mode="retrieval",
        decision=None,
        hits=hits,
    )

    return {
        "intent": intent,
        "mode": "retrieval",
        "topic": topic,
        "topic_confidence": topic_conf_f,
        "clarifying_questions": [],
        "answer_markdown": answer_md,
        "retrieved_count": len(hits),
        "memory": memory,
        "decision": None,
        "interaction": None,
    }
