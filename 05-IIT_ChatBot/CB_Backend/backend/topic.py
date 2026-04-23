from __future__ import annotations
"""Topic identification for IIT policy docs.

High-level approach:
1) Use ordered high-precision regexes so specific topics win before broad ones.
2) Fall back to fuzzy glossary similarity when no direct pattern matches.

"topic" is the Doc_ID / policy_topic (e.g., employment_cpt).
"category" is one of the dataset folders.
"""

from typing import Optional, Tuple, Dict, List
import re
import math

# Order matters: specific patterns must appear before broader ones.
_HIGH_PRECISION_PATTERNS: List[tuple[re.Pattern[str], str]] = [
    # STEM OPT application / filing / I-20 issuance
    (re.compile(r"\bstem\s+opt\b.*\b(extension|request|application|apply|uscis|i-765|within\s+60\s+days|60\s+days|valid)\b", re.I), "employment_stem_opt_application_procedures"),
    (re.compile(r"\b(within\s+60\s+days|60\s+days|uscis|i-765|stem\s+opt\s+i-20|before\s+receiving\s+the\s+stem\s+opt\s+i-20)\b", re.I), "employment_stem_opt_application_procedures"),

    # I-983 instructions / training plan content
    (re.compile(r"\b(i-?983|training\s+plan|naics|cip\s+code|sevis\s+school\s+code|dso\s+name|requested\s+period)\b", re.I), "employment_stem_opt_i983_instructions"),

    # OPT/STEM reporting requirements
    (re.compile(r"\b(report|reporting|report my employment|employment reporting|change of employer|changed employer|employer changed|address update|address changes|unemployment|validation report|material changes?|employment requirement|at least\s*20\s*hours|20\s*hours\s*/?\s*week|e-verify|e-verified|paid employment)\b", re.I), "employment_opt_reporting_requirements"),
    (re.compile(r"\b(received|got)\b.*\b(ead)\b.*\b(report|employment)\b", re.I), "employment_opt_reporting_requirements"),

    # SEVP portal topic
    (re.compile(r"\bsevp\b|\bsevp\s+portal\b", re.I), "employment_opt_sevp_portal"),

    (re.compile(r"processing times|processing time|how long does ogs|how long.*process|typically take to process|turnaround time", re.I), "f1_status_processing_times"),
    (re.compile(r"when should i submit|processed before .*deadline|before my program deadline|before .* deadline|submit immigration-related requests", re.I), "f1_status_processing_times"),

    # F-1 status maintenance questions about taking fewer classes / less than full-time
    # should route to RCL or enrollment pages before fuzzy matching drifts elsewhere.
    (re.compile(r"can i take fewer classes|take fewer classes|take fewer credits|less than full[- ]time|below full[- ]time|acceptable reasons.*rcl|reasons for rcl|reasons for reduced course load|grounds for rcl", re.I), "f1_status_reduced_course_load"),
    (re.compile(r"register(?:s|ed|ing)? for fewer credits|required credits|without informing ogs|without approval|below full[- ]time enrollment|full[- ]time enrollment requirement", re.I), "f1_status_enrollment_requirements"),

    # Portal/forms submission questions should win before CPT/OPT keyword matches.
    (re.compile(r"forms and requests|isss portal|request form|letter request|download my updated i-20|where do i submit|where can i submit|where can i download|where do i download", re.I), "f1_status_forms_and_requests"),

    # Health insurance / student life family should win before generic on-campus wording.
    (re.compile(r"\b(mandatory\s+student\s+health\s+insurance\s+fee|student\s+health\s+insurance\s+fee|health\s+insurance\s+fee|mandatory\s+and\s+other\s+fees|student\s+accounting|tuition\s+and\s+fees|health\s+insurance\s+charge|charge(?:d)?\s+(?:on|to)\s+(?:their|the)\s+bill|fee\s+charged\s+by\s+illinois\s+tech|view\s+the\s+mandatory\s+student\s+health\s+insurance\s+fee)\b", re.I), "health_insurance_student_fees"),
    (re.compile(r"\b(health\s+insurance\s+plan|student\s+health\s+insurance\s+plan|health\s+insurance\s+mandatory|mandatory\s+for\s+international\s+students|required\s+to\s+purchase|automatically\s+enrolled|added\s+to\s+the\s+illinois\s+tech\s+student\s+health\s+insurance\s+plan|proof\s+of\s+health\s+insurance|one\s+academic\s+credit\s+hour|one\s+billable\s+hour|comparable\s+coverage|waiver|waive|aetna|insurance\s+obligation|enrolled\s+in\s+the\s+illinois\s+tech\s+health\s+insurance\s+plan)\b", re.I), "health_insurance_ship_waiver"),
    (re.compile(r"\b(ssn|social\s+security|social\s+security\s+number|ss-?5|ssa|social\s+security\s+office|ssn\s+support\s+letter|seo\s+ssn\s+letter)\b", re.I), "health_insurance_ssn"),
    (re.compile(r"\b(student\s+health\s+and\s+wellness\s+center|shwc|care\s+hub|academic\s+live\s+care|schedule\s+an\s+appointment|appointment|medical\s+assistance|medical\s+care|counseling|wellness\s+center|urgent\s+emotional\s+concerns|vaccine|immunization)\b", re.I), "health_insurance_shwc_services"),
    (re.compile(r"\b(ogs\s+email|office\s+of\s+global\s+services\s+email|global@illinoistech\.edu|globalservices@illinoistech\.edu)\b", re.I), "f1_status_forms_i20_ds2019"),

    # On-campus employment should require job/work cues, not just the phrase 'on campus'.
    (re.compile(r"(?:\bon[ -]?campus\b|\bcampus employment\b).*(?:\bemployment\b|\bjob\b|\bwork\b|\bworking\b|\bhours\b|\boffer\b|\bseo\b)|(?:\bemployment\b|\bjob\b|\bwork\b|\bworking\b|\bhours\b|\boffer\b|\bseo\b).*(?:\bon[ -]?campus\b|\bcampus employment\b)", re.I), "employment_on_campus"),

    # CPT
    (re.compile(r"\bcpt\b|curricular practical training", re.I), "employment_cpt"),

    # Generic OPT only after more specific OPT-family topics above
    (re.compile(r"\bopt\b|optional practical training", re.I), "employment_opt"),

    # Travel / status / other families
    (re.compile(r"\btravel\b|re-?entry|earliest\s+date.*enter|enter\s+the\s+united\s+states|30\s+days\s+before.*program\s+start", re.I), "f1_status_Travel"),
    (re.compile(r"change to f-1|change of status|b-2 tourist visa|i-539|g-1145|change to j-1", re.I), "f1_status_change_to_f1_j1"),
    (re.compile(r"full[- ]time enrollment|enrollment requirements|online classes?|hybrid classes?|distance learning|credit hours|teaching assistantship|assistantship", re.I), "f1_status_enrollment_requirements"),
        (re.compile(r"\bi-20\b|\bds-2019\b|\bds2019\b", re.I), "f1_status_forms_i20_ds2019"),
    (re.compile(r"check[- ]in", re.I), "f1_status_new_student_check_in"),
    (re.compile(r"pre-enrollment fee", re.I), "f1_status_pre_enrollment_fee"),
        (re.compile(r"reduced course load|\brcl\b|below full[- ]time|drop a class|withdraw a class|only one class|only 1 class|medical condition", re.I), "f1_status_reduced_course_load"),
]

TOPIC_GLOSSARY: Dict[str, str] = {
    "employment_cpt": "Curricular Practical Training (CPT), internship/work authorization during study, academic curriculum, authorization required",
    "employment_on_campus": "On-campus employment, on-campus job, campus work, employment hours, SEO, employment offer, student job",
    "employment_opt": "Optional Practical Training (OPT), work authorization for F-1 students, pre-completion and post-completion OPT",
    "employment_opt_sevp_portal": "SEVP Portal, updating employment address details during OPT, reporting changes",
    "employment_opt_reporting_requirements": "OPT reporting requirements, report employer, address, employment changes, timelines, unemployment, 20 hours per week, E-Verify, I-983",
    "employment_stem_opt_application_procedures": "STEM OPT extension procedures, eligibility, application steps, USCIS filing, 60 day validity, I-765, request documents",
    "employment_stem_opt_i983_instructions": "Form I-983 training plan instructions, STEM OPT compliance, training plan fields, requested period, NAICS, CIP code, signatures",
    "f1_status_Travel": "Travel and re-entry, documents for travel, I-20 travel signature, visa, entry rules, 30 day entry window",
    "f1_status_change_to_f1_j1": "Change of status to F-1 or J-1, B-2 to F-1, I-539, G-1145, required steps, timelines, documents",
    "f1_status_enrollment_requirements": "Enrollment requirements to maintain status, full-time credits, online or hybrid class limits, assistantship, summer rules",
    "f1_status_forms_and_requests": "Student forms and requests, ISSS portal, where to submit requests, where to download updated I-20, request letters, processing",
    "f1_status_forms_i20_ds2019": "I-20 and DS-2019 forms, what they are, how to request, corrections, OGS email, global@illinoistech.edu",
    "f1_status_new_student_check_in": "New student check-in, arrival requirements, orientation, immigration check-in steps, within 24 hours, check in after arrival",
    "f1_status_pre_enrollment_fee": "Pre-enrollment fee, when it applies, payment instructions",
    "f1_status_processing_times": "Processing times for requests, typical timelines, delays, 7 business days, apply as early as possible, late or rushed requests",
    "f1_status_reduced_course_load": "Reduced Course Load (RCL), eligibility reasons, how to request, approvals",
    "health_insurance_ship_waiver": "Illinois Tech Student Health Insurance Plan SHIP, automatic enrollment, waiver, comparable coverage, Aetna Student Health, mandatory for international students",
    "health_insurance_shwc_services": "Student Health and Wellness Center services, SHWC, appointments, medical assistance, counseling, Care Hub, health support",
    "health_insurance_ssn": "Social Security Number (SSN), social security number process for students, eligibility, job offer steps, SSA, SS-5, required documents",
    "health_insurance_student_fees": "Student accounting mandatory and other fees, student health insurance fee, charges on student bill, tuition and fees, mandatory fee page",
}

_TOKEN_RE = re.compile(r"[a-z0-9\-]+")


def _tokens(s: str) -> List[str]:
    return _TOKEN_RE.findall((s or "").lower())


def _tf(tokens: List[str]) -> Dict[str, float]:
    d: Dict[str, float] = {}
    for t in tokens:
        d[t] = d.get(t, 0.0) + 1.0
    n = sum(d.values()) or 1.0
    for k in list(d.keys()):
        d[k] /= n
    return d


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, av in a.items():
        bv = b.get(k)
        if bv:
            dot += av * bv
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def detect_topic(query: str) -> Tuple[Optional[str], float]:
    q = (query or "").lower()

    for pat, doc_id in _HIGH_PRECISION_PATTERNS:
        if pat.search(q):
            return doc_id, 0.93

    q_tf = _tf(_tokens(query))
    best_id = None
    best = 0.0
    for doc_id, desc in TOPIC_GLOSSARY.items():
        sc = _cosine(q_tf, _tf(_tokens(desc)))
        if sc > best:
            best = sc
            best_id = doc_id

    if best_id is None:
        return None, 0.0
    if best >= 0.22:
        return best_id, min(0.78, 0.45 + best)
    if best >= 0.14:
        return best_id, 0.55
    return None, 0.0


def top_topic_candidates(query: str, n: int = 3) -> List[Tuple[str, float]]:
    q_tf = _tf(_tokens(query))
    scored = []
    for doc_id, desc in TOPIC_GLOSSARY.items():
        scored.append((doc_id, _cosine(q_tf, _tf(_tokens(desc)))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


def detect_category(doc_id: Optional[str]) -> Optional[str]:
    if not doc_id:
        return None
    if doc_id.startswith("employment_"):
        return "employment"
    if doc_id.startswith("f1_status_"):
        return "f1_status_maintenance"
    if doc_id.startswith("health_insurance_"):
        return "health_insurance_life_in_us"
    return None
