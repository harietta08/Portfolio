# ── llm/prompts.py ────────────────────────────────────────────────────────────
# ALL prompt templates live here and only here.
# Never put prompts in function bodies — untestable and unmaintainable.

SYSTEM_PROMPT = """You are a venture capital analyst assistant.
Extract structured information from startup descriptions.
Always respond in valid JSON only. No explanation. No markdown.
If information is not present, use null.
Never hallucinate metrics or facts not stated in the description.
If you cannot extract a field with confidence, return "insufficient information"."""

EXTRACTION_PROMPT_TEMPLATE = """Extract the following fields from this startup description.
Respond ONLY with valid JSON matching the schema exactly.

Schema:
{{
  "sector": string,
  "stage": string or null,
  "traction_score": integer 1-10 or null,
  "key_metrics": list of strings,
  "business_model": string,
  "target_customer": string,
  "moat": string or null,
  "risk_flags": list of strings,
  "investment_signal": string
}}

Few-shot examples:

Example 1:
Description: "AI-powered clinical documentation platform integrating with Epic EHR.
FDA 510k submitted. $2M ARR from 3 hospital pilots."
Output:
{{
  "sector": "Healthcare AI",
  "stage": null,
  "traction_score": 7,
  "key_metrics": ["$2M ARR", "3 hospital pilots", "FDA 510k submitted"],
  "business_model": "B2B SaaS sold to hospital systems",
  "target_customer": "Hospital systems using Epic EHR",
  "moat": "EHR integration depth and FDA clearance process",
  "risk_flags": ["FDA approval pending", "long hospital sales cycles"],
  "investment_signal": "Strong traction with regulated customer validation"
}}

Example 2:
Description: "Carbon accounting platform for mid-market companies.
SOC2 certified. 85 paying customers. $1.1M ARR."
Output:
{{
  "sector": "Climate Tech",
  "stage": null,
  "traction_score": 6,
  "key_metrics": ["$1.1M ARR", "85 paying customers", "SOC2 certified"],
  "business_model": "B2B SaaS subscription",
  "target_customer": "Mid-market companies with ESG reporting obligations",
  "moat": "SOC2 certification and compliance workflow integration",
  "risk_flags": ["regulatory dependency", "crowded market"],
  "investment_signal": "Solid early traction in structurally growing compliance market"
}}

Example 3:
Description: "Quantum error correction middleware for NISQ devices.
Research stage. 4 patents filed."
Output:
{{
  "sector": "Deep Tech",
  "stage": "Pre-revenue",
  "traction_score": 3,
  "key_metrics": ["4 patents filed", "NISQ device compatibility"],
  "business_model": "insufficient information",
  "target_customer": "Quantum computing hardware companies",
  "moat": "Patent portfolio and research expertise",
  "risk_flags": ["pre-revenue", "long time to market", "capital intensive"],
  "investment_signal": "Early but high-risk deep tech bet — thesis dependent"
}}

Now extract from this description:
Description: "{description}"
Output:"""

COMPARABLE_PROMPT_TEMPLATE = """You are a VC analyst identifying comparable companies.
Given this startup, suggest 3 comparable companies that are similar in:
- Business model
- Target customer
- Technology approach

Respond ONLY in valid JSON.

Startup: "{name}" — {description}

Output format:
{{
  "comparables": [
    {{"name": string, "reason": string}},
    {{"name": string, "reason": string}},
    {{"name": string, "reason": string}}
  ]
}}"""
