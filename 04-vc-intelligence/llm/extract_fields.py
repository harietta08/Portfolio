# ── llm/extract_fields.py ─────────────────────────────────────────────────────
# Purpose: Call HuggingFace Inference API, validate output with Pydantic
# Model: Mistral-7B-Instruct or Zephyr-7B-beta — free with rate limits
# Retry logic: tenacity handles HF rate limits automatically

import os
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from dotenv import load_dotenv

from llm.prompts import SYSTEM_PROMPT, EXTRACTION_PROMPT_TEMPLATE, COMPARABLE_PROMPT_TEMPLATE
from llm.validate_output import validate_extraction, validate_comparables, ExtractionResult

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1  # low temperature = more deterministic JSON output


# ── HuggingFace API call with retry ──────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _call_hf_api(prompt: str) -> str:
    """
    Call HuggingFace Inference API.
    tenacity retries on rate limits (429) with exponential backoff.
    """
    payload = {
        "inputs": f"<s>[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST]",
        "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "return_full_text": False,
            "do_sample": False,
        },
    }

    response = httpx.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)

    if response.status_code == 429:
        logger.warning("HF rate limit hit — retrying...")
        raise httpx.HTTPStatusError("Rate limited", request=response.request, response=response)

    if response.status_code != 200:
        logger.error(f"HF API error {response.status_code}: {response.text}")
        raise httpx.HTTPStatusError(
            f"API error {response.status_code}",
            request=response.request,
            response=response,
        )

    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "")

    raise ValueError(f"Unexpected API response format: {result}")


# ── Public functions ──────────────────────────────────────────────────────────
def extract_startup_fields(description: str) -> ExtractionResult:
    """
    Extract structured fields from startup description.
    Returns ExtractionResult — always succeeds structurally even if LLM fails.
    """
    if not description or len(description.strip()) < 10:
        return ExtractionResult(
            success=False,
            error="Description too short to extract meaningful fields",
        )

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(description=description)

    try:
        raw = _call_hf_api(prompt)
        logger.info(f"Raw LLM response length: {len(raw)} chars")
        return validate_extraction(raw)

    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        return ExtractionResult(
            success=False,
            error=f"LLM unavailable after retries: {str(e)}",
        )


def extract_comparables(name: str, description: str) -> dict:
    """Extract comparable companies for a startup."""
    prompt = COMPARABLE_PROMPT_TEMPLATE.format(name=name, description=description)

    try:
        raw = _call_hf_api(prompt)
        return validate_comparables(raw)
    except Exception as e:
        logger.error(f"Comparables extraction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": {"comparables": []},
        }
