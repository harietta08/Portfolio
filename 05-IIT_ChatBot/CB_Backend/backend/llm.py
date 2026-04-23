from __future__ import annotations
from typing import Any, Dict, Optional
import json
from openai import AzureOpenAI, OpenAI
from .config import SETTINGS

_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    if SETTINGS.azure_openai_api_key and SETTINGS.azure_openai_endpoint:
        _client = AzureOpenAI(
            api_key=SETTINGS.azure_openai_api_key,
            azure_endpoint=SETTINGS.azure_openai_endpoint,
            api_version=SETTINGS.azure_openai_api_version,
        )
    else:
        if not SETTINGS.openai_api_key:
            return None
        _client = OpenAI(api_key=SETTINGS.openai_api_key)
    return _client

def llm_json(*, system: str, user: str, schema_hint: Dict[str, Any], temperature: float = 0.0) -> Dict[str, Any]:
    client = _get_client()
    if client is None:
        return {}
    model = (
        SETTINGS.azure_openai_chat_deployment
        if SETTINGS.azure_openai_api_key
        else SETTINGS.openai_chat_model
    )
    guidance = (
        "You MUST respond with ONLY valid JSON. No markdown. No extra text. "
        "If unsure, use null values.\n\n"
        f"Schema hint (informal): {json.dumps(schema_hint)}"
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system + "\n\n" + guidance},
            {"role": "user", "content": user},
        ],
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(txt)
    except Exception:
        try:
            start = txt.find("{")
            end = txt.rfind("}")
            if start >= 0 and end > start:
                return json.loads(txt[start : end + 1])
        except Exception:
            pass
        return {}

def llm_text(*, system: str, user: str, temperature: float = 0.2) -> str:
    client = _get_client()
    if client is None:
        return ""
    model = (
        SETTINGS.azure_openai_chat_deployment
        if SETTINGS.azure_openai_api_key
        else SETTINGS.openai_chat_model
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()