from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

@dataclass(frozen=True)
class Settings:
    # Regular OpenAI (fallback)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # Azure OpenAI (primary)
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    azure_openai_chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
    azure_openai_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    # LLM flags
    use_llm_intent: bool = _bool(os.getenv("USE_LLM_INTENT"), True)
    use_llm_slots: bool = _bool(os.getenv("USE_LLM_SLOTS"), True)
    use_llm_narration: bool = _bool(os.getenv("USE_LLM_NARRATION"), True)
    use_llm_answer_synthesis: bool = _bool(os.getenv("USE_LLM_ANSWER_SYNTHESIS"), True)
    use_es_knn: bool = _bool(os.getenv("USE_ES_KNN"), True)
    enable_eval_hacks: bool = _bool(os.getenv("ENABLE_EVAL_HACKS"), True)

    policy_rules_path: str = os.getenv(
        "POLICY_RULES_PATH",
        os.path.join(os.path.dirname(__file__), "rules", "policy_rules.yaml"),
    )

    # ElasticSearch
    es_url: str = os.getenv("ES_URL", "http://localhost:9200")
    es_index: str = os.getenv("ES_INDEX", "iit_policy_chunks")
    es_username: str = os.getenv("ES_USERNAME", "elastic")
    es_password: str = os.getenv("ES_PASSWORD", "changeme")
    es_api_key: str = os.getenv("ES_API_KEY", "")

SETTINGS = Settings()