from __future__ import annotations
from openai import AzureOpenAI, OpenAI
from .config import SETTINGS

_client = None

def get_client():
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
        _client = OpenAI(api_key=SETTINGS.openai_api_key)
    return _client

def embed_text(text: str) -> list[float]:
    client = get_client()
    model = (
        SETTINGS.azure_openai_embedding_deployment
        if SETTINGS.azure_openai_api_key
        else SETTINGS.openai_embed_model
    )
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding