from __future__ import annotations
from elasticsearch import Elasticsearch
from .config import SETTINGS

def get_es() -> Elasticsearch:
    if SETTINGS.es_api_key:
        return Elasticsearch(
            SETTINGS.es_url,
            api_key=SETTINGS.es_api_key,
            request_timeout=60
        )
    return Elasticsearch(
        SETTINGS.es_url,
        basic_auth=(SETTINGS.es_username, SETTINGS.es_password),
        request_timeout=60
    )