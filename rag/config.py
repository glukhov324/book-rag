from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "deepvk/USER-bge-m3")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "yandex").strip().lower()
DEFAULT_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "prestuplenie_user_bge_m3")
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "300"))


def build_qdrant_client(
    *,
    url: str | None = None,
    api_key: str | None = None,
):
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=url or os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=api_key if api_key is not None else os.getenv("QDRANT_API_KEY") or None,
        timeout=QDRANT_TIMEOUT,
    )

RAG_TYPES = ("vanilla", "reranking", "hyde-reranking", "rag-query-extend")
