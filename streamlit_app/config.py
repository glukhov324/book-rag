from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"
CHAT_DB_PATH = APP_DIR / "data" / "chats.json"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Work:
    key: str
    title: str
    collection: str


def _collection(env_key: str, default: str) -> str:
    return os.getenv(env_key, default)


WORKS: dict[str, Work] = {
    "Преступление и наказание": Work(
        key="prestuplenie",
        title="Преступление и наказание",
        collection=_collection("QDRANT_COLLECTION_PRESTUPLENIE", "prestuplenie_user_bge_m3"),
    ),
    "Капитанская дочка": Work(
        key="kap_dochka",
        title="Капитанская дочка",
        collection=_collection("QDRANT_COLLECTION_KAP_DOCHKA", "kapitanskaya_dochka"),
    ),
    "Отцы и дети": Work(
        key="otci_i_deti",
        title="Отцы и дети",
        collection=_collection("QDRANT_COLLECTION_OTCI_I_DETI", "otci_i_deti"),
    ),
    "Герой нашего времени": Work(
        key="geroy_nashego_vremeni",
        title="Герой нашего времени",
        collection=_collection("QDRANT_COLLECTION_GEROY", "geroy_nashego_vremeni"),
    ),
}

TOP_K_DOCS = int(os.getenv("STREAMLIT_TOP_K_DOCS", "10"))
CANDIDATE_K = int(os.getenv("STREAMLIT_CANDIDATE_K", "20"))
HISTORY_TURNS = int(os.getenv("STREAMLIT_HISTORY_TURNS", "6"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

RAG_TYPES: dict[str, str] = {
    "Vanilla": "vanilla",
    "Reranking": "reranking",
    "HyDE + reranking": "hyde-reranking",
    "Query extend + reranking": "rag-query-extend",
}
DEFAULT_RAG_LABEL = "HyDE + reranking"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "yandex").strip().lower()
