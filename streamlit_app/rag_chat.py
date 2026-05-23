from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CANDIDATE_K,
    HISTORY_TURNS,
    LLM_PROVIDER,
    QDRANT_API_KEY,
    QDRANT_URL,
    RERANKER_MODEL_NAME,
    TOP_K_DOCS,
    Work,
)
from rag import answer_with_memory, build_rag


@lru_cache(maxsize=32)
def get_rag(work_key: str, collection: str, title: str, rag_type: str):
    return build_rag(
        rag_type=rag_type,
        collection_name=collection,
        book_title=title,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        top_k_docs=TOP_K_DOCS,
        candidate_k=CANDIDATE_K,
        reranker_model_name=RERANKER_MODEL_NAME,
        verify_collection=False,
    )


def answer_question(
    work: Work,
    question: str,
    history: list[dict[str, str]],
    rag_type: str,
) -> dict[str, Any]:
    rag = get_rag(work.key, work.collection, work.title, rag_type)
    result = answer_with_memory(
        rag,
        work.title,
        question,
        history,
        history_turns=HISTORY_TURNS,
    )
    result["rag_type"] = rag_type
    result["llm_provider"] = LLM_PROVIDER
    return result
