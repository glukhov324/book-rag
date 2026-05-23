from rag.chat import CHAT_ANSWER_PROMPT, answer_with_memory, format_history
from rag.config import DEFAULT_QDRANT_COLLECTION, HF_EMBEDDING_MODEL, RAG_TYPES
from rag.factory import build_rag, inspect_collection
from rag.llm import ChatLLM, OpenAICompatibleLLM, YandexGPT, build_llm
from rag.pipelines import (
    BaseRAG,
    HydeRAG,
    HydeRerankingRAG,
    RagQueryExtendRAG,
    RerankingRAG,
    VanillaRAG,
    dedupe_documents,
)
from rag.prompts import ANSWER_PROMPT_TEMPLATE
from rag.indexing import BOOK_PRESETS, index_pdf_to_qdrant, resolve_preset
from rag.retriever import QdrantRetriever, build_retrieval_embeddings

__all__ = [
    "ANSWER_PROMPT_TEMPLATE",
    "CHAT_ANSWER_PROMPT",
    "DEFAULT_QDRANT_COLLECTION",
    "HF_EMBEDDING_MODEL",
    "RAG_TYPES",
    "BOOK_PRESETS",
    "BaseRAG",
    "HydeRAG",
    "HydeRerankingRAG",
    "QdrantRetriever",
    "RagQueryExtendRAG",
    "RerankingRAG",
    "VanillaRAG",
    "ChatLLM",
    "OpenAICompatibleLLM",
    "YandexGPT",
    "build_llm",
    "answer_with_memory",
    "build_rag",
    "index_pdf_to_qdrant",
    "build_retrieval_embeddings",
    "dedupe_documents",
    "format_history",
    "inspect_collection",
    "resolve_preset",
]
