from __future__ import annotations

import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from rag.config import build_qdrant_client
from rag.llm import ChatLLM, build_llm
from rag.pipelines import (
    BaseRAG,
    HydeRerankingRAG,
    RagQueryExtendRAG,
    RerankingRAG,
    VanillaRAG,
)
from rag.retriever import QdrantRetriever, build_retrieval_embeddings


def inspect_collection(client: QdrantClient, collection_name: str) -> None:
    if not client.collection_exists(collection_name):
        raise ValueError(f"Qdrant collection not found: {collection_name}")

    count_result = client.count(collection_name=collection_name, exact=True)
    points_count = int(getattr(count_result, "count", 0) or 0)
    print(f"collection_name: {collection_name}")
    print(f"points_count: {points_count}")
    if points_count == 0:
        raise ValueError(
            f"Qdrant collection '{collection_name}' is empty. "
            "Load documents into it or pass a populated collection."
        )


def build_rag(
    *,
    rag_type: str,
    collection_name: str,
    book_title: str,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    top_k_docs: int = 10,
    candidate_k: int = 20,
    reranker_model_name: str | None = None,
    client: QdrantClient | None = None,
    llm: ChatLLM | None = None,
    embeddings: HuggingFaceEmbeddings | None = None,
    verify_collection: bool = False,
) -> BaseRAG:
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = qdrant_api_key if qdrant_api_key is not None else os.getenv("QDRANT_API_KEY") or None
    reranker_model_name = reranker_model_name or os.getenv(
        "RERANKER_MODEL_NAME",
        "BAAI/bge-reranker-v2-m3",
    )

    client = client or build_qdrant_client(url=qdrant_url, api_key=qdrant_api_key)
    if verify_collection:
        inspect_collection(client, collection_name)
    elif not client.collection_exists(collection_name):
        raise ValueError(f"Qdrant collection not found: {collection_name}")

    embeddings = embeddings or build_retrieval_embeddings()
    retriever = QdrantRetriever(client, collection_name, embeddings)
    llm = llm or build_llm()

    if rag_type == "vanilla":
        return VanillaRAG(llm, retriever, top_k_docs, book_title)
    if rag_type == "reranking":
        return RerankingRAG(
            llm, retriever, top_k_docs, candidate_k, book_title, reranker_model_name
        )
    if rag_type == "hyde-reranking":
        return HydeRerankingRAG(
            llm, retriever, top_k_docs, candidate_k, book_title, reranker_model_name
        )
    if rag_type == "rag-query-extend":
        return RagQueryExtendRAG(
            llm, retriever, top_k_docs, candidate_k, book_title, reranker_model_name
        )
    raise ValueError(f"Unknown RAG type: {rag_type}")
