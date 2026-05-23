from __future__ import annotations

from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from rag.config import HF_EMBEDDING_MODEL


def build_retrieval_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


class QdrantRetriever:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: HuggingFaceEmbeddings,
    ):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings

    def retrieve(self, query: str, k: int) -> list[Document]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embeddings.embed_query(query),
            query_filter=None,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", response)
        return [self._point_to_document(point) for point in points]

    @staticmethod
    def _point_to_document(point: Any) -> Document:
        payload = dict(getattr(point, "payload", {}) or {})
        metadata = dict(payload.get("metadata", {}) or {})
        page_content = payload.get("page_content") or payload.get("text") or payload.get("content") or ""
        if not metadata:
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"page_content", "text", "content"}
            }
        score = getattr(point, "score", None)
        if score is not None:
            metadata["qdrant_score"] = float(score)
        return Document(page_content=page_content, metadata=metadata)
