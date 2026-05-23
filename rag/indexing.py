from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from rag.config import PROJECT_ROOT, build_qdrant_client
from rag.retriever import build_retrieval_embeddings

T = TypeVar("T")

EVAL_DIR = PROJECT_ROOT / "evaluation"
TEST_BOOKS_DIR = EVAL_DIR / "test_books"

BOOK_PRESETS: dict[str, tuple[str, str, str]] = {
    "prestuplenie": (
        "Преступление и наказание.pdf",
        "QDRANT_COLLECTION_PRESTUPLENIE",
        "prestuplenie_user_bge_m3",
    ),
    "kap_dochka": (
        "Капитанская_дочка.pdf",
        "QDRANT_COLLECTION_KAP_DOCHKA",
        "kapitanskaya_dochka",
    ),
    "otci_i_deti": (
        "Отцы_и_дети.pdf",
        "QDRANT_COLLECTION_OTCI_I_DETI",
        "otci_i_deti",
    ),
    "geroy_nashego_vremeni": (
        "М.Лермонтов_Герой_нашего_времени.pdf",
        "QDRANT_COLLECTION_GEROY",
        "geroy_nashego_vremeni",
    ),
}


def _retry_on_timeout(call: Callable[[], T], *, attempts: int = 3) -> T:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1:
                raise
            message = str(exc).lower()
            if "timeout" not in message and "timed out" not in message:
                raise
            time.sleep(2.0 * (2**attempt))
    assert last_exc is not None
    raise last_exc


def resolve_preset(preset_key: str) -> tuple[Path, str]:
    if preset_key not in BOOK_PRESETS:
        known = ", ".join(sorted(BOOK_PRESETS))
        raise ValueError(f"Unknown preset '{preset_key}'. Available: {known}")
    pdf_name, env_key, default_collection = BOOK_PRESETS[preset_key]
    pdf_path = TEST_BOOKS_DIR / pdf_name
    collection_name = os.getenv(env_key, default_collection)
    return pdf_path, collection_name


def load_pdf_pages(pdf_path: Path | str) -> list[Document]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages: list[Document] = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(
                Document(
                    page_content=text,
                    metadata={"page": page_index, "source": pdf_path.name},
                )
            )
    return pages


def chunk_documents(
    pages: list[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index
    return chunks


def _embedding_vector_size(embeddings: Any) -> int:
    vector_size = getattr(embeddings, "vector_size", None)
    if vector_size is not None:
        return int(vector_size)
    return len(embeddings.embed_documents(["."])[0])


def _ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)
        vectors = info.config.params.vectors
        existing_size = int(getattr(vectors, "size", 0))
        if existing_size != vector_size:
            raise ValueError(
                f"Collection '{collection_name}' has vector size {existing_size}, "
                f"but embeddings produce size {vector_size}."
            )
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


def _point_id(collection_name: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection_name}:{chunk_index}"))


def index_pdf_to_qdrant(
    pdf_path: Path | str,
    collection_name: str,
    *,
    client: QdrantClient | None = None,
    embeddings: Any | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    upload_batch_size: int = 32,
    on_batch_uploaded: Any | None = None,
) -> tuple[int, int]:
    pdf_path = Path(pdf_path)
    client = client or build_qdrant_client()
    embeddings = embeddings or build_retrieval_embeddings()

    pages = load_pdf_pages(pdf_path)
    if not pages:
        raise ValueError(f"No text extracted from PDF: {pdf_path}")

    chunks = chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError(f"No chunks produced from PDF: {pdf_path}")

    vector_size = _embedding_vector_size(embeddings)

    if client.collection_exists(collection_name):
        _retry_on_timeout(
            lambda: client.delete_collection(collection_name=collection_name),
        )

    _ensure_collection(client, collection_name, vector_size)

    for start in range(0, len(chunks), upload_batch_size):
        batch = chunks[start : start + upload_batch_size]
        vectors = embeddings.embed_documents([chunk.page_content for chunk in batch])
        points = [
            qm.PointStruct(
                id=_point_id(collection_name, int(chunk.metadata["chunk_index"])),
                vector=vector,
                payload={
                    "page_content": chunk.page_content,
                    "metadata": dict(chunk.metadata),
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]
        _retry_on_timeout(
            lambda pts=points: client.upsert(
                collection_name=collection_name, points=pts, wait=True
            ),
        )
        uploaded = min(start + upload_batch_size, len(chunks))
        if on_batch_uploaded is not None:
            on_batch_uploaded(uploaded, len(chunks))

    return len(pages), len(chunks)
