from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from rag.llm import ChatLLM
from rag.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    HYDE_SYSTEM_PROMPT,
    invoke_hyde_paragraph,
    invoke_query_paraphrases,
)
from rag.retriever import QdrantRetriever


def dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in documents:
        key = (doc.page_content or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


class BaseRAG:
    def __init__(
        self,
        llm: ChatLLM,
        retriever: QdrantRetriever,
        top_k_docs: int,
        book_title: str,
    ):
        self.llm = llm
        self.retriever = retriever
        self.top_k_docs = top_k_docs
        self.book_title = book_title

    def retrieve_documents(self, question: str) -> list[Document]:
        raise NotImplementedError

    @staticmethod
    def format_docs_context(documents: list[Document]) -> str:
        if not documents:
            return "## Фрагменты текста книги\nРелевантные фрагменты не найдены."

        chunks = []
        for idx, doc in enumerate(documents, start=1):
            page = doc.metadata.get("page")
            header = f"[{idx}]"
            if page is not None:
                header += f" page={page}"
            chunks.append(f"{header}\n{doc.page_content}")
        return "## Фрагменты текста книги\n" + "\n\n---\n\n".join(chunks)

    def generate_answer(self, question: str, documents: list[Document]) -> str:
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            book_title=self.book_title,
            context=self.format_docs_context(documents),
            question=question,
        )
        return self.llm.invoke(prompt)

    def run_rag_pipeline(self, question: str) -> dict[str, Any]:
        documents = dedupe_documents(self.retrieve_documents(question))
        answer = self.generate_answer(question, documents) if documents else ""
        return {
            "answer": answer,
            "context": [doc.page_content for doc in documents],
            "documents": documents,
        }


class VanillaRAG(BaseRAG):
    def retrieve_documents(self, question: str) -> list[Document]:
        return self.retriever.retrieve(question, self.top_k_docs)


class RerankingRAG(BaseRAG):
    def __init__(
        self,
        llm: ChatLLM,
        retriever: QdrantRetriever,
        top_k_docs: int,
        candidate_k: int,
        book_title: str,
        reranker_model_name: str,
    ):
        super().__init__(llm, retriever, top_k_docs, book_title)
        self.candidate_k = candidate_k
        self.reranker = CrossEncoder(reranker_model_name)

    def retrieve_documents(self, question: str) -> list[Document]:
        candidates = self.retriever.retrieve(question, self.candidate_k)
        return self.rerank_documents(question, candidates)[: self.top_k_docs]

    def rerank_documents(self, question: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        scores = self.reranker.predict([(question, doc.page_content) for doc in documents])
        scored = []
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)
            scored.append((doc, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored]


class HydeRAG(BaseRAG):
    def __init__(self, llm: ChatLLM, retriever: QdrantRetriever, top_k_docs: int, book_title: str):
        super().__init__(llm, retriever, top_k_docs, book_title)
        self.hyde_system_prompt = HYDE_SYSTEM_PROMPT.format(book_title=book_title)

    def generate_hyde_query(self, question: str) -> str:
        return invoke_hyde_paragraph(self.llm, self.hyde_system_prompt, question)

    def retrieve_documents(self, question: str) -> list[Document]:
        hyde_paragraph = self.generate_hyde_query(question)
        hyde_query = (hyde_paragraph or "").strip() or question.strip()
        return self.retriever.retrieve(hyde_query, self.top_k_docs)


class HydeRerankingRAG(RerankingRAG):
    def __init__(
        self,
        llm: ChatLLM,
        retriever: QdrantRetriever,
        top_k_docs: int,
        candidate_k: int,
        book_title: str,
        reranker_model_name: str,
    ):
        super().__init__(llm, retriever, top_k_docs, candidate_k, book_title, reranker_model_name)
        self.hyde_system_prompt = HYDE_SYSTEM_PROMPT.format(book_title=book_title)

    def generate_hyde_query(self, question: str) -> str:
        return invoke_hyde_paragraph(self.llm, self.hyde_system_prompt, question)

    def retrieve_documents(self, question: str) -> list[Document]:
        hyde_paragraph = self.generate_hyde_query(question)
        hyde_query = (hyde_paragraph or "").strip() or question.strip()
        k = self.candidate_k
        from_question = self.retriever.retrieve(question, k)
        from_hyde = self.retriever.retrieve(hyde_query, k)
        merged = dedupe_documents(from_question + from_hyde)
        return self.rerank_documents(question, merged)[: self.top_k_docs]


class RagQueryExtendRAG(RerankingRAG):
    def __init__(
        self,
        llm: ChatLLM,
        retriever: QdrantRetriever,
        top_k_docs: int,
        candidate_k: int,
        book_title: str,
        reranker_model_name: str,
    ):
        super().__init__(llm, retriever, top_k_docs, candidate_k, book_title, reranker_model_name)

    def generate_paraphrases(self, question: str) -> list[str]:
        return invoke_query_paraphrases(self.llm, self.book_title, question)

    def retrieve_documents(self, question: str) -> list[Document]:
        q0 = question.strip()
        paraphrases = self.generate_paraphrases(q0)
        queries: list[str] = [q0]
        seen = {q0.lower()}
        for pq in paraphrases:
            t = (pq or "").strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            queries.append(t)
        k = self.candidate_k
        pooled: list[Document] = []
        for q in queries:
            pooled.extend(self.retriever.retrieve(q, k))
        merged = dedupe_documents(pooled)
        return self.rerank_documents(q0, merged)[: self.top_k_docs]
