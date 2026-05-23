from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from chat_store import append_message, clear_messages, get_messages
from config import CANDIDATE_K, DEFAULT_RAG_LABEL, LLM_PROVIDER, RAG_TYPES, TOP_K_DOCS, WORKS
from rag_chat import answer_question

st.set_page_config(page_title="RAG: русская классика", page_icon="📚", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

RAG_LABELS = list(RAG_TYPES.keys())


def _sources_key(work_key: str, rag_type: str) -> str:
    return f"last_sources_{work_key}_{rag_type}_{LLM_PROVIDER}"


with st.sidebar:
    work_title = st.selectbox("Произведение", list(WORKS.keys()), key="work_title")
    work = WORKS[work_title]

    rag_label = st.selectbox(
        "Тип RAG",
        RAG_LABELS,
        index=RAG_LABELS.index(DEFAULT_RAG_LABEL) if DEFAULT_RAG_LABEL in RAG_LABELS else 0,
        key="rag_label",
    )
    rag_type = RAG_TYPES[rag_label]

    st.markdown(f"**Коллекция:** `{work.collection}`")
    st.caption(
        f"LLM: `{LLM_PROVIDER}` · RAG: `{rag_type}` · top_k={TOP_K_DOCS} · candidate_k={CANDIDATE_K}"
    )

    if st.button("Очистить историю диалога", use_container_width=True):
        clear_messages(st.session_state.session_id, work.key)
        st.session_state.pop(f"chat_messages_{work.key}", None)
        for label in RAG_LABELS:
            st.session_state.pop(_sources_key(work.key, RAG_TYPES[label]), None)
        st.rerun()

st.title("Вопросы по произведению")
st.caption(f"Память диалога · TinyDB · LLM: **{LLM_PROVIDER}** · RAG: **{rag_label}**")

session_id: str = st.session_state.session_id
cache_key = f"chat_messages_{work.key}"
if cache_key not in st.session_state:
    st.session_state[cache_key] = get_messages(session_id, work.key)

messages: list[dict[str, str]] = st.session_state[cache_key]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Задайте вопрос по книге…"):
    append_message(session_id, work.key, "user", prompt)

    with st.spinner("Готовлю запросы, ищу фрагменты и формирую ответ…"):
        try:
            history = get_messages(session_id, work.key)
            result = answer_question(work, prompt, history, rag_type)
            answer = (result.get("answer") or "").strip() or "Не удалось сформировать ответ."
            st.session_state[_sources_key(work.key, rag_type)] = result.get("documents") or []
        except Exception as exc:
            answer = f"Ошибка: {exc}"
            st.session_state[_sources_key(work.key, rag_type)] = []

    append_message(session_id, work.key, "assistant", answer)
    st.session_state[cache_key] = get_messages(session_id, work.key)
    st.rerun()

sources = st.session_state.get(_sources_key(work.key, rag_type)) or []
if sources and messages and messages[-1].get("role") == "assistant":
    with st.expander("Источники к последнему ответу"):
        for idx, doc in enumerate(sources, start=1):
            page = doc.metadata.get("page")
            header = f"**[{idx}]**"
            if page is not None:
                header += f" стр. {page}"
            rerank = doc.metadata.get("rerank_score")
            if rerank is not None:
                header += f" · rerank={float(rerank):.3f}"
            st.markdown(header)
            text = doc.page_content
            st.text(text[:1200] + ("…" if len(text) > 1200 else ""))
