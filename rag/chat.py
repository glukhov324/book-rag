from __future__ import annotations

from typing import Any

from rag.pipelines import BaseRAG, dedupe_documents

CHAT_ANSWER_PROMPT = """Ты — эксперт по русской классической литературе. Отвечай по произведению «{book_title}», опираясь на фрагменты книги ниже и историю диалога с читателем.

## Фрагменты текста книги
{context}

## История диалога
{history}

## Текущий вопрос
{question}

Правила ответа
1. Опирайся преимущественно на фрагменты книги. Не выдумывай сюжет, имён и цитат, которых нет в них.
2. Учитывай историю диалога: на уточняющие вопросы («а почему?», «кто это?») отвечай в том же контексте беседы.
3. Если фрагменты частично отвечают — дай лучший возможный ответ из того, что в них явно следует.
4. Откажись коротко только если во фрагментах нет ни одной зацепки по сути вопроса.
5. Пиши по-русски, ясно и по делу.

Ответ:"""


def format_history(messages: list[dict[str, str]], max_turns: int = 6) -> str:
    if not messages:
        return "(диалог только начинается)"
    tail = messages[-max_turns * 2 :]
    lines: list[str] = []
    for msg in tail:
        role = msg.get("role", "user")
        label = "Читатель" if role == "user" else "Ассистент"
        lines.append(f"{label}: {msg.get('content', '').strip()}")
    return "\n".join(lines)


def answer_with_memory(
    rag: BaseRAG,
    book_title: str,
    question: str,
    history: list[dict[str, str]],
    *,
    history_turns: int = 6,
) -> dict[str, Any]:
    documents = dedupe_documents(rag.retrieve_documents(question))
    context = BaseRAG.format_docs_context(documents)
    history_for_prompt = history[:-1] if history and history[-1].get("role") == "user" else history
    prompt = CHAT_ANSWER_PROMPT.format(
        book_title=book_title,
        context=context,
        history=format_history(history_for_prompt, max_turns=history_turns),
        question=question.strip(),
    )
    answer = rag.llm.invoke(prompt)
    return {
        "answer": answer,
        "documents": documents,
    }
