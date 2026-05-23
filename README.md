# RAG по русской классике

RAG-система для ответов на вопросы по художественным произведениям на русском языке. Пользователь задаёт вопрос о книге; система извлекает релевантные фрагменты из Qdrant и генерирует ответ с опорой на контекст.

**Стек:** Qdrant, `deepvk/USER-bge-m3` (эмбеддинги), `BAAI/bge-reranker-v2-m3` (reranking), YandexGPT / OpenAI-compatible LLM, Streamlit, DeepEval (оценка).

## Датасет

### Состав

Четыре произведения, **169** пар «вопрос / эталонный ответ» (число вопросов по книгам различается).

| JSON-файл | Произведение | Вопросов | Коллекция Qdrant (по умолчанию) |
|-----------|--------------|---------:|----------------------------------|
| `prestuplenie.json` | Преступление и наказание | 44 | `prestuplenie_user_bge_m3` |
| `kap_dochka.json` | Капитанская дочка | 37 | `kapitanskaya_dochka` |
| `otci_i_deti.json` | Отцы и дети | 43 | `otci_i_deti` |
| `geroy_nashego_vremeni.json` | Герой нашего времени | 45 | `geroy_nashego_vremeni` |
| **Итого** | 4 произведения | **169** | |

PDF для индексации лежат в `evaluation/test_books/` (имена файлов с подчёркиваниями, см. `rag/indexing.py`).

### Формат примера

```json
{
  "question": "Как зовут главного героя «Капитанской дочки»?",
  "answer": "Главного героя повести зовут Пётр Андреевич Гринёв."
}
```

- `question`: вопрос читателя;
- `answer`: эталонный ответ для сравнения и для метрик DeepEval.

### Типы вопросов

В каждом датасете смешаны типы, чтобы проверить не только факты, но и рассуждение по тексту:

- **Фактологические**: имена, места, события с прямой опорой на текст.
- **Причинно-следственные**: «почему», мотивы, причины.
- **Событийно-описательные**: что происходит, как ведёт себя герой.
- **Интерпретационные**: роль персонажа, смысл образа.
- **Проверочные (ловушки)**: неверная предпосылка; модель не должна «додумывать».

### Типы RAG в приложении

| Режим в UI | Ключ | Описание |
|------------|------|----------|
| Vanilla | `vanilla` | Dense retrieval + генерация |
| Reranking | `reranking` | Больше кандидатов + cross-encoder rerank |
| HyDE + reranking | `hyde-reranking` | Гипотетический абзац для поиска + rerank |
| Query extend + reranking | `rag-query-extend` | Перефразы запроса + rerank |

---

## Метрики оценки

Оценка: **DeepEval**, LLM-as-a-Judge (GPT-4o-mini через OpenRouter). Порог успеха каждой метрики: **0.5**. Прогон на **169** вопросах по четырём романам.

Для каждого примера считаются три бинарные метрики:

| Метрика | Что измеряет |
|---------|----------------|
| **Answer Relevancy** | Ответ релевантен вопросу |
| **Faithfulness** | Ответ опирается только на извлечённый контекст |
| **Contextual Recall** | Контекст достаточен для эталонного ответа |

**Success rate** (как в отчёте, `report/report.tex`): среднее трёх метрик в процентах:

`(Answer Relevancy + Faithfulness + Contextual Recall) / 3`

### Success rate по книгам, %

| Произведение | Baseline | Rerank | HyDE | Query extend |
|--------------|---------:|-------:|-----:|-------------:|
| Преступление и наказание | 92.4 | 93.2 | 93.2 | 93.2 |
| Капитанская дочка | 90.1 | 91.0 | 91.9 | **94.6** |
| Отцы и дети | 89.1 | 95.3 | **97.7** | 96.9 |
| Герой нашего времени | 93.3 | 94.1 | **95.6** | 94.8 |
| **Macro average** | **91.3** | **93.5** | **94.7** | **94.9** |

Источник: отчёт (`report/report.pdf`). Сырые прогоны по отдельным метрикам: `evaluation/evaluation_results/`.

---

## Настройка окружения

```bash
cp .env.example .env
```

Заполните `.env` (см. `.env.example`). Минимум для приложения:

- **Qdrant:** `QDRANT_URL`, при необходимости `QDRANT_API_KEY`
- **LLM:** `LLM_PROVIDER=yandex`: `YANDEX_FOLDER_ID`, `YANDEX_API_KEY`; или `openai`: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL_NAME`
- **Оценка:** `OPENROUTER_JUDGE_MODEL`, `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`

---

## Запуск

### Полный стек (Qdrant + индексация + Streamlit)

```bash
docker compose -f docker-compose.deploy.yaml up -d --build
```

Порядок запуска:

1. **qdrant**: векторная БД  
2. **indexer**: `scripts/index_to_qdrant.py` (пропуск, если коллекции уже заполнены)  
3. **streamlit**: UI после успешного indexer  

Приложение: **http://localhost:8510**

В `.env` укажите ключи LLM и при необходимости OpenRouter. `QDRANT_URL` для контейнеров задаётся в `docker-compose.deploy.yaml` (`http://qdrant:6333`).

Остановка:

```bash
docker compose -f docker-compose.deploy.yaml down
```

### Только Qdrant (для оценки)

```bash
docker compose -f docker-compose.local.yaml up -d
```

В `.env` для `evaluate_rag.py`: `QDRANT_URL=http://localhost:6333`. Индексация в контейнере: `--profile init`.

---

## Оценка RAG (DeepEval)

Из корня репозитория:

```bash
cd evaluation
python evaluate_rag.py \
  --dataset test_data/prestuplenie.json \
  --book-title "Преступление и наказание" \
  --collection-name prestuplenie_user_bge_m3 \
  --rag hyde-reranking \
  --top-k-docs 10 \
  --candidate-k 10
```

### Параметры CLI

| Параметр | Описание |
|----------|----------|
| `--dataset` | Путь к JSON датасета |
| `--book-title` | Название книги (для отчёта) |
| `--collection-name` | Коллекция Qdrant |
| `--rag` | `vanilla`, `reranking`, `hyde-reranking`, `rag-query-extend` |
| `--top-k-docs` | Число фрагментов в контексте (по умолчанию 10) |
| `--candidate-k` | Кандидаты до rerank (по умолчанию 10) |
| `--max-cases` | Лимит примеров; `0` или не указывать: весь датасет |
| `--threshold` | Порог метрик DeepEval (по умолчанию 0.5) |
| `--test-cases-cache` | Путь к pickle с готовыми test cases (без повторного RAG) |