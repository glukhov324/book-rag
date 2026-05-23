from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from rag import (
    BaseRAG,
    DEFAULT_QDRANT_COLLECTION,
    build_rag as _build_rag,
    build_retrieval_embeddings,
    inspect_collection,
)
from rag.retriever import QdrantRetriever


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Add {name} to .env")
    return value


@dataclass
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    openrouter_judge_model: str = os.getenv("OPENROUTER_JUDGE_MODEL", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "")
    reranker_model_name: str = os.getenv(
        "RERANKER_MODEL_NAME",
        "BAAI/bge-reranker-v2-m3",
    )
    tmp_dir: str = str(EVAL_DIR / "tmp")
    evaluation_results_dir: str = str(EVAL_DIR / "evaluation_results")


class ScorerLLM(DeepEvalBaseLLM):
    def __init__(self, settings: Settings):
        judge_model = settings.openrouter_judge_model or require_env("OPENROUTER_JUDGE_MODEL")
        judge_key = settings.openrouter_api_key or require_env("OPENROUTER_API_KEY")
        judge_base = settings.openrouter_base_url or require_env("OPENROUTER_BASE_URL")
        self._client = ChatOpenAI(
            model=judge_model,
            openai_api_key=judge_key,
            openai_api_base=judge_base,
            temperature=0.0,
            seed=42,
            timeout=3600,
        )
        self._model = judge_model

    def load_model(self):
        return self._client

    def get_model_name(self):
        return self._model

    def generate(self, prompt: str) -> str:
        return self._client.invoke([HumanMessage(content=prompt)]).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


def load_eval_dataset(dataset_path: str) -> list[dict[str, str]]:
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    rows = data.get("dataset", data)
    return [{"question": row["question"], "answer": row["answer"]} for row in rows]


def create_test_cases(
    dataset: list[dict[str, str]],
    rag: BaseRAG,
    max_cases: int | None,
) -> list[LLMTestCase]:
    rows = dataset[:max_cases] if max_cases else dataset
    test_cases = []

    for idx, row in enumerate(rows, start=1):
        question = row["question"]
        expected_answer = row["answer"]
        print(f"[{idx}/{len(rows)}] Running RAG: {question}")

        try:
            result = rag.run_rag_pipeline(question=question)
        except Exception as error:
            print(f"[FAIL] RAG error: {error}")
            continue

        answer = result.get("answer")
        context = result.get("context") or []
        if not answer or not context:
            print("[SKIP] Empty answer or retrieval context")
            continue

        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=expected_answer,
                retrieval_context=[str(chunk) for chunk in context],
            )
        )

    return test_cases


def normalize_metric_name(name: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in name)
    return "_".join(part for part in normalized.split("_") if part)


def stringify_list(value: list[str] | None) -> str:
    if not value:
        return ""
    return "\n\n---\n\n".join(str(item) for item in value)


BASE_RESULT_FIELDNAMES = [
    "book_title",
    "rag_type",
    "question",
    "true_answer",
    "model_context",
    "model_answer",
]

_INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*')


def sanitize_filename_part(text: str, max_length: int = 100) -> str:
    out: list[str] = []
    for ch in text.strip():
        if ch.isspace():
            out.append("_")
        elif ch in _INVALID_FILENAME_CHARS or ord(ch) < 32:
            out.append("_")
        else:
            out.append(ch)
    collapsed = "_".join(part for part in "".join(out).split("_") if part)
    if len(collapsed) > max_length:
        collapsed = collapsed[:max_length].rstrip("_")
    return collapsed or "untitled"


def make_run_artifact_stem(
    book_title: str,
    rag_type: str,
    when: datetime,
) -> str:
    title = sanitize_filename_part(book_title)
    rag = sanitize_filename_part(rag_type)
    return f"{title}_{rag}_{when:%Y%m%d_%H%M%S}"


def make_evaluation_results_path(
    output_dir: str,
    book_title: str,
    rag_type: str,
    when: datetime,
) -> Path:
    Path(output_dir).mkdir(exist_ok=True)
    stem = make_run_artifact_stem(book_title, rag_type, when)
    return Path(output_dir) / f"{stem}.csv"


def evaluation_result_to_rows(
    evaluation_result: Any,
    book_title: str,
    rag_type: str,
) -> list[dict[str, Any]]:
    rows = []
    for test_result in getattr(evaluation_result, "test_results", []):
        row: dict[str, Any] = {
            "book_title": book_title,
            "rag_type": rag_type,
            "question": getattr(test_result, "input", "") or "",
            "true_answer": getattr(test_result, "expected_output", "") or "",
            "model_context": stringify_list(getattr(test_result, "retrieval_context", None)),
            "model_answer": getattr(test_result, "actual_output", "") or "",
        }

        for metric_data in getattr(test_result, "metrics_data", None) or []:
            metric_name = normalize_metric_name(getattr(metric_data, "name", "metric"))
            metric_values = {
                f"{metric_name}_success": getattr(metric_data, "success", ""),
            }
            row.update(metric_values)

        rows.append(row)

    return rows


def write_evaluation_results_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    metric_columns = {
        column
        for row in rows
        for column in row
        if column not in BASE_RESULT_FIELDNAMES
    }
    fieldnames = [*BASE_RESULT_FIELDNAMES, *sorted(metric_columns)]

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rag(args: argparse.Namespace, settings: Settings) -> BaseRAG:
    return _build_rag(
        rag_type=args.rag,
        collection_name=args.collection_name,
        book_title=args.book_title,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        top_k_docs=args.top_k_docs,
        candidate_k=args.candidate_k,
        reranker_model_name=settings.reranker_model_name,
        verify_collection=True,
    )


def build_metrics(args: argparse.Namespace, judge_model: ScorerLLM) -> list[Any]:
    return [
        AnswerRelevancyMetric(threshold=args.threshold, model=judge_model, include_reason=True),
        FaithfulnessMetric(threshold=args.threshold, model=judge_model, include_reason=True),
        ContextualRecallMetric(threshold=args.threshold, model=judge_model, include_reason=True),
    ]


def run_evaluation(args: argparse.Namespace) -> Any:
    settings = Settings()
    if args.test_cases_cache:
        with open(args.test_cases_cache, "rb") as file:
            test_cases = pickle.load(file)
        print(f"Loaded test cases cache: {args.test_cases_cache}")
    else:
        dataset = load_eval_dataset(args.dataset)
        rag = build_rag(args, settings)
        test_cases = create_test_cases(dataset, rag, args.max_cases)

    if not test_cases:
        raise ValueError(
            "No test cases created. "
            f"Check collection_name='{args.collection_name}' and loaded documents."
        )

    run_timestamp = datetime.now()
    artifact_stem = make_run_artifact_stem(args.book_title, args.rag, run_timestamp)

    Path(settings.tmp_dir).mkdir(exist_ok=True)
    cache_path = Path(settings.tmp_dir) / f"{artifact_stem}.pickle"
    if not args.test_cases_cache:
        with open(cache_path, "wb") as file:
            pickle.dump(test_cases, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved test cases cache: {cache_path}")

    results_path = make_evaluation_results_path(
        settings.evaluation_results_dir,
        args.book_title,
        args.rag,
        run_timestamp,
    )

    judge_model = ScorerLLM(settings)
    async_config = AsyncConfig(run_async=False)
    result_rows: list[dict[str, Any]] = []
    last_evaluation_result = None

    for idx, test_case in enumerate(test_cases, start=1):
        print(f"[{idx}/{len(test_cases)}] Running DeepEval: {test_case.input}")
        try:
            last_evaluation_result = evaluate(
                test_cases=[test_case],
                metrics=build_metrics(args, judge_model),
                async_config=async_config,
            )
        except Exception as error:
            print(f"[FAIL] DeepEval error: {error}")
            continue

        result_rows.extend(evaluation_result_to_rows(last_evaluation_result, args.book_title, args.rag))
        write_evaluation_results_csv(result_rows, results_path)
        print(f"Saved partial evaluation results CSV: {results_path}")

    if not result_rows:
        raise ValueError(
            "No test cases evaluated successfully. "
            f"Check collection_name='{args.collection_name}' and loaded documents."
        )

    print(f"Saved evaluation results CSV: {results_path}")
    return last_evaluation_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG answers with DeepEval LLM-as-a-Judge.")
    parser.add_argument("--dataset", default=str(EVAL_DIR / "test_data" / "prestuplenie.json"))
    parser.add_argument("--collection-name", default=DEFAULT_QDRANT_COLLECTION)
    parser.add_argument("--book-title", default="Преступление и наказание")
    parser.add_argument(
        "--rag",
        choices=["vanilla", "reranking", "hyde-reranking", "rag-query-extend"],
        default="hyde-reranking",
    )
    parser.add_argument("--top-k-docs", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-cases-cache", default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    run_evaluation(parse_args())
