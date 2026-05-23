#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient

from rag.config import build_qdrant_client
from rag.indexing import BOOK_PRESETS, index_pdf_to_qdrant, resolve_preset
from rag.retriever import build_retrieval_embeddings


def wait_for_qdrant(*, attempts: int = 60, delay_sec: float = 2.0) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            build_qdrant_client().get_collections()
            print("Qdrant готов.")
            return
        except Exception as exc:
            last_error = exc
            print(f"[{attempt}/{attempts}] ждём Qdrant: {exc}")
            time.sleep(delay_sec)
    raise RuntimeError(
        f"Qdrant не ответил за {attempts * delay_sec:.0f} с: {last_error}"
    ) from last_error


def _collection_points_count(client: QdrantClient, collection_name: str) -> int:
    if not client.collection_exists(collection_name):
        return 0
    result = client.count(collection_name=collection_name, exact=True)
    return int(getattr(result, "count", 0) or 0)


def presets_needing_index(client: QdrantClient) -> list[str]:
    needed: list[str] = []
    for preset_key in sorted(BOOK_PRESETS):
        pdf_path, collection_name = resolve_preset(preset_key)
        if not pdf_path.exists():
            continue
        if _collection_points_count(client, collection_name) == 0:
            needed.append(preset_key)
    return needed


def _print_index_status(client: QdrantClient) -> None:
    for preset_key in sorted(BOOK_PRESETS):
        pdf_path, collection_name = resolve_preset(preset_key)
        if not pdf_path.exists():
            continue
        count = _collection_points_count(client, collection_name)
        print(f"  {preset_key}: {collection_name} ({count} points)")


def _print_presets() -> None:
    print("Встроенные книги (загружаются все при запуске скрипта):")
    for key in sorted(BOOK_PRESETS):
        pdf_name, env_key, default = BOOK_PRESETS[key]
        pdf_path, collection = resolve_preset(key)
        exists = "ok" if pdf_path.exists() else "PDF не найден"
        print(f"  {key}")
        print(f"    pdf:        {pdf_path}")
        print(f"    collection: {collection}  ({env_key} или {default})")
        print(f"    status:     {exists}")


def _index_preset(
    preset_key: str,
    args: argparse.Namespace,
    *,
    client,
    embeddings,
) -> tuple[int, int]:
    pdf_path, collection_name = resolve_preset(preset_key)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF для '{preset_key}' не найден: {pdf_path}\n"
            f"Ожидается: evaluation/test_books/{BOOK_PRESETS[preset_key][0]}"
        )

    def on_progress(uploaded: int, total: int) -> None:
        print(f"  uploaded {uploaded}/{total} chunks")

    print(f"[{preset_key}] PDF: {pdf_path}")
    print(f"[{preset_key}] Collection: {collection_name}")

    n_pages, n_chunks = index_pdf_to_qdrant(
        pdf_path,
        collection_name,
        client=client,
        embeddings=embeddings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        upload_batch_size=args.batch_size,
        on_batch_uploaded=on_progress,
    )
    print(f"[{preset_key}] Готово: {n_pages} стр. {n_chunks} чанков")
    return n_pages, n_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Загрузить все встроенные книги в Qdrant (evaluation/test_books/).",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Только показать список книг, без загрузки",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Не ждать готовности Qdrant перед загрузкой",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Переиндексировать все книги, даже если коллекции уже заполнены",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16, help="Размер батча embed + upsert")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_presets:
        _print_presets()
        return

    if not args.no_wait:
        wait_for_qdrant()

    client = build_qdrant_client()

    if args.force:
        to_index = sorted(
            key for key in BOOK_PRESETS if resolve_preset(key)[0].exists()
        )
    else:
        to_index = presets_needing_index(client)
        if not to_index:
            print("Все коллекции уже заполнены — индексация не нужна.")
            _print_index_status(client)
            print("\nДля принудительной переиндексации: --force")
            return

    embeddings = build_retrieval_embeddings()
    failed: list[str] = []

    print(f"Загрузка {len(to_index)} книг в Qdrant…\n")
    for preset_key in to_index:
        print(f"=== {preset_key} ===")
        try:
            _index_preset(preset_key, args, client=client, embeddings=embeddings)
        except FileNotFoundError as exc:
            print(f"[{preset_key}] Пропуск: {exc}")
            failed.append(preset_key)
        except Exception as exc:
            print(f"[{preset_key}] Ошибка: {exc}")
            failed.append(preset_key)
        print()

    if failed:
        raise SystemExit(f"Не загружены: {', '.join(failed)}")
    print(f"Индексация завершена ({len(to_index)} книг).")


if __name__ == "__main__":
    main()
