from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tinydb import Query, TinyDB

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import CHAT_DB_PATH

Message = dict[str, str]


def _db() -> TinyDB:
    CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return TinyDB(CHAT_DB_PATH)


def get_messages(session_id: str, work_key: str) -> list[Message]:
    row = _db().search(
        (Query().session_id == session_id) & (Query().work_key == work_key)
    )
    if not row:
        return []
    return list(row[0].get("messages", []))


def save_messages(session_id: str, work_key: str, messages: list[Message]) -> None:
    db = _db()
    q = Query()
    payload: dict[str, Any] = {
        "session_id": session_id,
        "work_key": work_key,
        "messages": messages,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    existing = db.search((q.session_id == session_id) & (q.work_key == work_key))
    if existing:
        db.update(payload, (q.session_id == session_id) & (q.work_key == work_key))
    else:
        db.insert(payload)


def append_message(session_id: str, work_key: str, role: str, content: str) -> list[Message]:
    messages = get_messages(session_id, work_key)
    messages.append({"role": role, "content": content})
    save_messages(session_id, work_key, messages)
    return messages


def clear_messages(session_id: str, work_key: str) -> None:
    db = _db()
    q = Query()
    db.remove((q.session_id == session_id) & (q.work_key == work_key))
