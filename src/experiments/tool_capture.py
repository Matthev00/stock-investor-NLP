import json
import threading
from typing import Any

_lock = threading.Lock()
_active = False
_store: dict[str, Any] = {}


def start() -> None:
    global _active, _store
    with _lock:
        _active = True
        _store = {}


def record(key: str, json_str: str) -> None:
    global _store
    with _lock:
        if not _active:
            return
        try:
            _store[key] = json.loads(json_str)
        except Exception:
            _store[key] = json_str


def collect() -> dict[str, Any]:
    global _active, _store
    with _lock:
        _active = False
        return dict(_store)
