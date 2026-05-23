import json
import threading
from typing import Any

_local = threading.local()


def start() -> None:
    """Activate capture for the current thread and reset the store."""
    _local.active = True
    _local.store = {}


def record(key: str, json_str: str) -> None:
    """Store a JSON string result under the given key (only when capture is active)."""
    if not getattr(_local, "active", False):
        return
    try:
        _local.store[key] = json.loads(json_str)
    except Exception:
        _local.store[key] = json_str


def collect() -> dict[str, Any]:
    """Return captured data and deactivate capture for the current thread."""
    _local.active = False
    return dict(getattr(_local, "store", {}))
