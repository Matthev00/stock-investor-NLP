"""Utilities package."""

import json
import math
from typing import Any


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Infinity float values with None so the result is valid JSON."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def dumps(obj: Any, **kwargs) -> str:
    """json.dumps wrapper that sanitizes NaN/Infinity before serialization."""
    return json.dumps(sanitize_for_json(obj), **kwargs)
