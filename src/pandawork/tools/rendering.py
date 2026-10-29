"""Turning tool results into what the model sees: compact text and small images.

Formatting is deterministic (same result, same bytes), which keeps prompt caching and replay
reliable.
"""
import base64
import io
import json
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .base import ToolResult

DIGITS = 4  # significant decimals kept for floats


def format_value(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        # "+ 0.0" turns -0.0 (e.g. tiny noise rounded away) into 0.0, so it never prints "-0"
        return np.format_float_positional(float(value) + 0.0, precision=DIGITS, trim="-")
    if isinstance(value, np.ndarray):
        return format_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {format_value(v)}" for k, v in value.items()) + "}"
    return str(value)


def format_data(data: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {format_value(value)}" for key, value in data.items())


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def image_block(image: np.ndarray) -> dict:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def render_result(result: "ToolResult") -> list[dict]:
    if result.error is not None:
        lines = [f"error [{result.error.code}]: {result.error.message}"]
        if result.error.hint:
            lines.append(f"hint: {result.error.hint}")
        if result.data:
            lines.append(format_data(result.data))
        text = "\n".join(lines)
    else:
        text = result.text if result.text is not None else (format_data(result.data) or "ok")
    return [text_block(text)] + [image_block(img) for img in result.images]


def clean_schema(schema: Any, parent_key: str | None = None) -> Any:
    """Drop pydantic's auto-generated "title" entries: they cost tokens and say nothing."""
    if isinstance(schema, dict):
        return {k: clean_schema(v, k) for k, v in schema.items()
                if not (k == "title" and parent_key != "properties")}
    if isinstance(schema, list):
        return [clean_schema(v) for v in schema]
    return schema


def to_json(value: Any) -> str:
    """Deterministic JSON for logging tool arguments and data (numpy-aware)."""
    def default(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(f"not JSON serialisable: {type(v).__name__}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)
