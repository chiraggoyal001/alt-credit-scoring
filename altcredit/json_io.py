from __future__ import annotations

import json
from pathlib import Path
from typing import Any


JSON_ENCODINGS = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")


def decode_json_bytes(data: bytes) -> str:
    """Decode JSON bytes from common upload and Windows redirection encodings."""
    errors: list[str] = []
    for encoding in JSON_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    joined = "; ".join(errors)
    raise ValueError(f"Could not decode JSON file as UTF-8 or UTF-16. Details: {joined}")


def loads_json_bytes(data: bytes) -> Any:
    return json.loads(decode_json_bytes(data))


def load_json_file(path: str | Path) -> Any:
    return loads_json_bytes(Path(path).read_bytes())
