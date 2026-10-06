"""Canonical JSON and SHA-256 digests for the Verifier's records (AS-830, AS-831).

A record that is hashed must have exactly one byte representation, otherwise the same content could
carry two digests and a chain could not be re-verified from its text. `canonical_json` therefore fixes
every free choice of JSON serialisation:

    keys sorted, separators "," and ":" without spaces, ASCII only (non-ASCII escaped),
    tuples written as lists, non-finite floats written as the tagged objects {"$float": "nan"},
    {"$float": "inf"} and {"$float": "-inf"} (plain JSON has no NaN or infinity),
    floats written by Python's shortest repr, which round-trips exactly (IEEE double).

`from_canonical` restores the tagged floats, so `from_canonical(canonical_json(x))` equals the
JSON-ready form of x. Only plain data is accepted (None, bool, int, float, str, lists, tuples and
mappings with string keys); anything else raises, so an object never reaches a digest through an
implicit and possibly unstable string conversion.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

_FLOAT_TAG = "$float"


def json_ready(obj: Any) -> Any:
    """Plain JSON data of `obj` with tuples as lists and non-finite floats tagged (module docstring)."""
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return {_FLOAT_TAG: "nan"}
        if math.isinf(obj):
            return {_FLOAT_TAG: "inf" if obj > 0 else "-inf"}
        return float(obj)
    if isinstance(obj, list | tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"canonical JSON needs string keys, got {k!r}")
            if k == _FLOAT_TAG:
                raise TypeError(f"the key {_FLOAT_TAG!r} is reserved for tagged floats")
            out[k] = json_ready(v)
        return out
    raise TypeError(f"canonical JSON accepts plain data only, got {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """The one canonical text of `obj` (module docstring)."""
    return json.dumps(json_ready(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _restore(obj: Any) -> Any:
    # Inverse of the float tagging of `json_ready`.
    if isinstance(obj, list):
        return [_restore(v) for v in obj]
    if isinstance(obj, dict):
        if set(obj) == {_FLOAT_TAG}:
            tag = obj[_FLOAT_TAG]
            if tag == "nan":
                return math.nan
            if tag == "inf":
                return math.inf
            if tag == "-inf":
                return -math.inf
            raise ValueError(f"unknown tagged float {tag!r}")
        return {k: _restore(v) for k, v in obj.items()}
    return obj


def from_canonical(text: str) -> Any:
    """Parse canonical JSON text, restoring tagged non-finite floats."""
    return _restore(json.loads(text))


def sha256_hex(data: str | bytes) -> str:
    """SHA-256 hex digest of bytes, or of the UTF-8 encoding of a string."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def digest_of(obj: Any) -> str:
    """SHA-256 of the canonical JSON of `obj`."""
    return sha256_hex(canonical_json(obj))


__all__ = ["canonical_json", "digest_of", "from_canonical", "json_ready", "sha256_hex"]
