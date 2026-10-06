"""Versioned, checksummed storage of fitted LR models (one .npz file, no pickled objects).

A bundle holds named NumPy arrays and one JSON header. The header records the format name and version,
every array's dtype, shape and SHA-256 digest, and one digest over all arrays: SHA-256 of the
concatenation, in sorted name order, of name, dtype string, shape and the C-ordered bytes of each array.
Loading recomputes every digest and refuses a bundle whose format, version or content does not match, so
a stored model cannot be silently altered or read by an incompatible version of the code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from nagahana.core.errors import InvariantViolation

BUNDLE_FORMAT = "nagahana.lr.model"
BUNDLE_VERSION = 1
_HEADER_KEY = "__header__"


def _digest(name: str, a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(a.dtype.str.encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _total(digests: dict[str, str]) -> str:
    h = hashlib.sha256()
    for name in sorted(digests):
        h.update(name.encode())
        h.update(digests[name].encode())
    return h.hexdigest()


def save_bundle(path: str | Path, header: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    """Write arrays and header; returns the overall digest."""
    if _HEADER_KEY in arrays:
        raise InvariantViolation(f"{_HEADER_KEY} is reserved")
    clean: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        a = np.asarray(value)
        if a.dtype == object:
            raise InvariantViolation(f"array {name} has dtype object; bundles store no pickled objects")
        clean[name] = a
    digests = {n: _digest(n, a) for n, a in clean.items()}
    full = dict(header)
    full.update({"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION,
                 "arrays": {n: {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": digests[n]} for n, a in clean.items()},
                 "sha256": _total(digests)})
    np.savez_compressed(Path(path), allow_pickle=False, **{_HEADER_KEY: np.asarray(json.dumps(full, default=_json_default))},
                        **clean)
    return str(full["sha256"])


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def load_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read and verify a bundle (module docstring)."""
    with np.load(Path(path), allow_pickle=False) as data:
        if _HEADER_KEY not in data.files:
            raise InvariantViolation(f"{path} is not an LR model bundle")
        header = json.loads(str(data[_HEADER_KEY]))
        arrays = {k: np.asarray(data[k]) for k in data.files if k != _HEADER_KEY}
    if header.get("format") != BUNDLE_FORMAT:
        raise InvariantViolation(f"{path}: format {header.get('format')!r}, expected {BUNDLE_FORMAT!r}")
    if int(header.get("version", -1)) != BUNDLE_VERSION:
        raise InvariantViolation(f"{path}: bundle version {header.get('version')}, this code reads version {BUNDLE_VERSION}")
    manifest = header["arrays"]
    if set(manifest) != set(arrays):
        raise InvariantViolation(f"{path}: the stored arrays do not match the manifest")
    digests = {}
    for name, a in arrays.items():
        d = _digest(name, a)
        if d != manifest[name]["sha256"]:
            raise InvariantViolation(f"{path}: checksum mismatch for array {name}")
        digests[name] = d
    if _total(digests) != header["sha256"]:
        raise InvariantViolation(f"{path}: overall checksum mismatch")
    return header, arrays


__all__ = ["BUNDLE_FORMAT", "BUNDLE_VERSION", "load_bundle", "save_bundle"]
