"""Safe, exact serialisation of training state: a JSON manifest plus a tensor-only file (AS-576).

Why not pickle
--------------
`torch.load(..., weights_only=False)` unpickles arbitrary objects, so a crafted checkpoint executes
code when it is opened. Checkpoints here never contain pickled objects:

- every tensor (and every NumPy array) goes into one flat `dict[str, Tensor]` saved with `torch.save`
  and read back with `torch.load(weights_only=True)`, which only rebuilds tensors and plain containers;
- the structure around the tensors (dicts, lists, dataclasses, the stream carry, optimiser state, RNG
  states) is encoded as JSON with explicit type tags, and rebuilt by `decode` without running any
  constructor of a class: objects are created with `object.__new__` and their attribute dictionary is
  filled directly, and only classes of the `nagahana` package (or registered explicitly) are rebuilt;
- the file's SHA-256 digest is written to a sidecar file and checked *before* the file is opened, so a
  truncated or altered checkpoint is refused before deserialisation starts.

File layout
-----------
    <name>          torch.save({"__manifest__": uint8 tensor of the UTF-8 JSON manifest, "t0": ..., "t1": ...})
    <name>.sha256   "<hex digest>  <file name>\\n"

Both are written atomically: to a temporary file in the same directory, flushed and fsynced, then
moved into place with `os.replace`; the data file first and the sidecar second, so a file without a
sidecar is an incomplete write and is never loaded.

Type tags of the manifest (JSON values are encoded as themselves when unambiguous)
---------------------------------------------------------------------------------
    {"$f": "inf" | "-inf" | "nan"}           non-finite float (JSON has none)
    {"$tuple": [...]}  {"$set": [...]}  {"$frozenset": [...]}  {"$deque": [...], "maxlen": n}
    {"$dict": [[key, value], ...]}         any mapping (keys may be ints, tuples, ...)
    {"$odict": [[k, v], ...], "meta": m}   OrderedDict (a state dict; `_metadata` kept)
    {"$tensor": key, "param": bool, "grad": bool}
    {"$ndarray": key, "dtype": str} {"$ndarray_obj": [...], "shape": [...]} {"$ndarray_str": [...], "dtype": str}
    {"$npscalar": value, "dtype": str}  {"$device": str}  {"$dtype": str}  {"$bytes": hex}
    {"$enum": "module:Class", "value": v}
    {"$dataclass": "module:Class", "fields": {...}}   {"$object": "module:Class", "state": {...}}
    {"$dataframe": {"columns": [...], "data": [...], "index": v}}

Invariant (tests/test_training_checkpoint.py): decode(encode(x)) == x for every supported type,
tensors bit-identical with dtype and shape; a tampered file is refused before `torch.load` runs.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import hashlib
import importlib
import io
import json
import math
import os
import tempfile
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation

MANIFEST_KEY = "__manifest__"
FORMAT = "nagahana-safe-state-v1"
#: Modules whose classes `decode` may rebuild (by prefix). Registered types extend this explicitly.
TRUSTED_PREFIXES: tuple[str, ...] = ("nagahana.",)
_REGISTERED: dict[str, type] = {}


def register_type(cls: type) -> type:
    """Allow `decode` to rebuild instances of `cls` even outside the trusted prefixes."""
    _REGISTERED[f"{cls.__module__}:{cls.__qualname__}"] = cls
    return cls


def _class_path(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_class(path: str) -> type:
    """The class named "module:Qual.Name", refused unless trusted (module docstring)."""
    if path in _REGISTERED:
        return _REGISTERED[path]
    module_name, _, qual = path.partition(":")
    if not qual or not any(module_name.startswith(p) for p in TRUSTED_PREFIXES):
        raise InvariantViolation(f"refusing to rebuild {path!r}: not a trusted class")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qual.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            raise InvariantViolation(f"refusing to rebuild {path!r}: no such class")
    if not isinstance(obj, type):
        raise InvariantViolation(f"refusing to rebuild {path!r}: not a class")
    return obj


class Encoder:
    """Encode an object graph into (JSON-able tree, tensor table)."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}

    def _put(self, t: torch.Tensor) -> str:
        key = f"t{len(self.tensors)}"
        # A detached, contiguous CPU tensor that owns exactly its bytes: torch.save writes a tensor's
        # whole storage, so a view into a larger storage is copied first (the file then holds only the
        # tensor's own values); a tensor that already owns its storage is saved without a copy.
        x = t.detach().to("cpu").contiguous()
        if x.storage_offset() != 0 or x.untyped_storage().nbytes() != x.numel() * x.element_size():
            x = x.clone()
        self.tensors[key] = x
        return key

    def encode(self, obj: Any) -> Any:
        """The JSON tree of `obj` (tensors moved into `self.tensors`)."""
        if obj is None or isinstance(obj, bool | str):
            return obj
        if isinstance(obj, enum.Enum):
            return {"$enum": _class_path(obj), "value": self.encode(obj.value)}
        if isinstance(obj, int):
            return int(obj)
        if isinstance(obj, float):
            if math.isfinite(obj):
                return float(obj)
            return {"$f": "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")}
        if isinstance(obj, torch.Tensor):
            return {"$tensor": self._put(obj), "param": isinstance(obj, torch.nn.Parameter),
                    "grad": bool(obj.requires_grad) if isinstance(obj, torch.nn.Parameter) else False}
        if isinstance(obj, np.ndarray):
            return self._ndarray(obj)
        if isinstance(obj, np.generic):
            return {"$npscalar": self.encode(obj.item()), "dtype": str(obj.dtype)}
        if isinstance(obj, torch.device):
            return {"$device": str(obj)}
        if isinstance(obj, torch.dtype):
            return {"$dtype": str(obj).removeprefix("torch.")}
        if isinstance(obj, bytes | bytearray):
            return {"$bytes": bytes(obj).hex()}
        if isinstance(obj, tuple) and not hasattr(obj, "_fields"):
            return {"$tuple": [self.encode(x) for x in obj]}
        if isinstance(obj, list):
            return [self.encode(x) for x in obj]
        if isinstance(obj, collections.deque):
            return {"$deque": [self.encode(x) for x in obj], "maxlen": obj.maxlen}
        if isinstance(obj, frozenset | set):
            items = list(obj)
            try:
                items = sorted(items)
            except TypeError:
                pass
            tag = "$frozenset" if isinstance(obj, frozenset) else "$set"
            return {tag: [self.encode(x) for x in items]}
        if isinstance(obj, collections.OrderedDict):
            meta = getattr(obj, "_metadata", None)
            return {"$odict": [[self.encode(k), self.encode(v)] for k, v in obj.items()],
                    "meta": self.encode(dict(meta)) if meta is not None else None}
        if isinstance(obj, dict):
            return {"$dict": [[self.encode(k), self.encode(v)] for k, v in obj.items()]}
        if _is_dataframe(obj):
            return self._dataframe(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {"$dataclass": _class_path(obj),
                    "fields": {f.name: self.encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}}
        if isinstance(obj, types.FunctionType | types.MethodType | types.BuiltinFunctionType | type):
            raise InvariantViolation(f"refusing to serialise a callable or class ({obj!r}): state must be data")
        state = getattr(obj, "__dict__", None)
        if state is None:
            raise InvariantViolation(f"cannot serialise {type(obj).__name__}: no data attributes")
        return {"$object": _class_path(obj), "state": {k: self.encode(v) for k, v in state.items()}}

    def _ndarray(self, a: np.ndarray) -> Any:
        if a.dtype == object:
            return {"$ndarray_obj": [self.encode(x) for x in a.reshape(-1).tolist()], "shape": list(a.shape)}
        if a.dtype.kind in ("U", "S"):
            return {"$ndarray_str": a.reshape(-1).tolist() if a.dtype.kind == "U" else [x.hex() for x in a.reshape(-1).tolist()],
                    "dtype": a.dtype.str, "shape": list(a.shape)}
        # A read-only array (for example a pandas-owned buffer) is copied: torch tensors are writable.
        arr = np.ascontiguousarray(a) if a.flags.writeable else np.array(a, copy=True, order="C")
        return {"$ndarray": self._put(torch.from_numpy(arr)), "dtype": a.dtype.str}

    def _dataframe(self, df: Any) -> Any:
        import pandas as pd

        index = None
        if not isinstance(df.index, pd.RangeIndex) or df.index.start != 0 or df.index.step != 1:
            index = self.encode(df.index.to_numpy())
        cols = [str(c) for c in df.columns]
        if len(set(cols)) != len(cols) or any(not isinstance(c, str) for c in df.columns):
            raise InvariantViolation("data frames with non-string or duplicate column names are not serialised")
        data = []
        for c in df.columns:
            s = df[c]
            dtype = str(s.dtype)
            data.append({"dtype": dtype, "values": self.encode(s.to_numpy(dtype=object if dtype == "string" else None))})
        return {"$dataframe": {"columns": cols, "data": data, "index": index, "n": int(len(df))}}


def _is_dataframe(obj: Any) -> bool:
    # isinstance, not the class's module name: pandas 3 reports `DataFrame.__module__` as "pandas".
    import pandas as pd

    return isinstance(obj, pd.DataFrame)


class Decoder:
    """Rebuild an object graph from (JSON tree, tensor table) without running class constructors."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def _tensor(self, key: str) -> torch.Tensor:
        if key not in self.tensors:
            raise InvariantViolation(f"manifest names tensor {key!r}, which the file does not hold")
        return self.tensors[key]

    def decode(self, node: Any) -> Any:
        """The object encoded by `node`."""
        if node is None or isinstance(node, bool | int | float | str):
            return node
        if isinstance(node, list):
            return [self.decode(x) for x in node]
        if not isinstance(node, dict) or len(node) == 0:
            raise InvariantViolation(f"malformed manifest node {node!r}")
        tag = next(iter(node))
        handler = _HANDLERS.get(tag)
        if handler is None:
            raise InvariantViolation(f"unknown manifest tag {tag!r}")
        return handler(self, node)


def _dec_float(d: Decoder, n: dict[str, Any]) -> float:
    return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[n["$f"]]


def _dec_tensor(d: Decoder, n: dict[str, Any]) -> torch.Tensor:
    t = d._tensor(n["$tensor"])
    if n.get("param"):
        return torch.nn.Parameter(t, requires_grad=bool(n.get("grad")))
    return t


def _dec_ndarray(d: Decoder, n: dict[str, Any]) -> np.ndarray:
    a = d._tensor(n["$ndarray"]).numpy()
    return a.astype(np.dtype(n["dtype"]), copy=False)


def _dec_ndarray_obj(d: Decoder, n: dict[str, Any]) -> np.ndarray:
    items = [d.decode(x) for x in n["$ndarray_obj"]]
    out = np.empty(len(items), dtype=object)
    out[:] = items
    return out.reshape(n["shape"])


def _dec_ndarray_str(d: Decoder, n: dict[str, Any]) -> np.ndarray:
    dt = np.dtype(n["dtype"])
    vals = n["$ndarray_str"] if dt.kind == "U" else [bytes.fromhex(x) for x in n["$ndarray_str"]]
    return np.asarray(vals, dtype=dt).reshape(n["shape"])


def _dec_npscalar(d: Decoder, n: dict[str, Any]) -> Any:
    return np.dtype(n["dtype"]).type(d.decode(n["$npscalar"]))


def _dec_dtype(d: Decoder, n: dict[str, Any]) -> torch.dtype:
    dt = getattr(torch, n["$dtype"], None)
    if not isinstance(dt, torch.dtype):
        raise InvariantViolation(f"unknown torch dtype {n['$dtype']!r}")
    return dt


def _dec_dict(d: Decoder, n: dict[str, Any]) -> dict[Any, Any]:
    return {d.decode(k): d.decode(v) for k, v in n["$dict"]}


def _dec_odict(d: Decoder, n: dict[str, Any]) -> collections.OrderedDict[Any, Any]:
    out: collections.OrderedDict[Any, Any] = collections.OrderedDict((d.decode(k), d.decode(v)) for k, v in n["$odict"])
    if n.get("meta") is not None:
        out._metadata = d.decode(n["meta"])  # type: ignore[attr-defined]
    return out


def _dec_enum(d: Decoder, n: dict[str, Any]) -> Any:
    cls = _resolve_class(n["$enum"])
    if not issubclass(cls, enum.Enum):
        raise InvariantViolation(f"{n['$enum']} is not an enum")
    return cls(d.decode(n["value"]))


def _new_instance(cls: type) -> Any:
    """`object.__new__(cls)`, refused for classes that customise instance creation."""
    if cls.__new__ is not object.__new__:
        raise InvariantViolation(f"refusing to rebuild {cls.__qualname__}: it customises __new__")
    return object.__new__(cls)


def _set_state(obj: Any, state: dict[str, Any]) -> None:
    # The attribute dictionary is filled directly: no __setattr__, property or descriptor code runs.
    if not hasattr(obj, "__dict__"):
        raise InvariantViolation(f"{type(obj).__qualname__} has no attribute dictionary (slots are not rebuilt)")
    obj.__dict__.update(state)


def _dec_dataclass(d: Decoder, n: dict[str, Any]) -> Any:
    cls = _resolve_class(n["$dataclass"])
    if not dataclasses.is_dataclass(cls):
        raise InvariantViolation(f"{n['$dataclass']} is not a dataclass")
    expected = {f.name for f in dataclasses.fields(cls)}
    got = set(n["fields"])
    if got != expected:
        raise InvariantViolation(f"{n['$dataclass']}: fields {sorted(got ^ expected)} differ from the class")
    obj = _new_instance(cls)
    _set_state(obj, {k: d.decode(v) for k, v in n["fields"].items()})
    return obj


def _dec_object(d: Decoder, n: dict[str, Any]) -> Any:
    cls = _resolve_class(n["$object"])
    obj = _new_instance(cls)
    _set_state(obj, {k: d.decode(v) for k, v in n["state"].items()})
    return obj


def _dec_dataframe(d: Decoder, n: dict[str, Any]) -> Any:
    import pandas as pd

    spec = n["$dataframe"]
    cols: dict[str, Any] = {}
    for name, col in zip(spec["columns"], spec["data"], strict=True):
        values = d.decode(col["values"])
        cols[name] = pd.array(values, dtype=col["dtype"]) if col["dtype"] in ("string", "category") else values
    df = pd.DataFrame(cols, columns=spec["columns"])
    if len(df) != spec["n"] and spec["columns"]:
        raise InvariantViolation("data frame row count does not match its manifest")
    if not spec["columns"]:
        df = pd.DataFrame(index=pd.RangeIndex(spec["n"]))
    if spec["index"] is not None:
        df.index = d.decode(spec["index"])
    for name, col in zip(spec["columns"], spec["data"], strict=True):
        if col["dtype"] not in ("string", "category", "object") and str(df[name].dtype) != col["dtype"]:
            df[name] = df[name].astype(col["dtype"])
    return df


_HANDLERS: dict[str, Callable[[Decoder, dict[str, Any]], Any]] = {
    "$f": _dec_float,
    "$tuple": lambda d, n: tuple(d.decode(x) for x in n["$tuple"]),
    "$set": lambda d, n: {d.decode(x) for x in n["$set"]},
    "$frozenset": lambda d, n: frozenset(d.decode(x) for x in n["$frozenset"]),
    "$deque": lambda d, n: collections.deque((d.decode(x) for x in n["$deque"]), maxlen=n["maxlen"]),
    "$dict": _dec_dict,
    "$odict": _dec_odict,
    "$tensor": _dec_tensor,
    "$ndarray": _dec_ndarray,
    "$ndarray_obj": _dec_ndarray_obj,
    "$ndarray_str": _dec_ndarray_str,
    "$npscalar": _dec_npscalar,
    "$device": lambda d, n: torch.device(n["$device"]),
    "$dtype": _dec_dtype,
    "$bytes": lambda d, n: bytes.fromhex(n["$bytes"]),
    "$enum": _dec_enum,
    "$dataclass": _dec_dataclass,
    "$object": _dec_object,
    "$dataframe": _dec_dataframe,
}


def sha256_file(path: str | Path, *, chunk: int = 1 << 22) -> str:
    """Hex SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _fsync_dir(directory: Path) -> None:
    # Make the rename durable on POSIX file systems; Windows has no directory handles for fsync.
    if os.name == "nt":
        return
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes | Callable[[io.BufferedWriter], None]) -> None:
    """Write `data` (bytes, or a function writing to a binary file) to `path` atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            if callable(data):
                data(fh)  # type: ignore[arg-type]
            else:
                fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        _fsync_dir(target.parent)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text atomically."""
    atomic_write_bytes(path, text.encode("utf-8"))


def sidecar(path: str | Path) -> Path:
    """The digest sidecar of a state file."""
    p = Path(path)
    return p.with_name(p.name + ".sha256")


class _HashingWriter:
    """A binary file wrapper that hashes every byte written (torch.save writes its archive sequentially)."""

    def __init__(self, fh: io.BufferedWriter) -> None:
        self.fh = fh
        self.sha = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        self.sha.update(data)
        self.size += len(data)
        return self.fh.write(data)

    def flush(self) -> None:
        self.fh.flush()


def save_state(path: str | Path, obj: Any, *, kind: str, meta: dict[str, Any] | None = None) -> str:
    """Write `obj` as a safe state file (module docstring). Returns the file's SHA-256 digest.

    The archive is streamed to a temporary file while it is hashed, so a large state (the full
    optimiser state of the L model) is never held twice in memory.
    """
    enc = Encoder()
    tree = enc.encode(obj)
    manifest = {"format": FORMAT, "kind": kind, "meta": enc.encode(dict(meta or {})), "tree": tree,
                "tensors": sorted(enc.tensors)}
    text = json.dumps(manifest, separators=(",", ":"), allow_nan=False)
    payload: dict[str, torch.Tensor] = {MANIFEST_KEY: torch.frombuffer(bytearray(text.encode("utf-8")), dtype=torch.uint8)}
    payload.update(enc.tensors)
    holder: dict[str, str] = {}

    def write(fh: io.BufferedWriter) -> None:
        w = _HashingWriter(fh)
        torch.save(payload, w)  # type: ignore[arg-type]
        holder["digest"] = w.sha.hexdigest()

    atomic_write_bytes(path, write)
    digest = holder["digest"]
    atomic_write_text(sidecar(path), f"{digest}  {Path(path).name}\n")
    return digest


def verify_state(path: str | Path) -> str:
    """Check the sidecar digest of a state file; returns the digest. Raises before anything is unpickled."""
    p = Path(path)
    side = sidecar(p)
    if not p.is_file():
        raise InvariantViolation(f"{p}: no such state file")
    if not side.is_file():
        raise InvariantViolation(f"{p}: no digest sidecar ({side.name}); an incomplete write is never loaded")
    recorded = side.read_text(encoding="utf-8").split()
    if not recorded or len(recorded[0]) != 64:
        raise InvariantViolation(f"{side}: malformed digest")
    actual = sha256_file(p)
    if actual != recorded[0]:
        raise InvariantViolation(f"{p}: SHA-256 {actual[:16]}... does not match its sidecar {recorded[0][:16]}... "
                                 "(corrupt or altered file); refusing to load it")
    return actual


def load_state(path: str | Path, *, kind: str | None = None, map_location: str | torch.device = "cpu",
               mmap: bool = False) -> tuple[Any, dict[str, Any]]:
    """Read a safe state file: verify its digest, load tensors with weights_only=True, rebuild. Returns (obj, meta)."""
    verify_state(path)
    payload = torch.load(str(path), map_location=map_location, weights_only=True, mmap=mmap)
    if not isinstance(payload, dict) or MANIFEST_KEY not in payload:
        raise InvariantViolation(f"{path}: not a NagaHana state file")
    manifest = json.loads(payload[MANIFEST_KEY].numpy().tobytes().decode("utf-8"))
    if manifest.get("format") != FORMAT:
        raise InvariantViolation(f"{path}: unknown state format {manifest.get('format')!r}")
    if kind is not None and manifest.get("kind") != kind:
        raise InvariantViolation(f"{path}: holds {manifest.get('kind')!r}, expected {kind!r}")
    tensors = {k: v for k, v in payload.items() if k != MANIFEST_KEY}
    if sorted(tensors) != manifest["tensors"]:
        raise InvariantViolation(f"{path}: tensor table does not match its manifest")
    dec = Decoder(tensors)
    meta = dec.decode(manifest["meta"])
    return dec.decode(manifest["tree"]), meta


def encode_json(obj: Any) -> str:
    """JSON text of a tensor-free object (the same tags; raises if a tensor or array is inside)."""
    enc = Encoder()
    tree = enc.encode(obj)
    if enc.tensors:
        raise InvariantViolation("encode_json holds no tensors; use save_state for tensor data")
    return json.dumps(tree, separators=(",", ":"), allow_nan=False)


def decode_json(text: str) -> Any:
    """Inverse of `encode_json`."""
    return Decoder({}).decode(json.loads(text))


__all__ = ["Decoder", "Encoder", "FORMAT", "atomic_write_bytes", "atomic_write_text", "decode_json", "encode_json",
           "load_state", "register_type", "save_state", "sha256_file", "sidecar", "verify_state"]
