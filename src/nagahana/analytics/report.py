"""Analysis reports: scalar summaries, tables and plot-ready figure data, written as JSON, CSV and Markdown.

Every analysis of the package returns one `Report`. A report holds

    summary     scalar findings (counts, test statistics, p-values, estimates)
    tables      named pandas DataFrames (one row per field, per pair, per split, ...)
    figures     plot-ready data: a long-format DataFrame plus the names of its x, y and series columns
                and the kind of chart it is meant for. No image is rendered; any plotting tool can draw
                the figure from its CSV.
    notes       plain sentences that a reader must see next to the numbers (estimator caveats, inputs
                that were skipped and why)
    provenance  the configuration, the inputs and the seeds that produced the report

Encoding rules
--------------
JSON has no representation for NaN and infinities. A non-finite float is written as null (NaN), the
string "inf" or the string "-inf". Persistence diagrams use "inf" for classes that are still alive at
the end of a truncated filtration. NumPy scalars and arrays, pandas objects, dataclasses, enums, sets,
paths and bytes are converted to plain JSON types (bytes as lower-case hex).

CSV files: one per table (`<kind>__<table>.csv`) and one per figure (`<kind>__fig__<figure>.csv`).
Markdown: one file per report with the summary, every table (long tables are cut at `max_rows` rows
with a pointer to their CSV) and the list of figures with their columns.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Output formats a report can be written in.
FORMATS: tuple[str, ...] = ("json", "csv", "md")
#: Chart kinds a figure may declare (consumers dispatch on it).
FIGURE_KINDS: frozenset[str] = frozenset(
    {"line", "step", "scatter", "bar", "histogram", "heatmap", "diagram", "barcode", "area", "matrix"}
)
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(text: str) -> str:
    """A file-system-safe identifier: runs of characters outside [A-Za-z0-9_.-] become one underscore."""
    out = _SAFE.sub("_", str(text)).strip("_")
    if not out:
        raise ValueError(f"cannot form a file name from {text!r}")
    return out


def json_safe(obj: Any) -> Any:
    """Convert `obj` into plain JSON types (module docstring, "Encoding rules")."""
    # Floats first: the non-finite values have no JSON literal.
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        v = float(obj)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, enum.Enum):
        return json_safe(obj.value)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes | bytearray):
        return bytes(obj).hex()
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()] if obj.ndim <= 1 else [json_safe(r) for r in obj]
    if isinstance(obj, pd.DataFrame):
        return [{str(k): json_safe(v) for k, v in row.items()} for row in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return {str(k): json_safe(v) for k, v in obj.to_dict().items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: json_safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, set | frozenset):
        items = [json_safe(v) for v in obj]
        return sorted(items, key=lambda v: (str(type(v)), str(v)))
    if isinstance(obj, Iterable):
        return [json_safe(v) for v in obj]
    return str(obj)


def _fmt(value: Any, digits: int) -> str:
    """One Markdown cell: floats with `digits` significant digits, non-finite values spelled out."""
    if isinstance(value, float | np.floating):
        v = float(value)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.{digits}g}"
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(frame: pd.DataFrame, *, max_rows: int | None = None, digits: int = 6) -> str:
    """A GitHub-style pipe table of `frame` (index dropped), cut at `max_rows` rows when given."""
    shown = frame if max_rows is None or len(frame) <= max_rows else frame.iloc[:max_rows]
    cols = [str(c) for c in shown.columns]
    if not cols:
        return "(no columns)"
    lines = ["| " + " | ".join(_fmt(c, digits) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row) + " |")
    if len(shown) < len(frame):
        lines.append(f"\n({len(frame) - len(shown)} more rows in the CSV file)")
    return "\n".join(lines)


@dataclass
class Table:
    """A named table of results.

    Attributes
    ----------
    name : str
        Identifier, used in file names (made file-safe).
    frame : pandas.DataFrame
        The rows.
    title : str
        Human-readable heading.
    description : str
        What one row is and how its columns are defined.
    """

    name: str
    frame: pd.DataFrame
    title: str = ""
    description: str = ""


@dataclass
class Figure:
    """Plot-ready data for one chart (no image is rendered).

    Attributes
    ----------
    name : str
        Identifier, used in file names.
    kind : str
        One of `FIGURE_KINDS`.
    data : pandas.DataFrame
        Long-format data; one row per plotted point (or cell, for heatmaps).
    x, y : str
        Column names of the horizontal and vertical coordinates (for a heatmap: the two axes).
    series : str or None
        Column that separates several series (lines, bars, colours), if any.
    value : str or None
        Column with the cell value of a heatmap or matrix.
    title, x_label, y_label, description : str
        Text for the chart.
    log_x, log_y : bool
        Whether the axis is meant to be logarithmic.
    """

    name: str
    kind: str
    data: pd.DataFrame
    x: str
    y: str
    series: str | None = None
    value: str | None = None
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    description: str = ""
    log_x: bool = False
    log_y: bool = False

    def __post_init__(self) -> None:
        if self.kind not in FIGURE_KINDS:
            raise ValueError(f"figure kind {self.kind!r} is not one of {sorted(FIGURE_KINDS)}")
        for col in (self.x, self.y, self.series, self.value):
            if col is not None and col not in self.data.columns:
                raise ValueError(f"figure {self.name!r}: column {col!r} is not in its data {list(self.data.columns)}")


@dataclass
class Report:
    """The result of one analysis (module docstring)."""

    kind: str
    title: str
    summary: dict[str, Any] = field(default_factory=dict)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def add_table(self, name: str, frame: pd.DataFrame, *, title: str = "", description: str = "") -> Table:
        """Append a table and return it; names must be unique within the report."""
        if any(t.name == name for t in self.tables):
            raise ValueError(f"report {self.kind!r} already has a table named {name!r}")
        table = Table(name=name, frame=frame.reset_index(drop=True), title=title or name, description=description)
        self.tables.append(table)
        return table

    def add_figure(self, figure: Figure) -> Figure:
        """Append a figure; names must be unique within the report."""
        if any(f.name == figure.name for f in self.figures):
            raise ValueError(f"report {self.kind!r} already has a figure named {figure.name!r}")
        self.figures.append(figure)
        return figure

    def table(self, name: str) -> pd.DataFrame:
        """The frame of the table called `name` (KeyError if absent)."""
        for t in self.tables:
            if t.name == name:
                return t.frame
        raise KeyError(f"report {self.kind!r} has no table {name!r}; tables: {[t.name for t in self.tables]}")

    def figure(self, name: str) -> Figure:
        """The figure called `name` (KeyError if absent)."""
        for f in self.figures:
            if f.name == name:
                return f
        raise KeyError(f"report {self.kind!r} has no figure {name!r}; figures: {[f.name for f in self.figures]}")

    def merge(self, other: Report, *, prefix: str) -> None:
        """Fold `other` into this report; its names get `prefix` + "." so that nothing collides."""
        for k, v in other.summary.items():
            self.summary[f"{prefix}.{k}"] = v
        for t in other.tables:
            self.add_table(f"{prefix}.{t.name}", t.frame, title=t.title, description=t.description)
        for f in other.figures:
            self.add_figure(dataclasses.replace(f, name=f"{prefix}.{f.name}"))
        self.notes.extend(f"[{prefix}] {n}" for n in other.notes)
        if other.provenance:
            self.provenance[prefix] = other.provenance

    def to_dict(self) -> dict[str, Any]:
        """The whole report as plain JSON types."""
        return {
            "kind": self.kind,
            "title": self.title,
            "summary": json_safe(self.summary),
            "notes": list(self.notes),
            "provenance": json_safe(self.provenance),
            "tables": [
                {"name": t.name, "title": t.title, "description": t.description,
                 "columns": [str(c) for c in t.frame.columns], "rows": json_safe(t.frame)}
                for t in self.tables
            ],
            "figures": [
                {"name": f.name, "kind": f.kind, "title": f.title, "description": f.description,
                 "x": f.x, "y": f.y, "series": f.series, "value": f.value, "x_label": f.x_label,
                 "y_label": f.y_label, "log_x": f.log_x, "log_y": f.log_y,
                 "columns": [str(c) for c in f.data.columns], "rows": json_safe(f.data)}
                for f in self.figures
            ],
        }

    def to_json(self) -> str:
        """The report as a JSON document (strictly valid JSON: no NaN literals)."""
        return json.dumps(self.to_dict(), indent=2, allow_nan=False, ensure_ascii=True)

    def to_markdown(self, *, max_rows: int = 50, digits: int = 6) -> str:
        """The report as Markdown (module docstring)."""
        out = [f"# {self.title}", ""]
        if self.summary:
            out += ["## Summary", ""]
            flat = pd.DataFrame({"quantity": list(self.summary), "value": [_summary_cell(v) for v in self.summary.values()]})
            out += [markdown_table(flat, digits=digits), ""]
        if self.notes:
            out += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        for t in self.tables:
            out += [f"## {t.title}", ""]
            if t.description:
                out += [t.description, ""]
            out += [markdown_table(t.frame, max_rows=max_rows, digits=digits), ""]
        if self.figures:
            out += ["## Figures (plot-ready data)", ""]
            rows = [{"figure": f.name, "kind": f.kind, "x": f.x, "y": f.y, "series": f.series or "",
                     "points": len(f.data), "file": f"{safe_name(self.kind)}__fig__{safe_name(f.name)}.csv",
                     "description": f.description or f.title} for f in self.figures]
            out += [markdown_table(pd.DataFrame(rows), digits=digits), ""]
        return "\n".join(out).rstrip() + "\n"

    def write(self, out_dir: str | Path, formats: Sequence[str] = FORMATS) -> list[Path]:
        """Write the report in the given formats under `out_dir`; returns the files written."""
        bad = [f for f in formats if f not in FORMATS]
        if bad:
            raise ValueError(f"unknown output formats {bad}; known: {list(FORMATS)}")
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        stem = safe_name(self.kind)
        written: list[Path] = []
        if "json" in formats:
            p = root / f"{stem}.json"
            p.write_text(self.to_json(), encoding="utf-8")
            written.append(p)
        if "csv" in formats:
            for t in self.tables:
                p = root / f"{stem}__{safe_name(t.name)}.csv"
                t.frame.to_csv(p, index=False)
                written.append(p)
            for f in self.figures:
                p = root / f"{stem}__fig__{safe_name(f.name)}.csv"
                f.data.to_csv(p, index=False)
                written.append(p)
        if "md" in formats:
            p = root / f"{stem}.md"
            p.write_text(self.to_markdown(), encoding="utf-8")
            written.append(p)
        return written


def _summary_cell(value: Any) -> Any:
    """Summary values that are containers are shown as compact JSON in the Markdown table."""
    if isinstance(value, Mapping | list | tuple | np.ndarray | set | frozenset):
        return json.dumps(json_safe(value), allow_nan=False, ensure_ascii=True)
    return value


__all__ = [
    "FIGURE_KINDS", "FORMATS", "Figure", "Report", "Table", "json_safe", "markdown_table", "safe_name",
]
