"""Format-aware parser for telecom tables embedded in challenge questions."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

import pandas as pd


_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")


@dataclass(frozen=True)
class ParsedTable:
    """One parsed pipe-delimited table and its provenance."""

    kind: str
    frame: pd.DataFrame
    original_columns: tuple[str, ...]
    start_line: int
    malformed_rows: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedQuestion:
    """Structured parsing result; failures are explicit and auditable."""

    format_name: str
    tables: dict[str, ParsedTable] = field(default_factory=dict)
    unknown_tables: tuple[ParsedTable, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def is_telecom(self) -> bool:
        return "drive_test" in self.tables

    @property
    def parsed_cleanly(self) -> bool:
        return self.is_telecom and not self.errors


def _cells(line: str) -> list[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in text.split("|")]


def _is_separator_row(cells: Iterable[str]) -> bool:
    values = list(cells)
    return bool(values) and all(_SEPARATOR_CELL.fullmatch(v.replace(" ", "")) for v in values)


def _pipe_blocks(text: str) -> list[tuple[int, list[str]]]:
    """Extract contiguous groups of lines containing at least two pipes."""
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.count("|") >= 2:
            if not current:
                start = line_number
            current.append(line)
        elif current:
            blocks.append((start, current))
            current = []
    if current:
        blocks.append((start, current))
    return blocks


def _normalized_header(columns: Iterable[str]) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", col.lower()).strip() for col in columns)


def _classify_table(columns: list[str]) -> str:
    header = _normalized_header(columns)
    tokens = set(header.split())
    if "throughput" in header and ({"timestamp", "time"} & tokens):
        return "drive_test"
    if "gnodeb" in header and "pci" in tokens and (
        "tilt" in tokens or "downtilt" in tokens or "azimuth" in tokens
    ):
        return "engineering"
    if "gnodeb" in header and "neighbor" in header and (
        "a3" in header or "a5" in header or "handover" in header
    ):
        return "configuration"
    if ({"timestamp", "time"} & tokens) and ("event" in tokens or "handover" in header):
        return "signaling"
    return "unknown"


def _parse_block(start_line: int, lines: list[str]) -> ParsedTable | None:
    rows = [_cells(line) for line in lines]
    rows = [row for row in rows if row and not _is_separator_row(row)]
    if len(rows) < 2:
        return None

    columns = rows[0]
    width = len(columns)
    valid_rows: list[list[str]] = []
    malformed: list[str] = []
    for offset, row in enumerate(rows[1:], start=1):
        if len(row) != width:
            malformed.append(
                f"line {start_line + offset}: expected {width} cells, found {len(row)}"
            )
            continue
        valid_rows.append(row)

    frame = pd.DataFrame(valid_rows, columns=columns)
    return ParsedTable(
        kind=_classify_table(columns),
        frame=frame,
        original_columns=tuple(columns),
        start_line=start_line,
        malformed_rows=tuple(malformed),
    )


def parse_question(text: str) -> ParsedQuestion:
    """Parse all recognized tables from a question without hiding failures."""
    if not isinstance(text, str) or not text.strip():
        return ParsedQuestion(format_name="empty", errors=("Question is empty",))

    parsed = [
        table
        for start, lines in _pipe_blocks(text)
        if (table := _parse_block(start, lines)) is not None
    ]
    known: dict[str, ParsedTable] = {}
    unknown: list[ParsedTable] = []
    errors: list[str] = []

    for table in parsed:
        if table.malformed_rows:
            errors.extend(f"{table.kind}: {msg}" for msg in table.malformed_rows)
        if table.kind == "unknown":
            unknown.append(table)
        elif table.kind in known:
            errors.append(f"Duplicate {table.kind} table at line {table.start_line}")
        else:
            known[table.kind] = table

    if "drive_test" in known and "engineering" in known:
        format_name = "markdown" if text.lstrip().startswith("|") or "**Drive Test Data**" in text else "standard"
    elif "drive_test" in known:
        format_name = "partial_telecom"
        errors.append("Drive-test table found without engineering table")
    elif parsed:
        format_name = "unrecognized_tables"
        errors.append("Pipe tables found but no drive-test table recognized")
    else:
        format_name = "non_telecom"

    return ParsedQuestion(
        format_name=format_name,
        tables=known,
        unknown_tables=tuple(unknown),
        errors=tuple(errors),
    )
