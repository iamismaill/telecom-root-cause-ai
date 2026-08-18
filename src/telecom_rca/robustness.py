"""Label-preserving transformations for generalization stress tests."""

from __future__ import annotations

import random
import re


_OPTION_LINE = re.compile(r"^\s*([A-Z][1-9]?|[1-9])\s*[:.)]\s*(.+?)\s*$", re.IGNORECASE)
_TIMESTAMP = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")


def shuffle_and_relabel_options(question: str, seed: int) -> str:
    """Shuffle option order and replace labels while preserving descriptions."""
    lines = question.splitlines()
    positions: list[int] = []
    descriptions: list[str] = []
    for index, line in enumerate(lines):
        if "|" in line:
            continue
        match = _OPTION_LINE.match(line)
        if match:
            positions.append(index)
            descriptions.append(match.group(2).strip())
    if len(positions) < 2:
        raise ValueError("Question has fewer than two relabelable options")
    rng = random.Random(seed)
    rng.shuffle(descriptions)
    labels = [str(index) for index in range(1, len(descriptions) + 1)]
    rng.shuffle(labels)
    for position, label, description in zip(positions, labels, descriptions):
        lines[position] = f"{label}: {description}"
    return "\n".join(lines)


def _pipe_ranges(lines: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, line in enumerate(lines):
        if line.count("|") >= 2:
            if start is None:
                start = index
        elif start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(lines)))
    return ranges


def swap_first_two_pipe_tables(question: str) -> str:
    """Swap the first two complete table blocks without editing their contents."""
    lines = question.splitlines()
    ranges = _pipe_ranges(lines)
    if len(ranges) < 2:
        raise ValueError("Question has fewer than two pipe tables")
    (a_start, a_end), (b_start, b_end) = ranges[:2]
    first = lines[a_start:a_end]
    second = lines[b_start:b_end]
    between = lines[a_end:b_start]
    return "\n".join(lines[:a_start] + second + between + first + lines[b_end:])


def normalize_pipe_spacing(question: str) -> str:
    """Add harmless spaces around pipe delimiters in every table row."""
    lines = []
    for line in question.splitlines():
        if line.count("|") >= 2:
            outer_left = line.lstrip().startswith("|")
            outer_right = line.rstrip().endswith("|")
            cells = line.strip().strip("|").split("|")
            rebuilt = " | ".join(cell.strip() for cell in cells)
            if outer_left:
                rebuilt = "| " + rebuilt
            if outer_right:
                rebuilt = rebuilt + " |"
            lines.append(rebuilt)
        else:
            lines.append(line)
    return "\n".join(lines)


def shift_dates(question: str) -> str:
    """Replace irrelevant calendar dates while retaining time and row order."""
    return _TIMESTAMP.sub("2030-01-15", question)


def reverse_table_data_rows(question: str) -> str:
    """Reverse table observations while keeping headers/alignment rows fixed."""
    lines = question.splitlines()
    for start, end in reversed(_pipe_ranges(lines)):
        first_data = start + 1
        if first_data < end and re.fullmatch(r"\s*\|?[:\-\s|]+\|?\s*", lines[first_data]):
            first_data += 1
        lines[first_data:end] = reversed(lines[first_data:end])
    return "\n".join(lines)


def rotate_table_columns(question: str) -> str:
    """Rotate every table's columns consistently across all of its rows."""
    lines = question.splitlines()
    for start, end in reversed(_pipe_ranges(lines)):
        header = [cell.strip() for cell in lines[start].strip().strip("|").split("|")]
        if len(header) < 3:
            continue
        order = list(range(1, len(header))) + [0]
        for line_index in range(start, end):
            raw = lines[line_index]
            cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
            if len(cells) != len(header):
                continue
            rebuilt = "|".join(cells[index] for index in order)
            if raw.lstrip().startswith("|"):
                rebuilt = "|" + rebuilt
            if raw.rstrip().endswith("|"):
                rebuilt += "|"
            lines[line_index] = rebuilt
    return "\n".join(lines)


def duplicate_table_data_rows(question: str) -> str:
    """Duplicate observations without changing their measured values or order."""
    lines = question.splitlines()
    for start, end in reversed(_pipe_ranges(lines)):
        first_data = start + 1
        if first_data < end and re.fullmatch(r"\s*\|?[:\-\s|]+\|?\s*", lines[first_data]):
            first_data += 1
        expanded: list[str] = []
        for line in lines[first_data:end]:
            expanded.extend([line, line])
        lines[first_data:end] = expanded
    return "\n".join(lines)


def anonymize_gnodeb_ids(question: str) -> str:
    """Rename site IDs consistently while preserving co-location equality."""
    lines = question.splitlines()
    ranges = _pipe_ranges(lines)
    mapping: dict[str, str] = {}
    for start, end in ranges:
        header = [cell.strip() for cell in lines[start].strip().strip("|").split("|")]
        if not header or re.sub(r"[^a-z0-9]", "", header[0].lower()) != "gnodebid":
            continue
        for index in range(start + 1, end):
            raw = lines[index]
            if set(raw.replace("|", "").replace(":", "").replace("-", "").replace(" ", "")) == set():
                continue
            outer_left = raw.lstrip().startswith("|")
            outer_right = raw.rstrip().endswith("|")
            cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
            if not cells:
                continue
            original = cells[0]
            mapping.setdefault(original, f"SITE_{len(mapping) + 1:03d}")
            cells[0] = mapping[original]
            rebuilt = "|".join(cells)
            if outer_left:
                rebuilt = "|" + rebuilt
            if outer_right:
                rebuilt += "|"
            lines[index] = rebuilt
    return "\n".join(lines)


def rename_supported_columns(question: str) -> str:
    """Rename standard columns to supported alternative-format aliases."""
    aliases = {
        "Timestamp": "Time",
        "5G KPI PCell RF Serving PCI": "Serving PCI",
        "5G KPI PCell RF Serving SS-RSRP [dBm]": "Serving RSRP(dBm)",
        "5G KPI PCell RF Serving SS-SINR [dB]": "Serving SINR(dB)",
        "5G KPI PCell Layer2 MAC DL Throughput [Mbps]": "Throughput(Mbps)",
        "5G KPI PCell Layer1 DL RB Num (Including 0)": "RB/slot",
        "Mechanical Downtilt": "Mech Tilt(deg)",
        "Digital Tilt": "Elec Tilt(deg)",
        "Height": "Ant Height(m)",
    }
    for rank in range(1, 6):
        aliases[
            f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {rank} PCI"
        ] = f"Neighbor {rank} PCI"
        aliases[
            f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {rank} Filtered Tx BRSRP [dBm]"
        ] = f"Neighbor {rank} RSRP(dBm)"
    lines = question.splitlines()
    for start, _ in _pipe_ranges(lines):
        cells = [cell.strip() for cell in lines[start].strip().strip("|").split("|")]
        changed = [aliases.get(cell, cell) for cell in cells]
        outer_left = lines[start].lstrip().startswith("|")
        outer_right = lines[start].rstrip().endswith("|")
        rebuilt = "|".join(changed)
        if outer_left:
            rebuilt = "|" + rebuilt
        if outer_right:
            rebuilt += "|"
        lines[start] = rebuilt
    return "\n".join(lines)


def drop_irrelevant_engineering_columns(question: str) -> str:
    """Remove unused engineering fields while retaining all diagnostic inputs."""
    ignored = {
        "cell id",
        "mechanical azimuth",
        "digital azimuth",
        "txrx mode",
        "max transmit power",
        "antenna model",
    }
    lines = question.splitlines()
    for start, end in reversed(_pipe_ranges(lines)):
        header = [cell.strip() for cell in lines[start].strip().strip("|").split("|")]
        normalized = [re.sub(r"[^a-z0-9]+", " ", cell.lower()).strip() for cell in header]
        if not normalized or normalized[0] != "gnodeb id" or "pci" not in normalized:
            continue
        keep = [index for index, name in enumerate(normalized) if name not in ignored]
        for line_index in range(start, end):
            raw = lines[line_index]
            outer_left = raw.lstrip().startswith("|")
            outer_right = raw.rstrip().endswith("|")
            cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
            if len(cells) != len(header):
                continue
            rebuilt = "|".join(cells[index] for index in keep)
            if outer_left:
                rebuilt = "|" + rebuilt
            if outer_right:
                rebuilt += "|"
            lines[line_index] = rebuilt
    return "\n".join(lines)


def combined_stress(question: str, seed: int) -> str:
    """Apply all independent harmless transformations deterministically."""
    transformed = shuffle_and_relabel_options(question, seed)
    transformed = swap_first_two_pipe_tables(transformed)
    transformed = normalize_pipe_spacing(transformed)
    transformed = shift_dates(transformed)
    transformed = anonymize_gnodeb_ids(transformed)
    transformed = rename_supported_columns(transformed)
    return drop_irrelevant_engineering_columns(transformed)
