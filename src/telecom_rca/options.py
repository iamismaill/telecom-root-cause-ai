"""Parse shuffled choices and map semantic C1-C8 diagnoses safely."""

from __future__ import annotations

from dataclasses import dataclass
import re


_OPTION = re.compile(r"^\s*([A-Z][1-9]?|[1-9])\s*[:.)]\s*(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Option:
    label: str
    description: str


def parse_options(question: str) -> tuple[Option, ...]:
    """Extract one-line multiple-choice options without consuming table rows."""
    options: list[Option] = []
    for line in question.splitlines():
        if "|" in line:
            continue
        match = _OPTION.match(line)
        if match:
            options.append(Option(match.group(1), match.group(2).strip()))
    return tuple(options)


_SIGNATURES: dict[str, tuple[tuple[str, ...], ...]] = {
    "C1": (("downtilt", "too large"), ("weak coverage", "far end")),
    "C2": (("coverage distance", "1km"), ("over-shooting",)),
    "C3": (("neighboring cell", "higher throughput"),),
    "C4": (("non-colocated", "co-frequency"), ("severe overlapping coverage",)),
    "C5": (("frequent handover",),),
    "C6": (("same pci mod 30",), ("pci modulo-30",), ("mod 30", "interference")),
    "C7": (("speed", "40km/h"), ("vehicle speed", "throughput")),
    "C8": (("scheduled rbs", "160"), ("resource blocks", "160")),
}


def semantic_cause(description: str) -> str | None:
    """Return the unique C1-C8 meaning of an option description, if present."""
    lowered = description.lower()
    matches = [
        label
        for label, signatures in _SIGNATURES.items()
        if any(all(term in lowered for term in signature) for signature in signatures)
    ]
    return matches[0] if len(matches) == 1 else None


def standard_option_map(question: str) -> dict[str, str]:
    """Map every recognized semantic C-label to its displayed label."""
    mapping: dict[str, str] = {}
    for option in parse_options(question):
        meaning = semantic_cause(option.description)
        if meaning is None:
            continue
        if meaning in mapping:
            raise ValueError(f"Duplicate semantic option for {meaning}")
        mapping[meaning] = option.label
    return mapping


def map_standard_cause(question: str, semantic_label: str) -> str:
    """Map a C1-C8 meaning to its displayed option label.

    Exact C-label options are accepted, but description signatures take
    precedence for shuffled numeric/letter choices.
    """
    options = parse_options(question)
    if not options:
        raise ValueError("No answer options were recognized")
    if semantic_label not in _SIGNATURES:
        raise ValueError(f"Unsupported standard root cause: {semantic_label}")
    mapping = standard_option_map(question)
    if semantic_label in mapping:
        return mapping[semantic_label]

    exact = [option.label for option in options if option.label.upper() == semantic_label]
    if len(exact) == 1:
        return exact[0]
    raise ValueError(f"No option description matches {semantic_label}")
