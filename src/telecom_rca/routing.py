"""Structural routing for the three challenge question families."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .options import Option, parse_options, standard_option_map
from .parser import ParsedQuestion, parse_question


class Route(str, Enum):
    STANDARD_TELECOM = "standard_telecom"
    MARKDOWN_TELECOM = "markdown_telecom"
    GENERAL_KNOWLEDGE = "general_knowledge"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    parsed: ParsedQuestion
    options: tuple[Option, ...]
    evidence: str


def route_question(question: str) -> RouteDecision:
    """Route by parsed tables and option structure, never loose keywords."""
    parsed = parse_question(question)
    options = parse_options(question)
    labels = {option.label.upper() for option in options}

    if parsed.is_telecom:
        if parsed.format_name == "markdown" and len(options) == 9 and len(labels) == 9:
            return RouteDecision(
                Route.MARKDOWN_TELECOM,
                parsed,
                options,
                "Markdown drive/engineering tables with a complete nine-choice option domain",
            )
        semantic_options = standard_option_map(question)
        if 2 <= len(options) <= 8 and len(semantic_options) == len(options):
            return RouteDecision(
                Route.STANDARD_TELECOM,
                parsed,
                options,
                "Drive/engineering tables with a recognized C1-C8 semantic option domain",
            )
        raise ValueError(
            f"Telecom tables were recognized but option structure is unsupported: "
            f"format={parsed.format_name}, labels={sorted(labels)}"
        )

    if parsed.format_name == "non_telecom" and 2 <= len(options) <= 9:
        return RouteDecision(
            Route.GENERAL_KNOWLEDGE,
            parsed,
            options,
            "No telecom tables and a valid multiple-choice option set",
        )
    raise ValueError(
        f"Question cannot be routed safely: format={parsed.format_name}, options={len(options)}"
    )
