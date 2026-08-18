"""Non-answering taxonomy for supplied general-knowledge test questions."""

from __future__ import annotations

import re


def categorize_general_question(question: str) -> str:
    """Assign a coarse subject category without solving the question."""
    text = question.lower()
    math_signals = (
        "$", "\\frac", "log_", "equation", "function", "polynomial", "remainder",
        "greatest common", "probability", "integer", "triangle", "square", "calculate",
        "value of", "how many", "divided by",
    )
    physics_signals = ("velocity", "acceleration", "force", "energy", "circuit", "voltage", "wave", "physics")
    chemistry_signals = ("element", "molecule", "chemical", "acid", "atomic", "reaction", "compound")
    biology_signals = ("cell", "organism", "gene", "species", "protein", "biology", "plant", "animal")
    humanities_signals = ("history", "war", "president", "country", "capital", "author", "philosophy", "government")
    if any(signal in text for signal in math_signals) or re.search(r"\d\s*[+*/^=-]\s*\d", text):
        return "mathematics"
    if any(signal in text for signal in physics_signals):
        return "physics"
    if any(signal in text for signal in chemistry_signals):
        return "chemistry"
    if any(signal in text for signal in biology_signals):
        return "biology"
    if any(signal in text for signal in humanities_signals):
        return "humanities"
    return "other"

