"""Two-pass local reasoning prompts for general-knowledge retention questions."""

from __future__ import annotations

from .options import parse_options
from .qwen import extract_boxed_answer


def normalize_option_text(value: str) -> str:
    """Normalize harmless presentation differences for exact option matching."""
    value = value.strip().strip('"\'').strip()
    if value.lower().startswith("final option text:"):
        value = value.split(":", 1)[1].strip().strip('"\'').strip()
    return " ".join(value.lower().split()).rstrip(".")


def resolve_exact_option_text(text: str, question: str) -> str:
    """Map an exact returned option description to its displayed label."""
    options = parse_options(question)
    normalized = normalize_option_text(text)
    matches = [
        option.label
        for option in options
        if normalize_option_text(option.description) == normalized
    ]
    if len(matches) != 1:
        raise ValueError(f"Returned text does not match exactly one offered option: {text!r}")
    return matches[0]


def resolve_boxed_choice(text: str, question: str) -> str:
    """Resolve a boxed label, or an exact boxed option value, to its label."""
    options = parse_options(question)
    allowed = {option.label for option in options}
    boxed = extract_boxed_answer(text)
    if boxed in allowed:
        return boxed
    normalized = " ".join(boxed.strip().lower().split())
    matches = [
        option.label
        for option in options
        if " ".join(option.description.strip().lower().split()) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Boxed value {boxed!r} is neither a label nor one exact offered value")


def deliberation_messages(question: str) -> list[dict[str, str]]:
    """Ask the required local model to solve before committing to a choice."""
    options = parse_options(question)
    if not options:
        raise ValueError("General-knowledge question has no recognized choices")
    return [
        {
            "role": "system",
            "content": (
                "You are a careful multiple-choice problem solver. Work through the problem "
                "step by step, check calculations and factual distinctions, and identify the "
                "single best offered choice. This is a private scratch analysis. Be concise: "
                "use at most six short reasoning sentences and finish with the chosen label."
            ),
        },
        {"role": "user", "content": question + "\n\nReason carefully before choosing."},
    ]


def repair_messages(question: str, failed_response: str) -> list[dict[str, str]]:
    """Repair syntax without changing the underlying multiple-choice task."""
    options = parse_options(question)
    labels = ", ".join(option.label for option in options)
    return [
        {
            "role": "system",
            "content": "Return exactly one boxed choice label and no other text.",
        },
        {
            "role": "user",
            "content": (
                question
                + f"\n\nAllowed labels: {labels}. A previous verifier response was malformed: "
                + repr(failed_response)
                + "\nRecheck the answer and output only \\boxed{CHOICE}."
            ),
        },
    ]


def option_text_selection_messages(question: str, deliberation: str) -> list[dict[str, str]]:
    """Select by copying option text so code—not the model—maps it to a label."""
    options = parse_options(question)
    option_text = "\n".join(f"{option.label}: {option.description}" for option in options)
    return [
        {
            "role": "system",
            "content": (
                "Recheck the problem and scratch analysis. Select the best offered option. "
                "Output only the exact option text after the colon, copied verbatim. Do not "
                "output its label, explanation, prefix, or punctuation not present in the option."
            ),
        },
        {
            "role": "user",
            "content": (
                "QUESTION:\n"
                + question
                + "\n\nSCRATCH ANALYSIS:\n"
                + deliberation
                + "\n\nOFFERED OPTIONS:\n"
                + option_text
                + "\n\nCopy only the exact text of the single best option."
            ),
        },
    ]


def verification_messages(question: str, deliberation: str) -> list[dict[str, str]]:
    """Independently verify the scratch work and emit one strict boxed choice."""
    options = parse_options(question)
    if not options:
        raise ValueError("General-knowledge question has no recognized choices")
    labels = ", ".join(option.label for option in options)
    example = options[0].label
    return [
        {
            "role": "system",
            "content": (
                "You verify multiple-choice solutions. Recheck the original question and the "
                "scratch analysis. Correct any mistake. Return exactly one boxed offered choice "
                "and no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                "ORIGINAL QUESTION:\n"
                + question
                + "\n\nSCRATCH ANALYSIS:\n"
                + deliberation
                + f"\n\nAllowed labels: {labels}. Your entire response must be exactly "
                + rf"\boxed{{CHOICE}}, for example \boxed{{{example}}}."
            ),
        },
    ]
