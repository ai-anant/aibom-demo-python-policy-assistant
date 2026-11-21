"""Input/output guardrails: PII redaction and prompt-injection screening."""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){9,12}\d(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard your system prompt",
    "reveal your system prompt",
    "you are now",
)

_BLOCKED_TOPICS = (
    "how to launder money",
    "how to hide a data breach",
    "bypass the audit log",
)


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str = ""
    text: str = ""


def redact_pii(text: str) -> str:
    """Mask e-mail addresses, phone numbers and card-like digit runs."""

    text = _EMAIL.sub("[EMAIL_REDACTED]", text)
    text = _CARD.sub("[CARD_REDACTED]", text)
    text = _PHONE.sub("[PHONE_REDACTED]", text)
    return text


def screen_input(text: str) -> GuardrailResult:
    """Reject prompt-injection attempts and clearly disallowed requests."""

    lowered = text.lower()
    for marker in _INJECTION_MARKERS:
        if marker in lowered:
            return GuardrailResult(False, "prompt-injection pattern detected")
    for topic in _BLOCKED_TOPICS:
        if topic in lowered:
            return GuardrailResult(False, "request outside acceptable-use policy")
    return GuardrailResult(True, text=text)


def screen_output(text: str, redact: bool = True) -> GuardrailResult:
    """Clean model output before it is returned to the caller."""

    if redact:
        text = redact_pii(text)
    return GuardrailResult(True, text=text)


def moderate_with_openai(text: str) -> bool:
    """Call the OpenAI moderation endpoint; returns True when flagged.

    Only used when an API key is configured. The moderation model is a separate
    AI asset from the chat model and is reported as such in the AIBOM.
    """

    from openai import OpenAI

    client = OpenAI()
    result = client.moderations.create(
        model="omni-moderation-latest", input=text
    )
    return bool(result.results[0].flagged)
