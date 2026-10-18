"""
Input Sanitizer: Hardens Verdict claim inputs against malformed, adversarial,
or excessively long strings (null bytes, control characters, multi-thousand char floods).
"""

from __future__ import annotations
import re
from typing import Any


def sanitize_claim(raw_claim: Any, max_length: int = 1000) -> str:
    """
    Sanitizes user-provided claim input:
    - Handles None or non-string inputs safely
    - Strips null bytes (\\x00) and unprintable ASCII control characters
    - Normalizes excessive whitespace and newlines
    - Safely truncates claims exceeding max_length on word boundary
    """
    if raw_claim is None:
        return ""

    text = str(raw_claim)
    # Strip null bytes and non-printable control characters (\x00-\x08, \x0b-\x0c, \x0e-\x1f, \x7f)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Guard against excessively long claims
    if len(text) > max_length:
        text = text[:max_length].rsplit(" ", 1)[0].strip()
    return text
