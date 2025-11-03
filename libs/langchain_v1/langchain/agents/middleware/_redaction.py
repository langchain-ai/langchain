"""Shared redaction utilities for middleware components."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from typing_extensions import TypedDict

RedactionStrategy = Literal["block", "redact", "mask", "hash"]
"""Supported strategies for handling detected sensitive values."""


class PIIMatch(TypedDict):
    """Represents an individual match of sensitive data."""

    type: str
    value: str
    start: int
    end: int


class PIIDetectionError(Exception):
    """Raised when configured to block on detected sensitive values."""

    def __init__(self, pii_type: str, matches: Sequence[PIIMatch]) -> None:
        """Initialize the exception with match context.

        Args:
            pii_type: Name of the detected sensitive type.
            matches: All matches that were detected for that type.
        """
        self.pii_type = pii_type
        self.matches = list(matches)
        count = len(matches)
        msg = f"Detected {count} instance(s) of {pii_type} in text content"
        super().__init__(msg)


Detector = Callable[[str], list[PIIMatch]]
"""Callable signature for detectors that locate sensitive values."""


def detect_email(content: str) -> list[PIIMatch]:
    """Detect email addresses in content."""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return [
        PIIMatch(
            type="email",
            value=match.group(),
            start=match.start(),
            end=match.end(),
        )
        for match in re.finditer(pattern, content)
    ]


def detect_credit_card(content: str) -> list[PIIMatch]:
    """Detect credit card numbers in content using Luhn validation."""
    pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    matches = []

    for match in re.finditer(pattern, content):
        card_number = match.group()
        if _passes_luhn(card_number):
            matches.append(
                PIIMatch(
                    type="credit_card",
                    value=card_number,
                    start=match.start(),
                    end=match.end(),
                )
            )

    return matches


def detect_ip(content: str) -> list[PIIMatch]:
    """Detect IPv4 or IPv6 addresses in content."""
    matches: list[PIIMatch] = []
    ipv4_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"

    for match in re.finditer(ipv4_pattern, content):
        ip_candidate = match.group()
        try:
            ipaddress.ip_address(ip_candidate)
        except ValueError:
            continue
        matches.append(
            PIIMatch(
                type="ip",
                value=ip_candidate,
                start=match.start(),
                end=match.end(),
            )
        )

    return matches


def detect_mac_address(content: str) -> list[PIIMatch]:
    """Detect MAC addresses in content."""
    pattern = r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"
    return [
        PIIMatch(
            type="mac_address",
            value=match.group(),
            start=match.start(),
            end=match.end(),
        )
        for match in re.finditer(pattern, content)
    ]


def detect_url(content: str) -> list[PIIMatch]:
    """Detect URLs in content using regex and stdlib validation."""
    matches: list[PIIMatch] = []

    # Pattern 1: URLs with scheme (http:// or https://)
    scheme_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"

    for match in re.finditer(scheme_pattern, content):
        url = match.group()
        result = urlparse(url)
        if result.scheme in ("http", "https") and result.netloc:
            matches.append(
                PIIMatch(
                    type="url",
                    value=url,
                    start=match.start(),
                    end=match.end(),
                )
            )

    # Pattern 2: URLs without scheme (www.example.com or example.com/path)
    # More conservative to avoid false positives
    bare_pattern = (
        r"\b(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
        r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?:/[^\s]*)?"
    )

    for match in re.finditer(bare_pattern, content):
        start, end = match.start(), match.end()
        # Skip if already matched with scheme
        if any(m["start"] <= start < m["end"] or m["start"] < end <= m["end"] for m in matches):
            continue

        url = match.group()
        # Only accept if it has a path or starts with www
        # This reduces false positives like "example.com" in prose
        if "/" in url or url.startswith("www."):
            # Add scheme for validation (required for urlparse to work correctly)
            test_url = f"http://{url}"
            result = urlparse(test_url)
            if result.netloc and "." in result.netloc:
                matches.append(
                    PIIMatch(
                        type="url",
                        value=url,
                        start=start,
                        end=end,
                    )
                )

    return matches


BUILTIN_DETECTORS: dict[str, Detector] = {
    "email": detect_email,
    "credit_card": detect_credit_card,
    "ip": detect_ip,
    "mac_address": detect_mac_address,
    "url": detect_url,
}
"""Registry of built-in detectors keyed by type name."""

_CARD_NUMBER_MIN_DIGITS = 13
_CARD_NUMBER_MAX_DIGITS = 19


def _passes_luhn(card_number: str) -> bool:
    """Validate credit card number using the Luhn checksum."""
    digits = [int(d) for d in card_number if d.isdigit()]
    if not _CARD_NUMBER_MIN_DIGITS <= len(digits) <= _CARD_NUMBER_MAX_DIGITS:
        return False

    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        value = digit
        if index % 2 == 1:
            value *= 2
            if value > 9:  # noqa: PLR2004
                value -= 9
        checksum += value
    return checksum % 10 == 0


def _apply_redact_strategy(content: str, matches: list[PIIMatch]) -> str:
    result = content
    for match in sorted(matches, key=lambda item: item["start"], reverse=True):
        replacement = f"[REDACTED_{match['type'].upper()}]"
        result = result[: match["start"]] + replacement + result[match["end"] :]
    return result


_UNMASKED_CHAR_NUMBER = 4
_IPV4_PARTS_NUMBER = 4


def _apply_mask_strategy(content: str, matches: list[PIIMatch]) -> str:
    result = content
    for match in sorted(matches, key=lambda item: item["start"], reverse=True):
        value = match["value"]
        pii_type = match["type"]
        if pii_type == "email":
            parts = value.split("@")
            if len(parts) == 2:  # noqa: PLR2004
                domain_parts = parts[1].split(".")
                masked = (
                    f"{parts[0]}@****.{domain_parts[-1]}"
                    if len(domain_parts) > 1
                    else f"{parts[0]}@****"
                )
            else:
                masked = "****"
        elif pii_type == "credit_card":
            digits_only = "".join(c for c in value if c.isdigit())
            separator = "-" if "-" in value else " " if " " in value else ""
            if separator:
                masked = (
                    f"****{separator}****{separator}****{separator}"
                    f"{digits_only[-_UNMASKED_CHAR_NUMBER:]}"
                )
            else:
                masked = f"************{digits_only[-_UNMASKED_CHAR_NUMBER:]}"
        elif pii_type == "ip":
            octets = value.split(".")
            masked = f"*.*.*.{octets[-1]}" if len(octets) == _IPV4_PARTS_NUMBER else "****"
        elif pii_type == "mac_address":
            separator = ":" if ":" in value else "-"
            masked = (
                f"**{separator}**{separator}**{separator}**{separator}**{separator}{value[-2:]}"
            )
        elif pii_type == "url":
            masked = "[MASKED_URL]"
        else:
            masked = (
                f"****{value[-_UNMASKED_CHAR_NUMBER:]}"
                if len(value) > _UNMASKED_CHAR_NUMBER
                else "****"
            )
        result = result[: match["start"]] + masked + result[match["end"] :]
    return result


def _apply_hash_strategy(content: str, matches: list[PIIMatch]) -> str:
    result = content
    for match in sorted(matches, key=lambda item: item["start"], reverse=True):
        digest = hashlib.sha256(match["value"].encode()).hexdigest()[:8]
        replacement = f"<{match['type']}_hash:{digest}>"
        result = result[: match["start"]] + replacement + result[match["end"] :]
    return result


def apply_strategy(
    content: str,
    matches: list[PIIMatch],
    strategy: RedactionStrategy,
) -> str:
    """Apply the configured strategy to matches within content."""
    if not matches:
        return content
    if strategy == "redact":
        return _apply_redact_strategy(content, matches)
    if strategy == "mask":
        return _apply_mask_strategy(content, matches)
    if strategy == "hash":
        return _apply_hash_strategy(content, matches)
    if strategy == "block":
        raise PIIDetectionError(matches[0]["type"], matches)
    msg = f"Unknown redaction strategy: {strategy}"
    raise ValueError(msg)


def resolve_detector(pii_type: str, detector: Detector | str | None) -> Detector:
    """Return a callable detector for the given configuration."""
    if detector is None:
        if pii_type not in BUILTIN_DETECTORS:
            msg = (
                f"Unknown PII type: {pii_type}. "
                f"Must be one of {list(BUILTIN_DETECTORS.keys())} or provide a custom detector."
            )
            raise ValueError(msg)
        return BUILTIN_DETECTORS[pii_type]
    if isinstance(detector, str):
        pattern = re.compile(detector)

        def regex_detector(content: str) -> list[PIIMatch]:
            return [
                PIIMatch(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
                for match in pattern.finditer(content)
            ]

        return regex_detector
    return detector


@dataclass(frozen=True)
class RedactionRule:
    """Configuration for handling a single PII type."""

    pii_type: str
    strategy: RedactionStrategy = "redact"
    detector: Detector | str | None = None

    def resolve(self) -> ResolvedRedactionRule:
        """Resolve runtime detector and return an immutable rule."""
        resolved_detector = resolve_detector(self.pii_type, self.detector)
        return ResolvedRedactionRule(
            pii_type=self.pii_type,
            strategy=self.strategy,
            detector=resolved_detector,
        )


@dataclass(frozen=True)
class ResolvedRedactionRule:
    """Resolved redaction rule ready for execution."""

    pii_type: str
    strategy: RedactionStrategy
    detector: Detector

    def apply(self, content: str) -> tuple[str, list[PIIMatch]]:
        """Apply this rule to content, returning new content and matches."""
        matches = self.detector(content)
        if not matches:
            return content, []
        updated = apply_strategy(content, matches, self.strategy)
        return updated, matches


__all__ = [
    "PIIDetectionError",
    "PIIMatch",
    "RedactionRule",
    "ResolvedRedactionRule",
    "apply_strategy",
    "detect_credit_card",
    "detect_email",
    "detect_ip",
    "detect_mac_address",
    "detect_url",
]
