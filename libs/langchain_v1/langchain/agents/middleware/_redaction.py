"""PII (Personally Identifiable Information) detection utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class PIIMatch:
    """Represents a detected PII value in content."""

    value: str
    start: int
    end: int
    pii_type: str


def detect_email(content: str) -> list[PIIMatch]:
    """Detect email addresses in content.

    Args:
        content: Text content to scan for email addresses.

    Returns:
        List of PIIMatch objects for each detected email.
    """
    # Use ASCII-only lookbehind/lookahead instead of \b so that detection
    # works when the email is immediately preceded or followed by a
    # non-ASCII character (CJK, Cyrillic, Arabic, accented Latin, etc.).
    # Python's \b treats all Unicode letters as \w, so there is no word
    # boundary between e.g. a Chinese character and the start of an
    # ASCII email local-part.
    pattern = r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9._%+\-])"
    return [
        PIIMatch(
            value=match.group(),
            start=match.start(),
            end=match.end(),
            pii_type="email",
        )
        for match in re.finditer(pattern, content)
    ]


def detect_credit_card(content: str) -> list[PIIMatch]:
    """Detect credit card numbers in content.

    Args:
        content: Text content to scan for credit card numbers.

    Returns:
        List of PIIMatch objects for each detected credit card.
    """
    pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    matches = []
    for match in re.finditer(pattern, content):
        matches.append(
            PIIMatch(
                value=match.group(),
                start=match.start(),
                end=match.end(),
                pii_type="credit_card",
            )
        )
    return matches


def detect_ip(content: str) -> list[PIIMatch]:
    """Detect IP addresses in content.

    Args:
        content: Text content to scan for IP addresses.

    Returns:
        List of PIIMatch objects for each detected IP address.
    """
    # Same Unicode-boundary fix as detect_email: use explicit digit/dot
    # lookbehind/lookahead rather than \b.
    ipv4_pattern = r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])"
    matches = []
    for match in re.finditer(ipv4_pattern, content):
        # Validate that each octet is 0-255
        octets = match.group().split(".")
        if all(0 <= int(o) <= 255 for o in octets):
            matches.append(
                PIIMatch(
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    pii_type="ip",
                )
            )
    return matches


def detect_mac_address(content: str) -> list[PIIMatch]:
    """Detect MAC addresses in content.

    Args:
        content: Text content to scan for MAC addresses.

    Returns:
        List of PIIMatch objects for each detected MAC address.
    """
    # Same Unicode-boundary fix: use hex-digit/colon lookbehind/lookahead.
    pattern = r"(?<![0-9A-Fa-f:])([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])"
    return [
        PIIMatch(
            value=match.group(),
            start=match.start(),
            end=match.end(),
            pii_type="mac_address",
        )
        for match in re.finditer(pattern, content)
    ]


def detect_url(content: str) -> list[PIIMatch]:
    """Detect URLs in content using regex and stdlib validation.

    Args:
        content: Text content to scan for URLs.

    Returns:
        List of PIIMatch objects for each detected URL.
    """
    from urllib.parse import urlparse

    matches = []

    # Match explicit scheme URLs
    scheme_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"
    for match in re.finditer(scheme_pattern, content):
        try:
            parsed = urlparse(match.group())
            if parsed.netloc:
                matches.append(
                    PIIMatch(
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        pii_type="url",
                    )
                )
        except Exception:
            pass

    # Match bare domain names starting with www.
    bare_pattern = (
        r"\b(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
        r"(?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?"
    )
    for match in re.finditer(bare_pattern, content):
        url = match.group()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            parsed = urlparse(url)
            if parsed.netloc and "." in parsed.netloc:
                # Avoid re-matching already-found scheme URLs
                if not any(
                    m.start <= match.start() < m.end for m in matches
                ):
                    matches.append(
                        PIIMatch(
                            value=match.group(),
                            start=match.start(),
                            end=match.end(),
                            pii_type="url",
                        )
                    )
        except Exception:
            pass

    return matches


_BUILTIN_DETECTORS = {
    "email": detect_email,
    "credit_card": detect_credit_card,
    "ip": detect_ip,
    "mac_address": detect_mac_address,
    "url": detect_url,
}


def get_detector(
    pii_type: str,
    detector: Optional[object] = None,
) -> object:
    """Get a detector function for the specified PII type.

    Args:
        pii_type: Type of PII to detect.
        detector: Optional custom detector or regex pattern. If ``None``, a
            built-in detector is used when available.

    Returns:
        A callable that accepts a string and returns a list of PIIMatch.

    Raises:
        ValueError: If an unknown PII type is specified without a custom
            detector or regex.
    """
    if detector is not None:
        if callable(detector):
            return detector
        if isinstance(detector, str):
            pattern = re.compile(detector)

            def regex_detector(content: str) -> list[PIIMatch]:
                return [
                    PIIMatch(
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        pii_type=pii_type,
                    )
                    for match in pattern.finditer(content)
                ]

            return regex_detector
        raise ValueError(
            f"detector must be a callable or regex string, got {type(detector)}"
        )

    if pii_type in _BUILTIN_DETECTORS:
        return _BUILTIN_DETECTORS[pii_type]

    raise ValueError(
        f"Unknown PII type '{pii_type}'. Provide a custom detector or use one of: "
        f"{list(_BUILTIN_DETECTORS)}"
    )


__all__ = [
    "PIIMatch",
    "detect_email",
    "detect_ip",
    "detect_mac_address",
    "detect_url",
    "detect_credit_card",
    "get_detector",
]
