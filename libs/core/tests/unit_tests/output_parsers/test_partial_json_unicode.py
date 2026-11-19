"""Regression tests for JSON streams split inside Unicode escapes.

Examples represent common tool-call data rather than only placeholder strings.
See RFC 8259 section 7 (JSON escapes) and Unicode's UTF-16 guidance.
"""

import json
import re

import pytest

from langchain_core.messages import AIMessageChunk
from langchain_core.utils.json import parse_partial_json


@pytest.mark.parametrize(
    ("domain", "field", "value"),
    [
        pytest.param("geocoding", "destination", "München", id="geocoding-munich"),
        pytest.param("shipping", "destination", "Zürich", id="shipping-zurich"),
        pytest.param("customer_crm", "customer_name", "José García", id="crm-spanish-name"),
        pytest.param("customer_crm", "customer_name", "François", id="crm-french-name"),
        pytest.param("localization", "city", "東京都", id="localization-japanese"),
        pytest.param("localization", "city", "القاهرة", id="localization-arabic"),
        pytest.param("localization", "city", "मुंबई", id="localization-devanagari"),
        pytest.param("billing", "currency_symbol", "€", id="billing-euro"),
        pytest.param("legal", "reference", "§ 12", id="legal-section"),
        pytest.param("analytics", "comparison", "≤ 0.05", id="analytics-comparison"),
        pytest.param("notifications", "message", "Launch 🚀", id="emoji-surrogate-pair"),
        pytest.param("support", "message", "谢谢 🙏", id="support-cjk-and-emoji"),
    ],
)
def test_streamed_unicode_escape_preserves_existing_field(
    domain: str, field: str, value: str
) -> None:
    """Any partial \uXXXX escape must not erase previously streamed content."""
    # ensure_ascii=True models providers that serialize non-ASCII as JSON escapes.
    encoded = json.dumps({field: value}, ensure_ascii=True)
    assert "\\u" in encoded, domain

    for match in re.finditer(r"\\u[0-9a-fA-F]{4}", encoded):
        for offset in range(2, 6):
            # End after \u, \uX, \uXX or \uXXX (all incomplete).
            chunk = encoded[: match.start() + offset]
            valid_prefix = encoded[: match.start()]
            expected = json.loads(valid_prefix + '"}')
            assert parse_partial_json(chunk) == expected, (domain, chunk)

        # The completed escape must be preserved, including surrogate code units.
        complete_chunk = encoded[: match.end()]
        expected_complete = json.loads(complete_chunk + '"}')
        assert parse_partial_json(complete_chunk) == expected_complete


def test_unicode_escape_in_nested_business_payload() -> None:
    args = '{"shipment": {"id": "PKG-123", "destination": "M\\u00'
    assert parse_partial_json(args) == {
        "shipment": {"id": "PKG-123", "destination": "M"}
    }


def test_escaped_backslash_is_not_a_unicode_escape() -> None:
    # Windows paths and regexes may contain literal backslash-u sequences.
    path = r"C:\users\u00fc"
    encoded = json.dumps({"file_path": path})
    assert parse_partial_json(encoded[:-2]) == {"file_path": path}
    assert parse_partial_json(encoded) == {"file_path": path}


def test_streamed_tool_call_retains_field_during_unicode_escape() -> None:
    # Exercise the consumer named in #41545, not just the utility itself.
    for args, expected in [
        ('{"city": "M', {"city": "M"}),
        ('{"city": "M\\u00', {"city": "M"}),
        ('{"city": "M\\u00fc', {"city": "Mü"}),
    ]:
        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "weather", "args": args, "id": "call-1", "index": 0}
            ],
        )
        assert chunk.tool_calls[0]["args"] == expected


def test_malformed_hex_does_not_consume_following_quote() -> None:
    # An invalid Unicode digit must not leave the string-state machine open.
    # Existing best-effort recovery may discard malformed data; it must not crash.
    malformed = '{"city": "M\\u00Z", "country": "DE"}'
    assert parse_partial_json(malformed) is not None
