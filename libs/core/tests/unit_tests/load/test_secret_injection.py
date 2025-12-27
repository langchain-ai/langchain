"""Tests for secret injection prevention in serialization.

Verify that user-provided data containing secret-like structures cannot be used to
extract environment variables during deserialization.
"""

import json
import os
import re
from typing import Any
from unittest import mock

import pytest
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.load import dumpd, dumps, load
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration

SENTINEL_ENV_VAR = "TEST_SECRET_INJECTION_VAR"
"""Sentinel value that should NEVER appear in serialized output."""

SENTINEL_VALUE = "LEAKED_SECRET_MEOW_12345"
"""Sentinel value that should NEVER appear in serialized output."""

MALICIOUS_SECRET_DICT: dict[str, Any] = {
    "lc": 1,
    "type": "secret",
    "id": [SENTINEL_ENV_VAR],
}
"""The malicious secret-like dict that tries to read the env var"""


@pytest.fixture(autouse=True)
def _set_sentinel_env_var() -> Any:
    """Set the sentinel env var for all tests in this module."""
    with mock.patch.dict(os.environ, {SENTINEL_ENV_VAR: SENTINEL_VALUE}):
        yield


def _assert_no_secret_leak(payload: Any) -> None:
    """Assert that serializing/deserializing payload doesn't leak the secret."""
    # First serialize
    serialized = dumps(payload)

    # Deserialize with secrets_from_env=True (the dangerous setting)
    deserialized = load(serialized, secrets_from_env=True)

    # Re-serialize to string
    reserialized = dumps(deserialized)

    assert SENTINEL_VALUE not in reserialized, (
        f"Secret was leaked! Found '{SENTINEL_VALUE}' in output.\n"
        f"Original payload type: {type(payload)}\n"
        f"Reserialized output: {reserialized[:500]}..."
    )

    assert SENTINEL_VALUE not in repr(deserialized), (
        f"Secret was leaked in deserialized object! Found '{SENTINEL_VALUE}'.\n"
        f"Deserialized: {deserialized!r}"
    )


class TestSerializableTopLevel:
    """Tests with `Serializable` objects at the top level."""

    def test_human_message_with_secret_in_content(self) -> None:
        """`HumanMessage` with secret-like dict in `content`."""
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": MALICIOUS_SECRET_DICT},
            ]
        )
        _assert_no_secret_leak(msg)

    def test_human_message_with_secret_in_additional_kwargs(self) -> None:
        """`HumanMessage` with secret-like dict in `additional_kwargs`."""
        msg = HumanMessage(
            content="Hello",
            additional_kwargs={"data": MALICIOUS_SECRET_DICT},
        )
        _assert_no_secret_leak(msg)

    def test_human_message_with_secret_in_nested_additional_kwargs(self) -> None:
        """`HumanMessage` with secret-like dict nested in `additional_kwargs`."""
        msg = HumanMessage(
            content="Hello",
            additional_kwargs={"nested": {"deep": MALICIOUS_SECRET_DICT}},
        )
        _assert_no_secret_leak(msg)

    def test_human_message_with_secret_in_list_in_additional_kwargs(self) -> None:
        """`HumanMessage` with secret-like dict in a list in `additional_kwargs`."""
        msg = HumanMessage(
            content="Hello",
            additional_kwargs={"items": [MALICIOUS_SECRET_DICT]},
        )
        _assert_no_secret_leak(msg)

    def test_ai_message_with_secret_in_response_metadata(self) -> None:
        """`AIMessage` with secret-like dict in respo`nse_metadata."""
        msg = AIMessage(
            content="Hello",
            response_metadata={"data": MALICIOUS_SECRET_DICT},
        )
        _assert_no_secret_leak(msg)

    def test_document_with_secret_in_metadata(self) -> None:
        """Document with secret-like dict in `metadata`."""
        doc = Document(
            page_content="Hello",
            metadata={"data": MALICIOUS_SECRET_DICT},
        )
        _assert_no_secret_leak(doc)

    def test_nested_serializable_with_secret(self) -> None:
        """`AIMessage` containing `dumpd(HumanMessage)` with secret in kwargs."""
        inner = HumanMessage(
            content="Hello",
            additional_kwargs={"secret": MALICIOUS_SECRET_DICT},
        )
        outer = AIMessage(
            content="Outer",
            additional_kwargs={"nested": [dumpd(inner)]},
        )
        _assert_no_secret_leak(outer)


class TestDictTopLevel:
    """Tests with plain dicts at the top level."""

    def test_dict_with_serializable_containing_secret(self) -> None:
        """Dict containing a `Serializable` with secret-like dict."""
        msg = HumanMessage(
            content="Hello",
            additional_kwargs={"data": MALICIOUS_SECRET_DICT},
        )
        payload = {"message": msg}
        _assert_no_secret_leak(payload)

    def test_dict_with_secret_no_serializable(self) -> None:
        """Dict with secret-like dict, no `Serializable` objects."""
        payload = {"data": MALICIOUS_SECRET_DICT}
        _assert_no_secret_leak(payload)

    def test_dict_with_nested_secret_no_serializable(self) -> None:
        """Dict with nested secret-like dict, no `Serializable` objects."""
        payload = {"outer": {"inner": MALICIOUS_SECRET_DICT}}
        _assert_no_secret_leak(payload)

    def test_dict_with_secret_in_list(self) -> None:
        """Dict with secret-like dict in a list."""
        payload = {"items": [MALICIOUS_SECRET_DICT]}
        _assert_no_secret_leak(payload)

    def test_dict_mimicking_lc_constructor_with_secret(self) -> None:
        """Dict that looks like an LC constructor containing a secret."""
        payload = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain_core", "messages", "ai", "AIMessage"],
            "kwargs": {
                "content": "Hello",
                "additional_kwargs": {"secret": MALICIOUS_SECRET_DICT},
            },
        }
        _assert_no_secret_leak(payload)


class TestPydanticModelTopLevel:
    """Tests with Pydantic models (non-`Serializable`) at the top level."""

    def test_pydantic_model_with_serializable_containing_secret(self) -> None:
        """Pydantic model containing a `Serializable` with secret-like dict."""

        class MyModel(BaseModel):
            message: Any

        msg = HumanMessage(
            content="Hello",
            additional_kwargs={"data": MALICIOUS_SECRET_DICT},
        )
        payload = MyModel(message=msg)
        _assert_no_secret_leak(payload)

    def test_pydantic_model_with_secret_dict(self) -> None:
        """Pydantic model containing a secret-like dict directly."""

        class MyModel(BaseModel):
            data: dict[str, Any]

        payload = MyModel(data=MALICIOUS_SECRET_DICT)
        _assert_no_secret_leak(payload)

        # Test treatment of "parsed" in additional_kwargs
        msg = AIMessage(content=[], additional_kwargs={"parsed": payload})
        gen = ChatGeneration(message=msg)
        _assert_no_secret_leak(gen)
        round_trip = load(dumpd(gen))
        assert MyModel(**(round_trip.message.additional_kwargs["parsed"])) == payload

    def test_pydantic_model_with_nested_secret(self) -> None:
        """Pydantic model with nested secret-like dict."""

        class MyModel(BaseModel):
            nested: dict[str, Any]

        payload = MyModel(nested={"inner": MALICIOUS_SECRET_DICT})
        _assert_no_secret_leak(payload)


class TestNonSerializableClassTopLevel:
    """Tests with classes at the top level."""

    def test_custom_class_with_serializable_containing_secret(self) -> None:
        """Custom class containing a `Serializable` with secret-like dict."""

        class MyClass:
            def __init__(self, message: Any) -> None:
                self.message = message

        msg = HumanMessage(
            content="Hello",
            additional_kwargs={"data": MALICIOUS_SECRET_DICT},
        )
        payload = MyClass(message=msg)
        # This will serialize as not_implemented, but let's verify no leak
        _assert_no_secret_leak(payload)

    def test_custom_class_with_secret_dict(self) -> None:
        """Custom class containing a secret-like dict directly."""

        class MyClass:
            def __init__(self, data: dict[str, Any]) -> None:
                self.data = data

        payload = MyClass(data=MALICIOUS_SECRET_DICT)
        _assert_no_secret_leak(payload)


class TestDumpdInKwargs:
    """Tests for the specific pattern of `dumpd()` result stored in kwargs."""

    def test_dumpd_human_message_in_ai_message_kwargs(self) -> None:
        """`AIMessage` with `dumpd(HumanMessage)` in `additional_kwargs`."""
        h = HumanMessage("Hello")
        a = AIMessage("foo", additional_kwargs={"bar": [dumpd(h)]})
        _assert_no_secret_leak(a)

    def test_dumpd_human_message_with_secret_in_ai_message_kwargs(self) -> None:
        """`AIMessage` with `dumpd(HumanMessage w/ secret)` in `additional_kwargs`."""
        h = HumanMessage(
            "Hello",
            additional_kwargs={"secret": MALICIOUS_SECRET_DICT},
        )
        a = AIMessage("foo", additional_kwargs={"bar": [dumpd(h)]})
        _assert_no_secret_leak(a)

    def test_double_dumpd_nesting(self) -> None:
        """Double nesting: `dumpd(AIMessage(dumpd(HumanMessage)))`."""
        h = HumanMessage(
            "Hello",
            additional_kwargs={"secret": MALICIOUS_SECRET_DICT},
        )
        a = AIMessage("foo", additional_kwargs={"bar": [dumpd(h)]})
        outer = AIMessage("outer", additional_kwargs={"nested": [dumpd(a)]})
        _assert_no_secret_leak(outer)


class TestRoundTrip:
    """Tests that verify round-trip serialization preserves data structure."""

    def test_human_message_with_secret_round_trip(self) -> None:
        """Verify secret-like dict is preserved as dict after round-trip."""
        msg = HumanMessage(
            content="Hello",
            additional_kwargs={"data": MALICIOUS_SECRET_DICT},
        )

        serialized = dumpd(msg)
        deserialized = load(serialized, secrets_from_env=True)

        # The secret-like dict should be preserved as a plain dict
        assert deserialized.additional_kwargs["data"] == MALICIOUS_SECRET_DICT
        assert isinstance(deserialized.additional_kwargs["data"], dict)

    def test_document_with_secret_round_trip(self) -> None:
        """Verify secret-like dict in `Document` metadata is preserved."""
        doc = Document(
            page_content="Hello",
            metadata={"data": MALICIOUS_SECRET_DICT},
        )

        serialized = dumpd(doc)
        deserialized = load(
            serialized, secrets_from_env=True, allowed_objects=[Document]
        )

        # The secret-like dict should be preserved as a plain dict
        assert deserialized.metadata["data"] == MALICIOUS_SECRET_DICT
        assert isinstance(deserialized.metadata["data"], dict)

    def test_plain_dict_with_secret_round_trip(self) -> None:
        """Verify secret-like dict in plain dict is preserved."""
        payload = {"data": MALICIOUS_SECRET_DICT}

        serialized = dumpd(payload)
        deserialized = load(serialized, secrets_from_env=True)

        # The secret-like dict should be preserved as a plain dict
        assert deserialized["data"] == MALICIOUS_SECRET_DICT
        assert isinstance(deserialized["data"], dict)


class TestEscapingEfficiency:
    """Tests that escaping doesn't cause excessive nesting."""

    def test_no_triple_escaping(self) -> None:
        """Verify dumpd doesn't cause triple/multiple escaping."""
        h = HumanMessage(
            "Hello",
            additional_kwargs={"bar": [MALICIOUS_SECRET_DICT]},
        )
        a = AIMessage("foo", additional_kwargs={"bar": [dumpd(h)]})
        d = dumpd(a)

        serialized = json.dumps(d)
        # Count nested escape markers -
        # should be max 2 (one for HumanMessage, one for secret)
        # Not 3+ which would indicate re-escaping of already-escaped content
        escape_count = len(re.findall(r"__lc_escaped__", serialized))

        # The HumanMessage dict gets escaped (1), the secret inside gets escaped (1)
        # Total should be 2, not 4 (which would mean triple nesting)
        assert escape_count <= 2, (
            f"Found {escape_count} escape markers, expected <= 2. "
            f"This indicates unnecessary re-escaping.\n{serialized}"
        )

    def test_double_nesting_no_quadruple_escape(self) -> None:
        """Verify double dumpd nesting doesn't explode escape markers."""
        h = HumanMessage(
            "Hello",
            additional_kwargs={"secret": MALICIOUS_SECRET_DICT},
        )
        a = AIMessage("middle", additional_kwargs={"nested": [dumpd(h)]})
        outer = AIMessage("outer", additional_kwargs={"deep": [dumpd(a)]})
        d = dumpd(outer)

        serialized = json.dumps(d)
        escape_count = len(re.findall(r"__lc_escaped__", serialized))

        # Should be:
        # outer escapes middle (1),
        # middle escapes h (1),
        # h escapes secret (1) = 3
        # Not 6+ which would indicate re-escaping
        assert escape_count <= 3, (
            f"Found {escape_count} escape markers, expected <= 3. "
            f"This indicates unnecessary re-escaping."
        )


class TestConstructorInjection:
    """Tests for constructor-type injection (not just secrets)."""

    def test_constructor_in_metadata_not_instantiated(self) -> None:
        """Verify constructor-like dict in metadata is not instantiated."""
        malicious_constructor = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain_core", "messages", "ai", "AIMessage"],
            "kwargs": {"content": "injected"},
        }

        doc = Document(
            page_content="Hello",
            metadata={"data": malicious_constructor},
        )

        serialized = dumpd(doc)
        deserialized = load(
            serialized,
            secrets_from_env=True,
            allowed_objects=[Document, AIMessage],
        )

        # The constructor-like dict should be a plain dict, NOT an AIMessage
        assert isinstance(deserialized.metadata["data"], dict)
        assert deserialized.metadata["data"] == malicious_constructor

    def test_constructor_in_content_not_instantiated(self) -> None:
        """Verify constructor-like dict in message content is not instantiated."""
        malicious_constructor = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain_core", "messages", "human", "HumanMessage"],
            "kwargs": {"content": "injected"},
        }

        msg = AIMessage(
            content="Hello",
            additional_kwargs={"nested": malicious_constructor},
        )

        serialized = dumpd(msg)
        deserialized = load(
            serialized,
            secrets_from_env=True,
            allowed_objects=[AIMessage, HumanMessage],
        )

        # The constructor-like dict should be a plain dict, NOT a HumanMessage
        assert isinstance(deserialized.additional_kwargs["nested"], dict)
        assert deserialized.additional_kwargs["nested"] == malicious_constructor


def test_allowed_objects() -> None:
    # Core object
    msg = AIMessage(content="foo")
    serialized = dumpd(msg)
    assert load(serialized) == msg
    assert load(serialized, allowed_objects=[AIMessage]) == msg
    assert load(serialized, allowed_objects="core") == msg

    with pytest.raises(ValueError, match="not allowed"):
        load(serialized, allowed_objects=[])
    with pytest.raises(ValueError, match="not allowed"):
        load(serialized, allowed_objects=[Document])
