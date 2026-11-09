"""Tests for adapting to the installed `anthropic` SDK major version."""

from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import anthropic
import pytest

from langchain_anthropic import _sdk_compat
from langchain_anthropic._sdk_compat import (
    _aparse,
    _route_unsupported_sampling_params,
    _sdk_major_version,
    _unsupported_sampling_params,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.0.0", 1),
        ("v1.0.0", 1),  # tag-derived strings build tooling leaves unstripped
        ("0.125.0", 0),
        ("2.0.0rc1", 2),
        ("not-a-version", None),
    ],
)
def test_sdk_major_version_parsing(version: str, expected: int | None) -> None:
    with patch.object(anthropic, "__version__", version):
        assert _sdk_major_version() == expected


def test_unsupported_sampling_params_matches_the_sdk_signature() -> None:
    """The dropped set is exactly what `Messages.create` no longer accepts."""
    from anthropic.resources.messages import Messages

    accepted = inspect.signature(Messages.create).parameters
    assert _unsupported_sampling_params() == frozenset(
        p for p in ("temperature", "top_p", "top_k") if p not in accepted
    )


def test_sampling_params_end_up_where_the_sdk_accepts_them() -> None:
    """Sampling params stay top-level on 0.x and move to `extra_body` on 1.x."""
    payload = _route_unsupported_sampling_params(
        {"model": "claude-sonnet-4-5", "temperature": 0.5, "top_p": 0.9, "top_k": 5}
    )

    if _unsupported_sampling_params():
        assert payload == {
            "model": "claude-sonnet-4-5",
            "extra_body": {"temperature": 0.5, "top_p": 0.9, "top_k": 5},
        }
    else:
        assert payload == {
            "model": "claude-sonnet-4-5",
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 5,
        }


def test_routing_is_a_no_op_without_sampling_params() -> None:
    """No `extra_body` key is invented for payloads that set no sampling params."""
    assert _route_unsupported_sampling_params({"model": "claude-sonnet-4-5"}) == {
        "model": "claude-sonnet-4-5"
    }


def test_explicit_extra_body_wins_over_relocated_params() -> None:
    """`extra_body` is the SDK's documented escape hatch; don't overwrite it."""
    with patch.object(
        _sdk_compat, "_unsupported_sampling_params", lambda: frozenset({"temperature"})
    ):
        payload = _route_unsupported_sampling_params(
            {"temperature": 0.5, "extra_body": {"temperature": 0.1, "foo": "bar"}}
        )

    assert payload == {"extra_body": {"temperature": 0.1, "foo": "bar"}}


def test_non_dict_mapping_extra_body_still_receives_relocated_params() -> None:
    """`extra_body` accepts any Mapping, not just `dict`."""
    with patch.object(
        _sdk_compat, "_unsupported_sampling_params", lambda: frozenset({"temperature"})
    ):
        payload = _route_unsupported_sampling_params(
            {"temperature": 0.5, "extra_body": MappingProxyType({"foo": "bar"})}
        )

    assert payload == {"extra_body": {"temperature": 0.5, "foo": "bar"}}


def test_non_dict_extra_body_leaves_payload_untouched() -> None:
    """A malformed `extra_body` is the SDK's to reject, not ours to merge into.

    Crucially the sampling params must survive: relocating them into a value
    that cannot hold them would drop them silently.
    """
    with patch.object(
        _sdk_compat, "_unsupported_sampling_params", lambda: frozenset({"temperature"})
    ):
        payload = _route_unsupported_sampling_params(
            {"temperature": 0.5, "extra_body": "not-a-dict"}
        )

    assert payload == {"temperature": 0.5, "extra_body": "not-a-dict"}


def test_unsupported_sampling_params_falls_back_when_introspection_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unreadable signature guesses from the major version instead of raising."""
    _unsupported_sampling_params.cache_clear()
    try:
        with (
            patch.object(inspect, "signature", side_effect=ValueError("no signature")),
            patch.object(anthropic, "__version__", "1.0.0"),
        ):
            assert _unsupported_sampling_params() == frozenset(
                {"temperature", "top_p", "top_k"}
            )
    finally:
        _unsupported_sampling_params.cache_clear()

    assert "Could not introspect" in caplog.text


class _SyncRawResponse:
    def parse(self) -> str:
        return "sync"


class _AsyncRawResponse:
    async def parse(self) -> str:
        return "async"


@pytest.mark.parametrize(
    ("raw_response", "expected"),
    [(_SyncRawResponse(), "sync"), (_AsyncRawResponse(), "async")],
)
async def test_aparse_handles_both_raw_response_classes(
    raw_response: Any, expected: str
) -> None:
    """`anthropic<1` parsed synchronously; `anthropic>=1` returns a coroutine."""
    assert await _aparse(raw_response) == expected
