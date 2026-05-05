"""Tests for SecretMiddleware."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolCallRequest

from langchain.agents.factory import create_agent
from langchain.agents.middleware.secret import (
    BUILTIN_SECRET_TYPES,
    SecretMatch,
    SecretMiddleware,
    find_secrets,
)
from tests.unit_tests.agents.model import FakeToolCallingModel

# ---------------------------------------------------------------------------
# Fake credentials with the right shape but obviously not real values.
# ---------------------------------------------------------------------------

FAKE_GHP = "ghp_" + "A" * 36
FAKE_GHO = "gho_" + "B" * 36
FAKE_GHS = "ghs_" + "C" * 36
FAKE_GHU = "ghu_" + "D" * 36
FAKE_GHR = "ghr_" + "E" * 36
FAKE_GH_PAT = "github_pat_" + "F" * 22 + "_" + "G" * 59
FAKE_LSV2_PT = "lsv2_pt_" + "a" * 32 + "_" + "b" * 16
FAKE_LSV2_SK = "lsv2_sk_" + "1" * 32 + "_" + "2" * 16
FAKE_ANTHROPIC = "sk-ant-api03-" + "X" * 80
FAKE_OPENAI_PROJECT = "sk-proj-" + "y" * 60
FAKE_OPENAI_LEGACY = "sk-" + "Z" * 48
FAKE_AWS_AKIA = "AKIA" + "0" * 16
FAKE_AWS_ASIA = "ASIA" + "9" * 16
FAKE_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY.signaturePart12345678"


def _request(name: str, args: dict, call_id: str = "tc-1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"id": call_id, "name": name, "args": args, "type": "tool_call"},
        tool=None,
        state=None,
        runtime=None,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# find_secrets — pattern coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("secret_type", "token"),
    [
        ("github_classic_token", FAKE_GHP),
        ("github_classic_token", FAKE_GHO),
        ("github_classic_token", FAKE_GHS),
        ("github_classic_token", FAKE_GHU),
        ("github_classic_token", FAKE_GHR),
        ("github_fine_grained_pat", FAKE_GH_PAT),
        ("langsmith_key", FAKE_LSV2_PT),
        ("langsmith_key", FAKE_LSV2_SK),
        ("anthropic_key", FAKE_ANTHROPIC),
        ("openai_project_key", FAKE_OPENAI_PROJECT),
        ("openai_legacy_key", FAKE_OPENAI_LEGACY),
        ("aws_access_key_id", FAKE_AWS_AKIA),
        ("aws_access_key_id", FAKE_AWS_ASIA),
        ("jwt", FAKE_JWT),
    ],
)
def test_find_secrets_detects_each_builtin_pattern(secret_type: str, token: str) -> None:
    matches = find_secrets(token)
    assert any(m.secret_type == secret_type for m in matches), (
        f"expected {secret_type} match in {token!r}, got {[m.secret_type for m in matches]}"
    )


def test_find_secrets_walks_nested_dict_with_path() -> None:
    payload = {
        "actions": [{"body": {"code_evaluators": [{"code": f"K = '{FAKE_GHP}'"}]}}],
    }
    matches = find_secrets(payload)
    assert any(m.secret_type == "github_classic_token" for m in matches)  # noqa: S105
    assert any(m.path == "actions[0].body.code_evaluators[0].code" for m in matches)


def test_find_secrets_returns_offsets_into_matched_string() -> None:
    text = f"prefix {FAKE_GHP} suffix"
    matches = find_secrets(text)
    assert len(matches) == 1
    m = matches[0]
    assert text[m.start : m.end] == FAKE_GHP


@pytest.mark.parametrize(
    "value",
    [
        "ghp_",  # bare prefix, no body
        "lsv2_",
        "sk-ant-",
        "AKIA",
        "",
        " ",
        "the docs mention sk- as a prefix",
        "00000000-0000-0000-0000-000000000000",  # zero UUID
        "1f2a3b4c5d6e7f8090a1b2c3d4e5f607",  # plain 32-hex
        "1f2a3b4c5d6e7f8090a1b2c3d4e5f6071f2a3b4c",  # git SHA shape (40 hex)
        "01HV2C8M0F7R3K9X4N5W6E7T8B",  # ULID
        "sk-test",
        "TEST" + "0" * 16,  # 4-letter prefix that isn't AKIA/ASIA
        "eyJhbGciOi.eyJzdWIiOi",  # 2-segment "JWT" — not a real JWT
        '{"lc": 1, "type": "constructor"}',
    ],
)
def test_find_secrets_negative_corpus(value: str) -> None:
    assert find_secrets(value) == [], (
        f"unexpected match in {value!r}: {[m.secret_type for m in find_secrets(value)]}"
    )


def test_find_secrets_ignores_non_string_scalars() -> None:
    assert find_secrets(None) == []
    assert find_secrets(42) == []
    assert find_secrets(value=True) == []
    assert find_secrets(3.14) == []


def test_secret_match_is_frozen() -> None:
    m = SecretMatch(secret_type="github_classic_token", path="body", start=0, end=40)  # noqa: S106
    with pytest.raises(FrozenInstanceError):
        m.secret_type = "other"  # type: ignore[misc]  # noqa: S105


# ---------------------------------------------------------------------------
# Middleware — direct hook tests
# ---------------------------------------------------------------------------


def test_block_strategy_short_circuits_handler() -> None:
    middleware = SecretMiddleware()  # default strategy="block"
    handler = MagicMock(side_effect=AssertionError("handler should not run"))

    result = middleware.wrap_tool_call(
        _request("save_ao", {"content": f"# AO\n{FAKE_GHP}\n"}),
        handler,
    )

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "tc-1"
    assert result.name == "save_ao"
    assert "save_ao" in str(result.content)
    assert "github_classic_token" in str(result.content)
    # Critically: the secret substring must not be echoed back into context.
    assert FAKE_GHP not in str(result.content)


def test_clean_payload_passes_through_handler() -> None:
    middleware = SecretMiddleware()
    expected = ToolMessage(content="ok", tool_call_id="tc-2", name="save_ao")
    handler = MagicMock(return_value=expected)

    result = middleware.wrap_tool_call(
        _request("save_ao", {"content": "no secrets here"}, "tc-2"),
        handler,
    )

    handler.assert_called_once()
    assert result is expected


def test_redact_strategy_rewrites_args_and_calls_handler() -> None:
    middleware = SecretMiddleware(strategy="redact")
    captured: list[ToolCallRequest] = []

    def handler(req: ToolCallRequest) -> ToolMessage:
        captured.append(req)
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"], name="t")

    request = _request(
        "t",
        {
            "outer": f"prefix {FAKE_GHP} suffix",
            "nested": {"k": [f"a{FAKE_LSV2_PT}b"]},
        },
        "tc-3",
    )
    result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.content == "ok"
    assert len(captured) == 1
    new_args = captured[0].tool_call["args"]
    # Top-level string was redacted in place, surrounding text preserved.
    assert new_args["outer"] == "prefix [REDACTED_GITHUB_CLASSIC_TOKEN] suffix"
    # Nested string was redacted under the right path.
    assert new_args["nested"] == {"k": ["a[REDACTED_LANGSMITH_KEY]b"]}
    # The original request object is left intact (immutable override pattern).
    assert request.tool_call["args"]["outer"] == f"prefix {FAKE_GHP} suffix"


def test_tool_filter_skips_unlisted_tools() -> None:
    middleware = SecretMiddleware(tools=["post_to_slack"])
    handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="tc-4", name="search"))

    result = middleware.wrap_tool_call(
        _request("search", {"query": f"why is {FAKE_GHP} leaking"}, "tc-4"),
        handler,
    )

    handler.assert_called_once()
    assert isinstance(result, ToolMessage)
    assert result.content == "ok"


def test_tool_filter_applies_to_listed_tools() -> None:
    middleware = SecretMiddleware(tools=["post_to_slack"])
    handler = MagicMock(side_effect=AssertionError("handler should not run"))

    result = middleware.wrap_tool_call(
        _request("post_to_slack", {"text": f"check this: {FAKE_GHP}"}, "tc-5"),
        handler,
    )

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert result.status == "error"


def test_secret_types_filter_disables_others() -> None:
    """With secret_types=['aws_access_key_id'], a GitHub token should pass."""
    middleware = SecretMiddleware(secret_types=["aws_access_key_id"])
    expected = ToolMessage(content="ok", tool_call_id="tc-6", name="t")
    handler = MagicMock(return_value=expected)

    result = middleware.wrap_tool_call(
        _request("t", {"content": f"only github here: {FAKE_GHP}"}, "tc-6"),
        handler,
    )

    handler.assert_called_once()
    assert result is expected


def test_secret_types_filter_still_catches_listed() -> None:
    middleware = SecretMiddleware(secret_types=["aws_access_key_id"])
    handler = MagicMock(side_effect=AssertionError("handler should not run"))

    result = middleware.wrap_tool_call(
        _request("t", {"content": f"aws here: {FAKE_AWS_AKIA}"}, "tc-7"),
        handler,
    )

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert "aws_access_key_id" in str(result.content)


def test_unknown_secret_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown secret_type"):
        SecretMiddleware(secret_types=["not_a_real_type"])


def test_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError, match="strategy must be"):
        SecretMiddleware(strategy="mask")  # type: ignore[arg-type]


def test_custom_detector_runs_alongside_builtins() -> None:
    pattern = re.compile(r"\bACME-[A-Z0-9]{20}\b")

    def acme_finder(s: str):
        for m in pattern.finditer(s):
            yield m.start(), m.end()

    middleware = SecretMiddleware(custom_detectors={"acme_internal_token": acme_finder})
    handler = MagicMock(side_effect=AssertionError("handler should not run"))

    fake_acme = "ACME-" + "Z" * 20
    result = middleware.wrap_tool_call(
        _request("t", {"content": f"saw {fake_acme} in trace"}, "tc-8"),
        handler,
    )

    handler.assert_not_called()
    assert isinstance(result, ToolMessage)
    assert "acme_internal_token" in str(result.content)


def test_custom_detector_only_disables_builtins() -> None:
    """secret_types=[] disables all built-ins; only custom detectors run."""
    pattern = re.compile(r"\bACME-[A-Z0-9]{20}\b")

    def acme_finder(s: str):
        for m in pattern.finditer(s):
            yield m.start(), m.end()

    middleware = SecretMiddleware(
        secret_types=[],
        custom_detectors={"acme_internal_token": acme_finder},
    )
    expected = ToolMessage(content="ok", tool_call_id="tc-9", name="t")
    handler = MagicMock(return_value=expected)

    result = middleware.wrap_tool_call(
        _request("t", {"content": f"github token {FAKE_GHP} but no acme"}, "tc-9"),
        handler,
    )

    handler.assert_called_once()
    assert result is expected


def test_missing_args_field_is_handled() -> None:
    middleware = SecretMiddleware()
    expected = ToolMessage(content="ok", tool_call_id="tc-10", name="ls")
    handler = MagicMock(return_value=expected)

    request = ToolCallRequest(
        tool_call={"id": "tc-10", "name": "ls", "type": "tool_call"},  # type: ignore[typeddict-item]
        tool=None,
        state=None,
        runtime=None,  # type: ignore[arg-type]
    )
    result = middleware.wrap_tool_call(request, handler)

    handler.assert_called_once()
    assert result is expected


def test_redact_collapses_overlapping_matches() -> None:
    """Overlapping detector spans get merged in a single redaction.

    If two detectors hit overlapping spans, redaction merges them rather
    than double-substituting (which would corrupt offsets).
    """

    def covers_full_string(s: str):
        if s:
            yield 0, len(s)

    captured: list[ToolCallRequest] = []

    def handler(req: ToolCallRequest) -> ToolMessage:
        captured.append(req)
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"], name="t")

    middleware_redact = SecretMiddleware(
        strategy="redact",
        secret_types=["github_classic_token"],
        custom_detectors={"covers_all": covers_full_string},
    )
    request = _request("t", {"x": f"{FAKE_GHP}"}, "tc-11")
    result = middleware_redact.wrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)
    # The merged span ends up labelled with the first-sorted match's label
    # — what matters is that the output is a single redaction marker, not
    # nested or double-substituted text.
    assert captured[0].tool_call["args"]["x"].count("[REDACTED_") == 1


# ---------------------------------------------------------------------------
# Async variant
# ---------------------------------------------------------------------------


_HANDLER_CALLED_MSG = "handler should not run"


async def test_async_block_short_circuits_handler() -> None:
    middleware = SecretMiddleware()

    async def handler(_req: ToolCallRequest) -> ToolMessage:
        raise AssertionError(_HANDLER_CALLED_MSG)

    result = await middleware.awrap_tool_call(
        _request("t", {"content": FAKE_GHP}, "tc-async-1"),
        handler,
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert FAKE_GHP not in str(result.content)


async def test_async_clean_payload_awaits_handler() -> None:
    middleware = SecretMiddleware()
    awaited: list[ToolCallRequest] = []

    async def handler(req: ToolCallRequest) -> ToolMessage:
        awaited.append(req)
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"], name="t")

    result = await middleware.awrap_tool_call(
        _request("t", {"content": "no secrets"}, "tc-async-2"),
        handler,
    )

    assert len(awaited) == 1
    assert isinstance(result, ToolMessage)
    assert result.content == "ok"


# ---------------------------------------------------------------------------
# create_agent end-to-end
# ---------------------------------------------------------------------------


@tool
def echo_tool(value: str) -> str:
    """Echo the value back."""
    return f"echoed: {value}"


def test_e2e_block_returns_error_tool_message() -> None:
    """End-to-end: a poisoned tool call surfaces as a single error ToolMessage.

    The middleware integrates with create_agent: a poisoned tool call
    surfaces as a single ``ToolMessage(status='error')`` and the tool body
    is never reached (echoed value would otherwise appear).
    """
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="echo_tool", args={"value": FAKE_GHP}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[echo_tool],
        middleware=[SecretMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result: dict[str, Any] = agent.invoke(
        {"messages": [HumanMessage("Echo the test value")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    msg = tool_messages[0]
    assert msg.status == "error"
    assert "echo_tool" in str(msg.content)
    assert "github_classic_token" in str(msg.content)
    # Secret must not survive into the message stream.
    assert FAKE_GHP not in str(msg.content)
    # Tool body never ran.
    assert not any("echoed:" in str(m.content) for m in tool_messages)


def test_e2e_redact_runs_tool_with_redacted_args() -> None:
    """End-to-end: with strategy='redact', the tool runs on rewritten args.

    With ``strategy='redact'``, the tool body executes against the
    rewritten args. The agent receives the tool's normal output, computed
    over the redacted string.
    """
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="echo_tool", args={"value": FAKE_GHP}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[echo_tool],
        middleware=[SecretMiddleware(strategy="redact")],
        checkpointer=InMemorySaver(),
    )

    result: dict[str, Any] = agent.invoke(
        {"messages": [HumanMessage("Echo the test value")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    content = str(tool_messages[0].content)
    assert "echoed:" in content
    assert "[REDACTED_GITHUB_CLASSIC_TOKEN]" in content
    assert FAKE_GHP not in content


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_builtin_secret_types_is_frozen_set_of_known_names() -> None:
    assert isinstance(BUILTIN_SECRET_TYPES, frozenset)
    expected = {
        "github_classic_token",
        "github_fine_grained_pat",
        "langsmith_key",
        "anthropic_key",
        "openai_project_key",
        "openai_legacy_key",
        "aws_access_key_id",
        "jwt",
    }
    assert expected == BUILTIN_SECRET_TYPES
