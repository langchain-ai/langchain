"""Secret detection middleware for agent tool calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.messages import ToolMessage
from typing_extensions import override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from langchain_core.messages import ToolCall
    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain.tools import BaseTool


# ---------------------------------------------------------------------------
# Built-in patterns
# ---------------------------------------------------------------------------
# Each entry anchors on a fixed, high-entropy token prefix plus a shape
# constraint. Common high-entropy strings that are *not* credentials —
# commit SHAs, UUIDs, ULIDs, base64 blobs without a known prefix, the
# bare fragment "sk-" in prose — must not match. False positives here
# block legitimate tool calls, so the regexes are intentionally tight.

# No `\b` word-boundary anchors: the prefix + length + alphabet of each
# pattern is tight enough on its own, and `\b` would block matches when
# the secret is concatenated with alphanumeric characters (e.g. inside
# a JSON blob the agent built up by string interpolation). The negative
# corpus in the tests demonstrates the FP rate stays at zero without it.
_BUILTIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # GitHub classic tokens: ghp_ / gho_ / ghu_ / ghs_ / ghr_ + 36 base62.
    ("github_classic_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36}")),
    # GitHub fine-grained PATs: github_pat_ + 22 + _ + 59 chars.
    ("github_fine_grained_pat", re.compile(r"github_pat_[A-Za-z0-9_]{82}")),
    # LangSmith API keys: lsv2_(pt|sk)_<32hex>_<16hex>.
    ("langsmith_key", re.compile(r"lsv2_(?:pt|sk)_[a-f0-9]{32}_[a-f0-9]{16}")),
    # Anthropic API keys: sk-ant-api<NN>- + long base64url-ish blob.
    ("anthropic_key", re.compile(r"sk-ant-api\d{2}-[A-Za-z0-9_\-]{60,}")),
    # OpenAI project keys: sk-proj- + long alphanumeric blob.
    ("openai_project_key", re.compile(r"sk-proj-[A-Za-z0-9_\-]{40,}")),
    # OpenAI legacy keys: sk- + exactly 48 base62.
    ("openai_legacy_key", re.compile(r"sk-[A-Za-z0-9]{48}")),
    # AWS Access Key IDs (long-term and STS-temporary).
    ("aws_access_key_id", re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}")),
    # JWT: three base64url segments, header begins with eyJ (b64url of `{"`).
    (
        "jwt",
        re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    ),
)

#: Names of the built-in secret detectors, suitable for the ``secret_types`` argument.
BUILTIN_SECRET_TYPES: frozenset[str] = frozenset(name for name, _ in _BUILTIN_PATTERNS)


@dataclass(frozen=True)
class SecretMatch:
    """One detected secret occurrence inside a tool-call argument tree.

    Attributes:
        secret_type: Built-in or custom detector name (e.g. ``"github_classic_token"``).
        path: Dotted/bracketed location string for nested args
            (e.g. ``"actions[0].body.code"``). Empty string for a top-level scalar.
        start: Byte offset where the match begins within the matched string.
        end: Byte offset where the match ends within the matched string.
    """

    secret_type: str
    path: str
    start: int
    end: int


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _walk(value: Any, path: str) -> Iterable[tuple[str, str]]:
    """Yield (path, string-value) pairs for every string scalar reachable from ``value``."""
    if isinstance(value, str):
        yield path, value
        return
    if isinstance(value, dict):
        for k, v in value.items():
            child = f"{path}.{k}" if path else str(k)
            yield from _walk(v, child)
        return
    if isinstance(value, list | tuple):
        for i, v in enumerate(value):
            child = f"{path}[{i}]" if path else f"[{i}]"
            yield from _walk(v, child)


def find_secrets(
    value: Any,
    *,
    detectors: Iterable[tuple[str, Callable[[str], Iterable[tuple[int, int]]]]] | None = None,
) -> list[SecretMatch]:
    """Recursively scan strings inside ``value`` for known secret patterns.

    Walks dicts, lists, and tuples; non-string scalars (numbers, bools, ``None``)
    are ignored.

    Args:
        value: Any Python object. Strings reachable through dict/list/tuple
            recursion are scanned; other scalars are ignored.
        detectors: Optional iterable of ``(secret_type, finder)`` pairs replacing
            the built-in detectors. Each ``finder`` takes a string and returns an
            iterable of ``(start, end)`` byte offsets for each match. If ``None``,
            the built-in pattern set is used.

    Returns:
        One :class:`SecretMatch` per detected occurrence. A single string can
        contribute multiple matches.
    """
    pairs: list[tuple[str, Callable[[str], Iterable[tuple[int, int]]]]]
    if detectors is None:
        pairs = [(name, _make_regex_finder(pat)) for name, pat in _BUILTIN_PATTERNS]
    else:
        pairs = list(detectors)

    out: list[SecretMatch] = []
    for path, s in _walk(value, ""):
        for secret_type, finder in pairs:
            for start, end in finder(s):
                out.append(SecretMatch(secret_type=secret_type, path=path, start=start, end=end))
    return out


def _make_regex_finder(
    pattern: re.Pattern[str],
) -> Callable[[str], Iterable[tuple[int, int]]]:
    def finder(s: str) -> Iterable[tuple[int, int]]:
        for m in pattern.finditer(s):
            yield m.start(), m.end()

    return finder


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


_Strategy = Literal["block", "redact"]


class SecretMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    r"""Detect known credential patterns in tool-call arguments.

    Models can be steered by attacker-controllable input (system prompts,
    retrieved documents, tool-result content) into emitting tool calls that
    embed credentials they have read from elsewhere — exfiltrating them
    through the legitimate tool surface. This middleware is a chokepoint
    in front of tool execution that catches that.

    By default it ships with detectors for GitHub tokens (classic and
    fine-grained), LangSmith keys, Anthropic keys, OpenAI keys (legacy and
    project), AWS Access Key IDs, and JWTs. Each detector is anchored on
    a fixed, high-entropy prefix, so commit SHAs, UUIDs, ULIDs, and other
    legitimate high-entropy strings are not flagged.

    Strategies:

    - ``block`` (default): Return a ``ToolMessage(status="error")`` instead
      of executing the tool. The agent sees a generic rejection — the
      matched substring is **not** echoed back, which would re-publish the
      secret into the agent's context window.
    - ``redact``: Replace each matched substring with
      ``[REDACTED_<SECRET_TYPE>]`` in the tool args and execute the tool
      with the rewritten args. Useful when the tool legitimately needs to
      receive the surrounding string but must not see the credential
      itself.

    Examples:
        !!! example "Default — block any tool call carrying a known secret"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import SecretMiddleware

            agent = create_agent(
                "openai:gpt-5",
                tools=[...],
                middleware=[SecretMiddleware()],
            )
            ```

        !!! example "Redact instead of block"

            ```python
            agent = create_agent(
                "openai:gpt-5",
                tools=[...],
                middleware=[SecretMiddleware(strategy="redact")],
            )
            ```

        !!! example "Limit which tools the middleware applies to"

            ```python
            agent = create_agent(
                "openai:gpt-5",
                tools=[search_tool, post_to_slack],
                middleware=[SecretMiddleware(tools=[post_to_slack])],
            )
            ```

        !!! example "Limit which secret types are checked"

            ```python
            from langchain.agents.middleware import BUILTIN_SECRET_TYPES, SecretMiddleware

            print(BUILTIN_SECRET_TYPES)
            # frozenset({"github_classic_token", "github_fine_grained_pat",
            #            "langsmith_key", "anthropic_key", ...})

            agent = create_agent(
                "openai:gpt-5",
                tools=[...],
                middleware=[
                    SecretMiddleware(secret_types=["github_classic_token", "aws_access_key_id"])
                ],
            )
            ```

        !!! example "Add a custom detector"

            ```python
            import re
            from langchain.agents.middleware import SecretMiddleware

            INTERNAL_TOKEN = re.compile(r"\\bACME-[A-Z0-9]{20}\\b")

            agent = create_agent(
                "openai:gpt-5",
                tools=[...],
                middleware=[
                    SecretMiddleware(
                        custom_detectors={
                            "acme_internal_token": lambda s: [
                                (m.start(), m.end()) for m in INTERNAL_TOKEN.finditer(s)
                            ],
                        },
                    )
                ],
            )
            ```
    """

    def __init__(
        self,
        *,
        strategy: _Strategy = "block",
        tools: list[BaseTool | str] | None = None,
        secret_types: Iterable[str] | None = None,
        custom_detectors: dict[str, Callable[[str], Iterable[tuple[int, int]]]] | None = None,
    ) -> None:
        """Initialize ``SecretMiddleware``.

        Args:
            strategy: How to handle a tool call whose args contain a detected secret.

                - ``"block"`` (default): Short-circuit with
                  ``ToolMessage(status="error")``; the tool is never executed.
                - ``"redact"``: Replace each matched substring with
                  ``[REDACTED_<SECRET_TYPE>]`` and execute the tool with the
                  rewritten args.

            tools: Optional list of tools (``BaseTool`` instances or tool name
                strings) to apply the check to. If ``None``, all tools are
                checked.
            secret_types: Optional iterable of built-in secret-type names to
                enable. If ``None``, all built-in detectors are enabled. Pass
                an empty iterable to disable built-ins (use only
                ``custom_detectors``). See :data:`BUILTIN_SECRET_TYPES` for
                names.
            custom_detectors: Optional mapping of ``secret_type`` →
                ``finder``. Each finder takes a string and returns an iterable
                of ``(start, end)`` byte-offset pairs for each match. Custom
                detectors run alongside the enabled built-ins.

        Raises:
            ValueError: If ``secret_types`` references an unknown built-in name,
                or if ``strategy`` is not ``"block"`` / ``"redact"``.
        """
        super().__init__()

        if strategy not in ("block", "redact"):
            msg = f"strategy must be 'block' or 'redact', got {strategy!r}"
            raise ValueError(msg)

        self.strategy: _Strategy = strategy

        if tools is not None:
            self._tool_filter: set[str] | None = {
                t.name if not isinstance(t, str) else t for t in tools
            }
        else:
            self._tool_filter = None

        self.tools = []  # The middleware does not register any tools of its own.

        # Resolve enabled built-ins.
        builtin_lookup = dict(_BUILTIN_PATTERNS)
        if secret_types is None:
            enabled_builtins = list(_BUILTIN_PATTERNS)
        else:
            enabled_builtins = []
            for name in secret_types:
                if name not in builtin_lookup:
                    msg = (
                        f"Unknown secret_type {name!r}. "
                        f"Built-ins: {sorted(builtin_lookup)}. "
                        f"For non-built-in detectors, use custom_detectors."
                    )
                    raise ValueError(msg)
                enabled_builtins.append((name, builtin_lookup[name]))

        self._detectors: list[tuple[str, Callable[[str], Iterable[tuple[int, int]]]]] = [
            (name, _make_regex_finder(pat)) for name, pat in enabled_builtins
        ]
        if custom_detectors:
            self._detectors.extend(custom_detectors.items())

    # ------------------------------------------------------------------
    # AgentMiddleware hooks
    # ------------------------------------------------------------------

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Block or redact secret-bearing tool calls before execution."""
        new_request, rejection = self._handle(request)
        if rejection is not None:
            return rejection
        return handler(new_request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async version of :meth:`wrap_tool_call`."""
        new_request, rejection = self._handle(request)
        if rejection is not None:
            return rejection
        return await handler(new_request)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _handle(self, request: ToolCallRequest) -> tuple[ToolCallRequest, ToolMessage | None]:
        """Decide whether to short-circuit, rewrite, or pass through.

        Returns ``(request, None)`` for pass-through (no secrets, or tool
        not in the filter). Returns ``(request, ToolMessage)`` for ``block``.
        Returns ``(rewritten_request, None)`` for ``redact``.
        """
        call = request.tool_call
        name = call.get("name") or "tool"

        if self._tool_filter is not None and name not in self._tool_filter:
            return request, None

        args = call.get("args") or {}
        matches = find_secrets(args, detectors=self._detectors)
        if not matches:
            return request, None

        if self.strategy == "block":
            return request, ToolMessage(
                content=_format_block_message(name, matches),
                tool_call_id=call["id"],
                name=name,
                status="error",
            )

        # Strategy is "redact" — rewrite args and let handler run on the result.
        new_args = _redact(args, self._detectors)
        new_call = cast("ToolCall", {**call, "args": new_args})
        return request.override(tool_call=new_call), None


# ---------------------------------------------------------------------------
# Helpers — block message + redaction
# ---------------------------------------------------------------------------


def _format_block_message(tool_name: str, matches: list[SecretMatch]) -> str:
    kinds = sorted({m.secret_type for m in matches})
    return (
        f"Tool call to '{tool_name}' was blocked: arguments contained content "
        f"matching a known credential pattern ({', '.join(kinds)}). The agent "
        "must not pass raw API keys, GitHub tokens, JWTs, or other secrets as "
        "tool arguments. Reissue the call without the credential value."
    )


def _redact(
    value: Any,
    detectors: list[tuple[str, Callable[[str], Iterable[tuple[int, int]]]]],
) -> Any:
    """Return a deep copy of ``value`` with every detector match replaced.

    Strings are scanned; matches across all detectors are merged and replaced
    in a single left-to-right pass with ``[REDACTED_<SECRET_TYPE>]``. Dicts,
    lists, and tuples are walked recursively. Non-string scalars are
    returned unchanged.
    """
    if isinstance(value, str):
        spans: list[tuple[int, int, str]] = []
        for secret_type, finder in detectors:
            for start, end in finder(value):
                spans.append((start, end, secret_type))
        if not spans:
            return value
        spans.sort()
        merged: list[tuple[int, int, str]] = []
        for span in spans:
            if merged and span[0] < merged[-1][1]:
                # Overlap: keep the earlier match's label, extend the end.
                prev = merged[-1]
                merged[-1] = (prev[0], max(prev[1], span[1]), prev[2])
            else:
                merged.append(span)
        out: list[str] = []
        cursor = 0
        for start, end, secret_type in merged:
            out.append(value[cursor:start])
            out.append(f"[REDACTED_{secret_type.upper()}]")
            cursor = end
        out.append(value[cursor:])
        return "".join(out)
    if isinstance(value, dict):
        return {k: _redact(v, detectors) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, detectors) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact(v, detectors) for v in value)
    return value


__all__ = [
    "BUILTIN_SECRET_TYPES",
    "SecretMatch",
    "SecretMiddleware",
    "find_secrets",
]
