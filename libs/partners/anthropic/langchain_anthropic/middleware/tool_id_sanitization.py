"""Anthropic tool-call ID sanitization middleware."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal
from warnings import warn

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from langchain_anthropic.chat_models import ChatAnthropic

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        ModelCallResult,
        ModelRequest,
        ModelResponse,
    )
except ModuleNotFoundError as e:
    msg = (
        "AnthropicToolIdSanitizationMiddleware requires 'langchain' to be "
        "installed. This middleware is designed for use with LangChain agents. "
        "Install it with: pip install langchain"
    )
    raise ImportError(msg) from e


logger = logging.getLogger(__name__)


_ANTHROPIC_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
"""Character set Anthropic enforces server-side on `tool_use.id`.

Mirrors the regex echoed in the API errors returned for non-conforming IDs:
`messages.N.content.M.tool_use.id: String should match pattern
'^[a-zA-Z0-9_-]+$'`.
"""

_INVALID_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")
"""Complement of the character class in `_ANTHROPIC_ID_PATTERN`.

Matches any single illegal character. Used by `_make_safe_id` to substitute
offending characters with `_`.
"""


def _is_tool_use_type(block_type: Any) -> bool:
    """Return `True` if `block_type` names an ID-bearing tool-use block.

    Covers `tool_use` (client tools), `server_tool_use` (Anthropic-emitted
    server tools), and `mcp_tool_use` (MCP). Uses suffix matching so future
    `*_tool_use` variants are picked up automatically.
    """
    return isinstance(block_type, str) and block_type.endswith("tool_use")


def _is_client_tool_use_type(block_type: Any) -> bool:
    """Return `True` if `block_type` is a client `tool_use` block."""
    return block_type == "tool_use"


def _is_tool_result_type(block_type: Any) -> bool:
    """Return `True` if `block_type` names a `tool_use_id`-bearing result block.

    Covers `tool_result`, `mcp_tool_result`, and Anthropic server-tool result
    variants like `web_search_tool_result`, `code_execution_tool_result`, and
    `bash_tool_result`. Uses suffix matching for forward compatibility.
    """
    return isinstance(block_type, str) and block_type.endswith("tool_result")


def _is_valid_id(tool_id: str | None) -> bool:
    """Return `True` if `tool_id` matches Anthropic's required pattern."""
    if not tool_id:
        return False
    return _ANTHROPIC_ID_PATTERN.match(tool_id) is not None


def _make_safe_id(original: str, used: set[str]) -> str:
    """Derive an Anthropic-safe id, avoiding collisions within a single call.

    Replaces every illegal character with `_`. If the resulting `base` already
    appears in `used`, appends a sha256-derived suffix (and a counter on
    further collisions) so each invalid input gets a distinct safe output
    within one sanitization pass.

    Args:
        original: The invalid tool-call id to sanitize.
        used: The set of ids already taken in this pass — both pre-existing
            valid ids and previously-allocated safe ids.

    Returns:
        A regex-conformant id distinct from every entry in `used`.
    """
    base = _INVALID_CHAR_PATTERN.sub("_", original) or "tool"
    if base not in used:
        return base
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}_{digest}"
    counter = 0
    while candidate in used:
        counter += 1
        candidate = f"{base}_{digest}_{counter}"
    return candidate


class AnthropicToolIdSanitizationMiddleware(AgentMiddleware):
    """Rewrite illegal tool-call IDs before sending to Anthropic.

    Anthropic enforces `tool_use.id` matching `^[a-zA-Z0-9_-]+$`. Conversation
    histories produced by other providers can violate this — e.g. Kimi-K2 emits
    IDs of the form `functions.<name>:<idx>`, where `.` and `:` are illegal.
    Replaying such a thread against Claude raises a 400 error.

    This middleware scans `request.messages` for offending IDs, builds a
    deterministic `bad_id -> safe_id` map, and rewrites every occurrence:

    - `AIMessage.tool_calls[*]["id"]`
    - `AIMessage.content[*]["id"]` for `tool_use` / `server_tool_use` /
        `mcp_tool_use` blocks
    - `ToolMessage.tool_call_id`
    - `ToolMessage.content[*]["tool_use_id"]` for `tool_result` and the
        `*_tool_result` variants (`web_search_tool_result`,
        `code_execution_tool_result`, `bash_tool_result`, `mcp_tool_result`)

    Within an `AIMessage`, position-paired `tool_calls[i]` and `tool_use`
    content blocks are forced to share the same final id even when their
    inputs disagree (drift), so Anthropic never receives a mismatched pair.

    Only the outgoing `ModelRequest` is modified — graph state and persisted
    checkpoints are left untouched, so the sanitization is idempotent across
    turns and safe to combine with HITL resume.
    """

    def __init__(
        self,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> None:
        """Initialize the middleware.

        Args:
            unsupported_model_behavior: Behavior when the bound model is not
                `ChatAnthropic`.

                `'ignore'` skips sanitization silently (default — other
                providers may accept the original IDs).

                `'warn'` emits a warning and skips sanitization.

                `'raise'` raises `ValueError`.

        Raises:
            ValueError: If `unsupported_model_behavior` is not one of the
                three allowed strings. `Literal` is enforced statically only;
                this guards against typos at runtime.
        """
        allowed = ("ignore", "warn", "raise")
        if unsupported_model_behavior not in allowed:
            msg = (
                f"unsupported_model_behavior must be one of {allowed}; "
                f"got {unsupported_model_behavior!r}"
            )
            raise ValueError(msg)
        self.unsupported_model_behavior = unsupported_model_behavior

    def _should_run(self, request: ModelRequest) -> bool:
        """Return `True` if the bound model is `ChatAnthropic`.

        Args:
            request: The model request to inspect.

        Returns:
            `True` when sanitization should run; `False` to bypass.

        Raises:
            ValueError: When the bound model is not `ChatAnthropic` and
                `unsupported_model_behavior='raise'`.
        """
        if isinstance(request.model, ChatAnthropic):
            return True
        msg = (
            "AnthropicToolIdSanitizationMiddleware only supports Anthropic "
            f"models, not instances of {type(request.model)}"
        )
        if self.unsupported_model_behavior == "raise":
            raise ValueError(msg)
        if self.unsupported_model_behavior == "warn":
            warn(msg, stacklevel=3)
        else:
            # `ignore` mode — leave a breadcrumb so users debugging missing
            # sanitization can confirm the middleware was bypassed and why.
            logger.debug(
                "AnthropicToolIdSanitizationMiddleware skipped: bound model "
                "is %s, not ChatAnthropic.",
                type(request.model).__name__,
            )
        return False

    def _build_id_map(self, messages: list[AnyMessage]) -> dict[str, str]:
        """Build a deterministic mapping from invalid IDs to safe ones.

        Args:
            messages: The outgoing message list to scan.

        Returns:
            A dict mapping each invalid id to its safe replacement, or an
            empty dict when every id already conforms.
        """
        invalid: list[str] = []
        seen: set[str] = set()
        valid_ids: set[str] = set()

        for tool_id in _iter_all_ids(messages):
            if _is_valid_id(tool_id):
                valid_ids.add(tool_id)
            elif tool_id and tool_id not in seen:
                invalid.append(tool_id)
                seen.add(tool_id)

        if not invalid:
            return {}

        used = set(valid_ids)
        mapping: dict[str, str] = {}
        for original in invalid:
            new = _make_safe_id(original, used)
            mapping[original] = new
            used.add(new)
        return mapping

    def _sanitize(self, request: ModelRequest) -> ModelRequest:
        """Return a copy of `request` with sanitized tool-call IDs.

        Args:
            request: The model request to sanitize.

        Returns:
            A new request when any id was rewritten, otherwise the original.
        """
        mapping = self._build_id_map(request.messages)
        rewritten, changed = _rewrite_messages(request.messages, mapping)
        if not changed:
            return request
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Sanitized %d tool-call id(s) for Anthropic compatibility: %s",
                len(mapping),
                {k: mapping[k] for k in sorted(mapping)},
            )
        return request.override(messages=rewritten)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Sanitize tool-call IDs before delegating to the handler.

        Args:
            request: The model request to potentially modify.
            handler: The handler to execute the (possibly rewritten) request.

        Returns:
            The model response produced by `handler`.
        """
        if not self._should_run(request):
            return handler(request)
        return handler(self._sanitize(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Sanitize tool-call IDs before delegating to the async handler.

        Args:
            request: The model request to potentially modify.
            handler: The async handler to execute the (possibly rewritten)
                request.

        Returns:
            The model response produced by `handler`.
        """
        if not self._should_run(request):
            return await handler(request)
        return await handler(self._sanitize(request))


def _iter_all_ids(messages: list[AnyMessage]) -> list[str]:
    """Collect every tool-call ID present in the message list."""
    ids: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                tid = tc.get("id")
                if tid:
                    ids.append(tid)
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and _is_tool_use_type(block.get("type")):
                        tid = block.get("id")
                        if isinstance(tid, str) and tid:
                            ids.append(tid)
        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id:
                ids.append(msg.tool_call_id)
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and _is_tool_result_type(
                        block.get("type")
                    ):
                        tid = block.get("tool_use_id")
                        if isinstance(tid, str) and tid:
                            ids.append(tid)
    return ids


def _rewrite_messages(
    messages: list[AnyMessage], mapping: dict[str, str]
) -> tuple[list[AnyMessage], bool]:
    """Return messages rewritten with sanitized IDs and local drift aliases."""
    rewritten: list[AnyMessage] = []
    changed = False
    active_result_aliases: dict[str, str] = {}

    for msg in messages:
        new_msg: AnyMessage
        if isinstance(msg, AIMessage):
            new_msg, active_result_aliases = _rewrite_ai_message(msg, mapping)
        elif isinstance(msg, ToolMessage):
            result_mapping = {**mapping, **active_result_aliases}
            new_msg = _rewrite_tool_message(msg, result_mapping)
        else:
            active_result_aliases = {}
            new_msg = msg

        if new_msg is not msg:
            changed = True
        rewritten.append(new_msg)

    return rewritten, changed


def _rewrite_ai_message(
    msg: AIMessage, mapping: dict[str, str]
) -> tuple[AIMessage, dict[str, str]]:
    """Return a copy of `msg` with `tool_calls` and tool-use blocks aligned.

    Applies `mapping` to each id, then enforces position-paired alignment so
    `tool_calls[i].id` and the i-th client `tool_use` content block share the
    same final id even if their inputs drifted.
    """
    result_aliases: dict[str, str] = {}
    new_tool_calls: list[Any] | None = None
    if msg.tool_calls and any(tc.get("id") in mapping for tc in msg.tool_calls):
        new_tool_calls = []
        for tc in msg.tool_calls:
            tid = tc.get("id")
            if tid is not None and tid in mapping:
                new_tool_calls.append({**tc, "id": mapping[tid]})
            else:
                new_tool_calls.append(tc)

    new_content: list[Any] | None = None
    if isinstance(msg.content, list) and any(
        isinstance(b, dict)
        and _is_tool_use_type(b.get("type"))
        and b.get("id") in mapping
        for b in msg.content
    ):
        new_content = [
            {**b, "id": mapping[b["id"]]}
            if (
                isinstance(b, dict)
                and _is_tool_use_type(b.get("type"))
                and b.get("id") in mapping
            )
            else b
            for b in msg.content
        ]

    # Drift correction: even if `tool_calls[i].id` and the i-th client
    # `tool_use` block disagreed pre-mapping (or post-mapping due to distinct
    # invalid inputs collapsing through different sanitization branches), force
    # the content block to adopt the canonical `tool_calls[i].id`. `tool_calls`
    # is the langchain-canonical view.
    effective_tool_calls = (
        new_tool_calls if new_tool_calls is not None else msg.tool_calls
    )
    effective_content = new_content if new_content is not None else msg.content
    if effective_tool_calls and isinstance(effective_content, list):
        aligned, result_aliases, drift_seen = _align_client_tool_use_blocks(
            effective_tool_calls,
            effective_content,
            msg.content if isinstance(msg.content, list) else effective_content,
        )
        if drift_seen:
            new_content = aligned

    updates: dict[str, Any] = {}
    if new_tool_calls is not None:
        updates["tool_calls"] = new_tool_calls
    if new_content is not None:
        updates["content"] = new_content
    return (msg.model_copy(update=updates) if updates else msg), result_aliases


def _align_client_tool_use_blocks(
    tool_calls: list[Any], content: list[Any], original_content: list[Any]
) -> tuple[list[Any], dict[str, str], bool]:
    """Force tool-use content blocks to share IDs with corresponding tool_calls.

    Returns the (possibly rewritten) content list plus aliases from drifted
    content-block IDs to the canonical tool-call ID. These aliases are applied
    only to the following `ToolMessage` results for this assistant turn.
    """
    result_aliases: dict[str, str] = {}
    drift_seen = False
    aligned: list[Any] = list(content)
    tu_indices = [
        idx
        for idx, b in enumerate(content)
        if isinstance(b, dict) and _is_client_tool_use_type(b.get("type"))
    ]
    for pair_idx, content_idx in enumerate(tu_indices):
        if pair_idx >= len(tool_calls):
            break
        canonical_id = tool_calls[pair_idx].get("id")
        block = aligned[content_idx]
        if not isinstance(block, dict):
            continue
        if canonical_id and block.get("id") != canonical_id:
            aligned[content_idx] = {**block, "id": canonical_id}
            drift_seen = True
            original_block = original_content[content_idx]
            if isinstance(original_block, dict):
                original_id = original_block.get("id")
                if isinstance(original_id, str) and original_id != canonical_id:
                    result_aliases[original_id] = canonical_id
            current_id = block.get("id")
            if isinstance(current_id, str) and current_id != canonical_id:
                result_aliases[current_id] = canonical_id
    return aligned, result_aliases, drift_seen


def _rewrite_tool_message(msg: ToolMessage, mapping: dict[str, str]) -> ToolMessage:
    """Return a copy of `msg` with `tool_call_id` and tool-result blocks rewritten."""
    updates: dict[str, Any] = {}

    if msg.tool_call_id in mapping:
        updates["tool_call_id"] = mapping[msg.tool_call_id]

    if isinstance(msg.content, list) and any(
        isinstance(b, dict)
        and _is_tool_result_type(b.get("type"))
        and b.get("tool_use_id") in mapping
        for b in msg.content
    ):
        updates["content"] = [
            {**b, "tool_use_id": mapping[b["tool_use_id"]]}
            if (
                isinstance(b, dict)
                and _is_tool_result_type(b.get("type"))
                and b.get("tool_use_id") in mapping
            )
            else b
            for b in msg.content
        ]

    return msg.model_copy(update=updates) if updates else msg
