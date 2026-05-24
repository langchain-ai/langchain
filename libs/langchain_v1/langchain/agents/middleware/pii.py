"""PII detection and handling middleware for agents."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.stream import StreamTransformer
from typing_extensions import override

from langchain.agents.middleware._redaction import (
    PIIDetectionError,
    PIIMatch,
    RedactionRule,
    ResolvedRedactionRule,
    apply_strategy,
    detect_credit_card,
    detect_email,
    detect_ip,
    detect_mac_address,
    detect_url,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.runtime import Runtime
    from langgraph.stream._types import ProtocolEvent


_DEFAULT_STREAM_LOOKBACK = 128
"""Default trailing-buffer size for cross-delta PII detection.

The transformer always holds the last `lookback` characters in a per-content
block buffer so that PII patterns straddling delta boundaries are detected
before any text is released downstream. 128 comfortably covers the built-in
detectors (the credit-card regex tops out at 19 characters; URLs and emails
are typically well under 100) while bounding first-token latency.
"""


def _redact_tool_call_list(
    calls: list[Any] | None, *, rule: ResolvedRedactionRule
) -> tuple[list[Any], bool]:
    """Walk a list of tool-call (or invalid-tool-call) dicts."""
    if not calls:
        return calls or [], False
    new_calls: list[Any] = []
    changed = False
    for tc in calls:
        if isinstance(tc, dict) and "args" in tc and tc["args"] is not None:
            redacted = _redact_value(tc["args"], rule=rule)
            if redacted != tc["args"]:
                new_tc = dict(tc)
                new_tc["args"] = redacted
                new_calls.append(new_tc)
                changed = True
                continue
        new_calls.append(tc)
    return new_calls, changed


def _redact_value(value: Any, *, rule: ResolvedRedactionRule) -> Any:
    """Recursively redact PII in string leaves of a nested structure.

    Returns a new value where every `str` leaf that contains PII has
    been replaced (or emptied under `block`). Non-string leaves and
    the structure itself are preserved.

    `BaseMessage` payloads (typically `ToolMessage` from
    `tool-finished.output`, or any message reached via the `values`
    channel) return a fresh copy with `.content` redacted plus
    `AIMessage.tool_calls[*].args` / `invalid_tool_calls[*].args`
    walked. The original object stays intact for state-level
    enforcers (`after_model`, `before_model` with
    `apply_to_tool_results`) to act on independently.

    Scope mirrors the pre-streaming state-level surfaces:
    `.content` (string or list-of-content-blocks) and `tool_calls`
    args. Other message attributes (`additional_kwargs`,
    `response_metadata`, `ToolMessage.artifact`) are intentionally
    not walked here — they aren't scrubbed in graph state by the
    existing hooks, so scrubbing them on the wire would create
    a wire/state divergence.
    """
    if isinstance(value, str):
        if not value:
            return value
        matches = rule.detector(value)
        if not matches:
            return value
        # `apply_strategy` raises `PIIDetectionError` under `block`
        # — the run fails immediately rather than buffering until a
        # state-level hook can raise.
        return apply_strategy(value, matches, rule.strategy)
    if isinstance(value, BaseMessage):
        return _redact_base_message(value, rule=rule)
    if isinstance(value, dict):
        return {k: _redact_value(v, rule=rule) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v, rule=rule) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v, rule=rule) for v in value)
    return value


def _redact_base_message(value: BaseMessage, *, rule: ResolvedRedactionRule) -> BaseMessage:
    """Return a fresh copy of `value` with PII-carrying surfaces redacted."""
    update: dict[str, Any] = {}

    content = value.content
    if isinstance(content, str) and content:
        matches = rule.detector(content)
        if matches:
            update["content"] = apply_strategy(content, matches, rule.strategy)
    elif isinstance(content, list) and content:
        # Structured content-blocks shape:
        # `[{"type": "text", "text": "..."}, {"type": "tool_call", ...}, ...]`.
        redacted_content = _redact_value(content, rule=rule)
        if redacted_content != content:
            update["content"] = redacted_content

    # `AIMessage.tool_calls` and `.invalid_tool_calls` carry PII in
    # `args` independently of `.content`. `tool_call.args` is a
    # dict; `invalid_tool_call.args` is a raw JSON string —
    # `_redact_value` handles both shapes via the recursion.
    if isinstance(value, AIMessage):
        new_tc_list, tc_changed = _redact_tool_call_list(value.tool_calls, rule=rule)
        if tc_changed:
            update["tool_calls"] = new_tc_list
        new_inv_list, inv_changed = _redact_tool_call_list(
            value.invalid_tool_calls, rule=rule
        )
        if inv_changed:
            update["invalid_tool_calls"] = new_inv_list

    if not update:
        return value
    return value.model_copy(update=update)


class _PIIStreamTransformer(StreamTransformer):
    """Mutates `content-block-delta` text on `messages` events in flight.

    Runs before built-in stream transformers so the redacted text is what
    every downstream consumer sees — both the main protocol event log and
    the `run.messages` projection that `MessagesTransformer` snapshots into.

    Holds a sliding buffer of the most recent text per (run_id, content
    block index) so PII patterns that straddle delta boundaries are caught.
    Anything older than `lookback` characters is redacted with the resolved
    rule's strategy and emitted as the new delta text; the trailing tail
    stays in the buffer until a later delta extends it past the cap or the
    block's finish event flushes the snapshot.
    """

    before_builtins: ClassVar[bool] = True
    required_stream_modes: ClassVar[tuple[str, ...]] = ("messages", "tools", "values")

    def __init__(
        self,
        scope: tuple[str, ...] = (),
        *,
        rule: ResolvedRedactionRule,
        lookback: int = _DEFAULT_STREAM_LOOKBACK,
    ) -> None:
        super().__init__(scope)
        self._rule = rule
        self._lookback = lookback
        # Text/reasoning deltas keyed by `(run_id, content_block_index)`.
        self._buffers: dict[tuple[str, int], str] = {}
        # Tool-output-delta buffers keyed by `tool_call_id`. Held in a
        # separate dict so `_drop_run` on the messages channel can't
        # sweep active tool-output state.
        self._tool_buffers: dict[str, str] = {}

    def init(self) -> dict[str, Any]:
        # No projection — this transformer mutates events in place rather
        # than building a derived view.
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        method = event["method"]
        if method == "messages":
            return self._process_messages_event(event)
        if method == "tools":
            return self._process_tools_event(event)
        if method == "values":
            return self._process_values_event(event)
        return True

    def _process_values_event(self, event: ProtocolEvent) -> bool:
        """Redact the state snapshot on the `values` channel.

        State snapshots emitted between nodes carry the full state dict,
        which typically includes the messages list. Walking the snapshot
        with `_redact_value` returns a fresh structure where every
        message has a redacted copy of its content — the original
        objects in graph state remain intact for the state-level
        enforcer (`apply_to_tool_results` via `before_model`) to act on
        independently when the agent loops back.
        """
        data = event["params"].get("data")
        if data is None:
            return True
        event["params"]["data"] = self._redact_value(data)
        return True

    def _process_messages_event(self, event: ProtocolEvent) -> bool:
        params = event["params"]
        data = params.get("data")
        if not isinstance(data, tuple) or len(data) != 2:  # noqa: PLR2004
            return True
        payload, metadata = data

        # Legacy `(BaseMessage, metadata)` shape: the langgraph→langchain
        # integration emits this when a model only implements `_generate`
        # (or when its `_astream` falls back), producing a single event
        # carrying the full message rather than streamed content-block
        # deltas. Swap in a redacted copy so the consumer sees scrubbed
        # text on the wire while the original stays intact in graph state
        # for `after_model` to act on independently. Under `block`,
        # `_redact_base_message` raises `PIIDetectionError` via
        # `apply_strategy` before we get here.
        if isinstance(payload, BaseMessage):
            redacted = self._redact_base_message(payload)
            if redacted is not payload:
                params["data"] = (redacted, metadata)
            return True

        if not isinstance(payload, dict):
            return True
        kind = payload.get("event")
        run_id = str(metadata.get("run_id") or "") if metadata else ""

        if kind == "content-block-delta":
            self._mutate_delta(payload, run_id)
        elif kind == "content-block-finish":
            self._finalize_block(payload, run_id)
        elif kind in {"message-finish", "error"}:
            self._drop_run(run_id)
        return True

    def _process_tools_event(self, event: ProtocolEvent) -> bool:
        data = event["params"].get("data")
        if not isinstance(data, dict):
            return True
        kind = data.get("event")
        tool_call_id = data.get("tool_call_id")

        if kind == "tool-started":
            # Tool inputs may be a dict (multi-arg tools), a string
            # (single-arg tools — `BaseTool._parse_input` passes the
            # raw string through), or a list (array-input tools).
            # `_redact_value` handles all three uniformly.
            if "input" in data:
                data["input"] = self._redact_value(data["input"])
        elif kind == "tool-output-delta":
            # Use the tool_call_id as buffer key when present; fall back
            # to a None-keyed slot for the rare malformed/custom emitter
            # case (the buffer becomes shared but at least redaction runs).
            self._mutate_tool_output_delta(
                data, tool_call_id if isinstance(tool_call_id, str) else ""
            )
        elif kind == "tool-finished":
            if "output" in data:
                data["output"] = self._redact_value(data["output"])
            if isinstance(tool_call_id, str):
                self._tool_buffers.pop(tool_call_id, None)
        elif kind == "tool-error":
            msg = data.get("message")
            if isinstance(msg, str) and msg:
                matches = self._rule.detector(msg)
                if matches:
                    data["message"] = apply_strategy(msg, matches, self._rule.strategy)
            if isinstance(tool_call_id, str):
                self._tool_buffers.pop(tool_call_id, None)

        return True

    def _mutate_tool_output_delta(self, data: dict[str, Any], tool_call_id: str) -> None:
        """Redact a `tool-output-delta` payload.

        String deltas go through the same lookback machinery as
        text-deltas, keyed by `tool_call_id` in the disjoint
        `_tool_buffers` dict so `_drop_run` on the messages channel
        can't sweep active tool-output state.

        Structured deltas (dict/list) walk recursively without
        buffering — they don't have a position-stable shape across
        deltas to buffer against.
        """
        delta = data.get("delta")
        if isinstance(delta, str):
            held = self._tool_buffers.get(tool_call_id, "")
            combined = held + delta

            matches = self._rule.detector(combined)
            if matches:
                # `apply_strategy` raises `PIIDetectionError` under
                # `strategy="block"`, failing the run immediately —
                # cleaner than withholding deltas until `after_model`
                # raises later.
                combined = apply_strategy(combined, matches, self._rule.strategy)

            emit_end = max(0, len(combined) - self._lookback)
            self._tool_buffers[tool_call_id] = combined[emit_end:]
            data["delta"] = combined[:emit_end]
        elif isinstance(delta, (dict, list)):
            data["delta"] = self._redact_value(delta)

    def _redact_tool_call_list(self, calls: list[Any] | None) -> tuple[list[Any], bool]:
        return _redact_tool_call_list(calls, rule=self._rule)

    def _redact_value(self, value: Any) -> Any:
        return _redact_value(value, rule=self._rule)

    def _redact_base_message(self, value: BaseMessage) -> BaseMessage:
        return _redact_base_message(value, rule=self._rule)

    def _mutate_delta(self, payload: dict[str, Any], run_id: str) -> None:
        delta = payload.get("delta")
        if not isinstance(delta, dict):
            return
        delta_type = delta.get("type")
        if delta_type == "text-delta":
            self._mutate_string_field_delta(delta, payload, run_id, "text")
            return
        if delta_type == "reasoning-delta":
            # Reasoning content (chain-of-thought from extended-thinking
            # models) is a real PII surface — models echo back
            # user-supplied data or synthesize it from context. Run the
            # same lookback machinery as text-delta against the
            # `reasoning` field. Block indices are unique within a
            # message regardless of block type, so the buffer key
            # `(run_id, index)` naturally disjoint from text-delta keys.
            self._mutate_string_field_delta(delta, payload, run_id, "reasoning")
            return
        if delta_type == "block-delta":
            fields = delta.get("fields")
            if isinstance(fields, dict) and fields.get("type") in {
                "tool_call_chunk",
                "server_tool_call_chunk",
            }:
                self._mutate_tool_call_chunk_delta(fields)
        # Other delta types (`data-delta`, vendor block types) pass
        # through. The pre-streaming middleware scrubbed `.content` text
        # on state messages only; binary payloads and provider-specific
        # block shapes are out of scope for parity with that surface.

    def _mutate_string_field_delta(
        self,
        delta: dict[str, Any],
        payload: dict[str, Any],
        run_id: str,
        field: str,
    ) -> None:
        """Apply the lookback-buffer redaction to a string field on a delta.

        Shared by `text-delta` (`field="text"`) and `reasoning-delta`
        (`field="reasoning"`). Buffer is keyed by `(run_id, block_index)`;
        block indices are unique within a message so different block
        types share the same key space without collision.
        """
        text = delta.get(field)
        if not isinstance(text, str) or not text:
            return
        index = payload.get("index")
        if not isinstance(index, int):
            return

        key = (run_id, index)
        held = self._buffers.get(key, "")
        combined = held + text

        # Run detection on the full accumulated buffer before splitting.
        # Detecting only on the about-to-emit prefix would miss matches
        # that straddle the lookback boundary — the detector's regex
        # needs a complete, boundary-anchored hit, so a truncated prefix
        # would fail to match and the partial PII would leak on the
        # wire. Under `strategy="block"`, `apply_strategy` raises
        # `PIIDetectionError` here, failing the run as soon as PII
        # arrives rather than buffering until `after_model`.
        matches = self._rule.detector(combined)
        if matches:
            combined = apply_strategy(combined, matches, self._rule.strategy)

        emit_end = max(0, len(combined) - self._lookback)
        self._buffers[key] = combined[emit_end:]
        delta[field] = combined[:emit_end]

    def _mutate_tool_call_chunk_delta(self, fields: dict[str, Any]) -> None:
        """Redact cumulative tool-call args with lookback withholding.

        Each `tool_call_chunk` `block-delta` event carries the full
        accumulated args string (verified against `_compat_bridge.py`
        — `delta_source = current` for these block types — and against
        the consumer-side `_merge_block_delta_into_store`, which
        replaces wholesale rather than appends).

        Detection runs on the full cumulative args so any complete PII
        anywhere in the string is redacted before emission. Lookback
        withholding then trims the trailing the lookback window characters
        from what reaches the consumer — those characters might be the
        start of a partial PII match that completes in a future
        cumulative delta. The trimmed tail surfaces at `content-block-
        finish` where `_finalize_block` redacts the parsed args dict.

        For args that fit within the lookback window (the typical case),
        this withholds the entire args string during streaming — the
        redacted args dict appears only at finalize. For args that
        exceed the lookback window, the safe prefix streams incrementally
        as the cumulative state grows. PII that appears more than
        the lookback window characters from the cumulative tail in a
        delta where it hasn't yet completed can still surface in the
        emit prefix — same residual exposure as PII longer than
        the lookback window on the text path. The `content-block-finish`
        snapshot redaction is the backstop.
        """
        args = fields.get("args")
        if not isinstance(args, str) or not args:
            return

        matches = self._rule.detector(args)
        if matches:
            # `apply_strategy` raises `PIIDetectionError` under
            # `strategy="block"` — the run fails the moment a complete
            # PII pattern surfaces in the cumulative args string.
            args = apply_strategy(args, matches, self._rule.strategy)

        emit_end = max(0, len(args) - self._lookback)
        fields["args"] = args[:emit_end]

    def _finalize_block(self, payload: dict[str, Any], run_id: str) -> None:
        index = payload.get("index")
        if not isinstance(index, int):
            return
        key = (run_id, index)
        # The finalized block carries the model's original concatenation
        # of deltas, not what we emitted on the wire. Re-run detection over
        # its full text so the snapshot matches the redacted stream.
        content = payload.get("content")
        if isinstance(content, dict):
            ctype = content.get("type")
            if ctype == "text":
                self._finalize_string_field(content, "text")
            elif ctype == "reasoning":
                self._finalize_string_field(content, "reasoning")
            elif (
                ctype in {"tool_call", "server_tool_call", "invalid_tool_call"}
                and "args" in content
                and content["args"] is not None
            ):
                # `tool_call` / `server_tool_call` args are dicts;
                # `invalid_tool_call.args` is the raw unparsed JSON
                # string. `_redact_value` handles both shapes.
                content["args"] = self._redact_value(content["args"])
        self._buffers.pop(key, None)

    def _finalize_string_field(self, content: dict[str, Any], field: str) -> None:
        """Re-redact a string content-block field on `content-block-finish`.

        Used for `text` and `reasoning` content blocks. Under
        `strategy="block"` `apply_strategy` raises `PIIDetectionError`,
        failing the run immediately.
        """
        text = content.get(field)
        if not isinstance(text, str) or not text:
            return
        matches = self._rule.detector(text)
        if not matches:
            return
        content[field] = apply_strategy(text, matches, self._rule.strategy)

    def _drop_run(self, run_id: str) -> None:
        # Release any buffered tails for this run_id — content-block-finish
        # should have already done so for normal completion, but message-finish
        # / error paths need an explicit sweep so abandoned blocks don't
        # accumulate in long-lived processes.
        stale = [key for key in self._buffers if key[0] == run_id]
        for key in stale:
            del self._buffers[key]

    def finalize(self) -> None:
        self._buffers.clear()
        self._tool_buffers.clear()

    def fail(self, err: BaseException) -> None:  # noqa: ARG002
        self._buffers.clear()
        self._tool_buffers.clear()


class PIIMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Detect and handle Personally Identifiable Information (PII) in conversations.

    This middleware detects common PII types and applies configurable strategies
    to handle them. It can detect emails, credit cards, IP addresses, MAC addresses, and
    URLs in both user input and agent output.

    Built-in PII types:

    - `email`: Email addresses
    - `credit_card`: Credit card numbers (validated with Luhn algorithm)
    - `ip`: IP addresses (validated with stdlib)
    - `mac_address`: MAC addresses
    - `url`: URLs (both `http`/`https` and bare URLs)

    Strategies:

    - `block`: Raise an exception when PII is detected
    - `redact`: Replace PII with `[REDACTED_TYPE]` placeholders
    - `mask`: Partially mask PII (e.g., `****-****-****-1234` for credit card)
    - `hash`: Replace PII with deterministic hash (e.g., `<email_hash:a1b2c3d4>`)

    Strategy Selection Guide:

    | Strategy | Preserves Identity? | Best For                                |
    | -------- | ------------------- | --------------------------------------- |
    | `block`  | N/A                 | Avoid PII completely                    |
    | `redact` | No                  | General compliance, log sanitization    |
    | `mask`   | No                  | Human readability, customer service UIs |
    | `hash`   | Yes (pseudonymous)  | Analytics, debugging                    |

    Example:
        ```python
        from langchain.agents.middleware import PIIMiddleware
        from langchain.agents import create_agent

        # Redact all emails in user input
        agent = create_agent(
            "openai:gpt-5.5",
            middleware=[
                PIIMiddleware("email", strategy="redact"),
            ],
        )

        # Use different strategies for different PII types
        agent = create_agent(
            "openai:gpt-5.5",
            middleware=[
                PIIMiddleware("credit_card", strategy="mask"),
                PIIMiddleware("url", strategy="redact"),
                PIIMiddleware("ip", strategy="hash"),
            ],
        )

        # Custom PII type with regex
        agent = create_agent(
            "openai:gpt-5.5",
            middleware=[
                PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),
            ],
        )
        ```
    """

    def __init__(
        self,
        # From a typing point of view, the literals are covered by 'str'.
        # Nonetheless, we escape PYI051 to keep hints and autocompletion for the caller.
        pii_type: Literal["email", "credit_card", "ip", "mac_address", "url"] | str,  # noqa: PYI051
        *,
        strategy: Literal["block", "redact", "mask", "hash"] = "redact",
        detector: Callable[[str], list[PIIMatch]] | str | None = None,
        apply_to_input: bool = True,
        apply_to_output: bool = False,
        apply_to_tool_results: bool = False,
    ) -> None:
        """Initialize the PII detection middleware.

        Args:
            pii_type: Type of PII to detect.

                Can be a built-in type (`email`, `credit_card`, `ip`, `mac_address`,
                `url`) or a custom type name.
            strategy: How to handle detected PII.

                Options:

                * `block`: Raise `PIIDetectionError` when PII is detected
                * `redact`: Replace with `[REDACTED_TYPE]` placeholders
                * `mask`: Partially mask PII (show last few characters)
                * `hash`: Replace with deterministic hash (format: `<type_hash:digest>`)

            detector: Custom detector function or regex pattern.

                * If `Callable`: Function that takes content string and returns
                    list of `PIIMatch` objects
                * If `str`: Regex pattern to match PII
                * If `None`: Uses built-in detector for the `pii_type`
            apply_to_input: Whether to check user messages before model call.
            apply_to_output: Whether to check AI messages after model call.

                When `True`, a stream transformer is also installed so
                that every wire surface of an agent run is redacted in
                flight:

                * Streamed AI text deltas (`content-block-delta` of type
                  `text-delta`)
                * Streamed tool-call arguments (`content-block-delta`
                  with `tool_call_chunk` / `server_tool_call_chunk`
                  fields, plus the finalized `tool_call` content block
                  on `content-block-finish`)
                * Tool execution events on the `tools` channel
                  (`tool-started.input`, `tool-output-delta`,
                  `tool-finished.output`, `tool-error.message`)
                * State snapshots on the `values` channel — message
                  lists are walked and each message's `.content` is
                  redacted on a fresh copy (state itself stays intact
                  for `before_model` / `after_model` to act on
                  independently)

                State-level redaction via `after_model` (and
                `before_model` with `apply_to_tool_results`) remains the
                canonical enforcer; the streaming transformer ensures
                consumers reading `astream_events(version="v3")` or
                `run.messages` / `run.tool_calls` / `run.values` never
                see PII on the wire.
            apply_to_tool_results: Whether to check tool result messages after tool execution.

        Raises:
            ValueError: If `pii_type` is not built-in and no detector is provided.
        """
        super().__init__()

        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output
        self.apply_to_tool_results = apply_to_tool_results

        self._resolved_rule: ResolvedRedactionRule = RedactionRule(
            pii_type=pii_type,
            strategy=strategy,
            detector=detector,
        ).resolve()
        self.pii_type = self._resolved_rule.pii_type
        self.strategy = self._resolved_rule.strategy
        self.detector = self._resolved_rule.detector

        # Stream transformer scrubs the streamed surface of the same
        # messages that the state-level hooks scrub in graph state.
        # Installed whenever any output-side scrubbing is enabled —
        # `apply_to_output` covers AI messages (text, tool-call args,
        # reasoning), `apply_to_tool_results` covers tool execution
        # (the `tools` channel + ToolMessage content on `values` and
        # `messages`). For `block` the transformer raises
        # `PIIDetectionError` directly from its event handler the
        # moment a complete PII pattern is detected, failing the run
        # via langgraph's `StreamMux.afail` path. The state-level
        # `after_model` / `before_model` hooks remain a backstop for
        # non-streaming consumers.
        if self.apply_to_output or self.apply_to_tool_results:
            self.transformers = (
                partial(
                    _PIIStreamTransformer,
                    rule=self._resolved_rule,
                ),
            )

    @property
    def name(self) -> str:
        """Name of the middleware."""
        return f"{self.__class__.__name__}[{self.pii_type}]"

    def _redact_base_message(self, value: BaseMessage) -> BaseMessage:
        return _redact_base_message(value, rule=self._resolved_rule)

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Check user messages and tool results for PII before model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or `None` if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        """
        if not self.apply_to_input and not self.apply_to_tool_results:
            return None

        messages = state["messages"]
        if not messages:
            return None

        new_messages = list(messages)
        any_modified = False

        # Check user input if enabled
        if self.apply_to_input:
            # Get last user message
            last_user_msg = None
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    last_user_msg = messages[i]
                    last_user_idx = i
                    break

            if last_user_idx is not None and last_user_msg:
                redacted = self._redact_base_message(last_user_msg)
                if redacted is not last_user_msg:
                    new_messages[last_user_idx] = redacted
                    any_modified = True

        # Check tool results if enabled
        if self.apply_to_tool_results:
            # Find the last AIMessage, then process all `ToolMessage` objects after it
            last_ai_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], AIMessage):
                    last_ai_idx = i
                    break

            if last_ai_idx is not None:
                # Get all tool messages after the last AI message
                for i in range(last_ai_idx + 1, len(messages)):
                    msg = messages[i]
                    if isinstance(msg, ToolMessage):
                        redacted = self._redact_base_message(msg)
                        if redacted is not msg:
                            new_messages[i] = redacted
                            any_modified = True

        if any_modified:
            return {"messages": new_messages}

        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async check user messages and tool results for PII before model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or `None` if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        """
        return self.before_model(state, runtime)

    @override
    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Check AI messages for PII after model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or None if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        """
        if not self.apply_to_output:
            return None

        messages = state["messages"]
        if not messages:
            return None

        # Get last AI message
        last_ai_msg = None
        last_ai_idx = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                last_ai_msg = msg
                last_ai_idx = i
                break

        if last_ai_idx is None or not last_ai_msg:
            return None

        redacted = self._redact_base_message(last_ai_msg)
        if redacted is last_ai_msg:
            return None

        # Return updated messages
        new_messages = list(messages)
        new_messages[last_ai_idx] = redacted

        return {"messages": new_messages}

    async def aafter_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async check AI messages for PII after model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or None if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        """
        return self.after_model(state, runtime)


__all__ = [
    "PIIDetectionError",
    "PIIMatch",
    "PIIMiddleware",
    "detect_credit_card",
    "detect_email",
    "detect_ip",
    "detect_mac_address",
    "detect_url",
]
