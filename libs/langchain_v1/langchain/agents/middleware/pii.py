"""PII detection and handling middleware for agents."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage
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
        # Keyed by `(run_id, content_block_index: int)` for text-deltas
        # and `(run_id, tool_call_id: str)` for tool-output-deltas.
        self._buffers: dict[tuple[str, int | str], str] = {}

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
        # deltas. For non-`block` strategies, scrub `.content` in place
        # so the consumer sees redacted text (after_model re-runs and is
        # idempotent on the now-redacted state). For `block`, replace the
        # event payload with an empty copy — the original message stays
        # in state for `after_model` to raise on.
        if isinstance(payload, BaseMessage):
            self._mutate_legacy_payload(event, payload, metadata)
            return True

        if not isinstance(payload, dict):
            return True
        kind = payload.get("event")
        run_id = str(metadata.get("run_id", "")) if metadata else ""

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
            if isinstance(data.get("input"), dict):
                data["input"] = self._redact_value(data["input"])
        elif kind == "tool-output-delta":
            if isinstance(tool_call_id, str):
                self._mutate_tool_output_delta(data, tool_call_id)
        elif kind == "tool-finished":
            if "output" in data:
                data["output"] = self._redact_value(data["output"])
            if isinstance(tool_call_id, str):
                self._buffers.pop(("", tool_call_id), None)
        elif kind == "tool-error":
            msg = data.get("message")
            if isinstance(msg, str) and msg:
                matches = self._rule.detector(msg)
                if matches:
                    if self._rule.strategy == "block":
                        data["message"] = ""
                    else:
                        data["message"] = apply_strategy(msg, matches, self._rule.strategy)
            if isinstance(tool_call_id, str):
                self._buffers.pop(("", tool_call_id), None)

        return True

    def _mutate_tool_output_delta(self, data: dict[str, Any], tool_call_id: str) -> None:
        """Redact a `tool-output-delta` payload.

        String deltas go through the same lookback machinery as
        text-deltas, keyed by `("", tool_call_id)` — tool_call_ids are
        unique within a run, so a flat key suffices and avoids the
        run_id propagation gap on the `tools` channel.

        Structured deltas (dict/list) walk recursively without
        buffering — they don't have a position-stable shape across
        deltas to buffer against.
        """
        delta = data.get("delta")
        if isinstance(delta, str):
            key: tuple[str, int | str] = ("", tool_call_id)
            held = self._buffers.get(key, "")
            combined = held + delta

            if self._rule.strategy == "block":
                self._buffers[key] = combined
                data["delta"] = ""
                return

            matches = self._rule.detector(combined)
            if matches:
                combined = apply_strategy(combined, matches, self._rule.strategy)

            emit_end = max(0, len(combined) - self._lookback)
            self._buffers[key] = combined[emit_end:]
            data["delta"] = combined[:emit_end]
        elif isinstance(delta, (dict, list)):
            data["delta"] = self._redact_value(delta)

    def _mutate_legacy_payload(
        self,
        event: ProtocolEvent,
        message: BaseMessage,
        metadata: Any,
    ) -> None:
        """Scrub a legacy `(BaseMessage, metadata)` payload.

        For non-`block` strategies, `.content` and any `tool_calls` on an
        `AIMessage` are mutated in place — `after_model` runs on the same
        object in graph state and is idempotent over already-redacted
        text.

        For `block`, the event's `data` tuple is replaced with one
        carrying a fresh empty-content message of the same class and id.
        That keeps the original message in graph state intact so
        `after_model` can still raise `PIIDetectionError`, while ensuring
        downstream stream consumers — including `MessagesTransformer`,
        which runs *after* this transformer but sees the same event
        object — never observe the PII.
        """
        content = message.content
        tool_calls = getattr(message, "tool_calls", None) or []

        # Plain-string content: detect via the str path so we can apply
        # the configured strategy directly. Structured content (list of
        # content blocks) walks via `_redact_value`, which handles every
        # string leaf inside.
        content_matches: list[PIIMatch] = []
        redacted_content_list: list[Any] | None = None
        if isinstance(content, str) and content:
            content_matches = self._rule.detector(content)
        elif isinstance(content, list) and content:
            walked = self._redact_value(content)
            if walked != content:
                redacted_content_list = walked

        tool_call_hits: list[int] = []
        for i, tc in enumerate(tool_calls):
            args = tc.get("args") if isinstance(tc, dict) else None
            if isinstance(args, dict) and self._args_contain_pii(args):
                tool_call_hits.append(i)

        content_dirty = bool(content_matches) or redacted_content_list is not None
        if not content_dirty and not tool_call_hits:
            return

        if self._rule.strategy == "block":
            empty = type(message)(content="", id=getattr(message, "id", None))
            event["params"]["data"] = (empty, metadata)
            return

        if content_matches and isinstance(content, str):
            message.content = apply_strategy(content, content_matches, self._rule.strategy)
        elif redacted_content_list is not None:
            message.content = redacted_content_list
        if tool_call_hits and isinstance(message, AIMessage):
            new_tool_calls: list[Any] = list(message.tool_calls)
            for i in tool_call_hits:
                tc = dict(new_tool_calls[i])
                tc["args"] = self._redact_value(tc.get("args") or {})
                new_tool_calls[i] = tc
            message.tool_calls = new_tool_calls

    def _args_contain_pii(self, args: dict[str, Any]) -> bool:
        """Cheap check: does any string leaf in args trigger detection?"""
        for v in args.values():
            if isinstance(v, str) and v and self._rule.detector(v):
                return True
            if isinstance(v, (dict, list, tuple)) and self._redact_value(v) != v:
                return True
        return False

    def _redact_value(self, value: Any) -> Any:
        """Recursively redact PII in string leaves of a nested structure.

        Returns a new value where every `str` leaf that contains PII has
        been replaced (or emptied under `block`). Non-string leaves and
        the structure itself are preserved.

        `BaseMessage` payloads (typically `ToolMessage` from
        `tool-finished.output`) return a fresh copy with `.content`
        redacted — never mutated in place. The original object stays
        intact for the state-level enforcer (`before_model` with
        `apply_to_tool_results`) to act on independently.
        """
        if isinstance(value, str):
            if not value:
                return value
            matches = self._rule.detector(value)
            if not matches:
                return value
            if self._rule.strategy == "block":
                return ""
            return apply_strategy(value, matches, self._rule.strategy)
        if isinstance(value, BaseMessage):
            update: dict[str, Any] = {}

            content = value.content
            if isinstance(content, str) and content:
                matches = self._rule.detector(content)
                if matches:
                    if self._rule.strategy == "block":
                        update["content"] = ""
                    else:
                        update["content"] = apply_strategy(content, matches, self._rule.strategy)
            elif isinstance(content, list) and content:
                # Structured content-blocks shape:
                # `[{"type": "text", "text": "..."}, {"type": "tool_call", ...}, ...]`.
                # Walk each block through the recursion so any string
                # leaf (text, reasoning, args values, …) gets redacted.
                redacted_content = self._redact_value(content)
                if redacted_content != content:
                    update["content"] = redacted_content

            # `AIMessage.tool_calls` carry PII in `args` independently of
            # `.content` (the common shape for tool-calling responses is
            # empty content + populated `tool_calls`). Walk each call's
            # args through the recursion so a state snapshot containing
            # such a message doesn't leak the args on the wire.
            if isinstance(value, AIMessage) and value.tool_calls:
                new_tool_calls: list[Any] = []
                tool_calls_changed = False
                for tc in value.tool_calls:
                    if isinstance(tc, dict):
                        args = tc.get("args")
                        if isinstance(args, dict):
                            redacted_args = self._redact_value(args)
                            if redacted_args != args:
                                new_tc = dict(tc)
                                new_tc["args"] = redacted_args
                                new_tool_calls.append(new_tc)
                                tool_calls_changed = True
                                continue
                    new_tool_calls.append(tc)
                if tool_calls_changed:
                    update["tool_calls"] = new_tool_calls

            if not update:
                return value
            return value.model_copy(update=update)
        if isinstance(value, dict):
            # Redact both keys and values — a sensitive identifier used
            # as a map key (e.g., an email-indexed result dict) would
            # otherwise stream unchanged. Non-string keys pass through
            # unchanged via the `_redact_value` identity branch.
            return {self._redact_value(k): self._redact_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._redact_value(v) for v in value)
        return value

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
        # `data-delta` (binary/base64 payloads) passes through — regex
        # detection on base64 strings would produce mostly false matches
        # and rarely catches real PII, since binary payloads aren't a
        # typical PII surface. Users with structured-data PII concerns
        # should attach a custom detector that understands their
        # payload format.

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

        if self._rule.strategy == "block":
            # `block` withholds every delta from the consumer and defers
            # the decision to `_finalize_block`. Detection runs once on
            # the assembled block so we don't pay regex cost per delta,
            # and the consumer sees no content until the block resolves
            # either to its full text (clean) or to empty (PII found —
            # `after_model` raises shortly after).
            self._buffers[key] = combined
            delta[field] = ""
            return

        # Run detection on the full accumulated buffer before splitting.
        # Detecting only on the about-to-emit prefix would miss matches
        # that straddle the lookback boundary — the detector's regex
        # needs a complete, boundary-anchored hit, so a truncated prefix
        # would fail to match and the partial PII would leak on the wire.
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
        withholding then trims the trailing `stream_lookback` characters
        from what reaches the consumer — those characters might be the
        start of a partial PII match that completes in a future
        cumulative delta. The trimmed tail surfaces at `content-block-
        finish` where `_finalize_block` redacts the parsed args dict.

        For args that fit within `stream_lookback` (the typical case),
        this withholds the entire args string during streaming — the
        redacted args dict appears only at finalize. For args that
        exceed `stream_lookback`, the safe prefix streams incrementally
        as the cumulative state grows. PII that appears more than
        `stream_lookback` characters from the cumulative tail in a
        delta where it hasn't yet completed can still surface in the
        emit prefix — same residual exposure as PII longer than
        `stream_lookback` on the text path. The `content-block-finish`
        snapshot redaction is the backstop.
        """
        args = fields.get("args")
        if not isinstance(args, str) or not args:
            return

        if self._rule.strategy == "block":
            # `block` withholds every wire surface during streaming;
            # `_finalize_block` empties the parsed args on PII and
            # `after_model` raises shortly after.
            fields["args"] = ""
            return

        matches = self._rule.detector(args)
        if matches:
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
            elif ctype in {"tool_call", "server_tool_call", "invalid_tool_call"}:
                args = content.get("args")
                if isinstance(args, dict):
                    content["args"] = self._redact_value(args)
        self._buffers.pop(key, None)

    def _finalize_string_field(self, content: dict[str, Any], field: str) -> None:
        """Re-redact a string content-block field on `content-block-finish`.

        Used for `text` and `reasoning` content blocks. `block` withholds
        the content from the consumer; `after_model` raises
        `PIIDetectionError` on the original state message shortly after.
        """
        text = content.get(field)
        if not isinstance(text, str) or not text:
            return
        matches = self._rule.detector(text)
        if not matches:
            return
        if self._rule.strategy == "block":
            content[field] = ""
        else:
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

    def fail(self, err: BaseException) -> None:  # noqa: ARG002
        self._buffers.clear()


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
        stream_lookback: int = _DEFAULT_STREAM_LOOKBACK,
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
            stream_lookback: Trailing-buffer size for cross-delta PII
                detection in the stream transformer. The transformer always
                holds the last `stream_lookback` characters back until the
                buffer extends past the cap or the block finishes, so the
                value sets both the longest reliably-caught pattern and the
                worst-case first-token latency. Patterns longer than this
                may slip past in-flight detection when split across deltas,
                but the finalize snapshot always re-runs detection over the
                full block text.

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
        # `messages`). State-level hooks remain canonical enforcers;
        # for `block` the transformer withholds every delta and lets
        # the matching hook raise on the original state message.
        self._stream_lookback = stream_lookback
        if self.apply_to_output or self.apply_to_tool_results:
            self.transformers = (
                partial(
                    _PIIStreamTransformer,
                    rule=self._resolved_rule,
                    lookback=self._stream_lookback,
                ),
            )

    @property
    def name(self) -> str:
        """Name of the middleware."""
        return f"{self.__class__.__name__}[{self.pii_type}]"

    def _process_content(self, content: str) -> tuple[str, list[PIIMatch]]:
        """Apply the configured redaction rule to the provided content."""
        matches = self.detector(content)
        if not matches:
            return content, []
        sanitized = apply_strategy(content, matches, self.strategy)
        return sanitized, matches

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

            if last_user_idx is not None and last_user_msg and last_user_msg.content:
                # Detect PII in message content
                content = str(last_user_msg.content)
                new_content, matches = self._process_content(content)

                if matches:
                    updated_message: AnyMessage = HumanMessage(
                        content=new_content,
                        id=last_user_msg.id,
                        name=last_user_msg.name,
                    )

                    new_messages[last_user_idx] = updated_message
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
                        tool_msg = msg
                        if not tool_msg.content:
                            continue

                        content = str(tool_msg.content)
                        new_content, matches = self._process_content(content)

                        if not matches:
                            continue

                        # Create updated tool message
                        updated_message = ToolMessage(
                            content=new_content,
                            id=tool_msg.id,
                            name=tool_msg.name,
                            tool_call_id=tool_msg.tool_call_id,
                        )

                        new_messages[i] = updated_message
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

        if last_ai_idx is None or not last_ai_msg or not last_ai_msg.content:
            return None

        # Detect PII in message content
        content = str(last_ai_msg.content)
        new_content, matches = self._process_content(content)

        if not matches:
            return None

        # Create updated message
        updated_message = AIMessage(
            content=new_content,
            id=last_ai_msg.id,
            name=last_ai_msg.name,
            tool_calls=last_ai_msg.tool_calls,
        )

        # Return updated messages
        new_messages = list(messages)
        new_messages[last_ai_idx] = updated_message

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
