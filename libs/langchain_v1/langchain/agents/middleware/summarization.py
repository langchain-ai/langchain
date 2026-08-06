"""Summarization middleware."""

import logging
import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from typing import Annotated, Any, Literal, TypedDict, cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.utils import (
    count_tokens_approximately,
    get_buffer_string,
    trim_messages,
)
from langgraph.graph.message import (
    REMOVE_ALL_MESSAGES,
)
from langgraph.runtime import Runtime
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
    ResponseT,
)
from langchain.chat_models import BaseChatModel, init_chat_model

logger = logging.getLogger(__name__)

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

DEFAULT_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to continue working toward your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.

You should structure your summary using the following sections. Each section acts as a checklist - you must populate it with relevant information or explicitly state "None" if there is nothing to report for that section:

## SESSION INTENT

What is the user's primary goal or request? What overall task are you trying to accomplish? This should be concise but complete enough to understand the purpose of the entire session.

## SUMMARY

Extract and record all of the most important context from the conversation history. Include important choices, conclusions, or strategies determined during this conversation. Include the reasoning behind key decisions. Document any rejected options and why they were not pursued.

## ARTIFACTS

What artifacts, files, or resources were created, modified, or accessed during this conversation? For file modifications, list specific file paths and briefly describe the changes made to each. This section prevents silent loss of artifact information.

## NEXT STEPS

What specific tasks remain to be completed to achieve the session intent? What should you do next?

</instructions>

The user will message you with the full message history from which you'll extract context to create a replacement. Carefully read through it all and think deeply about what information is most important to your overall goal and should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""  # noqa: E501
"""Default prompt used to summarize conversation history.

The `<messages>` marker (on its own line) and the `{messages}` placeholder are
part of this constant's public contract, not just cosmetic formatting.
Downstream consumers depend on them: for example, deep agents'
`SummarizationMiddleware` splices an extra instruction block in immediately
before the `<messages>` marker via `str.replace`. Removing, renaming, or
reformatting the marker (or the `{messages}` placeholder) is a breaking change
for those consumers even though it does not alter any function signature, so
treat edits to it accordingly.
"""

_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_FALLBACK_MESSAGE_COUNT = 15

# Some providers tag emitted messages with a `model_provider` string that differs from
# their LangSmith `ls_provider`. The reported-token check below compares the two, so we
# accept known aliases per `ls_provider`.
_LS_PROVIDER_ALIASES: dict[str, frozenset[str]] = {
    "amazon_bedrock": frozenset({"bedrock", "bedrock_converse"}),
    # ChatOpenAIMantle traces under ls_provider="openai-mantle" but its messages
    # inherit model_provider="openai" from BaseChatOpenAI.
    "openai-mantle": frozenset({"openai"}),
    # ChatAnthropicMantle traces under ls_provider="anthropic-mantle" but its
    # messages inherit model_provider="anthropic" from BaseChatAnthropic.
    "anthropic-mantle": frozenset({"anthropic"}),
}


def _provider_matches(message_provider: str, model_ls_provider: str | None) -> bool:
    if model_ls_provider is None:
        return False
    if message_provider == model_ls_provider:
        return True
    aliases = _LS_PROVIDER_ALIASES.get(model_ls_provider)
    return aliases is not None and message_provider in aliases


ContextFraction = tuple[Literal["fraction"], float]
"""Fraction of model's maximum input tokens.

Example:
    To specify 50% of the model's max input tokens:

    ```python
    ("fraction", 0.5)
    ```
"""

ContextTokens = tuple[Literal["tokens"], int]
"""Absolute number of tokens.

Example:
    To specify 3000 tokens:

    ```python
    ("tokens", 3000)
    ```
"""

ContextMessages = tuple[Literal["messages"], int]
"""Absolute number of messages.

Example:
    To specify 50 messages:

    ```python
    ("messages", 50)
    ```
"""

ContextSize = ContextFraction | ContextTokens | ContextMessages
"""Union type for context size specifications.

Can be either:

- [`ContextFraction`][langchain.agents.middleware.summarization.ContextFraction]: A
    fraction of the model's maximum input tokens.
- [`ContextTokens`][langchain.agents.middleware.summarization.ContextTokens]: An absolute
    number of tokens.
- [`ContextMessages`][langchain.agents.middleware.summarization.ContextMessages]: An
    absolute number of messages.

Depending on use with `trigger` or `keep` parameters, this type indicates either
when to trigger summarization or how much context to retain.

Example:
    ```python
    # ContextFraction
    context_size: ContextSize = ("fraction", 0.5)

    # ContextTokens
    context_size: ContextSize = ("tokens", 3000)

    # ContextMessages
    context_size: ContextSize = ("messages", 50)
    ```
"""


class TriggerClause(TypedDict, total=False):
    """Dictionary-based trigger specification for AND conditions.

    All specified thresholds in a single `TriggerClause` must be met for the clause to
    trigger summarization (AND semantics). When multiple clauses are provided in a list,
    summarization triggers if any clause is met (OR semantics).

    Example:
        ```python
        # AND: Trigger when tokens >= 4000 AND messages >= 10
        trigger_clause: TriggerClause = {"tokens": 4000, "messages": 10}

        # Use in a list for OR semantics:
        trigger_list: list[TriggerClause] = [
            {"tokens": 5000, "messages": 3},
            {"tokens": 3000, "messages": 6},
        ]
        ```
    """

    tokens: int
    """Trigger when the computed (or provider-reported) token count reaches or
    exceeds this value.
    """

    messages: int
    """Trigger when message count reaches or exceeds this value."""

    fraction: float
    """Trigger when the computed (or provider-reported) token count reaches or
    exceeds this fraction of the model's maximum input tokens.
    """


def _get_approximate_token_counter(model: BaseChatModel) -> TokenCounter:
    """Tune parameters of approximate token counter based on model type."""
    if model._llm_type.startswith("anthropic-chat"):  # noqa: SLF001
        # 3.3 was estimated in an offline experiment, comparing with Claude's token-counting
        # API: https://platform.claude.com/docs/en/build-with-claude/token-counting
        return partial(
            count_tokens_approximately, use_usage_metadata_scaling=True, chars_per_token=3.3
        )
    return partial(count_tokens_approximately, use_usage_metadata_scaling=True)


class SummarizationState(AgentState[ResponseT]):
    """State schema for `SummarizationMiddleware`.

    Extends `AgentState` with a counter tracking consecutive summary-generation
    failures for the current thread.

    Type Parameters:
        ResponseT: The type of the structured response. Defaults to `Any`.
    """

    summary_consecutive_failures: NotRequired[Annotated[int, PrivateStateAttr]]


def _build_summarization_failed_message(
    *,
    consecutive_failures: int,
    max_consecutive_summary_failures: int | None,
    total_tokens: int,
    hard_token_ceiling: int | None,
) -> str:
    """Build a message describing why summarization is being surfaced as an error.

    Args:
        consecutive_failures: Number of consecutive summary-generation failures.
        max_consecutive_summary_failures: Configured cap on consecutive failures.
        total_tokens: Current (pre-summarization) token count.
        hard_token_ceiling: Configured absolute token ceiling.

    Returns:
        A message describing which limit was exceeded.
    """
    if hard_token_ceiling is not None and total_tokens >= hard_token_ceiling:
        return (
            f"Summarization failed with token usage ({total_tokens}) at or above the "
            f"configured hard_token_ceiling ({hard_token_ceiling}). Refusing to silently "
            "skip summarization again - check logs for the underlying error(s)."
        )
    return (
        f"Summarization failed {consecutive_failures} consecutive time(s), reaching "
        f"max_consecutive_summary_failures ({max_consecutive_summary_failures}). Refusing to "
        "silently skip summarization again - check logs for the underlying error(s)."
    )


class SummarizationFailedError(Exception):
    """Raised when summary generation keeps failing and it is no longer safe to skip.

    A single summary-generation failure is not fatal: `SummarizationMiddleware` skips
    compaction for that turn and retries on the next one, so a transient error (e.g. a
    rate limit) never destroys or blocks the primary model call. This exception is
    raised instead once further silent retries are no longer safe - either
    `max_consecutive_summary_failures` consecutive failures have accumulated for this
    thread, or a failure occurs while token usage is already at or above
    `hard_token_ceiling` - so a persistently broken summarizer cannot let context grow
    unbounded without anyone noticing.
    """

    def __init__(
        self,
        *,
        consecutive_failures: int,
        max_consecutive_summary_failures: int | None,
        total_tokens: int,
        hard_token_ceiling: int | None,
    ) -> None:
        """Initialize the exception with failure and token usage information.

        Args:
            consecutive_failures: Number of consecutive summary-generation failures.
            max_consecutive_summary_failures: Configured cap on consecutive failures.
            total_tokens: Current (pre-summarization) token count.
            hard_token_ceiling: Configured absolute token ceiling.
        """
        self.consecutive_failures = consecutive_failures
        self.max_consecutive_summary_failures = max_consecutive_summary_failures
        self.total_tokens = total_tokens
        self.hard_token_ceiling = hard_token_ceiling

        msg = _build_summarization_failed_message(
            consecutive_failures=consecutive_failures,
            max_consecutive_summary_failures=max_consecutive_summary_failures,
            total_tokens=total_tokens,
            hard_token_ceiling=hard_token_ceiling,
        )
        super().__init__(msg)


class SummarizationMiddleware(AgentMiddleware[SummarizationState[ResponseT], ContextT, ResponseT]):
    """Summarizes conversation history when token limits are approached.

    This middleware monitors message token counts and automatically summarizes older
    messages when a threshold is reached, preserving recent messages and maintaining
    context continuity by ensuring AI/Tool message pairs remain together.

    A single summary-generation failure never blocks the primary model call: it is
    logged and summarization is skipped for that turn. However, failures are tracked
    per thread via `max_consecutive_summary_failures` and `hard_token_ceiling`, so a
    persistently broken summarizer raises
    [`SummarizationFailedError`][langchain.agents.middleware.summarization.SummarizationFailedError]
    instead of silently letting context usage grow unbounded.
    """

    state_schema = SummarizationState  # type: ignore[assignment]

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        trigger: (ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None) = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        max_consecutive_summary_failures: int | None = 3,
        hard_token_ceiling: int | None = None,
        **deprecated_kwargs: Any,
    ) -> None:
        """Initialize summarization middleware.

        Args:
            model: The language model to use for generating summaries.
            trigger: One or more thresholds that trigger summarization.

                Provide a single
                [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                tuple, or a single
                [`TriggerClause`][langchain.agents.middleware.summarization.TriggerClause]
                dict, or a list mixing either form.

                A `ContextSize` tuple expresses one threshold. A `TriggerClause` dict
                expresses multiple thresholds that must *all* be met (AND). When a list is
                provided, summarization runs if *any* item is met (OR).

                !!! example

                    ```python
                    # Trigger summarization when 50 messages is reached
                    ("messages", 50)

                    # Trigger summarization when 3000 tokens is reached
                    ("tokens", 3000)

                    # Trigger summarization either when 80% of model's max input tokens
                    # is reached or when 100 messages is reached (whichever comes first)
                    [("fraction", 0.8), ("messages", 100)]

                    # Trigger when tokens >= 4000 AND messages >= 10
                    {"tokens": 4000, "messages": 10}

                    # Trigger when (tokens >= 5000 AND messages >= 3) OR
                    # (tokens >= 3000 AND messages >= 6)
                    [{"tokens": 5000, "messages": 3}, {"tokens": 3000, "messages": 6}]
                    ```

                    See [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                    for more details.
            keep: Context retention policy applied after summarization.

                Provide a [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                tuple to specify how much history to preserve.

                Defaults to keeping the most recent `20` messages.

                Does not support multiple values like `trigger`.

                !!! example

                    ```python
                    # Keep the most recent 20 messages
                    ("messages", 20)

                    # Keep the most recent 3000 tokens
                    ("tokens", 3000)

                    # Keep the most recent 30% of the model's max input tokens
                    ("fraction", 0.3)
                    ```
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            trim_tokens_to_summarize: Maximum tokens to keep when preparing messages for
                the summarization call.

                Pass `None` to skip trimming entirely.
            max_consecutive_summary_failures: Maximum number of consecutive
                summary-generation failures tolerated for a given thread before
                raising
                [`SummarizationFailedError`][langchain.agents.middleware.summarization.SummarizationFailedError].

                Each individual failure is skipped (not raised) so a transient error
                never blocks the primary model call; the counter resets to `0` after
                any successful summarization. Pass `None` to retry indefinitely and
                never raise on consecutive failures alone.
            hard_token_ceiling: An absolute token count that, if reached or exceeded
                at the time a summarization attempt fails, immediately raises
                [`SummarizationFailedError`][langchain.agents.middleware.summarization.SummarizationFailedError]
                regardless of `max_consecutive_summary_failures`.

                Use this to fail fast when there is no headroom left to safely defer
                summarization to a later turn. Pass `None` (default) to disable this
                check.

        Raises:
            ValueError: If `max_consecutive_summary_failures` or `hard_token_ceiling`
                is not a positive integer.
        """
        if max_consecutive_summary_failures is not None and max_consecutive_summary_failures < 1:
            msg = (
                "max_consecutive_summary_failures must be a positive integer or None, "
                f"got {max_consecutive_summary_failures}."
            )
            raise ValueError(msg)

        if hard_token_ceiling is not None and hard_token_ceiling < 1:
            msg = (
                f"hard_token_ceiling must be a positive integer or None, got {hard_token_ceiling}."
            )
            raise ValueError(msg)

        # Handle deprecated parameters
        if "max_tokens_before_summary" in deprecated_kwargs:
            value = deprecated_kwargs["max_tokens_before_summary"]
            warnings.warn(
                "max_tokens_before_summary is deprecated. Use trigger=('tokens', value) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if trigger is None and value is not None:
                trigger = ("tokens", value)

        if "messages_to_keep" in deprecated_kwargs:
            value = deprecated_kwargs["messages_to_keep"]
            warnings.warn(
                "messages_to_keep is deprecated. Use keep=('messages', value) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if keep == ("messages", _DEFAULT_MESSAGES_TO_KEEP):
                keep = ("messages", value)

        super().__init__()

        if isinstance(model, str):
            model = init_chat_model(model)

        self.model = model

        self.trigger: ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None = (
            self._copy_trigger(trigger)
        )

        # Canonical trigger representation: AND within a clause, OR across clauses.
        self._trigger_clauses = self._normalize_trigger(self.trigger)
        # Legacy compatibility view for private consumers that inspected the previous
        # tuple-normalized representation. LangChain behavior is driven by
        # `_trigger_clauses`, not this attribute. Remove in LangChain 2.0.
        self._trigger_conditions = self._legacy_trigger_conditions(self.trigger)

        self.keep = self._validate_context_size(keep, "keep")
        if token_counter is count_tokens_approximately:
            self.token_counter = _get_approximate_token_counter(self.model)
            self._partial_token_counter: TokenCounter = partial(  # type: ignore[call-arg]
                self.token_counter, use_usage_metadata_scaling=False
            )
        else:
            self.token_counter = token_counter
            self._partial_token_counter = token_counter
        self.summary_prompt = summary_prompt
        self.trim_tokens_to_summarize = trim_tokens_to_summarize
        self.max_consecutive_summary_failures = max_consecutive_summary_failures
        self.hard_token_ceiling = hard_token_ceiling

        requires_profile = any("fraction" in clause for clause in self._trigger_clauses)
        if self.keep[0] == "fraction":
            requires_profile = True
        if requires_profile and self._get_profile_limits() is None:
            msg = (
                "Model profile information is required to use fractional token limits, "
                "and is unavailable for the specified model. Please use absolute token "
                "counts instead, or pass "
                '`\n\nChatModel(..., profile={"max_input_tokens": ...})`.\n\n'
                "with a desired integer value of the model's maximum input tokens."
            )
            raise ValueError(msg)

    @override
    def before_model(
        self, state: SummarizationState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Process messages before model invocation, potentially triggering summarization.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            An updated state with summarized messages if summarization was performed,
                or with an incremented failure counter if summarization failed but was
                skipped for this turn.

        Raises:
            SummarizationFailedError: If summary generation fails and either
                `max_consecutive_summary_failures` consecutive failures have
                accumulated for this thread, or `hard_token_ceiling` has been reached
                or exceeded.
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        summary = self._create_summary(messages_to_summarize)
        if summary is None:
            # Summary generation failed; leave the conversation untouched and retry
            # summarization on a later turn rather than destroying history, unless
            # retries are exhausted or we're already out of headroom.
            return self._handle_summary_failure(state, total_tokens)
        new_messages = self._build_new_messages(summary)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ],
            "summary_consecutive_failures": 0,
        }

    @override
    async def abefore_model(
        self, state: SummarizationState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Process messages before model invocation, potentially triggering summarization.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            An updated state with summarized messages if summarization was performed,
                or with an incremented failure counter if summarization failed but was
                skipped for this turn.

        Raises:
            SummarizationFailedError: If summary generation fails and either
                `max_consecutive_summary_failures` consecutive failures have
                accumulated for this thread, or `hard_token_ceiling` has been reached
                or exceeded.
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        summary = await self._acreate_summary(messages_to_summarize)
        if summary is None:
            # Summary generation failed; leave the conversation untouched and retry
            # summarization on a later turn rather than destroying history, unless
            # retries are exhausted or we're already out of headroom.
            return self._handle_summary_failure(state, total_tokens)
        new_messages = self._build_new_messages(summary)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ],
            "summary_consecutive_failures": 0,
        }

    def _handle_summary_failure(
        self, state: SummarizationState[Any], total_tokens: int
    ) -> dict[str, Any]:
        """Track a summary-generation failure and raise once it is no longer safe to skip.

        Args:
            state: The agent state, used to read the current consecutive-failure count.
            total_tokens: The (pre-summarization) token count for the conversation.

        Returns:
            A state update incrementing the consecutive-failure counter.

        Raises:
            SummarizationFailedError: If `max_consecutive_summary_failures` consecutive
                failures have accumulated for this thread, or `hard_token_ceiling` has
                been reached or exceeded.
        """
        consecutive_failures = state.get("summary_consecutive_failures", 0) + 1

        ceiling_reached = (
            self.hard_token_ceiling is not None and total_tokens >= self.hard_token_ceiling
        )
        retries_exhausted = (
            self.max_consecutive_summary_failures is not None
            and consecutive_failures >= self.max_consecutive_summary_failures
        )
        if ceiling_reached or retries_exhausted:
            raise SummarizationFailedError(
                consecutive_failures=consecutive_failures,
                max_consecutive_summary_failures=self.max_consecutive_summary_failures,
                total_tokens=total_tokens,
                hard_token_ceiling=self.hard_token_ceiling,
            )

        logger.warning(
            "Summarization failed (%d consecutive failure(s) for this thread); "
            "skipping summarization and retrying on a later turn.",
            consecutive_failures,
        )
        return {"summary_consecutive_failures": consecutive_failures}

    @staticmethod
    def _copy_trigger(
        trigger: ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None,
    ) -> ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None:
        """Copy mutable trigger containers so caller mutations do not affect this instance."""
        if isinstance(trigger, Mapping):
            return cast("TriggerClause", dict(trigger))
        if isinstance(trigger, list):
            return [
                cast("TriggerClause", dict(item)) if isinstance(item, Mapping) else item
                for item in trigger
            ]
        return trigger

    def _legacy_trigger_conditions(
        self,
        trigger: ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None,
    ) -> list[ContextSize]:
        """Project tuple-expressible triggers to the legacy private representation."""
        if trigger is None:
            return []
        if isinstance(trigger, tuple):
            return [self._validate_context_size(trigger, "trigger")]
        if isinstance(trigger, Mapping):
            if len(trigger) != 1:
                return []
            kind, value = next(iter(trigger.items()))
            return [self._validate_context_size(cast("ContextSize", (kind, value)), "trigger")]

        conditions: list[ContextSize] = []
        for item in trigger:
            if isinstance(item, tuple):
                conditions.append(self._validate_context_size(item, "trigger"))
            elif isinstance(item, Mapping) and len(item) == 1:
                kind, value = next(iter(item.items()))
                conditions.append(
                    self._validate_context_size(cast("ContextSize", (kind, value)), "trigger")
                )
        return conditions

    def _normalize_trigger(
        self,
        trigger: (ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None),
    ) -> list[TriggerClause]:
        """Normalize supported trigger inputs into list of Trigger clauses.

        - tuple ("tokens", 3000) -> [{"tokens": 3000}]
        - dict {"tokens": 4000, "messages": 10} -> [{"tokens": 4000, "messages": 10}]
        - list of either -> OR across items
        """
        if trigger is None:
            return []

        def _validate_and_convert_tuple(t: ContextSize) -> TriggerClause:
            kind, value = self._validate_context_size(t, "trigger")
            return cast("TriggerClause", {kind: value})

        def _validate_mapping(m: Mapping[str, Any]) -> TriggerClause:
            """Validate and convert a mapping to a TriggerClause.

            Type checks reject silent coercion (booleans, numeric strings, and
            fractional floats for integer metrics) so a misconfigured clause fails loudly
            at construction. Range and positivity checks are delegated to
            `_validate_context_size`, keeping a single source of truth for the rules and
            error messages shared with the tuple form.
            """
            if not m:
                msg = "trigger clause must specify at least one of 'tokens', 'messages', 'fraction'"
                raise ValueError(msg)
            out: dict[str, float | int] = {}
            for k, v in m.items():
                if k not in {"tokens", "messages", "fraction"}:
                    msg = f"Unsupported trigger metric: {k!r}"
                    raise ValueError(msg)
                # `bool` is an `int` subclass; reject it so `{"messages": True}` cannot
                # silently become a threshold of 1. Raise `ValueError` (not `TypeError`)
                # so every trigger-config error stays one catchable type.
                if isinstance(v, bool):
                    msg = f"{k} trigger value must be numeric, got {v!r}"
                    raise ValueError(msg)  # noqa: TRY004
                if k == "fraction":
                    if not isinstance(v, (int, float)):
                        msg = f"Fraction trigger values must be numeric, got {v!r}"
                        raise ValueError(msg)
                elif not isinstance(v, int):
                    # Reject floats and numeric strings rather than truncating/coercing.
                    msg = f"{k} trigger values must be integers, got {v!r}"
                    raise ValueError(msg)
                # Delegate range/positivity validation so dict and tuple forms share
                # identical rules and error messages.
                self._validate_context_size(cast("ContextSize", (k, v)), "trigger")
                out[k] = v
            return cast("TriggerClause", out)

        clauses: list[TriggerClause] = []
        # `trigger` may originate from untyped callers, so dispatch on the runtime type
        # and raise on anything unsupported.
        subject: Any = trigger
        if isinstance(subject, Mapping):
            clauses.append(_validate_mapping(subject))
        elif isinstance(subject, tuple):
            clauses.append(_validate_and_convert_tuple(cast("ContextSize", subject)))
        elif isinstance(subject, list):
            for item in subject:
                if isinstance(item, Mapping):
                    clauses.append(_validate_mapping(item))
                elif isinstance(item, tuple):
                    clauses.append(_validate_and_convert_tuple(cast("ContextSize", item)))
                else:
                    msg = f"Unsupported trigger item type: {type(item)}"
                    raise TypeError(msg)
        else:
            msg = f"Unsupported trigger type: {type(subject)}"
            raise TypeError(msg)
        return clauses

    def _should_summarize_based_on_reported_tokens(
        self, messages: list[AnyMessage], threshold: float
    ) -> bool:
        """Check if reported token usage from last AIMessage exceeds threshold."""
        last_ai_message = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )
        if (  # noqa: SIM103
            isinstance(last_ai_message, AIMessage)
            and last_ai_message.usage_metadata is not None
            and (reported_tokens := last_ai_message.usage_metadata.get("total_tokens", -1))
            and reported_tokens >= threshold
            and (message_provider := last_ai_message.response_metadata.get("model_provider"))
            and _provider_matches(
                message_provider,
                self.model._get_ls_params().get("ls_provider"),  # noqa: SLF001
            )
        ):
            return True
        return False

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Determine whether summarization should run for the current token usage."""
        if not self._trigger_clauses:
            return False

        for clause in self._trigger_clauses:
            clause_met = True
            for kind, value in clause.items():
                if kind == "messages" and len(messages) < cast("int", value):
                    clause_met = False
                    break
                if kind == "tokens":
                    threshold_tokens = cast("int", value)
                    # Trigger if total tokens exceed threshold OR reported tokens do
                    if (
                        total_tokens < threshold_tokens
                        and not self._should_summarize_based_on_reported_tokens(
                            messages, float(threshold_tokens)
                        )
                    ):
                        clause_met = False
                        break
                if kind == "fraction":
                    max_input_tokens = self._get_profile_limits()
                    if max_input_tokens is None:
                        clause_met = False
                        break
                    threshold = int(max_input_tokens * cast("float", value))
                    if threshold <= 0:
                        threshold = 1
                    if (
                        total_tokens < threshold
                        and not self._should_summarize_based_on_reported_tokens(
                            messages, float(threshold)
                        )
                    ):
                        clause_met = False
                        break
            if clause_met:
                return True
        return False

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        """Choose cutoff index respecting retention configuration."""
        kind, value = self.keep
        if kind in {"tokens", "fraction"}:
            token_based_cutoff = self._find_token_based_cutoff(messages)
            if token_based_cutoff is not None:
                return token_based_cutoff
            # None cutoff -> model profile data not available (caught in __init__ but
            # here for safety), fallback to message count
            return self._find_safe_cutoff(messages, _DEFAULT_MESSAGES_TO_KEEP)
        return self._find_safe_cutoff(messages, cast("int", value))

    def _find_token_based_cutoff(self, messages: list[AnyMessage]) -> int | None:
        """Find cutoff index based on target token retention."""
        if not messages:
            return 0

        kind, value = self.keep
        if kind == "fraction":
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return None
            target_token_count = int(max_input_tokens * value)
        elif kind == "tokens":
            target_token_count = int(value)
        else:
            return None

        if target_token_count <= 0:
            target_token_count = 1

        if self.token_counter(messages) <= target_token_count:
            return 0

        # Use binary search to identify the earliest message index that keeps the
        # suffix within the token budget.
        left, right = 0, len(messages)
        cutoff_candidate = len(messages)
        max_iterations = len(messages).bit_length() + 1
        for _ in range(max_iterations):
            if left >= right:
                break

            mid = (left + right) // 2
            if self._partial_token_counter(messages[mid:]) <= target_token_count:
                cutoff_candidate = mid
                right = mid
            else:
                left = mid + 1

        if cutoff_candidate == len(messages):
            cutoff_candidate = left

        if cutoff_candidate >= len(messages):
            if len(messages) == 1:
                return 0
            cutoff_candidate = len(messages) - 1

        # Advance past any ToolMessages to avoid splitting AI/Tool pairs
        return self._find_safe_cutoff_point(messages, cutoff_candidate)

    def _get_profile_limits(self) -> int | None:
        """Retrieve max input token limit from the model profile."""
        try:
            profile = self.model.profile
        except AttributeError:
            return None

        if not isinstance(profile, Mapping):
            return None

        max_input_tokens = profile.get("max_input_tokens")

        if not isinstance(max_input_tokens, int):
            return None

        return max_input_tokens

    @staticmethod
    def _validate_context_size(context: ContextSize, parameter_name: str) -> ContextSize:
        """Validate context configuration tuples."""
        kind, value = context
        if kind == "fraction":
            if not 0 < value <= 1:
                msg = f"Fractional {parameter_name} values must be between 0 and 1, got {value}."
                raise ValueError(msg)
        elif kind in {"tokens", "messages"}:
            if value <= 0:
                msg = f"{parameter_name} thresholds must be greater than 0, got {value}."
                raise ValueError(msg)
        else:
            msg = f"Unsupported context size type {kind} for {parameter_name}."
            raise ValueError(msg)
        return context

    @staticmethod
    def _build_new_messages(summary: str) -> list[HumanMessage]:
        return [
            HumanMessage(
                content=f"Here is a summary of the conversation to date:\n\n{summary}",
                additional_kwargs={"lc_source": "summarization"},
            )
        ]

    @staticmethod
    def _ensure_message_ids(messages: list[AnyMessage]) -> None:
        """Ensure all messages have unique IDs for the add_messages reducer."""
        for msg in messages:
            if msg.id is None:
                msg.id = str(uuid.uuid4())

    @staticmethod
    def _partition_messages(
        conversation_messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """Partition messages into those to summarize and those to preserve."""
        messages_to_summarize = conversation_messages[:cutoff_index]
        preserved_messages = conversation_messages[cutoff_index:]

        return messages_to_summarize, preserved_messages

    def _find_safe_cutoff(self, messages: list[AnyMessage], messages_to_keep: int) -> int:
        """Find safe cutoff point that preserves AI/Tool message pairs.

        Returns the index where messages can be safely cut without separating
        related AI and Tool messages. Returns `0` if no safe cutoff is found.

        This is aggressive with summarization - if the target cutoff lands in the
        middle of tool messages, we advance past all of them (summarizing more).
        """
        if len(messages) <= messages_to_keep:
            return 0

        target_cutoff = len(messages) - messages_to_keep
        return self._find_safe_cutoff_point(messages, target_cutoff)

    @staticmethod
    def _find_safe_cutoff_point(messages: list[AnyMessage], cutoff_index: int) -> int:
        """Find a safe cutoff point that doesn't split AI/Tool message pairs.

        If the message at `cutoff_index` is a `ToolMessage`, search backward for the
        `AIMessage` containing the corresponding `tool_calls` and adjust the cutoff to
        include it. This ensures tool call requests and responses stay together.

        Falls back to advancing forward past `ToolMessage` objects only if no matching
        `AIMessage` is found (edge case).
        """
        if cutoff_index >= len(messages) or not isinstance(messages[cutoff_index], ToolMessage):
            return cutoff_index

        # Collect tool_call_ids from consecutive ToolMessages at/after cutoff
        tool_call_ids: set[str] = set()
        idx = cutoff_index
        while idx < len(messages) and isinstance(messages[idx], ToolMessage):
            tool_msg = cast("ToolMessage", messages[idx])
            if tool_msg.tool_call_id:
                tool_call_ids.add(tool_msg.tool_call_id)
            idx += 1

        # Search backward for AIMessage with matching tool_calls
        for i in range(cutoff_index - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage) and msg.tool_calls:
                ai_tool_call_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
                if tool_call_ids & ai_tool_call_ids:
                    # Found the AIMessage - move cutoff to include it
                    return i

        # Fallback: no matching AIMessage found, advance past ToolMessages to avoid
        # orphaned tool responses
        return idx

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str | None:
        """Generate summary for the given messages.

        Args:
            messages_to_summarize: Messages to summarize.

        Returns:
            The generated summary, or `None` if summary generation failed. A `None`
            return must never be treated as a valid summary by callers.
        """
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        # Serialize as XML so URL-based multimodal blocks remain visible in the summary
        # prompt while excluding raw message metadata from the token budget.
        formatted_messages = get_buffer_string(trimmed_messages, format="xml")

        try:
            response = self.model.invoke(
                self.summary_prompt.format(messages=formatted_messages).rstrip(),
                config={"metadata": {"lc_source": "summarization"}},
            )
            return response.text.strip()
        except Exception:
            logger.warning("Summary generation failed; skipping summarization.", exc_info=True)
            return None

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str | None:
        """Generate summary for the given messages.

        Args:
            messages_to_summarize: Messages to summarize.

        Returns:
            The generated summary, or `None` if summary generation failed. A `None`
            return must never be treated as a valid summary by callers.
        """
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        # Serialize as XML so URL-based multimodal blocks remain visible in the summary
        # prompt while excluding raw message metadata from the token budget.
        formatted_messages = get_buffer_string(trimmed_messages, format="xml")

        try:
            response = await self.model.ainvoke(
                self.summary_prompt.format(messages=formatted_messages).rstrip(),
                config={"metadata": {"lc_source": "summarization"}},
            )
            return response.text.strip()
        except Exception:
            logger.warning("Summary generation failed; skipping summarization.", exc_info=True)
            return None

    def _trim_messages_for_summary(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Trim messages to fit within summary generation limits."""
        try:
            if self.trim_tokens_to_summarize is None:
                return messages
            return cast(
                "list[AnyMessage]",
                trim_messages(
                    messages,
                    max_tokens=self.trim_tokens_to_summarize,
                    token_counter=self.token_counter,
                    start_on="human",
                    strategy="last",
                    allow_partial=True,
                    include_system=True,
                ),
            )
        except Exception:
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]
