"""Summarization middleware."""

import logging
import uuid
import warnings
from collections.abc import Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, TypedDict, cast

from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
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
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables.base import coerce_to_runnable
from langgraph.graph.message import (
    REMOVE_ALL_MESSAGES,
)
from langgraph.runtime import Runtime
from typing_extensions import override

from langchain.agents.middleware._retry import default_retry_on
from langchain.agents.middleware.internal_call_transformer import (
    InternalCallTransformer,
    internal_call_metadata,
)
from langchain.agents.middleware.model_selection import (
    current_model_spec,
    init_model_from_spec,
    model_spec_from_model,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelSpec,
    ResponseT,
)
from langchain.chat_models import BaseChatModel, init_chat_model

logger = logging.getLogger(__name__)

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

Summarizer = (
    str
    | BaseChatModel
    | Runnable[list[AnyMessage], str]
    | Callable[[list[AnyMessage]], str]
    | Callable[[list[AnyMessage]], Coroutine[Any, Any, str]]
)
"""Generates a summary from the messages selected for summarization.

Either a chat model (or model string) prompted with the summary prompt, or a
`Runnable` or callable taking the messages and returning the summary text.
"""

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


def _summary_config() -> RunnableConfig:
    """Config for summary model calls."""
    return {"metadata": {"lc_source": "summarization", **internal_call_metadata()}}


def _check_summary(summary: object) -> str:
    """Validate and normalize the text returned by a custom `summarizer`."""
    if not isinstance(summary, str):
        msg = f"`summarizer` must return a str, got {type(summary).__name__}."
        raise TypeError(msg)
    return summary.strip()


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
        return _approximate_token_counter("anthropic")
    return _approximate_token_counter(None)


def _approximate_token_counter(provider: str | None) -> TokenCounter:
    """Approximate token counter tuned for a provider."""
    if provider in _ANTHROPIC_PROVIDERS:
        # 3.3 was estimated in an offline experiment, comparing with Claude's token-counting
        # API: https://platform.claude.com/docs/en/build-with-claude/token-counting
        return partial(
            count_tokens_approximately, use_usage_metadata_scaling=True, chars_per_token=3.3
        )
    return partial(count_tokens_approximately, use_usage_metadata_scaling=True)


_ANTHROPIC_PROVIDERS = frozenset({"anthropic", "google_anthropic_vertex"})


@dataclass(frozen=True)
class _CallModels:
    """Context and summary models resolved for one summarization check.

    Built per check from build-time configuration and the agent state, and passed
    down explicitly; never stored on the middleware, since one compiled agent
    serves many concurrent runs.
    """

    label: str | None
    """Identifier of the context model, or `None` when no model is known yet."""

    max_input_tokens: int | None
    """Context window of the context model, for `fraction` limits."""

    ls_provider: str | None
    """Provider of the context model, for provider-reported token counts."""

    token_counter: TokenCounter
    partial_token_counter: TokenCounter

    summary_chat_model: BaseChatModel | None
    """Chat model that writes summaries, when already built."""

    summary_runnable: Runnable[LanguageModelInput, BaseMessage] | None
    """`summary_chat_model` wrapped with retries."""

    spec: ModelSpec | None = None
    """Recorded context model, rebuilt to write summaries when nothing else is set."""


def _spec_label(spec: ModelSpec) -> str:
    return f"{spec['provider']}:{spec['model']}"


def _model_label(model: BaseChatModel) -> str:
    spec = model_spec_from_model(model)
    return _spec_label(spec) if spec else type(model).__name__


def _same_spec_model(first: ModelSpec, second: ModelSpec) -> bool:
    return first["provider"] == second["provider"] and first["model"] == second["model"]


class SummarizationMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Summarizes conversation history when token limits are approached.

    This middleware monitors message token counts and automatically summarizes older
    messages when a threshold is reached, preserving recent messages and maintaining
    context continuity by ensuring AI/Tool message pairs remain together.

    Summary-generation errors are retried in-process, up to 3 attempts total, via
    `Runnable.with_retry`, unless the error is a `ModelError` marked non-retryable
    (e.g. authentication or invalid request errors). If the summary prompt exceeds
    the model's context window (`ContextOverflowError`), the call is retried once with
    the middle of the history cut out. If a summary call still fails, the
    underlying error propagates rather than fabricating a summary.

    To generate summaries with something other than a chat model, pass a
    `summarizer`. The middleware still decides when to summarize and which
    messages to summarize; only the summary text comes from the summarizer.

    By default the middleware follows the agent's current model, which the agent
    records in state under `model_selection` (see
    [`ModelSelection`][langchain.agents.middleware.types.ModelSelection]) after
    every model call, including models swapped in by other middleware (a `/model`
    command, a router, a fallback). Before each call it resolves one *context
    model*, which drives trigger thresholds, token counting, provider-reported
    token checks and trimming, and one *summary model*, which writes the summary:
    `summarizer` if set, otherwise the context model.
    """

    transformers = (InternalCallTransformer,)
    """Keeps the summarization model call's tokens out of `run.messages`.

    Registered only when this middleware is used — see
    `InternalCallTransformer` for why the call needs tagging and filtering.
    """

    def __init__(
        self,
        model: str | BaseChatModel | None = None,
        *,
        trigger: (ContextSize | TriggerClause | list[ContextSize | TriggerClause] | None) = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        summarizer: Summarizer | None = None,
        **deprecated_kwargs: Any,
    ) -> None:
        """Initialize summarization middleware.

        Args:
            model: The context model: the model whose context window is managed.

                It drives trigger thresholds (including `fraction` limits), token
                counting, provider-reported token checks and trimming, and writes
                summaries unless `summarizer` is set.

                - `None`: follow the agent's current model, read from the
                    `model_selection` state key. Recommended. Until the agent has
                    recorded a model (before its first model call), only limits
                    that don't need a context window apply. Without `summarizer`,
                    summaries are written by the current model, rebuilt from its
                    recorded spec with `init_chat_model` (credentials come from the
                    environment); pass `summarizer` if models are built with
                    explicit credentials or custom endpoints.
                - A chat model, or a model string resolved with `init_chat_model`:
                    pin the context model, whatever the agent's current model.

                !!! warning "Pinning `model` to choose a summary-writing model"

                    If a pinned `model` differs from the agent's current model, a
                    `LangChainDeprecationWarning` is emitted. To summarize with a
                    cheaper model while following the agent's model, pass
                    `summarizer=` instead and omit `model`.

                !!! example

                    ```python
                    # Follow the agent's current model
                    SummarizationMiddleware(trigger=("fraction", 0.8))

                    # Follow the agent's current model, write summaries with gpt-6-luna
                    SummarizationMiddleware(
                        trigger=("fraction", 0.8),
                        summarizer="openai:gpt-6-luna",
                    )

                    # Pin the context model
                    SummarizationMiddleware(model="openai:gpt-6-sol", trigger=("fraction", 0.8))
                    ```
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
            summarizer: Optional replacement for the summarization model call.

                The context model (see `model`) is still used for its profile
                (fractional limits) and token counting, but is not called to
                write summaries. `summarizer` takes precedence over every form
                of `model`.

                - A chat model, or a model string resolved with
                    `init_chat_model`, writes summaries in place of the context
                    model. It is prompted with `summary_prompt` and retried like
                    the context model would be. Use this to summarize with a
                    different model than the one whose context window you're
                    managing. To summarize with a chat
                    model, pass its name or a `BaseChatModel` instance; wrapped
                    models such as `model.with_config(...)` are treated as a
                    `Runnable` (below).
                - A `Runnable` or function receives the messages to summarize
                    (after trimming) and returns the summary text. `summary_prompt`
                    is not used. Functions may be sync or async; an async function
                    requires async invocation (e.g. `ainvoke`), and a sync function
                    runs in a thread pool when invoked asynchronously.

                !!! example

                    ```python
                    # Manage the current model's context window, summarize with gpt-6-luna
                    SummarizationMiddleware(
                        trigger=("fraction", 0.8),
                        summarizer="openai:gpt-6-luna",
                    )


                    # Summarize with a non-LLM service
                    async def summarize(messages: Sequence[BaseMessage]) -> str:
                        return await my_summarization_service(get_buffer_string(messages))


                    SummarizationMiddleware(
                        trigger=("fraction", 0.8),
                        summarizer=summarize,
                    )
                    ```
        """
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

        self.model: BaseChatModel | None = None
        subject: Any = model
        if isinstance(subject, BaseChatModel) or hasattr(subject, "invoke"):
            self.model = subject
        elif subject is not None:
            msg = (
                "`model` must be a model string, a `BaseChatModel`, or `None`; "
                f"got {type(subject).__name__}."
            )
            raise TypeError(msg)
        self._pinned_spec: ModelSpec | None = (
            model_spec_from_model(self.model) if isinstance(self.model, BaseChatModel) else None
        )
        # Retry up to 3 attempts total, skipping errors marked non-retryable.
        self._summary_model: Runnable[LanguageModelInput, BaseMessage] | None = (
            self.model.with_retry(retry_if_exception_type=default_retry_on)
            if self.model is not None
            else None
        )

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
        self._uses_default_token_counter = token_counter is count_tokens_approximately
        if self._uses_default_token_counter:
            self.token_counter = (
                _get_approximate_token_counter(self.model)
                if self.model is not None
                else partial(count_tokens_approximately, use_usage_metadata_scaling=True)
            )
            self._partial_token_counter: TokenCounter = partial(  # type: ignore[call-arg]
                self.token_counter, use_usage_metadata_scaling=False
            )
        else:
            self.token_counter = token_counter
            self._partial_token_counter = token_counter
        self.summary_prompt = summary_prompt
        self.trim_tokens_to_summarize = trim_tokens_to_summarize
        if isinstance(summarizer, str):
            summarizer = init_chat_model(summarizer)
        # The chat model that writes summaries: the context model unless `summarizer`
        # replaces it. `None` when it is rebuilt from the agent's current model.
        self._summary_chat_model: BaseChatModel | None = self.model
        self.summarizer: Runnable[list[AnyMessage], str] | None = None
        if isinstance(summarizer, BaseChatModel):
            self._summary_chat_model = summarizer
            self._summary_model = summarizer.with_retry(retry_if_exception_type=default_retry_on)
        elif summarizer is not None:
            self.summarizer = coerce_to_runnable(summarizer)

        if self.model is not None and self._requires_profile and self._get_profile_limits() is None:
            msg = (
                "Model profile information is required to use fractional token limits, "
                "and is unavailable for the specified model. Please use absolute token "
                "counts instead, or pass "
                '`\n\nChatModel(..., profile={"max_input_tokens": ...})`.\n\n'
                "with a desired integer value of the model's maximum input tokens."
            )
            raise ValueError(msg)

    @property
    def _requires_profile(self) -> bool:
        """Whether `trigger` or `keep` uses limits relative to the model's context window."""
        return self.keep[0] == "fraction" or any(
            "fraction" in clause for clause in self._trigger_clauses
        )

    @override
    def before_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Process messages before model invocation, potentially triggering summarization.

        Thresholds and token counts use the context model: the pinned `model`, or else
        the model recorded in state under `model_selection`.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            An updated state with summarized messages if summarization was performed.

        Raises:
            Exception: If summary generation still fails once retries are exhausted.
        """
        models = self._resolve_call_models(state)
        plan = self._plan_summary(state["messages"], models)
        if plan is None:
            return None
        messages_to_summarize, preserved_messages = plan
        summary = self._traced_summarizer(models).invoke(
            messages_to_summarize, config=self._summary_run_config(models)
        )
        return self._summary_update(summary, preserved_messages)

    @override
    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Process messages before model invocation, potentially triggering summarization.

        Thresholds and token counts use the context model: the pinned `model`, or else
        the model recorded in state under `model_selection`.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            An updated state with summarized messages if summarization was performed.

        Raises:
            Exception: If summary generation still fails once retries are exhausted.
        """
        models = self._resolve_call_models(state)
        plan = self._plan_summary(state["messages"], models)
        if plan is None:
            return None
        messages_to_summarize, preserved_messages = plan
        summary = await self._traced_summarizer(models).ainvoke(
            messages_to_summarize, config=self._summary_run_config(models)
        )
        return self._summary_update(summary, preserved_messages)

    def _plan_summary(
        self, messages: list[AnyMessage], models: _CallModels
    ) -> tuple[list[AnyMessage], list[AnyMessage]] | None:
        """Split messages into those to summarize and those to keep, or `None` to skip."""
        if self.summarizer is None and models.summary_runnable is None and models.spec is None:
            return None
        self._ensure_message_ids(messages)
        total_tokens = models.token_counter(messages)
        if not self._should_summarize(messages, total_tokens, models=models):
            return None
        cutoff_index = self._determine_cutoff_index(messages, models=models)
        if cutoff_index <= 0:
            return None
        return self._partition_messages(messages, cutoff_index)

    def _summary_update(self, summary: str, preserved_messages: list[AnyMessage]) -> dict[str, Any]:
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *self._build_new_messages(summary),
                *preserved_messages,
            ]
        }

    def _resolve_call_models(self, state: Mapping[str, Any]) -> _CallModels:
        """Resolve the context and summary models for one summarization check."""
        current = current_model_spec(state)
        if self.model is not None:
            if current and self._pinned_spec and not _same_spec_model(current, self._pinned_spec):
                self._warn_pinned_model_differs(current)
            return self._fixed_call_models()
        return self._call_models_for_spec(current)

    def _warn_pinned_model_differs(self, current: ModelSpec) -> None:
        pinned = cast("ModelSpec", self._pinned_spec)
        msg = (
            f"`SummarizationMiddleware` is pinned to `model={_spec_label(pinned)}`, but "
            f"the agent's current model is `{_spec_label(current)}`. Trigger thresholds, "
            "token counting and summaries use the pinned model. Using `model=` to choose "
            "a model for writing summaries is deprecated: pass `summarizer=` instead and "
            "omit `model=` so thresholds follow the agent's current model. Keep `model=` "
            "only to deliberately pin the context model."
        )
        warnings.warn(msg, LangChainDeprecationWarning, stacklevel=3)

    def _call_models_for_spec(self, spec: ModelSpec | None) -> _CallModels:
        """Models for the context model recorded in state (or none recorded yet).

        Derived per check (cheap dict reads and `partial`), so nothing per-run is
        cached on the instance. Rebuilding a summary-writing model from the spec is
        deferred until a summary is actually written.
        """
        provider = spec["provider"] if spec else None
        if self._uses_default_token_counter:
            token_counter = _approximate_token_counter(provider)
            partial_token_counter: TokenCounter = partial(  # type: ignore[call-arg]
                token_counter, use_usage_metadata_scaling=False
            )
        else:
            token_counter = partial_token_counter = self.token_counter
        summary_chat_model = None if self.summarizer is not None else self._summary_chat_model
        return _CallModels(
            label=_spec_label(spec) if spec else None,
            max_input_tokens=spec.get("max_input_tokens") if spec else None,
            ls_provider=provider,
            token_counter=token_counter,
            partial_token_counter=partial_token_counter,
            summary_chat_model=summary_chat_model,
            summary_runnable=self._summary_model if summary_chat_model is not None else None,
            spec=spec,
        )

    def _fixed_call_models(self) -> _CallModels:
        """Models for the pinned `model`, read from instance attributes."""
        if self.model is None:
            msg = (
                "This `SummarizationMiddleware` has no pinned `model`; its context model "
                "is read from the agent state's `model_selection`."
            )
            raise ValueError(msg)
        uses_summarizer = self.summarizer is not None
        return _CallModels(
            label=_model_label(self.model),
            max_input_tokens=self._get_profile_limits(),
            ls_provider=self.model._get_ls_params().get("ls_provider"),  # noqa: SLF001
            token_counter=self.token_counter,
            partial_token_counter=self._partial_token_counter,
            summary_chat_model=None if uses_summarizer else self._summary_chat_model,
            summary_runnable=None if uses_summarizer else self._summary_model,
        )

    def _summary_runnable(self, models: _CallModels) -> Runnable[LanguageModelInput, BaseMessage]:
        """The retrying chat model that writes this summary."""
        if models.summary_runnable is not None:
            return models.summary_runnable
        if models.spec is None:
            msg = "`SummarizationMiddleware` has no model to write the summary with."
            raise ValueError(msg)
        try:
            model = init_model_from_spec(models.spec)
        except Exception as e:
            msg = (
                f"`SummarizationMiddleware` could not rebuild `{_spec_label(models.spec)}` "
                "from the agent state's `model_selection` to write a summary. Pass "
                "`summarizer=` (or pin `model=`) to choose the summary model explicitly."
            )
            raise ValueError(msg) from e
        return model.with_retry(retry_if_exception_type=default_retry_on)

    def _traced_summarizer(self, models: _CallModels) -> Runnable[list[AnyMessage], str]:
        """Wrap summary generation in its own run so its metadata shows in traces."""

        def summarize(messages: list[AnyMessage]) -> str:
            return self._create_summary(messages, models=models)

        async def asummarize(messages: list[AnyMessage]) -> str:
            return await self._acreate_summary(messages, models=models)

        return RunnableLambda(summarize, afunc=asummarize, name="summarization")

    def _summary_run_config(self, models: _CallModels) -> RunnableConfig:
        """Config for the summarization run, recording which models compacted."""
        metadata: dict[str, Any] = {
            "lc_source": "summarization",
            "summarization_context_model": models.label,
            "summarization_context_window": models.max_input_tokens,
        }
        if models.summary_chat_model is not None:
            metadata["summarization_model"] = _model_label(models.summary_chat_model)
        elif self.summarizer is None:
            metadata["summarization_model"] = models.label
        return {"metadata": metadata}

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
        self,
        messages: list[AnyMessage],
        threshold: float,
        *,
        models: _CallModels | None = None,
    ) -> bool:
        """Check if reported token usage from last AIMessage exceeds threshold.

        Only counts usage reported by the same provider as the context model.
        """
        ls_provider = (models or self._fixed_call_models()).ls_provider
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
            and _provider_matches(message_provider, ls_provider)
        ):
            return True
        return False

    def _should_summarize(
        self,
        messages: list[AnyMessage],
        total_tokens: int,
        *,
        models: _CallModels | None = None,
    ) -> bool:
        """Determine whether summarization should run for the current token usage."""
        if not self._trigger_clauses:
            return False
        models = models or self._fixed_call_models()

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
                            messages, float(threshold_tokens), models=models
                        )
                    ):
                        clause_met = False
                        break
                if kind == "fraction":
                    max_input_tokens = models.max_input_tokens
                    if max_input_tokens is None:
                        self._warn_missing_profile(models)
                        clause_met = False
                        break
                    threshold = int(max_input_tokens * cast("float", value))
                    if threshold <= 0:
                        threshold = 1
                    if (
                        total_tokens < threshold
                        and not self._should_summarize_based_on_reported_tokens(
                            messages, float(threshold), models=models
                        )
                    ):
                        clause_met = False
                        break
            if clause_met:
                return True
        return False

    def _determine_cutoff_index(
        self, messages: list[AnyMessage], *, models: _CallModels | None = None
    ) -> int:
        """Choose cutoff index respecting retention configuration."""
        kind, value = self.keep
        if kind in {"tokens", "fraction"}:
            token_based_cutoff = self._find_token_based_cutoff(messages, models=models)
            if token_based_cutoff is not None:
                return token_based_cutoff
            # None cutoff -> model profile data not available (caught in __init__ but
            # here for safety), fallback to message count
            return self._find_safe_cutoff(messages, _DEFAULT_MESSAGES_TO_KEEP)
        return self._find_safe_cutoff(messages, cast("int", value))

    def _find_token_based_cutoff(
        self, messages: list[AnyMessage], *, models: _CallModels | None = None
    ) -> int | None:
        """Find cutoff index based on target token retention."""
        if not messages:
            return 0

        models = models or self._fixed_call_models()
        kind, value = self.keep
        if kind == "fraction":
            max_input_tokens = models.max_input_tokens
            if max_input_tokens is None:
                self._warn_missing_profile(models)
                return None
            target_token_count = int(max_input_tokens * value)
        elif kind == "tokens":
            target_token_count = int(value)
        else:
            return None

        if target_token_count <= 0:
            target_token_count = 1

        if models.token_counter(messages) <= target_token_count:
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
            if models.partial_token_counter(messages[mid:]) <= target_token_count:
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

    def _get_profile_limits(self, model: BaseChatModel | None = None) -> int | None:
        """Retrieve max input token limit from the profile of `model` (default: `self.model`)."""
        model = model or self.model
        if model is None:
            return None
        try:
            profile = model.profile
        except AttributeError:
            return None

        if not isinstance(profile, Mapping):
            return None

        max_input_tokens = profile.get("max_input_tokens")

        if not isinstance(max_input_tokens, int):
            return None

        return max_input_tokens

    @staticmethod
    def _warn_missing_profile(models: _CallModels) -> None:
        if models.label is None:
            return
        msg = (
            f"`SummarizationMiddleware` uses fractional limits, but `{models.label}` "
            "has no `max_input_tokens` in its profile, so those limits are skipped for "
            "this call. Pass absolute token counts, or set "
            '`profile={"max_input_tokens": ...}` on the model.'
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

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

    def _create_summary(
        self, messages_to_summarize: list[AnyMessage], *, models: _CallModels | None = None
    ) -> str:
        """Generate summary for the given messages.

        Args:
            messages_to_summarize: Messages to summarize.
            models: Models resolved for the current call; defaults to the pinned
                `model`.

        Returns:
            The generated summary.

        Raises:
            Exception: If the summary model call still fails once retries (see
                `Runnable.with_retry`) are exhausted.
        """
        if not messages_to_summarize:
            return "No previous conversation history."

        models = models or self._fixed_call_models()
        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize, models=models)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        if self.summarizer is not None:
            return _check_summary(
                self.summarizer.invoke(trimmed_messages, config=_summary_config())
            )

        summary_model = self._summary_runnable(models)
        prompt = self._format_summary_prompt(trimmed_messages)
        try:
            response = summary_model.invoke(prompt, config=_summary_config())
        except ContextOverflowError:
            retry_prompt = self._overflow_retry_prompt(trimmed_messages, models=models)
            if retry_prompt is None:
                raise
            response = summary_model.invoke(retry_prompt, config=_summary_config())
        return response.text.strip()

    async def _acreate_summary(
        self, messages_to_summarize: list[AnyMessage], *, models: _CallModels | None = None
    ) -> str:
        """Generate summary for the given messages.

        Args:
            messages_to_summarize: Messages to summarize.
            models: Models resolved for the current call; defaults to the pinned
                `model`.

        Returns:
            The generated summary.

        Raises:
            Exception: If the summary model call still fails once retries (see
                `Runnable.with_retry`) are exhausted.
        """
        if not messages_to_summarize:
            return "No previous conversation history."

        models = models or self._fixed_call_models()
        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize, models=models)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        if self.summarizer is not None:
            summary = await self.summarizer.ainvoke(trimmed_messages, config=_summary_config())
            return _check_summary(summary)

        summary_model = self._summary_runnable(models)
        prompt = self._format_summary_prompt(trimmed_messages)
        try:
            response = await summary_model.ainvoke(prompt, config=_summary_config())
        except ContextOverflowError:
            retry_prompt = self._overflow_retry_prompt(trimmed_messages, models=models)
            if retry_prompt is None:
                raise
            response = await summary_model.ainvoke(retry_prompt, config=_summary_config())
        return response.text.strip()

    def _format_summary_prompt(self, messages: list[AnyMessage]) -> str:
        """Render the summary prompt for the given messages."""
        # Serialize as XML so URL-based multimodal blocks remain visible in the summary
        # prompt while excluding raw message metadata from the token budget.
        formatted_messages = get_buffer_string(messages, format="xml")
        return self.summary_prompt.format(messages=formatted_messages).rstrip()

    def _overflow_retry_prompt(
        self, messages: list[AnyMessage], *, models: _CallModels | None = None
    ) -> str | None:
        """Build a smaller summary prompt after the summary call overflowed.

        Cuts the middle out of the formatted history, keeping its start (where the
        objective is usually stated) and its most recent part, within half of its
        token count and 40% of the summary model's `max_input_tokens` when known.

        Returns:
            The retry prompt, or `None` if nothing fits the budget.
        """
        models = models or self._fixed_call_models()
        formatted = get_buffer_string(messages, format="xml")
        tokens = models.partial_token_counter([HumanMessage(content=formatted)])
        budget = tokens // 2
        max_input_tokens = (
            self._get_profile_limits(models.summary_chat_model)
            if models.summary_chat_model is not None
            else models.max_input_tokens
        )
        if max_input_tokens is not None:
            # 40% rather than half absorbs up to 1.25x usage-metadata scaling.
            budget = min(budget, max_input_tokens * 2 // 5)
        keep_chars = len(formatted) * budget // max(tokens, 1)
        if keep_chars <= 0:
            return None
        head_chars = keep_chars // 4
        tail_chars = keep_chars - head_chars
        logger.debug(
            "Summary prompt exceeded the model's context window; retrying with "
            "~%d of ~%d history tokens.",
            budget,
            tokens,
        )
        truncated = (
            f"{formatted[:head_chars]}\n\n[... middle of conversation omitted ...]\n\n"
            f"{formatted[-tail_chars:]}"
        )
        return self.summary_prompt.format(messages=truncated).rstrip()

    def _trim_messages_for_summary(
        self, messages: list[AnyMessage], *, models: _CallModels | None = None
    ) -> list[AnyMessage]:
        """Trim messages to fit within summary generation limits."""
        try:
            if self.trim_tokens_to_summarize is None:
                return messages
            models = models or self._fixed_call_models()
            return cast(
                "list[AnyMessage]",
                trim_messages(
                    messages,
                    max_tokens=self.trim_tokens_to_summarize,
                    token_counter=models.token_counter,
                    start_on="human",
                    strategy="last",
                    allow_partial=True,
                    include_system=True,
                ),
            )
        except Exception:
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]
