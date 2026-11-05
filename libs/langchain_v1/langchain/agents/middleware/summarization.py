"""Summarization middleware."""

import logging
import uuid
import warnings
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, Literal, TypedDict, cast

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
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import override

from langchain.agents.middleware._retry import default_retry_on
from langchain.agents.middleware.internal_call_transformer import (
    InternalCallTransformer,
    internal_call_metadata,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
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
        # 3.3 was estimated in an offline experiment, comparing with Claude's token-counting
        # API: https://platform.claude.com/docs/en/build-with-claude/token-counting
        return partial(
            count_tokens_approximately, use_usage_metadata_scaling=True, chars_per_token=3.3
        )
    return partial(count_tokens_approximately, use_usage_metadata_scaling=True)


@dataclass(frozen=True)
class ModelResolution(Generic[ContextT]):
    """What a `model` callable receives to choose the context model for one call.

    See the `model` argument of
    [`SummarizationMiddleware`][langchain.agents.middleware.summarization.SummarizationMiddleware].
    """

    runtime: Runtime[ContextT]
    """Runtime of the current agent run, including its `context`."""

    state: Mapping[str, Any]
    """Agent state at the time of the call."""

    request: ModelRequest[ContextT] | None = None
    """The model call being made, when summarization runs as part of one.

    `request.model` is the model the call actually goes to, after any model
    swaps made by middleware that runs before this one.
    """

    @property
    def context(self) -> ContextT:
        """Shortcut for `runtime.context`."""
        return self.runtime.context


ModelResolver = Callable[[ModelResolution[Any]], BaseChatModel]
"""Chooses the context model for each model call.

Must return a `BaseChatModel` instance, never a model string, so compaction never
builds a new (possibly unauthenticated) client on the fly.
"""


@dataclass(frozen=True)
class _CallModels:
    """Models and token counters resolved for a single model call.

    Built per call and passed down explicitly; never stored on the middleware, since
    one compiled agent serves many concurrent runs.
    """

    context_model: BaseChatModel
    """Drives thresholds, token counting, provider checks and trimming."""

    summary_chat_model: BaseChatModel | None
    """Chat model that writes summaries, or `None` when a custom `summarizer` does."""

    summary_runnable: Runnable[LanguageModelInput, BaseMessage] | None
    """`summary_chat_model` wrapped with retries."""

    token_counter: TokenCounter
    partial_token_counter: TokenCounter


@dataclass(frozen=True)
class _SummaryPlan:
    """What one model call summarizes, and how the stored history is updated."""

    messages_to_summarize: list[AnyMessage]
    """Summarized messages, using stored originals wherever they exist."""

    preserved_messages: list[AnyMessage]
    """Messages kept after the summary, as this call sees them."""

    anchor_id: str
    """ID of the first summarized stored message, which the summary replaces."""

    removed_ids: list[str]
    """IDs of the other summarized messages."""

    stored_ids: frozenset[str]
    """IDs of the messages stored in state when the call started."""


def _model_identifier(model: BaseChatModel) -> str:
    """Return a human-readable `provider:model` identifier for tracing and warnings."""
    params = _ls_params(model)
    provider = params.get("ls_provider")
    name = params.get("ls_model_name")
    if provider and name:
        return f"{provider}:{name}"
    return str(name or provider or type(model).__name__)


def _ls_params(model: BaseChatModel) -> Mapping[str, Any]:
    try:
        return model._get_ls_params()  # noqa: SLF001
    except Exception:
        return {}


def _same_model(first: BaseChatModel, second: BaseChatModel) -> bool:
    """Whether two model objects target the same provider model.

    Agents often build a fresh model object per call, so identity alone would
    report false mismatches; fall back to identity only when a model does not
    report its name.
    """
    if first is second:
        return True
    first_params, second_params = _ls_params(first), _ls_params(second)
    first_name = first_params.get("ls_model_name")
    if not first_name:
        return False
    return first_name == second_params.get("ls_model_name") and first_params.get(
        "ls_provider"
    ) == second_params.get("ls_provider")


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

    Summarization runs inside each model call (`wrap_model_call`), so by default it
    follows the model the call actually goes to, including models swapped in by
    other middleware (a `/model` command, a router, a fallback). For every call
    the middleware resolves one *context model*, which drives trigger thresholds,
    token counting, provider-reported token checks and trimming, and one *summary
    model*, which writes the summary: `summarizer` if set, otherwise the context
    model.

    The agent also records its model in state under `model_selection` (see
    [`ModelSelection`][langchain.agents.middleware.types.ModelSelection]). That
    record is informational: it is never used to choose the summarization model.
    """

    transformers = (InternalCallTransformer,)
    """Keeps the summarization model call's tokens out of `run.messages`.

    Registered only when this middleware is used — see
    `InternalCallTransformer` for why the call needs tagging and filtering.
    """

    def __init__(
        self,
        model: str | BaseChatModel | ModelResolver | None = None,
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

                - `None`: use the model each call goes to (`request.model`),
                    including models swapped in by earlier `wrap_model_call`
                    middleware. Recommended.
                - A chat model, or a model string resolved with `init_chat_model`:
                    pin the context model, whatever model a call goes to.
                - A callable taking a
                    [`ModelResolution`][langchain.agents.middleware.summarization.ModelResolution]
                    and returning a `BaseChatModel` instance: choose the context
                    model per call. Returning a model string raises `TypeError`.

                !!! warning "Pinning `model` to choose a summary-writing model"

                    If a pinned `model` differs from the model a call goes to, a
                    `LangChainDeprecationWarning` is emitted. To summarize with a
                    cheaper model while following the agent's model, pass
                    `summarizer=` instead and omit `model`.

                !!! example

                    ```python
                    # Follow whichever model each call goes to
                    SummarizationMiddleware(trigger=("fraction", 0.8))

                    # Follow each call's model, but write summaries with gpt-6-luna
                    SummarizationMiddleware(
                        trigger=("fraction", 0.8),
                        summarizer="openai:gpt-6-luna",
                    )

                    # Choose the context model per call
                    SummarizationMiddleware(
                        model=lambda resolution: MODELS[resolution.context.tier],
                        trigger=("fraction", 0.8),
                    )
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

                !!! note "Fitting the summarizer's context window"

                    The summarizer's input is first trimmed to
                    `trim_tokens_to_summarize` tokens, counted with the context
                    model's token counter. Choose a summarizer whose context window
                    fits that budget plus `summary_prompt`. If a chat model
                    summarizer still raises `ContextOverflowError`, the call is
                    retried once with the middle of the history cut out, sized to
                    its profile's `max_input_tokens` when known. A `Runnable` or
                    function summarizer gets no overflow retry, so it must handle
                    oversized input itself. With `trim_tokens_to_summarize=None`,
                    the whole summarized history is sent, so make sure the
                    summarizer can take it.

                !!! example

                    ```python
                    # Manage each call's context window, summarize with gpt-6-luna
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

        self._model_resolver: ModelResolver | None = None
        self.model: BaseChatModel | None = None
        subject: Any = model
        if isinstance(subject, BaseChatModel) or hasattr(subject, "invoke"):
            self.model = subject
        elif callable(subject):
            self._model_resolver = subject
        elif subject is not None:
            msg = (
                "`model` must be a model string, a `BaseChatModel`, a callable "
                f"returning a `BaseChatModel`, or `None`; got {type(subject).__name__}."
            )
            raise TypeError(msg)
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
        # replaces it. `None` until resolved per call when `model` is not pinned.
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
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse[ResponseT]:
        """Summarize older messages before the model call when a trigger is met.

        Thresholds and token counts use this call's context model, and the summary is
        written by `summarizer` or, failing that, the same context model.

        Args:
            request: The model request.
            handler: Calls the model (and any inner middleware).

        Returns:
            The model response, plus a state update replacing the summarized history
                with the summary when summarization ran.

        Raises:
            Exception: If summary generation still fails once retries are exhausted.
        """
        models = self._resolve_call_models(request)
        plan = self._plan_summary(request, models)
        if plan is None:
            return handler(request)
        summary = self._traced_summarizer(models).invoke(
            plan.messages_to_summarize, config=self._summary_run_config(models)
        )
        summary_message = self._build_summary_message(summary, plan)
        response = handler(request.override(messages=[summary_message, *plan.preserved_messages]))
        return self._with_summary_update(response, summary_message, plan)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse[ResponseT]:
        """Summarize older messages before the model call when a trigger is met.

        Thresholds and token counts use this call's context model, and the summary is
        written by `summarizer` or, failing that, the same context model.

        Args:
            request: The model request.
            handler: Calls the model (and any inner middleware).

        Returns:
            The model response, plus a state update replacing the summarized history
                with the summary when summarization ran.

        Raises:
            Exception: If summary generation still fails once retries are exhausted.
        """
        models = self._resolve_call_models(request)
        plan = self._plan_summary(request, models)
        if plan is None:
            return await handler(request)
        summary = await self._traced_summarizer(models).ainvoke(
            plan.messages_to_summarize, config=self._summary_run_config(models)
        )
        summary_message = self._build_summary_message(summary, plan)
        response = await handler(
            request.override(messages=[summary_message, *plan.preserved_messages])
        )
        return self._with_summary_update(response, summary_message, plan)

    def _plan_summary(self, request: ModelRequest[Any], models: _CallModels) -> _SummaryPlan | None:
        """Decide on this call's view whether and what to summarize, or `None` to skip.

        Triggers and the cutoff use `request.messages`, which is what the context
        model will see. The text to summarize uses the stored originals of those
        messages, so per-call edits by outer middleware (e.g. cleared tool results)
        never end up in the persisted summary.
        """
        messages = request.messages
        self._ensure_message_ids(messages)
        total_tokens = models.token_counter(messages)
        if not self._should_summarize(messages, total_tokens, models=models):
            return None
        cutoff_index = self._determine_cutoff_index(messages, models=models)
        if cutoff_index <= 0:
            return None
        summarized, preserved = self._partition_messages(messages, cutoff_index)
        stored = {msg.id: msg for msg in request.state.get("messages", []) if msg.id}
        anchor_id = next((msg.id for msg in summarized if msg.id in stored), None)
        if anchor_id is None:
            return None
        return _SummaryPlan(
            messages_to_summarize=[stored.get(msg.id or "", msg) for msg in summarized],
            preserved_messages=preserved,
            anchor_id=anchor_id,
            removed_ids=[msg.id for msg in summarized if msg.id and msg.id != anchor_id],
            stored_ids=frozenset(stored),
        )

    def _build_summary_message(self, summary: str, plan: _SummaryPlan) -> HumanMessage:
        """The summary message, taking the place of the first summarized stored message."""
        summary_message = self._build_new_messages(summary)[0]
        summary_message.id = plan.anchor_id
        return summary_message

    @staticmethod
    def _with_summary_update(
        response: ModelResponse[ResponseT],
        summary_message: HumanMessage,
        plan: _SummaryPlan,
    ) -> ExtendedModelResponse[ResponseT]:
        """Attach a state update that swaps the summarized messages for the summary.

        The update edits messages by ID instead of replacing the whole history, so
        the model response and any messages inner middleware add in the same step
        are kept, and stored messages outer middleware hid from this call are left
        alone. The summary replaces the first summarized stored message in place.
        Summarized messages that may not be stored (e.g. added for this call only)
        are removed through a placeholder, since removing an unknown ID raises.
        """
        updates: list[BaseMessage] = [summary_message]
        for message_id in plan.removed_ids:
            if message_id not in plan.stored_ids:
                updates.append(HumanMessage(content="", id=message_id))
            updates.append(RemoveMessage(id=message_id))
        return ExtendedModelResponse(
            model_response=response, command=Command(update={"messages": updates})
        )

    def _resolve_call_models(self, request: ModelRequest[Any]) -> _CallModels:
        """Resolve the context and summary models for one model call."""
        return self._call_models_for(self._resolve_context_model(request))

    def _resolve_context_model(self, request: ModelRequest[Any]) -> BaseChatModel:
        if self._model_resolver is not None:
            resolution = ModelResolution(
                runtime=request.runtime, state=request.state, request=request
            )
            return self._check_resolved_model(self._model_resolver(resolution))
        if self.model is None:
            return self._check_resolved_model(request.model)
        if not _same_model(self.model, request.model):
            self._warn_pinned_model_differs(request.model)
        return self.model

    @staticmethod
    def _check_resolved_model(model: object) -> BaseChatModel:
        if isinstance(model, str):
            msg = (
                "The `model` callable passed to `SummarizationMiddleware` returned the "
                f"string {model!r}. Return a `BaseChatModel` instance instead (for "
                "example one built once with `init_chat_model`), so summarization "
                "never constructs a new, possibly unauthenticated, model client."
            )
            raise TypeError(msg)
        if not isinstance(model, BaseChatModel):
            msg = (
                "`SummarizationMiddleware` needs a `BaseChatModel` as its context "
                f"model, got {type(model).__name__}. Pass `model=` to choose one."
            )
            raise TypeError(msg)
        return model

    def _warn_pinned_model_differs(self, call_model: BaseChatModel) -> None:
        pinned = cast("BaseChatModel", self.model)
        msg = (
            f"`SummarizationMiddleware` is pinned to `model={_model_identifier(pinned)}`, "
            f"but this call goes to `{_model_identifier(call_model)}`. Trigger "
            "thresholds, token counting and summaries use the pinned model. Using "
            "`model=` to choose a model for writing summaries is deprecated: pass "
            "`summarizer=` instead and omit `model=` so thresholds follow each "
            "call's model. Keep `model=` only to deliberately pin the context model."
        )
        warnings.warn(msg, LangChainDeprecationWarning, stacklevel=3)

    def _call_models_for(self, context_model: BaseChatModel) -> _CallModels:
        """Derive token counters and the summary model for a resolved context model.

        These are cheap to derive (attribute reads and `partial`), so they are rebuilt
        per call instead of cached: agents may build a new model object per call, and
        an `id()`-keyed cache could hand a recycled id another model's settings.
        """
        if context_model is self.model:
            return self._fixed_call_models()
        if self._uses_default_token_counter:
            token_counter = _get_approximate_token_counter(context_model)
            partial_token_counter: TokenCounter = partial(  # type: ignore[call-arg]
                token_counter, use_usage_metadata_scaling=False
            )
        else:
            token_counter = partial_token_counter = self.token_counter
        if self.summarizer is not None:
            summary_chat_model, summary_runnable = None, None
        elif self._summary_chat_model is not None:
            summary_chat_model, summary_runnable = self._summary_chat_model, self._summary_model
        else:
            summary_chat_model = context_model
            summary_runnable = context_model.with_retry(retry_if_exception_type=default_retry_on)
        return _CallModels(
            context_model=context_model,
            summary_chat_model=summary_chat_model,
            summary_runnable=summary_runnable,
            token_counter=token_counter,
            partial_token_counter=partial_token_counter,
        )

    def _fixed_call_models(self) -> _CallModels:
        """Models for the pinned `model`, read from instance attributes."""
        if self.model is None:
            msg = (
                "This `SummarizationMiddleware` has no pinned `model`; its context model "
                "is resolved per model call."
            )
            raise ValueError(msg)
        return _CallModels(
            context_model=self.model,
            summary_chat_model=None if self.summarizer is not None else self._summary_chat_model,
            summary_runnable=None if self.summarizer is not None else self._summary_model,
            token_counter=self.token_counter,
            partial_token_counter=self._partial_token_counter,
        )

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
            "summarization_context_model": _model_identifier(models.context_model),
            "summarization_context_window": self._get_profile_limits(models.context_model),
        }
        if models.summary_chat_model is not None:
            metadata["summarization_model"] = _model_identifier(models.summary_chat_model)
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
        context_model = (models or self._fixed_call_models()).context_model
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
                context_model._get_ls_params().get("ls_provider"),  # noqa: SLF001
            )
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
                    max_input_tokens = self._get_profile_limits(models.context_model)
                    if max_input_tokens is None:
                        self._warn_missing_profile(models.context_model)
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
            max_input_tokens = self._get_profile_limits(models.context_model)
            if max_input_tokens is None:
                self._warn_missing_profile(models.context_model)
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
    def _warn_missing_profile(model: BaseChatModel) -> None:
        msg = (
            f"`SummarizationMiddleware` uses fractional limits, but `{_model_identifier(model)}` "
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

        summary_model = cast("Runnable[LanguageModelInput, BaseMessage]", models.summary_runnable)
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

        summary_model = cast("Runnable[LanguageModelInput, BaseMessage]", models.summary_runnable)
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
        summary_chat_model = models.summary_chat_model or models.context_model
        if (max_input_tokens := self._get_profile_limits(summary_chat_model)) is not None:
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
