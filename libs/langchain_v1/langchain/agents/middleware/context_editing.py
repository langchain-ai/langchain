"""Context editing middleware.

Mirrors Anthropic's context editing capabilities by clearing older tool results once the
conversation grows beyond a configurable token threshold.

The implementation is intentionally model-agnostic so it can be used with any LangChain
chat model.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from typing_extensions import Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)

DEFAULT_TOOL_PLACEHOLDER = "[cleared]"

DEFAULT_SUMMARY_PROMPT = """Summarize the following tool output concisely. \
Keep only the essential information that would be useful for continuing \
a conversation. Be brief but preserve key facts, numbers, and conclusions.

Tool name: {tool_name}
Tool output:
{tool_output}

Provide a concise summary:"""


TokenCounter = Callable[
    [Sequence[BaseMessage]],
    int,
]


class ContextEdit(Protocol):
    """Protocol describing a context editing strategy."""

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """Apply an edit to the message list in place."""
        ...


@dataclass(slots=True)
class ClearToolUsesEdit(ContextEdit):
    """Configuration for clearing tool outputs when token limits are exceeded."""

    trigger: int = 100_000
    """Token count that triggers the edit."""

    clear_at_least: int = 0
    """Minimum number of tokens to reclaim when the edit runs."""

    keep: int = 3
    """Number of most recent tool results that must be preserved."""

    clear_tool_inputs: bool = False
    """Whether to clear the originating tool call parameters on the AI message."""

    exclude_tools: Sequence[str] = ()
    """List of tool names to exclude from clearing."""

    placeholder: str = DEFAULT_TOOL_PLACEHOLDER
    """Placeholder text inserted for cleared tool outputs."""

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """Apply the clear-tool-uses strategy."""
        tokens = count_tokens(messages)

        if tokens <= self.trigger:
            return

        candidates = [
            (idx, msg) for idx, msg in enumerate(messages) if isinstance(msg, ToolMessage)
        ]

        if self.keep >= len(candidates):
            candidates = []
        elif self.keep:
            candidates = candidates[: -self.keep]

        cleared_tokens = 0
        excluded_tools = set(self.exclude_tools)

        for idx, tool_message in candidates:
            if tool_message.response_metadata.get("context_editing", {}).get("cleared"):
                continue

            ai_message = next(
                (m for m in reversed(messages[:idx]) if isinstance(m, AIMessage)), None
            )

            if ai_message is None:
                continue

            tool_call = next(
                (
                    call
                    for call in ai_message.tool_calls
                    if call.get("id") == tool_message.tool_call_id
                ),
                None,
            )

            if tool_call is None:
                continue

            if (tool_message.name or tool_call["name"]) in excluded_tools:
                continue

            messages[idx] = tool_message.model_copy(
                update={
                    "artifact": None,
                    "content": self.placeholder,
                    "response_metadata": {
                        **tool_message.response_metadata,
                        "context_editing": {
                            "cleared": True,
                            "strategy": "clear_tool_uses",
                        },
                    },
                }
            )

            if self.clear_tool_inputs:
                messages[messages.index(ai_message)] = self._build_cleared_tool_input_message(
                    ai_message,
                    tool_message.tool_call_id,
                )

            if self.clear_at_least > 0:
                new_token_count = count_tokens(messages)
                cleared_tokens = max(0, tokens - new_token_count)
                if cleared_tokens >= self.clear_at_least:
                    break

        return

    def _build_cleared_tool_input_message(
        self,
        message: AIMessage,
        tool_call_id: str,
    ) -> AIMessage:
        updated_tool_calls = []
        cleared_any = False
        for tool_call in message.tool_calls:
            updated_call = dict(tool_call)
            if updated_call.get("id") == tool_call_id:
                updated_call["args"] = {}
                cleared_any = True
            updated_tool_calls.append(updated_call)

        metadata = dict(getattr(message, "response_metadata", {}))
        context_entry = dict(metadata.get("context_editing", {}))
        if cleared_any:
            cleared_ids = set(context_entry.get("cleared_tool_inputs", []))
            cleared_ids.add(tool_call_id)
            context_entry["cleared_tool_inputs"] = sorted(cleared_ids)
            metadata["context_editing"] = context_entry

        return message.model_copy(
            update={
                "tool_calls": updated_tool_calls,
                "response_metadata": metadata,
            }
        )


@dataclass(slots=True)
class SummarizeToolUsesEdit(ContextEdit):
    """Summarize older tool outputs when token limits are exceeded.

    Instead of clearing tool outputs entirely (like `ClearToolUsesEdit`), this strategy
    uses an LLM to summarize older tool results, preserving essential information while
    reducing context size.

    This is useful for long-running agents that need to maintain context about past tool
    interactions while staying within token limits.

    Examples:
        !!! example "Basic usage with a model instance"

            ```python
            from langchain.agents.middleware import (
                ContextEditingMiddleware,
                SummarizeToolUsesEdit,
            )
            from langchain.chat_models import init_chat_model

            summary_model = init_chat_model("openai:gpt-4o-mini")
            middleware = ContextEditingMiddleware(
                edits=[
                    SummarizeToolUsesEdit(
                        summarization_model=summary_model,
                        trigger=100_000,
                        keep=3,
                    ),
                ],
            )
            ```

        !!! example "Using a model string"

            ```python
            middleware = ContextEditingMiddleware(
                edits=[
                    SummarizeToolUsesEdit(
                        summarization_model="openai:gpt-4o-mini",
                        trigger=50_000,
                        summarize_at_least=10_000,
                        exclude_tools=["calculator"],
                    ),
                ],
            )
            ```

        !!! example "Custom summary prompt"

            ```python
            custom_prompt = '''Summarize this tool output briefly:
            Tool: {tool_name}
            Output: {tool_output}
            Summary:'''

            middleware = ContextEditingMiddleware(
                edits=[
                    SummarizeToolUsesEdit(
                        summarization_model="anthropic:claude-sonnet-4-5-20250929",
                        summary_prompt=custom_prompt,
                    ),
                ],
            )
            ```
    """

    summarization_model: BaseChatModel | str
    """The LLM model to use for summarizing tool outputs.

    Can be a `BaseChatModel` instance or a model identifier string
    (e.g., `'openai:gpt-4o-mini'`).
    """

    trigger: int = 100_000
    """Token count that triggers the summarization."""

    summarize_at_least: int = 0
    """Minimum number of tokens to reclaim when the edit runs.

    If set to 0, all eligible tool outputs will be summarized when triggered.
    """

    keep: int = 3
    """Number of most recent tool results that must be preserved (not summarized)."""

    exclude_tools: Sequence[str] = field(default_factory=tuple)
    """List of tool names to exclude from summarization."""

    max_output_length: int = 80000
    """Maximum length of tool output to summarize.

    Longer outputs are truncated before being sent to the summarization model.
    """

    min_content_length: int = 200
    """Minimum content length to consider for summarization.

    Tool outputs shorter than this are left unchanged.
    """

    summary_prefix: str = "[Summary] "
    """Prefix added to summarized content to indicate it's a summary."""

    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    """Prompt template for summarization.

    Must contain `{tool_name}` and `{tool_output}` placeholders.
    """

    _resolved_model: BaseChatModel | None = field(default=None, repr=False)

    def _get_model(self) -> BaseChatModel:
        """Lazily resolve the summarization model."""
        if self._resolved_model is not None:
            return self._resolved_model

        if isinstance(self.summarization_model, BaseChatModel):
            self._resolved_model = self.summarization_model
        else:
            from langchain.chat_models import init_chat_model

            self._resolved_model = init_chat_model(self.summarization_model)

        return self._resolved_model

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """Apply the summarize-tool-uses strategy."""
        tokens = count_tokens(messages)

        if tokens <= self.trigger:
            return

        candidates = [
            (idx, msg) for idx, msg in enumerate(messages) if isinstance(msg, ToolMessage)
        ]

        if self.keep >= len(candidates):
            candidates = []
        elif self.keep:
            candidates = candidates[: -self.keep]

        summarized_tokens = 0
        excluded_tools = set(self.exclude_tools)

        for idx, tool_message in candidates:
            if tool_message.response_metadata.get("context_editing", {}).get("summarized"):
                continue

            ai_message = next(
                (m for m in reversed(messages[:idx]) if isinstance(m, AIMessage)),
                None,
            )

            if ai_message is None:
                continue

            tool_call = next(
                (
                    call
                    for call in ai_message.tool_calls
                    if call.get("id") == tool_message.tool_call_id
                ),
                None,
            )

            if tool_call is None:
                continue

            tool_name = tool_message.name or tool_call.get("name", "unknown")

            if tool_name in excluded_tools:
                continue

            original_content = self._get_content_as_string(tool_message.content)

            if len(original_content) < self.min_content_length:
                continue

            content_to_summarize = original_content
            if len(content_to_summarize) > self.max_output_length:
                content_to_summarize = (
                    content_to_summarize[: self.max_output_length]
                    + "\n... [truncated for summarization]"
                )

            try:
                summary = self._summarize_content(tool_name, content_to_summarize)
            except Exception as e:
                logger.warning("Failed to summarize tool output for %s: %s", tool_name, e)
                summary = original_content[:500] + "... [truncated]"

            messages[idx] = tool_message.model_copy(
                update={
                    "content": f"{self.summary_prefix}{summary}",
                    "response_metadata": {
                        **tool_message.response_metadata,
                        "context_editing": {
                            "summarized": True,
                            "strategy": "summarize_tool_uses",
                            "original_length": len(original_content),
                        },
                    },
                }
            )

            if self.summarize_at_least > 0:
                new_token_count = count_tokens(messages)
                summarized_tokens = max(0, tokens - new_token_count)
                if summarized_tokens >= self.summarize_at_least:
                    break

    async def aapply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """Apply the summarize-tool-uses strategy asynchronously."""
        tokens = count_tokens(messages)

        if tokens <= self.trigger:
            return

        candidates = [
            (idx, msg) for idx, msg in enumerate(messages) if isinstance(msg, ToolMessage)
        ]

        if self.keep >= len(candidates):
            candidates = []
        elif self.keep:
            candidates = candidates[: -self.keep]

        summarized_tokens = 0
        excluded_tools = set(self.exclude_tools)

        for idx, tool_message in candidates:
            if tool_message.response_metadata.get("context_editing", {}).get("summarized"):
                continue

            ai_message = next(
                (m for m in reversed(messages[:idx]) if isinstance(m, AIMessage)),
                None,
            )

            if ai_message is None:
                continue

            tool_call = next(
                (
                    call
                    for call in ai_message.tool_calls
                    if call.get("id") == tool_message.tool_call_id
                ),
                None,
            )

            if tool_call is None:
                continue

            tool_name = tool_message.name or tool_call.get("name", "unknown")

            if tool_name in excluded_tools:
                continue

            original_content = self._get_content_as_string(tool_message.content)

            if len(original_content) < self.min_content_length:
                continue

            content_to_summarize = original_content
            if len(content_to_summarize) > self.max_output_length:
                content_to_summarize = (
                    content_to_summarize[: self.max_output_length]
                    + "\n... [truncated for summarization]"
                )

            try:
                summary = await self._asummarize_content(tool_name, content_to_summarize)
            except Exception as e:
                logger.warning("Failed to summarize tool output for %s: %s", tool_name, e)
                summary = original_content[:500] + "... [truncated]"

            messages[idx] = tool_message.model_copy(
                update={
                    "content": f"{self.summary_prefix}{summary}",
                    "response_metadata": {
                        **tool_message.response_metadata,
                        "context_editing": {
                            "summarized": True,
                            "strategy": "summarize_tool_uses",
                            "original_length": len(original_content),
                        },
                    },
                }
            )

            if self.summarize_at_least > 0:
                new_token_count = count_tokens(messages)
                summarized_tokens = max(0, tokens - new_token_count)
                if summarized_tokens >= self.summarize_at_least:
                    break

    def _get_content_as_string(self, content: str | list | object) -> str:
        """Convert tool message content to string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("text", str(item)))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    def _summarize_content(self, tool_name: str, content: str) -> str:
        """Use the LLM to summarize the tool output synchronously."""
        prompt = self.summary_prompt.format(tool_name=tool_name, tool_output=content)
        model = self._get_model()

        response = model.invoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": []},
        )

        response_content = response.content
        if isinstance(response_content, str):
            return response_content.strip()
        return self._get_content_as_string(response_content).strip()

    async def _asummarize_content(self, tool_name: str, content: str) -> str:
        """Use the LLM to summarize the tool output asynchronously."""
        prompt = self.summary_prompt.format(tool_name=tool_name, tool_output=content)
        model = self._get_model()

        response = await model.ainvoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": []},
        )

        response_content = response.content
        if isinstance(response_content, str):
            return response_content.strip()
        return self._get_content_as_string(response_content).strip()


class ContextEditingMiddleware(AgentMiddleware):
    """Automatically prune tool results to manage context size.

    The middleware applies a sequence of edits when the total input token count exceeds
    configured thresholds.

    Available edit strategies:

    - `ClearToolUsesEdit`: Clears older tool outputs, aligning with Anthropic's
      `clear_tool_uses_20250919` behavior
      [(read more)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool).
    - `SummarizeToolUsesEdit`: Summarizes older tool outputs using an LLM, preserving
      essential information while reducing context size.
    """

    edits: list[ContextEdit]
    token_count_method: Literal["approximate", "model"]
    persist_edits: bool

    def __init__(
        self,
        *,
        edits: Iterable[ContextEdit] | None = None,
        token_count_method: Literal["approximate", "model"] = "approximate",  # noqa: S107
        persist_edits: bool = True,
    ) -> None:
        """Initialize an instance of context editing middleware.

        Args:
            edits: Sequence of edit strategies to apply.

                Defaults to a single `ClearToolUsesEdit` mirroring Anthropic defaults.
            token_count_method: Whether to use approximate token counting
                (faster, less accurate) or exact counting implemented by the
                chat model (potentially slower, more accurate).
            persist_edits: Whether to persist edits back to the original message list.

                When `True` (the default), edited messages (summarized/cleared content
                and metadata) are written back to the original message list, preventing
                re-processing on subsequent turns.
        """
        super().__init__()
        self.edits = list(edits or (ClearToolUsesEdit(),))
        self.token_count_method = token_count_method
        self.persist_edits = persist_edits

    def _sync_edits_to_original(
        self,
        original_messages: Sequence[AnyMessage],
        edited_messages: list[AnyMessage],
    ) -> None:
        """Sync edited messages back to the original list to persist edits across turns."""
        if len(original_messages) != len(edited_messages):
            return

        if not isinstance(original_messages, list):
            return

        for i, (orig, edited) in enumerate(zip(original_messages, edited_messages, strict=True)):
            if not isinstance(orig, ToolMessage):
                continue

            edited_context = edited.response_metadata.get("context_editing", {})
            if not edited_context:
                continue

            orig_context = orig.response_metadata.get("context_editing", {})
            if orig_context == edited_context and orig.content == edited.content:
                continue

            original_messages[i] = edited

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Apply context edits before invoking the model via handler."""
        if not request.messages:
            return handler(request)

        if self.token_count_method == "approximate":  # noqa: S105

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return count_tokens_approximately(messages)

        else:
            system_msg = [request.system_message] if request.system_message else []

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return request.model.get_num_tokens_from_messages(
                    system_msg + list(messages), request.tools
                )

        edited_messages = deepcopy(list(request.messages))
        for edit in self.edits:
            edit.apply(edited_messages, count_tokens=count_tokens)

        if self.persist_edits:
            self._sync_edits_to_original(request.messages, edited_messages)

        return handler(request.override(messages=edited_messages))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Apply context edits before invoking the model via handler (async version)."""
        if not request.messages:
            return await handler(request)

        if self.token_count_method == "approximate":  # noqa: S105

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return count_tokens_approximately(messages)

        else:
            system_msg = [request.system_message] if request.system_message else []

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return request.model.get_num_tokens_from_messages(
                    system_msg + list(messages), request.tools
                )

        edited_messages = deepcopy(list(request.messages))
        for edit in self.edits:
            if hasattr(edit, "aapply"):
                await edit.aapply(edited_messages, count_tokens=count_tokens)
            else:
                edit.apply(edited_messages, count_tokens=count_tokens)

        if self.persist_edits:
            self._sync_edits_to_original(request.messages, edited_messages)

        return await handler(request.override(messages=edited_messages))


__all__ = [
    "ClearToolUsesEdit",
    "ContextEditingMiddleware",
    "SummarizeToolUsesEdit",
]
