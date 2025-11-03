"""Agent middleware that integrates OpenAI's moderation endpoint."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from openai import AsyncOpenAI, OpenAI
from openai.types import Moderation, ModerationModel

if TYPE_CHECKING:  # pragma: no cover
    from langgraph.runtime import Runtime

ViolationStage = Literal["input", "output", "tool"]

DEFAULT_VIOLATION_TEMPLATE = (
    "I'm sorry, but I can't comply with that request. It was flagged for {categories}."
)


class OpenAIModerationError(RuntimeError):
    """Raised when OpenAI flags content and `exit_behavior` is set to ``"error"``."""

    def __init__(
        self,
        *,
        content: str,
        stage: ViolationStage,
        result: Moderation,
        message: str,
    ) -> None:
        """Initialize the error with violation details.

        Args:
            content: The content that was flagged.
            stage: The stage where the violation occurred.
            result: The moderation result from OpenAI.
            message: The error message.
        """
        super().__init__(message)
        self.content = content
        self.stage = stage
        self.result = result


class OpenAIModerationMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Moderate agent traffic using OpenAI's moderation endpoint."""

    def __init__(
        self,
        *,
        model: ModerationModel = "omni-moderation-latest",
        check_input: bool = True,
        check_output: bool = True,
        check_tool_results: bool = False,
        exit_behavior: Literal["error", "end", "replace"] = "end",
        violation_message: str | None = None,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
    ) -> None:
        """Create the middleware instance.

        Args:
            model: OpenAI moderation model to use.
            check_input: Whether to check user input messages.
            check_output: Whether to check model output messages.
            check_tool_results: Whether to check tool result messages.
            exit_behavior: How to handle violations
                (`'error'`, `'end'`, or `'replace'`).
            violation_message: Custom template for violation messages.
            client: Optional pre-configured OpenAI client to reuse.
                If not provided, a new client will be created.
            async_client: Optional pre-configured AsyncOpenAI client to reuse.
                If not provided, a new async client will be created.
        """
        super().__init__()
        self.model = model
        self.check_input = check_input
        self.check_output = check_output
        self.check_tool_results = check_tool_results
        self.exit_behavior = exit_behavior
        self.violation_message = violation_message

        self._client = client
        self._async_client = async_client

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:  # type: ignore[override]
        """Moderate user input and tool results before the model is called.

        Args:
            state: Current agent state containing messages.
            runtime: Agent runtime context.

        Returns:
            Updated state with moderated messages, or `None` if no changes.
        """
        if not self.check_input and not self.check_tool_results:
            return None

        messages = list(state.get("messages", []))
        if not messages:
            return None

        return self._moderate_inputs(messages)

    @hook_config(can_jump_to=["end"])
    def after_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:  # type: ignore[override]
        """Moderate model output after the model is called.

        Args:
            state: Current agent state containing messages.
            runtime: Agent runtime context.

        Returns:
            Updated state with moderated messages, or `None` if no changes.
        """
        if not self.check_output:
            return None

        messages = list(state.get("messages", []))
        if not messages:
            return None

        return self._moderate_output(messages)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:  # type: ignore[override]
        """Async version of before_model.

        Args:
            state: Current agent state containing messages.
            runtime: Agent runtime context.

        Returns:
            Updated state with moderated messages, or `None` if no changes.
        """
        if not self.check_input and not self.check_tool_results:
            return None

        messages = list(state.get("messages", []))
        if not messages:
            return None

        return await self._amoderate_inputs(messages)

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:  # type: ignore[override]
        """Async version of after_model.

        Args:
            state: Current agent state containing messages.
            runtime: Agent runtime context.

        Returns:
            Updated state with moderated messages, or `None` if no changes.
        """
        if not self.check_output:
            return None

        messages = list(state.get("messages", []))
        if not messages:
            return None

        return await self._amoderate_output(messages)

    def _moderate_inputs(
        self, messages: Sequence[BaseMessage]
    ) -> dict[str, Any] | None:
        working = list(messages)
        modified = False

        if self.check_tool_results:
            action = self._moderate_tool_messages(working)
            if action:
                if "jump_to" in action:
                    return action
                working = cast("list[BaseMessage]", action["messages"])
                modified = True

        if self.check_input:
            action = self._moderate_user_message(working)
            if action:
                if "jump_to" in action:
                    return action
                working = cast("list[BaseMessage]", action["messages"])
                modified = True

        if modified:
            return {"messages": working}

        return None

    async def _amoderate_inputs(
        self, messages: Sequence[BaseMessage]
    ) -> dict[str, Any] | None:
        working = list(messages)
        modified = False

        if self.check_tool_results:
            action = await self._amoderate_tool_messages(working)
            if action:
                if "jump_to" in action:
                    return action
                working = cast("list[BaseMessage]", action["messages"])
                modified = True

        if self.check_input:
            action = await self._amoderate_user_message(working)
            if action:
                if "jump_to" in action:
                    return action
                working = cast("list[BaseMessage]", action["messages"])
                modified = True

        if modified:
            return {"messages": working}

        return None

    def _moderate_output(
        self, messages: Sequence[BaseMessage]
    ) -> dict[str, Any] | None:
        last_ai_idx = self._find_last_index(messages, AIMessage)
        if last_ai_idx is None:
            return None

        ai_message = messages[last_ai_idx]
        text = self._extract_text(ai_message)
        if not text:
            return None

        result = self._moderate(text)
        if not result.flagged:
            return None

        return self._apply_violation(
            messages, index=last_ai_idx, stage="output", content=text, result=result
        )

    async def _amoderate_output(
        self, messages: Sequence[BaseMessage]
    ) -> dict[str, Any] | None:
        last_ai_idx = self._find_last_index(messages, AIMessage)
        if last_ai_idx is None:
            return None

        ai_message = messages[last_ai_idx]
        text = self._extract_text(ai_message)
        if not text:
            return None

        result = await self._amoderate(text)
        if not result.flagged:
            return None

        return self._apply_violation(
            messages, index=last_ai_idx, stage="output", content=text, result=result
        )

    def _moderate_tool_messages(
        self, messages: Sequence[BaseMessage]
    ) -> dict[str, Any] | None:
        last_ai_idx = self._find_last_index(messages, AIMessage)
        if last_ai_idx is None:
            return None

        working = list(messages)
        modified = False

        for idx in range(last_ai_idx + 1, len(working)):
            msg = working[idx]
            if not isinstance(msg, ToolMessage):
                continue

            text = self._extract_text(msg)
            if not text:
                continue

            result = self._moderate(text)
            if not result.flagged:
                continue

            action = self._apply_violation(
                working, index=idx, stage="tool", content=text, result=result
            )
            if action:
                if "jump_to" in action:
                    return action
                working = cast("list[BaseMessage]", action["messages"])
                modified = True

        if modified:
            return {"messages": working}

        return None

    async def _amoderate_tool_messages(
        self, messages: Sequence[BaseMessage]
    ) -> dict[str, Any] | None:
        last_ai_idx = self._find_last_index(messages, AIMessage)
        if last_ai_idx is None:
            return None

        working = list(messages)
        modified = False

        for idx in range(last_ai_idx + 1, len(working)):
            msg = working[idx]
            if not isinstance(msg, ToolMessage):
                continue

            text = self._extract_text(msg)
            if not text:
                continue

            result = await self._amoderate(text)
            if not result.flagged:
                continue

            action = self._apply_violation(
                working, index=idx, stage="tool", content=text, result=result
            )
            if action:
                if "jump_to" in action:
                    return action
                working = cast("list[BaseMessage]", action["messages"])
                modified = True

        if modified:
            return {"messages": working}

        return None

    def _moderate_user_message(
        self, messages: Sequence[BaseMessage]
    ) -> dict[str, Any] | None:
        idx = self._find_last_index(messages, HumanMessage)
        if idx is None:
            return None

        message = messages[idx]
        text = self._extract_text(message)
        if not text:
            return None

        result = self._moderate(text)
        if not result.flagged:
            return None

        return self._apply_violation(
            messages, index=idx, stage="input", content=text, result=result
        )

    async def _amoderate_user_message(
        self, messages: Sequence[BaseMessage]
    ) -> dict[str, Any] | None:
        idx = self._find_last_index(messages, HumanMessage)
        if idx is None:
            return None

        message = messages[idx]
        text = self._extract_text(message)
        if not text:
            return None

        result = await self._amoderate(text)
        if not result.flagged:
            return None

        return self._apply_violation(
            messages, index=idx, stage="input", content=text, result=result
        )

    def _apply_violation(
        self,
        messages: Sequence[BaseMessage],
        *,
        index: int | None,
        stage: ViolationStage,
        content: str,
        result: Moderation,
    ) -> dict[str, Any] | None:
        violation_text = self._format_violation_message(content, result)

        if self.exit_behavior == "error":
            raise OpenAIModerationError(
                content=content,
                stage=stage,
                result=result,
                message=violation_text,
            )

        if self.exit_behavior == "end":
            return {"jump_to": "end", "messages": [AIMessage(content=violation_text)]}

        if index is None:
            return None

        new_messages = list(messages)
        original = new_messages[index]
        new_messages[index] = cast(
            BaseMessage, original.model_copy(update={"content": violation_text})
        )
        return {"messages": new_messages}

    def _moderate(self, text: str) -> Moderation:
        if self._client is None:
            self._client = self._build_client()
        response = self._client.moderations.create(model=self.model, input=text)
        return response.results[0]

    async def _amoderate(self, text: str) -> Moderation:
        if self._async_client is None:
            self._async_client = self._build_async_client()
        response = await self._async_client.moderations.create(
            model=self.model, input=text
        )
        return response.results[0]

    def _build_client(self) -> OpenAI:
        self._client = OpenAI()
        return self._client

    def _build_async_client(self) -> AsyncOpenAI:
        self._async_client = AsyncOpenAI()
        return self._async_client

    def _format_violation_message(self, content: str, result: Moderation) -> str:
        # Convert categories to dict and filter for flagged items
        categories_dict = result.categories.model_dump()
        categories = [
            name.replace("_", " ")
            for name, flagged in categories_dict.items()
            if flagged
        ]
        category_label = (
            ", ".join(categories) if categories else "OpenAI's safety policies"
        )
        template = self.violation_message or DEFAULT_VIOLATION_TEMPLATE
        scores_json = json.dumps(result.category_scores.model_dump(), sort_keys=True)
        try:
            message = template.format(
                categories=category_label,
                category_scores=scores_json,
                original_content=content,
            )
        except KeyError:
            message = template
        return message

    def _find_last_index(
        self, messages: Sequence[BaseMessage], message_type: type[BaseMessage]
    ) -> int | None:
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], message_type):
                return idx
        return None

    def _extract_text(self, message: BaseMessage) -> str | None:
        if message.content is None:
            return None
        text_accessor = getattr(message, "text", None)
        if text_accessor is None:
            return str(message.content)
        text = str(text_accessor)
        return text if text else None


__all__ = [
    "OpenAIModerationError",
    "OpenAIModerationMiddleware",
]
