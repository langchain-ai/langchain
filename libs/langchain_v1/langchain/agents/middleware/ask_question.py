"""Structured multi-choice questions middleware for human-in-the-loop agents."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, Field, model_validator
from typing_extensions import override

from langchain.agents.middleware._hitl_structured_tool import HITLStructuredTool
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

ASK_QUESTION_SCHEMA_VERSION = "1.1"


class QuestionOption(BaseModel):
    """A single selectable option within a question."""

    id: str = Field(description="Unique identifier for this option")
    label: str = Field(description="Display text for this option")
    requires_free_text: bool = Field(
        default=False,
        description=(
            "If true, selecting this option opens a free-text field so the user can "
            "add flexible detail. That text is merged into the resume payload."
        ),
    )
    free_text_placeholder: str | None = Field(
        default=None,
        description="Short hint for the text field when requires_free_text is true.",
    )
    description: str | None = Field(
        default=None,
        description="Optional secondary helper text for this option.",
    )
    disabled: bool = Field(
        default=False,
        description="If true, option is visible but not selectable.",
    )

    @model_validator(mode="after")
    def _validate_free_text_fields(self) -> QuestionOption:
        if self.requires_free_text and not (self.free_text_placeholder or "").strip():
            msg = "free_text_placeholder is required when requires_free_text is true"
            raise ValueError(msg)
        return self


class Question(BaseModel):
    """A single question with multiple-choice options."""

    id: str = Field(description="Unique identifier for this question")
    prompt: str = Field(description="The question text to display to the user")
    options: list[QuestionOption] = Field(
        description="Array of answer options (minimum 2 required)",
        min_length=2,
    )
    allow_multiple: bool | None = Field(
        default=False,
        description="If true, user can select multiple options. Defaults to false.",
    )
    required: bool = Field(
        default=True,
        description="If true, user must answer this question before submitting.",
    )
    min_select: int | None = Field(
        default=None,
        ge=0,
        description="Minimum selections required for multi-select questions.",
    )
    max_select: int | None = Field(
        default=None,
        ge=1,
        description="Maximum selections allowed for multi-select questions.",
    )
    validation_message: str | None = Field(
        default=None,
        description="Optional custom validation message shown to the user.",
    )

    @model_validator(mode="after")
    def _validate_question_constraints(self) -> Question:
        option_ids = [str(option.id).strip() for option in self.options]
        if len(option_ids) != len(set(option_ids)):
            msg = f"Question '{self.id}' has duplicate option ids"
            raise ValueError(msg)

        if not self.allow_multiple:
            if self.min_select is not None and self.min_select > 1:
                msg = (
                    f"Question '{self.id}' cannot set min_select > 1 "
                    "when allow_multiple is false"
                )
                raise ValueError(msg)
            if self.max_select is not None and self.max_select > 1:
                msg = (
                    f"Question '{self.id}' cannot set max_select > 1 "
                    "when allow_multiple is false"
                )
                raise ValueError(msg)
            return self

        option_count = len(self.options)
        if self.min_select is not None and self.min_select > option_count:
            msg = f"Question '{self.id}' min_select cannot exceed option count"
            raise ValueError(msg)
        if self.max_select is not None and self.max_select > option_count:
            msg = f"Question '{self.id}' max_select cannot exceed option count"
            raise ValueError(msg)
        if (
            self.min_select is not None
            and self.max_select is not None
            and self.min_select > self.max_select
        ):
            msg = f"Question '{self.id}' min_select cannot be greater than max_select"
            raise ValueError(msg)
        return self


class AskQuestionInput(BaseModel):
    """Input schema for the AskQuestion tool."""

    title: str | None = Field(
        default=None,
        description="Title for the questions form",
    )
    questions: list[Question] = Field(
        description="Array of questions to present to the user (minimum 1 required)",
        min_length=1,
    )
    context_note: str | None = Field(
        default=None,
        description="Optional explanatory text shown above questions.",
    )
    submit_label: str | None = Field(
        default=None,
        description="Optional custom submit button label.",
    )
    cancel_label: str | None = Field(
        default=None,
        description="Optional custom cancel button label.",
    )

    @model_validator(mode="after")
    def _validate_unique_question_ids(self) -> AskQuestionInput:
        question_ids = [str(question.id).strip() for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            msg = "AskQuestionInput contains duplicate question ids"
            raise ValueError(msg)
        return self


def build_ask_question_interrupt_payload(
    *,
    questions: list[Question],
    title: str | None = None,
    context_note: str | None = None,
    submit_label: str | None = None,
    cancel_label: str | None = None,
) -> dict[str, Any]:
    """Build the versioned interrupt payload surfaced to human-in-the-loop consumers."""
    return {
        "schema_version": ASK_QUESTION_SCHEMA_VERSION,
        "title": title,
        "context_note": context_note,
        "submit_label": submit_label,
        "cancel_label": cancel_label,
        "questions": [question.model_dump() for question in questions],
    }


def _normalize_interrupt_resume(human_input: Any) -> str:
    """Normalize interrupt resume values to a string for the model."""
    if isinstance(human_input, str):
        return human_input
    if isinstance(human_input, dict):
        messages = human_input.get("messages", [])
        if messages and hasattr(messages[0], "content"):
            return str(messages[0].content)
        return json.dumps(human_input)
    if isinstance(human_input, list):
        return json.dumps(human_input)
    return str(human_input)


def _ask_question(
    questions: list[Question],
    title: str | None = None,
    context_note: str | None = None,
    submit_label: str | None = None,
    cancel_label: str | None = None,
) -> str:
    """Present structured multiple-choice questions and wait for human answers."""
    interrupt_payload = build_ask_question_interrupt_payload(
        questions=questions,
        title=title,
        context_note=context_note,
        submit_label=submit_label,
        cancel_label=cancel_label,
    )
    human_input = interrupt(interrupt_payload)
    return _normalize_interrupt_resume(human_input)


async def _aask_question(
    questions: list[Question],
    title: str | None = None,
    context_note: str | None = None,
    submit_label: str | None = None,
    cancel_label: str | None = None,
) -> str:
    """Async variant of `_ask_question`."""
    return _ask_question(
        questions=questions,
        title=title,
        context_note=context_note,
        submit_label=submit_label,
        cancel_label=cancel_label,
    )


ASK_QUESTION_TOOL_DESCRIPTION = """Use this tool to ask the user one or more structured multiple-choice questions.

Use when you need specific information through a structured selection format — for example
to clarify ambiguous requests, confirm a date range, or choose between report types.

Each question must have:
- A unique id (used to match answers)
- A clear prompt
- At least 2 options, each with an id and label
- Optional `requires_free_text` on an option to collect typed detail from the user
- Optional `allow_multiple` for multi-select questions (defaults to false)

The tool pauses execution until the user submits answers. The return value is a JSON
string the agent can parse to continue processing."""


ASK_QUESTION_SYSTEM_PROMPT = """## `AskQuestion`

You have access to the `AskQuestion` tool to collect structured answers from the user.

Use it when free-form chat is insufficient and you need the user to pick from defined
options (single or multi-select). Prefer one call with multiple related questions rather
than many separate calls.

Do not use `AskQuestion` for simple yes/no confirmations that fit naturally in chat.
Do not use it when the user already provided the needed information."""


class AskQuestionMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Middleware that registers the `AskQuestion` interrupt tool for agents.

    This middleware adds an `AskQuestion` tool that pauses graph execution via
    LangGraph's `interrupt()` primitive and resumes with structured user answers.
    Consumers render the versioned interrupt payload (`schema_version: "1.1"`) as a
    form in their UI.

    Unlike `HumanInTheLoopMiddleware`, which intercepts tool execution for approval,
    `AskQuestion` lets the model proactively ask clarification questions mid-task.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import AskQuestionMiddleware
        from langgraph.checkpoint.memory import InMemorySaver

        agent = create_agent(
            "openai:gpt-4.1-mini",
            middleware=[AskQuestionMiddleware()],
            checkpointer=InMemorySaver(),
        )
        ```
    """

    def __init__(
        self,
        *,
        system_prompt: str = ASK_QUESTION_SYSTEM_PROMPT,
        tool_description: str = ASK_QUESTION_TOOL_DESCRIPTION,
    ) -> None:
        """Initialize `AskQuestionMiddleware`.

        Args:
            system_prompt: System prompt appended to guide when to use `AskQuestion`.
            tool_description: Description exposed to the model for the tool.
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description
        self.tools = [
            HITLStructuredTool.from_function(
                func=_ask_question,
                coroutine=_aask_question,
                name="AskQuestion",
                description=tool_description,
                args_schema=AskQuestionInput,
                infer_schema=False,
            )
        ]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Append AskQuestion guidance to the system prompt."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Async variant of `wrap_model_call`."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))

    @override
    def after_model(
        self, state: AgentState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Reject parallel `AskQuestion` calls in the same model turn."""
        del runtime
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        ask_question_calls = [
            tool_call for tool_call in last_ai_msg.tool_calls if tool_call["name"] == "AskQuestion"
        ]
        if len(ask_question_calls) <= 1:
            return None

        error_messages = [
            ToolMessage(
                content=(
                    "Error: Call `AskQuestion` only once per model turn. "
                    "Combine related questions into a single tool call."
                ),
                tool_call_id=tool_call["id"],
                status="error",
            )
            for tool_call in ask_question_calls
        ]
        return {"messages": error_messages}

    @override
    async def aafter_model(
        self, state: AgentState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async variant of `after_model`."""
        return self.after_model(state, runtime)


# Re-exported for tests and advanced callers.
AskQuestion = AskQuestionMiddleware().tools[0]
