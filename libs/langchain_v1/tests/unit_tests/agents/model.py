import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from typing import (
    Any,
    Literal,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    InvalidToolCall,
    ToolCall,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from typing_extensions import override


class FakeToolCallingModel(BaseChatModel):
    tool_calls: list[list[ToolCall]] | list[list[dict[str, Any]]] | None = None
    invalid_tool_calls: list[list[InvalidToolCall]] | list[list[dict[str, Any]]] | None = None
    structured_response: Any | None = None
    index: int = 0
    tool_style: Literal["openai", "anthropic"] = "openai"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call."""
        is_native = kwargs.get("response_format")

        if self.tool_calls:
            if is_native:
                tool_calls = (
                    self.tool_calls[self.index] if self.index < len(self.tool_calls) else []
                )
            else:
                tool_calls = self.tool_calls[self.index % len(self.tool_calls)]
        else:
            tool_calls = []

        if is_native and not tool_calls:
            if isinstance(self.structured_response, BaseModel):
                content_obj = self.structured_response.model_dump()
            elif is_dataclass(self.structured_response) and not isinstance(
                self.structured_response, type
            ):
                content_obj = asdict(self.structured_response)
            elif isinstance(self.structured_response, dict):
                content_obj = self.structured_response
            message = AIMessage(content=json.dumps(content_obj), id=str(self.index))
        else:
            messages_string = "-".join([m.text for m in messages])
            invalid_tc: list[InvalidToolCall] | list[dict[str, Any]] = []
            if self.invalid_tool_calls:
                invalid_tc = self.invalid_tool_calls[self.index % len(self.invalid_tool_calls)]
            message = AIMessage(
                content=messages_string,
                id=str(self.index),
                tool_calls=tool_calls.copy(),
                invalid_tool_calls=list(invalid_tc),
            )
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        if len(tools) == 0:
            msg = "Must provide at least one tool"
            raise ValueError(msg)

        tool_dicts = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_dicts.append(tool)
                continue
            if not isinstance(tool, BaseTool):
                msg = "Only BaseTool and dict is supported by FakeToolCallingModel.bind_tools"
                raise TypeError(msg)

            # NOTE: this is a simplified tool spec for testing purposes only
            if self.tool_style == "openai":
                tool_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                        },
                    }
                )
            elif self.tool_style == "anthropic":
                tool_dicts.append(
                    {
                        "name": tool.name,
                    }
                )

        return self.bind(tools=tool_dicts, **kwargs)
