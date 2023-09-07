from __future__ import annotations

import json
from typing import Iterator, List, Any, Mapping

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult, ChatGeneration, AIMessage
from langchain.tools import BaseTool


class FakeChatModel(BaseChatModel):
    """A fake chat model that returns a pre-defined response."""

    message_iter: Iterator[BaseMessage]

    @property
    def _llm_type(self) -> str:
        """The type of the model."""
        return "fake-openai-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response to the given messages."""
        message = next(self.message_iter)
        return ChatResult(generations=[ChatGeneration(message=message)])


def construct_func_invocation_message(
    tool: BaseTool, args: Mapping[str, Any]
) -> AIMessage:
    """Construct a function invocation message."""
    return AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": tool.name,
                "arguments": json.dumps(args),
            }
        },
    )
