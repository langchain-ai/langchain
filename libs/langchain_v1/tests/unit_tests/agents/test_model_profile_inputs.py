from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.callbacks import CallbackManagerForLLMRun

    from langchain.agents.middleware.types import ModelRequest, ModelResponse


class RecordingModel(GenericFakeChatModel):
    captured_messages: list[list[BaseMessage]] = Field(default_factory=list)

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.captured_messages.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage("done"))])


@pytest.mark.parametrize(
    ("startup_profile", "runtime_profile", "expected_type"),
    [
        ({"image_inputs": False}, {"image_inputs": True}, "image"),
        ({"image_inputs": True}, {"image_inputs": False}, "text"),
    ],
)
def test_filters_inputs_against_runtime_model_profile(
    startup_profile: dict[str, bool],
    runtime_profile: dict[str, bool],
    expected_type: str,
) -> None:
    startup = RecordingModel(messages=iter([]), profile=startup_profile)
    runtime = RecordingModel(messages=iter([]), profile=runtime_profile)

    class SwapModel(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            return handler(request.override(model=runtime))

    agent = create_agent(startup, middleware=[SwapModel()])
    image = {"type": "image", "base64": "aW1hZ2U=", "mime_type": "image/png"}

    agent.invoke({"messages": [HumanMessage(content=[image])]})

    assert not startup.captured_messages
    message = runtime.captured_messages[0][0]
    assert isinstance(message, HumanMessage)
    assert message.content_blocks[0]["type"] == expected_type


async def test_async_filters_inputs_against_runtime_model_profile() -> None:
    startup = RecordingModel(messages=iter([]), profile={"image_inputs": True})
    runtime = RecordingModel(messages=iter([]), profile={"image_inputs": False})

    class SwapModel(AgentMiddleware):
        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await handler(request.override(model=runtime))

    agent = create_agent(startup, middleware=[SwapModel()])
    image = {"type": "image", "base64": "aW1hZ2U=", "mime_type": "image/png"}

    await agent.ainvoke({"messages": [HumanMessage(content=[image])]})

    message = runtime.captured_messages[0][0]
    assert isinstance(message, HumanMessage)
    assert message.content_blocks[0]["type"] == "text"


def test_tool_message_profile_gate() -> None:
    model = RecordingModel(
        messages=iter([]),
        profile={"image_inputs": True, "image_tool_message": False},
    )
    agent = create_agent(model)
    image = {"type": "image", "base64": "aW1hZ2U=", "mime_type": "image/png"}
    tool_message = ToolMessage(content=[image], tool_call_id="call")

    agent.invoke({"messages": [tool_message]})

    message = model.captured_messages[0][0]
    assert isinstance(message, ToolMessage)
    assert message.content_blocks[0]["type"] == "text"
