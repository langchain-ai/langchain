"""Tests for `UnsupportedContentMiddleware`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, UnsupportedContentMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import AnyMessage, ContentBlock

    from langchain.agents.middleware.types import ModelRequest, ModelResponse

IMAGE: ContentBlock = {"type": "image", "base64": "aW1hZ2U=", "mime_type": "image/png"}
PDF: ContentBlock = {
    "type": "file",
    "base64": "cGRm",
    "mime_type": "application/pdf",
}


class RecordingModel(GenericFakeChatModel):
    """Fake model capturing the messages each request carried."""

    captured: list[list[BaseMessage]] = Field(default_factory=list)

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.captured.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage("done"))])


def _model(**profile: Any) -> RecordingModel:
    return RecordingModel(messages=iter([]), profile=profile)


def _swap(model: RecordingModel) -> AgentMiddleware:
    """Middleware standing in for a runtime model switch."""

    class SwapModel(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            return handler(request.override(model=model))

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await handler(request.override(model=model))

    return SwapModel()


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ({"image_inputs": False}, "text"),
        ({"image_inputs": True}, "image"),
        ({}, "image"),
    ],
)
def test_filters_against_profile(profile: dict[str, Any], expected: str) -> None:
    model = _model(**profile)
    agent = create_agent(model, middleware=[UnsupportedContentMiddleware()])

    agent.invoke({"messages": [HumanMessage(content=[IMAGE])]})

    message = model.captured[0][0]
    assert message.content_blocks[0]["type"] == expected


@pytest.mark.parametrize(
    ("startup", "runtime", "expected"),
    [
        ({"image_inputs": False}, {"image_inputs": True}, "image"),
        ({"image_inputs": True}, {"image_inputs": False}, "text"),
    ],
)
def test_filters_against_runtime_model(
    startup: dict[str, Any], runtime: dict[str, Any], expected: str
) -> None:
    """The innermost layer must see the model an outer middleware swapped in."""
    startup_model = _model(**startup)
    runtime_model = _model(**runtime)
    agent = create_agent(
        startup_model,
        middleware=[_swap(runtime_model), UnsupportedContentMiddleware()],
    )

    agent.invoke({"messages": [HumanMessage(content=[IMAGE])]})

    assert not startup_model.captured
    message = runtime_model.captured[0][0]
    assert message.content_blocks[0]["type"] == expected


async def test_async_filters_against_runtime_model() -> None:
    startup_model = _model(image_inputs=True)
    runtime_model = _model(image_inputs=False)
    agent = create_agent(
        startup_model,
        middleware=[_swap(runtime_model), UnsupportedContentMiddleware()],
    )

    await agent.ainvoke({"messages": [HumanMessage(content=[IMAGE])]})

    message = runtime_model.captured[0][0]
    assert message.content_blocks[0]["type"] == "text"


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ({"image_inputs": True, "image_tool_message": False}, "text"),
        ({"image_inputs": True, "image_tool_message": True}, "image"),
    ],
)
def test_tool_message_gate(profile: dict[str, Any], expected: str) -> None:
    """`image_tool_message` gates images only inside a `ToolMessage`."""
    model = _model(**profile)
    agent = create_agent(model, middleware=[UnsupportedContentMiddleware()])

    agent.invoke(
        {
            "messages": [
                HumanMessage(content=[IMAGE]),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "t", "args": {}, "id": "call", "type": "tool_call"}],
                ),
                ToolMessage(content=[IMAGE], tool_call_id="call"),
            ]
        }
    )

    human, _, tool = model.captured[0]
    assert human.content_blocks[0]["type"] == "image"
    assert tool.content_blocks[0]["type"] == expected


def test_pdf_tool_message_gate() -> None:
    model = _model(pdf_inputs=True, pdf_tool_message=False)
    agent = create_agent(model, middleware=[UnsupportedContentMiddleware()])

    agent.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "t", "args": {}, "id": "call", "type": "tool_call"}],
                ),
                ToolMessage(content=[PDF], tool_call_id="call"),
            ]
        }
    )

    assert model.captured[0][1].content_blocks[0]["type"] == "text"


def test_non_pdf_file_blocks_are_left_alone() -> None:
    """No profile field describes non-PDF payloads, so `pdf_inputs` must not gate them."""
    model = _model(pdf_inputs=False)
    docx: ContentBlock = {
        "type": "file",
        "base64": "ZG9jeA==",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    agent = create_agent(model, middleware=[UnsupportedContentMiddleware()])

    agent.invoke({"messages": [HumanMessage(content=[docx])]})

    assert model.captured[0][0].content_blocks[0]["type"] == "file"


def test_image_url_gate() -> None:
    model = _model(image_inputs=True, image_url_inputs=False)
    url_image: ContentBlock = {"type": "image", "url": "https://example.com/i.png"}
    agent = create_agent(model, middleware=[UnsupportedContentMiddleware()])

    agent.invoke({"messages": [HumanMessage(content=[url_image, IMAGE])]})

    blocks = model.captured[0][0].content_blocks
    assert [block["type"] for block in blocks] == ["text", "image"]


def test_string_content_is_untouched() -> None:
    """A string-content message keeps its string form rather than becoming a block list."""
    model = _model(image_inputs=False)
    agent = create_agent(model, middleware=[UnsupportedContentMiddleware()])

    agent.invoke({"messages": [HumanMessage(content="hello")]})

    assert model.captured[0][0].content == "hello"


def test_on_unsupported_customizes_the_notice() -> None:
    model = _model(image_inputs=False)

    def on_unsupported(block: ContentBlock, message: AnyMessage) -> str:
        assert isinstance(message, HumanMessage)
        return f"dropped a {block['type']}"

    agent = create_agent(
        model, middleware=[UnsupportedContentMiddleware(on_unsupported=on_unsupported)]
    )

    agent.invoke({"messages": [HumanMessage(content=[IMAGE])]})

    assert model.captured[0][0].content_blocks[0]["text"] == "dropped a image"


def test_on_unsupported_returning_none_drops_the_block() -> None:
    model = _model(image_inputs=False)
    agent = create_agent(
        model,
        middleware=[UnsupportedContentMiddleware(on_unsupported=lambda _block, _message: None)],
    )

    agent.invoke({"messages": [HumanMessage(content=[IMAGE, {"type": "text", "text": "hi"}])]})

    blocks = model.captured[0][0].content_blocks
    assert [block["type"] for block in blocks] == ["text"]


def test_subclass_can_extend_support_checks() -> None:
    """Subclasses gate on things no profile field covers (e.g. the provider class)."""

    class RejectsDocx(UnsupportedContentMiddleware):
        def is_supported(
            self, block: ContentBlock, *, model: Any, profile: Any, in_tool_message: bool
        ) -> bool:
            if block["type"] == "file" and "wordprocessingml" in block.get("mime_type", ""):
                return False
            return super().is_supported(
                block, model=model, profile=profile, in_tool_message=in_tool_message
            )

    model = _model()
    docx: ContentBlock = {
        "type": "file",
        "base64": "ZG9jeA==",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    agent = create_agent(model, middleware=[RejectsDocx()])

    agent.invoke({"messages": [HumanMessage(content=[docx])]})

    assert model.captured[0][0].content_blocks[0]["type"] == "text"
