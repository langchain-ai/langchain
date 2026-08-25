from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
)

from langchain.mcp.prompts import (
    convert_mcp_prompt_message_to_langchain_message,
    load_mcp_prompt,
)


@pytest.mark.parametrize(
    ("role", "text", "expected_cls"),
    [("assistant", "Hello", AIMessage), ("user", "Hello", HumanMessage)],
)
def test_convert_mcp_prompt_message_to_langchain_message_with_text_content(
    role: str,
    text: str,
    expected_cls: type,
) -> None:
    message = PromptMessage(role=role, content=TextContent(type="text", text=text))
    result = convert_mcp_prompt_message_to_langchain_message(message)
    assert isinstance(result, expected_cls)
    assert result.content == text


@pytest.mark.parametrize("role", ["assistant", "user"])
def test_convert_mcp_prompt_message_to_langchain_message_with_resource_content(
    role: str,
) -> None:
    message = PromptMessage(
        role=role,
        content=EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri="message://greeting",
                mimeType="text/plain",
                text="hi",
            ),
        ),
    )
    with pytest.raises(ValueError):
        convert_mcp_prompt_message_to_langchain_message(message)


@pytest.mark.parametrize("role", ["assistant", "user"])
def test_convert_mcp_prompt_message_to_langchain_message_with_image_content(role: str) -> None:
    message = PromptMessage(
        role=role,
        content=ImageContent(type="image", mimeType="image/png", data="base64data"),
    )
    with pytest.raises(ValueError):
        convert_mcp_prompt_message_to_langchain_message(message)


async def test_load_mcp_prompt() -> None:
    session = AsyncMock()
    session.get_prompt = AsyncMock(
        return_value=AsyncMock(
            messages=[
                PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
                PromptMessage(role="assistant", content=TextContent(type="text", text="Hi")),
            ],
        ),
    )
    result = await load_mcp_prompt(session, "test_prompt")
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello"
    assert isinstance(result[1], AIMessage)
    assert result[1].content == "Hi"
