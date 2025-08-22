from langchain_core.messages import HumanMessage
from langchain_core.messages import content as types
from tests.unit_tests.language_models.chat_models.test_base import (
    _content_blocks_equal_ignore_id,
)


def test_convert_to_v1_from_openai_input() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},
            {
                "type": "image",
                "source_type": "url",
                "url": "https://example.com/image.png",
            },
            {
                "type": "image",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "image/png",
            },
            {
                "type": "file",
                "source_type": "url",
                "url": "<document url>",
            },
            {
                "type": "file",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "application/pdf",
            },
            {
                "type": "audio",
                "source_type": "base64",
                "data": "<base64 data>",
                "mime_type": "audio/mpeg",
            },
            {
                "type": "file",
                "source_type": "id",
                "id": "<file id>",
            },
        ]
    )

    expected: list[types.ContentBlock] = [
        {"type": "text", "text": "Hello"},
        {
            "type": "image",
            "url": "https://example.com/image.png",
        },
        {
            "type": "image",
            "base64": "<base64 data>",
            "mime_type": "image/png",
        },
        {
            "type": "file",
            "url": "<document url>",
        },
        {
            "type": "file",
            "base64": "<base64 data>",
            "mime_type": "application/pdf",
        },
        {
            "type": "audio",
            "base64": "<base64 data>",
            "mime_type": "audio/mpeg",
        },
        {
            "type": "file",
            "file_id": "<file id>",
        },
    ]

    assert _content_blocks_equal_ignore_id(message.content_blocks, expected)
