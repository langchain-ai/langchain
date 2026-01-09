from langchain_core.messages import AIMessage


def test_pretty_repr_formats_content_blocks() -> None:
    message = AIMessage(
        content=[
            {"type": "text", "text": "Hello"},
            {"type": "image", "url": "https://example.com/image.png"},
        ]
    )

    rendered = message.pretty_repr()

    assert "Ai Message" in rendered
    assert '"type": "text"' in rendered
    assert '"type": "image"' in rendered
    assert "[\n" in rendered
