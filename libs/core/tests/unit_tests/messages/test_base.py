from langchain_core.messages import AIMessage, HumanMessage


def test_pretty_repr_string_content() -> None:
    msg = HumanMessage(content="What is the capital of France?")
    repr_str = msg.pretty_repr()
    assert "Human Message" in repr_str
    assert "What is the capital of France?" in repr_str


def test_pretty_repr_multimodal_content() -> None:
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
        ]
    )
    repr_str = msg.pretty_repr()
    assert "Human Message" in repr_str
    assert "What is in this image?" in repr_str
    assert "[image: https://example.com/image.png]" in repr_str
    assert "'type': 'text'" not in repr_str  # Ensure raw dict is not printed


def test_pretty_repr_all_blocks() -> None:
    msg = AIMessage(
        content=[
            {"type": "text", "text": "Look at these:"},
            {"type": "image", "url": "https://foo.com/bar.jpg"},
            {"type": "image", "base64": "ZXhhbXBsZQ=="},
            {"type": "video", "url": "https://foo.com/video.mp4"},
            {"type": "audio", "base64": "XYZ="},
            {"type": "reasoning", "reasoning": "This is a thought process."},
        ]
    )
    repr_str = msg.pretty_repr()
    assert "Ai Message" in repr_str
    assert "Look at these:" in repr_str
    assert "[image: https://foo.com/bar.jpg]" in repr_str
    assert "[image: base64 data]" in repr_str
    assert "[video: https://foo.com/video.mp4]" in repr_str
    assert "[audio: base64 data]" in repr_str
    assert "This is a thought process." in repr_str
