from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def test_pretty_repr_string_content() -> None:
    msg = HumanMessage(content="What is the capital of France?")
    result = msg.pretty_repr()
    assert "Human Message" in result
    assert "What is the capital of France?" in result


def test_pretty_repr_text_block() -> None:
    msg = HumanMessage(content=[{"type": "text", "text": "Hello"}])
    result = msg.pretty_repr()
    assert "Hello" in result
    assert "{'type': 'text'" not in result


def test_pretty_repr_image_url_block() -> None:
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "url": "https://example.com/image.png"},
        ]
    )
    result = msg.pretty_repr()
    assert "Describe this image" in result
    assert "[image: https://example.com/image.png]" in result
    assert "{'type': 'image_url'" not in result


def test_pretty_repr_image_url_dict_block() -> None:
    msg = HumanMessage(
        content=[
            {
                "type": "image_url",
                "url": {"url": "https://example.com/image.png"},
            }
        ]
    )
    result = msg.pretty_repr()
    assert "[image: https://example.com/image.png]" in result


def test_pretty_repr_image_base64_block() -> None:
    msg = HumanMessage(
        content=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBOR...",
                },
            }
        ]
    )
    result = msg.pretty_repr()
    assert "[image: base64 image/png]" in result


def test_pretty_repr_audio_base64_block() -> None:
    msg = HumanMessage(
        content=[
            {
                "type": "audio",
                "source": {
                    "type": "base64",
                    "media_type": "audio/mp3",
                    "data": "UklGRi...",
                },
            }
        ]
    )
    result = msg.pretty_repr()
    assert "[audio: base64 audio/mp3]" in result


def test_pretty_repr_video_base64_block() -> None:
    msg = HumanMessage(
        content=[
            {
                "type": "video",
                "source": {
                    "type": "base64",
                    "media_type": "video/mp4",
                    "data": "AAAA...",
                },
            }
        ]
    )
    result = msg.pretty_repr()
    assert "[video: base64 video/mp4]" in result


def test_pretty_repr_reasoning_block() -> None:
    msg = AIMessage(
        content=[
            {"type": "reasoning", "reasoning_text": "Let me think about this..."},
            {"type": "text", "text": "The answer is 42."},
        ]
    )
    result = msg.pretty_repr()
    assert "[reasoning: Let me think about this...]" in result
    assert "The answer is 42." in result


def test_pretty_repr_reasoning_block_empty() -> None:
    msg = AIMessage(
        content=[{"type": "reasoning", "reasoning_text": ""}]
    )
    result = msg.pretty_repr()
    assert "[reasoning]" in result


def test_pretty_repr_audio_no_source() -> None:
    msg = HumanMessage(content=[{"type": "audio"}])
    result = msg.pretty_repr()
    assert "[audio]" in result


def test_pretty_repr_video_no_source() -> None:
    msg = HumanMessage(content=[{"type": "video"}])
    result = msg.pretty_repr()
    assert "[video]" in result


def test_pretty_repr_image_no_url() -> None:
    msg = HumanMessage(content=[{"type": "image"}])
    result = msg.pretty_repr()
    assert "[image]" in result


def test_pretty_repr_mixed_string_and_blocks() -> None:
    msg = HumanMessage(
        content=["Hello", {"type": "text", "text": "World"}]
    )
    result = msg.pretty_repr()
    assert "Hello" in result
    assert "World" in result


def test_pretty_repr_unknown_block_type() -> None:
    msg = HumanMessage(content=[{"type": "custom", "data": "value"}])
    result = msg.pretty_repr()
    assert "custom" in result
    assert "value" in result


def test_pretty_repr_with_name() -> None:
    msg = HumanMessage(content="Hello", name="Alice")
    result = msg.pretty_repr()
    assert "Name: Alice" in result
    assert "Hello" in result


def test_pretty_repr_system_message() -> None:
    msg = SystemMessage(content="You are a helpful assistant.")
    result = msg.pretty_repr()
    assert "System Message" in result
    assert "You are a helpful assistant." in result


def test_pretty_repr_ai_message_multimodal() -> None:
    msg = AIMessage(
        content=[
            {"type": "text", "text": "Here is the analysis"},
            {"type": "reasoning", "reasoning_text": "I analyzed the data"},
        ]
    )
    result = msg.pretty_repr()
    assert "Ai Message" in result
    assert "Here is the analysis" in result
    assert "[reasoning: I analyzed the data]" in result
