import pytest

from langchain_core.messages import (
    NOTIFICATION_HEADER,
    HumanMessage,
    ImageContentBlock,
    TextContentBlock,
    create_notification_message,
    messages_from_dict,
    messages_to_dict,
)


def test_notification_starts_with_header_and_note() -> None:
    message = create_notification_message(
        "<status>success</status>", source="async_subagent", note="A task finished."
    )

    assert isinstance(message, HumanMessage)
    assert message.content == (
        f"{NOTIFICATION_HEADER}\nA task finished.\n<status>success</status>"
    )


def test_notification_without_header_keeps_content() -> None:
    message = create_notification_message(
        "Here is a summary.", source="summarization", header=False
    )

    assert message.content == "Here is a summary."
    assert message.additional_kwargs == {
        "lc_synthetic": True,
        "lc_source": "summarization",
    }


def test_notification_prepends_header_block_to_content_blocks() -> None:
    text: TextContentBlock = {"type": "text", "text": "Frame 1"}
    image: ImageContentBlock = {"type": "image", "url": "https://example.com/a.png"}
    message = create_notification_message(
        [text, image], source="read_file_media", note="Frames from a video."
    )

    assert message.content == [
        {"type": "text", "text": f"{NOTIFICATION_HEADER}\nFrames from a video."},
        text,
        image,
    ]


def test_notification_labels_survive_serialization() -> None:
    message = create_notification_message(
        "done",
        source="async_subagent",
        data={"task_id": "t_42", "status": "success"},
        id="msg_1",
    )

    [restored] = messages_from_dict(messages_to_dict([message]))

    assert restored == message
    assert restored.additional_kwargs == {
        "lc_synthetic": True,
        "lc_source": "async_subagent",
        "lc_data": {"task_id": "t_42", "status": "success"},
    }


def test_notification_rejects_note_without_header() -> None:
    with pytest.raises(ValueError, match="header=True"):
        create_notification_message(
            "summary", source="summarization", note="ignored", header=False
        )
