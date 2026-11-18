from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    add_notification_metadata,
    get_notification_metadata,
    messages_from_dict,
    messages_to_dict,
)


def test_metadata_is_added_to_a_copy() -> None:
    original = HumanMessage("done", id="msg_1", additional_kwargs={"other": 1})

    message = add_notification_metadata(
        original, source="async_subagent", data={"task_id": "t_42"}
    )

    assert message.id == "msg_1"
    assert message.content == "done"
    assert message.additional_kwargs == {
        "other": 1,
        "lc_source": "async_subagent",
        "lc_data": {"task_id": "t_42"},
    }
    assert original.additional_kwargs == {"other": 1}


def test_tool_metadata_preserves_message_and_survives_serialization() -> None:
    original = ToolMessage(
        "denied",
        id="msg_1",
        name="send_email",
        tool_call_id="call_1",
        status="error",
        artifact={"detail": "kept"},
        additional_kwargs={"other": 1},
    )

    message = add_notification_metadata(
        original, source="human_in_the_loop", data={"decision": "reject"}
    )

    assert isinstance(message, ToolMessage)
    assert message.model_dump(exclude={"additional_kwargs"}) == original.model_dump(
        exclude={"additional_kwargs"}
    )
    assert original.additional_kwargs == {"other": 1}
    assert message.additional_kwargs == {
        "other": 1,
        "lc_source": "human_in_the_loop",
        "lc_data": {"decision": "reject"},
    }
    [restored] = messages_from_dict(messages_to_dict([message]))
    assert restored == message
    assert get_notification_metadata(restored) == {
        "source": "human_in_the_loop",
        "data": {"decision": "reject"},
    }


def test_metadata_is_read_back() -> None:
    message = add_notification_metadata(
        HumanMessage("done"), source="async_subagent", data={"task_id": "t_42"}
    )

    assert get_notification_metadata(message) == {
        "source": "async_subagent",
        "data": {"task_id": "t_42"},
    }


def test_marking_again_replaces_source_and_data() -> None:
    message = add_notification_metadata(
        HumanMessage("x"), source="first", data={"a": 1}
    )

    remarked = add_notification_metadata(message, source="second")

    assert get_notification_metadata(remarked) == {"source": "second"}


def test_data_is_copied() -> None:
    data = {"status": "success"}
    message = add_notification_metadata(HumanMessage("x"), source="s", data=data)
    data["status"] = "error"

    assert get_notification_metadata(message) == {
        "source": "s",
        "data": {"status": "success"},
    }


def test_messages_without_a_string_source_have_no_metadata() -> None:
    assert get_notification_metadata(HumanMessage("hi")) is None
    assert (
        get_notification_metadata(HumanMessage("x", additional_kwargs={"lc_source": 1}))
        is None
    )


def test_non_dict_data_is_left_out() -> None:
    kwargs = {"lc_source": "s", "lc_data": "not a dict"}

    assert get_notification_metadata(HumanMessage("x", additional_kwargs=kwargs)) == {
        "source": "s"
    }


def test_metadata_survives_serialization() -> None:
    message = add_notification_metadata(
        HumanMessage("done", id="msg_1"), source="async_subagent", data={"a": 1}
    )

    [restored] = messages_from_dict(messages_to_dict([message]))

    assert restored == message
    assert get_notification_metadata(restored) == {
        "source": "async_subagent",
        "data": {"a": 1},
    }
