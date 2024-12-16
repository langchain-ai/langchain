"""Test ChatSnowflakeCortex."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_community.chat_models.snowflake import _convert_message_to_dict


def test_messages_to_prompt_dict_with_valid_messages() -> None:
    messages = [
        SystemMessage(content="System Prompt"),
        HumanMessage(content="User message #1"),
        AIMessage(content="AI message #1"),
        HumanMessage(content="User message #2"),
        AIMessage(content="AI message #2"),
    ]
    result = [_convert_message_to_dict(m) for m in messages]
    expected = [
        {"role": "system", "content": "System Prompt"},
        {"role": "user", "content": "User message #1"},
        {"role": "assistant", "content": "AI message #1"},
        {"role": "user", "content": "User message #2"},
        {"role": "assistant", "content": "AI message #2"},
    ]
    assert result == expected
