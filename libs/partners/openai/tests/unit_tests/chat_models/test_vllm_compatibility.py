from langchain_core.messages import ToolMessage

from langchain_openai.chat_models.base import _convert_message_to_dict


def test_tool_message_harmony_uses_standard_text_type() -> None:
    """
    ASSERTION: Even when 'responses' API or 'force_harmony' is used,
    ToolMessage content parts must have type 'text', NOT 'output_text'.

    This ensures compatibility with vLLM's strict Pydantic validation.
    """
    msg = ToolMessage(
        content="Satellite fact: Satellites monitor Earth.", tool_call_id="call_123"
    )

    converted = _convert_message_to_dict(msg, api="responses")

    assert converted["type"] == "tool_result"
    assert isinstance(converted["content"], list)

    assert converted["content"][0]["type"] == "text", (
        f"Expected content type 'text' for vLLM compatibility, "
        f"but got '{converted['content'][0]['type']}'"
    )
    assert (
        converted["content"][0]["text"] == "Satellite fact: Satellites monitor Earth."
    )
