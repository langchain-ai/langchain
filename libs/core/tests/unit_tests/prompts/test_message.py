from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts.message import _DictMessagePromptTemplate

CUR_DIR = Path(__file__).parent.absolute().resolve()


def test__dict_message_prompt_template_fstring() -> None:
    template = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "{text1}", "cache_control": {"type": "ephemeral"}},
        ],
        "name": "{name1}",
        "tool_calls": [
            {
                "name": "{tool_name1}",
                "args": {"arg1": "{tool_arg1}"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    }
    prompt = _DictMessagePromptTemplate(template=template, template_format="f-string")
    expected: BaseMessage = AIMessage(
        [
            {
                "type": "text",
                "text": "important message",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        name="foo",
        tool_calls=[
            {
                "name": "do_stuff",
                "args": {"arg1": "important arg1"},
                "id": "1",
                "type": "tool_call",
            }
        ],
    )
    actual = prompt.format_messages(
        text1="important message",
        name1="foo",
        tool_arg1="important arg1",
        tool_name1="do_stuff",
    )[0]
    assert actual == expected

    template = {
        "role": "tool",
        "content": "{content1}",
        "tool_call_id": "1",
        "name": "{name1}",
    }
    prompt = _DictMessagePromptTemplate(template=template, template_format="f-string")
    expected = ToolMessage("foo", name="bar", tool_call_id="1")
    actual = prompt.format_messages(content1="foo", name1="bar")[0]
    assert actual == expected
