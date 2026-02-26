from langchain_core.load import load
from langchain_core.prompts.dict import DictPromptTemplate


def test__dict_message_prompt_template_fstring() -> None:
    template = {
        "type": "text",
        "text": "{text1}",
        "cache_control": {"type": "{cache_type}"},
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    expected = {
        "type": "text",
        "text": "important message",
        "cache_control": {"type": "ephemeral"},
    }
    actual = prompt.format(text1="important message", cache_type="ephemeral")
    assert actual == expected


def test_deserialize_legacy() -> None:
    ser = {
        "type": "constructor",
        "lc": 1,
        "id": ["langchain_core", "prompts", "message", "_DictMessagePromptTemplate"],
        "kwargs": {
            "template_format": "f-string",
            "template": {"type": "audio", "audio": "{audio_data}"},
        },
    }
    expected = DictPromptTemplate(
        template={"type": "audio", "audio": "{audio_data}"}, template_format="f-string"
    )
    assert load(ser) == expected
