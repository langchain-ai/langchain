import json

import pytest

from langchain_core.load import load, loads
from langchain_core.prompts import PromptTemplate
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
    assert load(ser, allowed_objects=[DictPromptTemplate]) == expected


def test_dict_prompt_template_rejects_attribute_access_to_rich_objects() -> None:
    with pytest.raises(ValueError, match="Variable names cannot contain attribute"):
        DictPromptTemplate(
            template={"output": "{message.additional_kwargs[secret]}"},
            template_format="f-string",
        )


def test_dict_prompt_template_loads_payload_rejects_attribute_access() -> None:
    payload = json.dumps(
        {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain_core", "prompts", "dict", "DictPromptTemplate"],
            "kwargs": {
                "template": {"output": "{message.additional_kwargs[secret]}"},
                "template_format": "f-string",
            },
        }
    )

    with pytest.raises(ValueError, match="Variable names cannot contain attribute"):
        loads(payload, allowed_objects=[DictPromptTemplate])


def test_dict_prompt_template_dumpd_round_trip_rejects_attribute_access() -> None:
    payload = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain_core", "prompts", "dict", "DictPromptTemplate"],
        "kwargs": {
            "template": {"output": "{message.additional_kwargs[secret]}"},
            "template_format": "f-string",
        },
    }

    with pytest.raises(ValueError, match="Variable names cannot contain attribute"):
        load(payload, allowed_objects=[DictPromptTemplate])


def test_dict_prompt_template_deserialization_rejects_attribute_access() -> None:
    payload = json.dumps(
        {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain_core", "prompts", "dict", "DictPromptTemplate"],
            "kwargs": {
                "template": {"output": "{name.__class__.__name__}"},
                "template_format": "f-string",
            },
        }
    )

    with pytest.raises(ValueError, match="Variable names cannot contain attribute"):
        loads(payload, allowed_objects=[DictPromptTemplate])


def test_dict_prompt_template_legacy_deserialization_rejects_attribute_access() -> None:
    ser = {
        "type": "constructor",
        "lc": 1,
        "id": ["langchain_core", "prompts", "message", "_DictMessagePromptTemplate"],
        "kwargs": {
            "template_format": "f-string",
            "template": {"output": "{name.__class__.__name__}"},
        },
    }

    with pytest.raises(ValueError, match="Variable names cannot contain attribute"):
        load(ser, allowed_objects=[DictPromptTemplate])


def test_prompt_template_blocks_attribute_access() -> None:
    with pytest.raises(
        ValueError, match="Variable names cannot contain attribute access"
    ):
        PromptTemplate.from_template("{name.__class__}", template_format="f-string")


def test_dict_prompt_template_preserves_non_str_items_in_list() -> None:
    """Non-str/non-dict items inside lists must be preserved, not silently dropped.

    Regression test for #39152: int, float, bool, None and other scalars nested
    inside a list value were silently discarded by _insert_input_variables.
    """
    template = {
        "type": "tool_use",
        "id": "call_1",
        "name": "search",
        "input": {
            "query": "{q}",
            "top_k_scores": [1, 2, 3],
            "flags": [True, None],
        },
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format(q="cats")

    assert result["input"]["query"] == "cats"
    assert result["input"]["top_k_scores"] == [1, 2, 3]
    assert result["input"]["flags"] == [True, None]


def test_dict_prompt_template_preserves_mixed_list() -> None:
    """Mixed-type lists must survive formatting unchanged (except str interpolation)."""
    template = {
        "type": "x",
        "mixed": ["a {v}", 1, 2.5, {"k": "val"}, None, True],
    }
    prompt = DictPromptTemplate(template=template, template_format="f-string")
    result = prompt.format(v="b")

    assert result["mixed"] == ["a b", 1, 2.5, {"k": "val"}, None, True]


def test_dict_prompt_template_preserves_list_with_no_variables() -> None:
    """Items must survive even when the template has no variables at all."""
    template = {"nums": [1, 2]}
    prompt = DictPromptTemplate(template=template, template_format="mustache")
    assert prompt.format() == {"nums": [1, 2]}
