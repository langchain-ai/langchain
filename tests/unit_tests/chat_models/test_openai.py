"""Test OpenAI Chat API wrapper."""

import json

from langchain.chat_models.openai import (
    _convert_dict_to_message,
)
from langchain.schema import (
    FunctionMessage,
)


def test_function_message_dict_to_function_message() -> None:
    content = json.dumps({"result": "Example #1"})
    name = "test_function"
    result = _convert_dict_to_message(
        {
            "role": "function",
            "name": name,
            "content": content,
        }
    )
    assert isinstance(result, FunctionMessage)
    assert result.name == name
    assert result.content == content
