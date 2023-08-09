from typing import Any

import openai

from langchain.adapters import openai as lcopenai


def _test_no_stream(**kwargs: Any) -> None:
    result = openai.ChatCompletion.create(**kwargs)
    lc_result = lcopenai.ChatCompletion.create(**kwargs)
    if isinstance(lc_result, dict):
        if isinstance(result, dict):
            result_dict = result["choices"][0]["message"].to_dict_recursive()
            lc_result_dict = lc_result["choices"][0]["message"]
            assert result_dict == lc_result_dict
    return


def _test_stream(**kwargs: Any) -> None:
    result = []
    for c in openai.ChatCompletion.create(**kwargs):
        result.append(c["choices"][0]["delta"].to_dict_recursive())

    lc_result = []
    for c in lcopenai.ChatCompletion.create(**kwargs):
        lc_result.append(c["choices"][0]["delta"])
    assert result == lc_result


FUNCTIONS = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]


def _test_module(**kwargs: Any) -> None:
    _test_no_stream(**kwargs)
    _test_stream(stream=True, **kwargs)


def test_normal_call() -> None:
    _test_module(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-3.5-turbo",
        temperature=0,
    )


def test_function_calling() -> None:
    _test_module(
        messages=[{"role": "user", "content": "whats the weather in boston"}],
        model="gpt-3.5-turbo",
        functions=FUNCTIONS,
        temperature=0,
    )


def test_answer_with_function_calling() -> None:
    _test_module(
        messages=[
            {"role": "user", "content": "say hi, then whats the weather in boston"}
        ],
        model="gpt-3.5-turbo",
        functions=FUNCTIONS,
        temperature=0,
    )
