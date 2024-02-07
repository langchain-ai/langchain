"""Test ChatGoogleGenerativeAI function call."""

from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
)


def test_function_call() -> None:
    functions = [
        {
            "name": "get_weather",
            "description": "Determine weather in my location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["location"],
            },
        }
    ]
    llm = ChatGoogleGenerativeAI(model="gemini-pro").bind(functions=functions)
    res = llm.invoke("what weather is today in san francisco?")
    assert res
    assert res.additional_kwargs
    assert "function_call" in res.additional_kwargs
    assert "get_weather" == res.additional_kwargs["function_call"]["name"]
    arguments = res.additional_kwargs["function_call"]["arguments"]
    assert isinstance(arguments, dict)
    assert "location" in arguments
