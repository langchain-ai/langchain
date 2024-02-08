"""Test ChatGoogleGenerativeAI function call."""

import json

from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import tool

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
    arguments_str = res.additional_kwargs["function_call"]["arguments"]
    assert isinstance(arguments_str, str)
    arguments = json.loads(arguments_str)
    assert "location" in arguments


def test_tool_call() -> None:
    @tool
    def search_tool(query: str) -> str:
        """Searches the web for `query` and returns the result."""
        raise NotImplementedError

    llm = ChatGoogleGenerativeAI(model="gemini-pro").bind(functions=[search_tool])
    response = llm.invoke("weather in san francisco")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == ""
    function_call = response.additional_kwargs.get("function_call")
    assert function_call
    assert function_call["name"] == "search_tool"
    arguments_str = function_call.get("arguments")
    assert arguments_str
    arguments = json.loads(arguments_str)
    assert "query" in arguments


class MyModel(BaseModel):
    name: str
    age: int


def test_pydantic_call() -> None:
    llm = ChatGoogleGenerativeAI(model="gemini-pro").bind(functions=[MyModel])
    response = llm.invoke("my name is Erick and I am 27 years old")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == ""
    function_call = response.additional_kwargs.get("function_call")
    assert function_call
    assert function_call["name"] == "MyModel"
    arguments_str = function_call.get("arguments")
    assert arguments_str
    arguments = json.loads(arguments_str)
    assert arguments == {
        "name": "Erick",
        "age": 27.0,
    }
