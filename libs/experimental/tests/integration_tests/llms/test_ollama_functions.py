"""Test OllamaFunctions"""

import unittest

from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_experimental.llms.ollama_functions import (
    OllamaFunctions,
    convert_to_ollama_tool,
)


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


class TestOllamaFunctions(unittest.TestCase):
    """
    Test OllamaFunctions
    """

    def test_default_ollama_functions(self) -> None:
        base_model = OllamaFunctions(model="llama3", format="json")

        # bind functions
        model = base_model.bind_tools(
            tools=[
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, "
                                "e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ],
            function_call={"name": "get_current_weather"},
        )

        res = model.invoke("What's the weather in San Francisco?")

        function_call = res.additional_kwargs.get("function_call")
        assert function_call
        self.assertEqual(function_call.get("name"), "get_current_weather")

    def test_ollama_structured_output(self) -> None:
        model = OllamaFunctions(model="phi3")
        structured_llm = model.with_structured_output(Joke, include_raw=False)

        res = structured_llm.invoke("Tell me a joke about cats")
        assert isinstance(res, Joke)

    def test_ollama_structured_output_with_json(self) -> None:
        model = OllamaFunctions(model="phi3")
        joke_schema = convert_to_ollama_tool(Joke)
        structured_llm = model.with_structured_output(joke_schema, include_raw=False)

        res = structured_llm.invoke("Tell me a joke about cats")
        assert "setup" in res
        assert "punchline" in res

    def test_ollama_structured_output_raw(self) -> None:
        model = OllamaFunctions(model="phi3")
        structured_llm = model.with_structured_output(Joke, include_raw=True)

        res = structured_llm.invoke("Tell me a joke about cars")
        assert "raw" in res
        assert "parsed" in res
        assert isinstance(res["raw"], AIMessage)
        assert isinstance(res["parsed"], Joke)
