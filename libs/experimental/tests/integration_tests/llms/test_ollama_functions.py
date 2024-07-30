"""Test OllamaFunctions"""

import unittest

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.pubmed.tool import PubmedQueryRun
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
        base_model = OllamaFunctions(model="phi3", format="json")

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

        self.assertIsInstance(res, AIMessage)
        res = AIMessage(**res.__dict__)
        tool_calls = res.tool_calls
        assert tool_calls
        tool_call = tool_calls[0]
        assert tool_call
        self.assertEqual("get_current_weather", tool_call.get("name"))

    def test_ollama_functions_tools(self) -> None:
        base_model = OllamaFunctions(model="phi3", format="json")
        model = base_model.bind_tools(
            tools=[PubmedQueryRun(), DuckDuckGoSearchResults(max_results=2)]  # type: ignore[call-arg]
        )
        res = model.invoke("What causes lung cancer?")
        self.assertIsInstance(res, AIMessage)
        res = AIMessage(**res.__dict__)
        tool_calls = res.tool_calls
        assert tool_calls
        tool_call = tool_calls[0]
        assert tool_call
        self.assertEqual("pub_med", tool_call.get("name"))

    def test_default_ollama_functions_default_response(self) -> None:
        base_model = OllamaFunctions(model="phi3", format="json")

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
            ]
        )

        res = model.invoke("What is the capital of France?")

        self.assertIsInstance(res, AIMessage)
        res = AIMessage(**res.__dict__)
        tool_calls = res.tool_calls
        if len(tool_calls) > 0:
            tool_call = tool_calls[0]
            assert tool_call
            self.assertEqual("__conversational_response", tool_call.get("name"))

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
        assert isinstance(res, dict)
        assert "raw" in res
        assert "parsed" in res
        assert isinstance(res["raw"], AIMessage)
        assert isinstance(res["parsed"], Joke)
