"""Test OllamaFunctions"""

import unittest

from langchain_experimental.llms.ollama_functions import OllamaFunctions


class TestOllamaFunctions(unittest.TestCase):
    """
    Test OllamaFunctions
    """

    def test_default_ollama_functions(self) -> None:
        base_model = OllamaFunctions(
            model="llama3", format="json"
        )

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
