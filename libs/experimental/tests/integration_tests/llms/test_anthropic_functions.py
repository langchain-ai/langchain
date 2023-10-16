"""Test AnthropicFunctions"""

import unittest

from langchain.chat_models.anthropic import ChatAnthropic
from langchain.chat_models.bedrock import BedrockChat

from langchain_experimental.llms.anthropic_functions import AnthropicFunctions


class TestAnthropicFunctions(unittest.TestCase):
    """
    Test AnthropicFunctions with default llm (ChatAnthropic) as well as a passed-in llm
    """

    def test_default_chat_anthropic(self) -> None:
        base_model = AnthropicFunctions(model="claude-2")
        self.assertIsInstance(base_model.model, ChatAnthropic)

        # bind functions
        model = base_model.bind(
            functions=[
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
        self.assertEqual(
            function_call.get("arguments"),
            '{"location": "San Francisco, CA", "unit": "fahrenheit"}',
        )

    def test_bedrock_chat_anthropic(self) -> None:
        """
              const chatBedrock = new ChatBedrock({
          region: process.env.BEDROCK_AWS_REGION ?? "us-east-1",
          model: "anthropic.claude-v2",
          temperature: 0.1,
          credentials: {
            secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
            accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
          },
        });"""
        llm = BedrockChat(
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": 0.1},
            region_name="us-east-1",
        )
        base_model = AnthropicFunctions(llm=llm)
        assert isinstance(base_model.model, BedrockChat)

        # bind functions
        model = base_model.bind(
            functions=[
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
        self.assertEqual(
            function_call.get("arguments"),
            '{"location": "San Francisco, CA", "unit": "fahrenheit"}',
        )
