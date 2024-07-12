"""Standard LangChain interface tests"""

import base64
from typing import Type, cast

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_openai import ChatOpenAI


class TestOpenAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-4o", "stream_usage": True}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    # TODO: Add to standard tests if reliable token counting is added to other models.
    def test_image_token_counting_jpeg(self, model: BaseChatModel) -> None:
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
        expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
            "input_tokens"
        ]
        actual = model.get_num_tokens_from_messages([message])
        assert expected == actual

        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )
        expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
            "input_tokens"
        ]
        actual = model.get_num_tokens_from_messages([message])
        assert expected == actual

    def test_image_token_counting_png(self, model: BaseChatModel) -> None:
        image_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
        message = HumanMessage(
            content=[
                {"type": "text", "text": "how many dice are in this image"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
        expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
            "input_tokens"
        ]
        actual = model.get_num_tokens_from_messages([message])
        assert expected == actual

        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "how many dice are in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
            ]
        )
        expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
            "input_tokens"
        ]
        actual = model.get_num_tokens_from_messages([message])
        assert expected == actual
