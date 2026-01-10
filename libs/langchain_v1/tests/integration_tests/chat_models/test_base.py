from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_tests.integration_tests import ChatModelIntegrationTests
from pydantic import BaseModel

from langchain.chat_models import init_chat_model


class Multiply(BaseModel):
    """Product of two ints."""

    x: int
    y: int


@pytest.mark.requires("langchain_openai", "langchain_anthropic")
async def test_init_chat_model_chain() -> None:
    model = init_chat_model("gpt-4o", configurable_fields="any", config_prefix="bar")
    model_with_tools = model.bind_tools([Multiply])

    model_with_config = model_with_tools.with_config(
        RunnableConfig(tags=["foo"]),
        configurable={"bar_model": "claude-sonnet-4-5-20250929"},
    )
    prompt = ChatPromptTemplate.from_messages([("system", "foo"), ("human", "{input}")])
    chain = prompt | model_with_config
    output = chain.invoke({"input": "bar"})
    assert isinstance(output, AIMessage)
    events = [event async for event in chain.astream_events({"input": "bar"}, version="v2")]
    assert events


class TestStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return cast("type[BaseChatModel]", init_chat_model)

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {"model": "gpt-4o", "configurable_fields": "any"}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_structured_output(self) -> bool:
        return True
