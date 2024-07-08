from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig

from langchain.chat_models import init_chat_model


class multiply(BaseModel):
    """Product of two ints."""

    x: int
    y: int


def test_init_chat_model() -> None:
    model = init_chat_model("gpt-4o", configurable_fields="any", config_prefix="bar")
    model_with_tools = model.bind_tools([multiply])

    model_with_config = model_with_tools.with_config(
        RunnableConfig(tags=["foo"]),
        configurable={"bar_model": "claude-3-sonnet-20240229"},
    )
    prompt = ChatPromptTemplate.from_messages([("system", "foo"), ("human", "{input}")])
    chain = prompt | model_with_config
    output = chain.invoke({"input": "bar"})
    assert isinstance(output, AIMessage)
