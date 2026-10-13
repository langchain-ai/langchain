import pickle

from langchain_core.load import dumpd
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.prompt_values import (
    ChatPromptValue,
    ChatPromptValueConcrete,
    StringPromptValue,
    _get_prompt_template_info,
    _set_prompt_template_info,
)
from langchain_core.prompts import ChatPromptTemplate


def test_chat_prompt_value_concrete() -> None:
    messages = [
        AIMessage("foo"),
        HumanMessage("foo"),
        SystemMessage("foo"),
        ToolMessage("foo", tool_call_id="foo"),
        AIMessageChunk(content="foo"),
        HumanMessageChunk(content="foo"),
        SystemMessageChunk(content="foo"),
        ToolMessageChunk(content="foo", tool_call_id="foo"),
    ]
    assert ChatPromptValueConcrete(messages=messages).messages == messages


def test_prompt_template_info_does_not_change_the_prompt_value() -> None:
    """Template info is tracing-only: equality and serialization ignore it."""
    plain = ChatPromptValue(messages=[HumanMessage(content="Weather in Oslo?")])
    traced = ChatPromptTemplate.from_messages([("human", "Weather in {city}?")]).invoke(
        {"city": "Oslo"}
    )

    assert _get_prompt_template_info(traced) is not None
    assert traced == plain
    assert dumpd(traced) == dumpd(plain)
    assert traced.model_dump() == plain.model_dump()
    assert traced.model_json_schema() == plain.model_json_schema()


def test_prompt_template_info_dropped_on_pickle() -> None:
    traced = StringPromptValue(text="Weather in Oslo?")
    _set_prompt_template_info(traced, {"template": "Weather in {city}?"})

    restored = pickle.loads(pickle.dumps(traced))

    assert restored == traced
    assert _get_prompt_template_info(restored) is None
    assert _get_prompt_template_info(traced) == {"template": "Weather in {city}?"}
