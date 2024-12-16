"""Test Alibaba Tongyi Chat Model."""

from typing import Any, List, cast

from langchain_core.callbacks import CallbackManager
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, SecretStr
from pytest import CaptureFixture

from langchain_community.chat_models.tongyi import ChatTongyi
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

_FUNCTIONS: Any = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


def test_initialization() -> None:
    """Test chat model initialization."""
    for model in [
        ChatTongyi(model_name="qwen-turbo", api_key="xyz"),  # type: ignore[arg-type, call-arg]
        ChatTongyi(model="qwen-turbo", dashscope_api_key="xyz"),  # type: ignore[call-arg]
    ]:
        assert model.model_name == "qwen-turbo"
        assert cast(SecretStr, model.dashscope_api_key).get_secret_value() == "xyz"


def test_api_key_is_string() -> None:
    llm = ChatTongyi(dashscope_api_key="secret-api-key")  # type: ignore[call-arg]
    assert isinstance(llm.dashscope_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = ChatTongyi(dashscope_api_key="secret-api-key")  # type: ignore[call-arg]
    print(llm.dashscope_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_default_call() -> None:
    """Test default model call."""
    chat = ChatTongyi()  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_model() -> None:
    """Test model kwarg works."""
    chat = ChatTongyi(model="qwen-plus")  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_vision_model() -> None:
    """Test model kwarg works."""
    chat = ChatTongyi(model="qwen-vl-max")  # type: ignore[call-arg]
    response = chat.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "image": "https://python.langchain.com/v0.1/assets/images/run_details-806f6581cd382d4887a5bc3e8ac62569.png"
                    },
                    {"text": "Summarize the image"},
                ]
            )
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, list)


def test_functions_call_thoughts() -> None:
    chat = ChatTongyi(model="qwen-plus")  # type: ignore[call-arg]

    prompt_tmpl = "Use the given functions to answer following question: {input}"
    prompt_msgs = [
        HumanMessagePromptTemplate.from_template(prompt_tmpl),
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)  # type: ignore[arg-type, call-arg]

    chain = prompt | chat.bind(functions=_FUNCTIONS)

    message = HumanMessage(content="What's the weather like in Shanghai today?")
    response = chain.batch([{"input": message}])
    assert isinstance(response[0], AIMessage)
    assert "tool_calls" in response[0].additional_kwargs


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = ChatTongyi()  # type: ignore[call-arg]

    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_stream() -> None:
    """Test that stream works."""
    chat = ChatTongyi(streaming=True)  # type: ignore[call-arg]
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Who are you?"),
        ],
        stream=True,
        config={"callbacks": callback_manager},
    )
    assert callback_handler.llm_streams > 0
    assert isinstance(response.content, str)


def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = ChatTongyi()  # type: ignore[call-arg]
    message = HumanMessage(content="Hi, how are you.")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


class GenerateUsername(BaseModel):
    "Get a username based on someone's name and hair color."

    name: str
    hair_color: str


def test_tool_use() -> None:
    llm = ChatTongyi(model="qwen-turbo", temperature=0)  # type: ignore
    llm_with_tool = llm.bind_tools(tools=[GenerateUsername])
    msgs: List = [
        HumanMessage(content="Sally has green hair, what would her username be?")
    ]
    ai_msg = llm_with_tool.invoke(msgs)
    # assert ai_msg is None
    # ai_msg.content = " "

    assert isinstance(ai_msg, AIMessage)
    assert isinstance(ai_msg.tool_calls, list)
    assert len(ai_msg.tool_calls) == 1
    tool_call = ai_msg.tool_calls[0]
    assert "args" in tool_call

    tool_msg = ToolMessage(
        content="sally_green_hair",
        tool_call_id=ai_msg.tool_calls[0]["id"],  # type: ignore
        name=ai_msg.tool_calls[0]["name"],
    )
    msgs.extend([ai_msg, tool_msg])
    llm_with_tool.invoke(msgs)

    # Test streaming
    ai_messages = llm_with_tool.stream(msgs)
    first = True
    for message in ai_messages:
        if first:
            gathered = message
            first = False
        else:
            gathered = gathered + message  # type: ignore
    assert isinstance(gathered, AIMessageChunk)

    streaming_tool_msg = ToolMessage(
        content="sally_green_hair",
        name=tool_call["name"],
        tool_call_id=tool_call["id"] if tool_call["id"] else " ",
    )
    msgs.extend([gathered, streaming_tool_msg])
    llm_with_tool.invoke(msgs)


def test_manual_tool_call_msg() -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatTongyi(model="qwen-turbo", temperature=0)  # type: ignore
    llm_with_tool = llm.bind_tools(tools=[GenerateUsername])
    msgs: List = [
        HumanMessage(content="Sally has green hair, what would her username be?"),
        AIMessage(
            content=" ",
            tool_calls=[
                ToolCall(
                    name="GenerateUsername",
                    args={"name": "Sally", "hair_color": "green"},
                    id="foo",
                )
            ],
        ),
        ToolMessage(content="sally_green_hair", tool_call_id="foo"),
    ]
    output: AIMessage = cast(AIMessage, llm_with_tool.invoke(msgs))
    assert output.content
    # Should not have called the tool again.
    assert not output.tool_calls and not output.invalid_tool_calls


class AnswerWithJustification(BaseModel):
    """An answer to the user question along with justification for the answer."""

    answer: str
    justification: str


def test_chat_tongyi_with_structured_output() -> None:
    """Test ChatTongyi with structured output."""
    llm = ChatTongyi()  # type: ignore
    structured_llm = llm.with_structured_output(AnswerWithJustification)
    response = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(response, AnswerWithJustification)


def test_chat_tongyi_with_structured_output_include_raw() -> None:
    """Test ChatTongyi with structured output."""
    llm = ChatTongyi()  # type: ignore
    structured_llm = llm.with_structured_output(
        AnswerWithJustification, include_raw=True
    )
    response = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(response, dict)
    assert isinstance(response.get("raw"), AIMessage)
    assert isinstance(response.get("parsed"), AnswerWithJustification)
