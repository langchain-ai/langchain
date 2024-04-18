from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from langchain_community.chat_models.sparkllm import ChatSparkLLM

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


def test_functions_call_thoughts() -> None:
    chat = ChatSparkLLM()

    prompt_tmpl = "Use the given functions to answer following question: {input}"
    prompt_msgs = [
        HumanMessagePromptTemplate.from_template(prompt_tmpl),
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)

    chain = prompt | chat.bind(functions=_FUNCTIONS)

    message = HumanMessage(content="What's the weather like in Shanghai today?")
    response = chain.batch([{"input": message}])
    assert isinstance(response[0], AIMessage)
    assert "tool_calls" in response[0].additional_kwargs


def test_initialization() -> None:
    """Test chat model initialization."""
    for model in [
        ChatSparkLLM(timeout=30),
        ChatSparkLLM(request_timeout=30),
    ]:
        assert model.request_timeout == 30


def test_chat_spark_llm() -> None:
    chat = ChatSparkLLM()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_spark_llm_streaming() -> None:
    chat = ChatSparkLLM(streaming=True)
    for chunk in chat.stream("Hello!"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, str)


def test_chat_spark_llm_with_domain() -> None:
    chat = ChatSparkLLM(spark_llm_domain="generalv3")
    message = HumanMessage(content="Hello")
    response = chat([message])
    print(response)  # noqa: T201
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_spark_llm_with_temperature() -> None:
    chat = ChatSparkLLM(temperature=0.9, top_k=2)
    message = HumanMessage(content="Hello")
    response = chat([message])
    print(response)  # noqa: T201
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
