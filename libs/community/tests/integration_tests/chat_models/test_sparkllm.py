from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from langchain_community.chat_models.sparkllm import ChatSparkLLM


def test_initialization() -> None:
    """Test chat model initialization."""

    for model in [
        ChatSparkLLM(
            api_key="secret",
            temperature=0.5,
            timeout=30,
        ),
        ChatSparkLLM(
            spark_api_key="secret",
            request_timeout=30,
        ),  # type: ignore[call-arg]
    ]:
        assert model.request_timeout == 30
        assert model.spark_api_key == "secret"
        assert model.temperature == 0.5


def test_chat_spark_llm() -> None:
    chat = ChatSparkLLM()  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_spark_llm_streaming() -> None:
    chat = ChatSparkLLM(streaming=True)  # type: ignore[call-arg]
    for chunk in chat.stream("Hello!"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, str)


def test_chat_spark_llm_with_domain() -> None:
    chat = ChatSparkLLM(spark_llm_domain="generalv3")  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    print(response)  # noqa: T201
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_spark_llm_with_temperature() -> None:
    chat = ChatSparkLLM(temperature=0.9, top_k=2)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    print(response)  # noqa: T201
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_spark_llm_streaming_with_stream_method() -> None:
    chat = ChatSparkLLM()  # type: ignore[call-arg]
    for chunk in chat.stream("Hello!"):
        assert isinstance(chunk, AIMessageChunk)
        assert isinstance(chunk.content, str)
