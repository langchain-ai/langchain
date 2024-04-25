from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from langchain_community.chat_models.sparkllm import ChatSparkLLM


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
    response = chat.invoke([message])
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
    response = chat.invoke([message])
    print(response)  # noqa: T201
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_spark_llm_with_temperature() -> None:
    chat = ChatSparkLLM(temperature=0.9, top_k=2)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    print(response)  # noqa: T201
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
