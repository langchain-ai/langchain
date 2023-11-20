from langchain.chat_models.sparkllm import ChatSparkLLM
from langchain.schema.messages import AIMessage, AIMessageChunk, HumanMessage


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
    print(response)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_spark_llm_with_temperature() -> None:
    chat = ChatSparkLLM(temperature=0.9, top_k=2)
    message = HumanMessage(content="Hello")
    response = chat([message])
    print(response)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_extra_kwargs() -> None:
    chat = ChatSparkLLM(
        spark_api_url="ws://test_url",
        spark_llm_domain="general",
        temperature=0.88,
        top_k=2,
        spark_user_id="test-lc-user",
        streaming=True,
        request_timeout=10,
    )
    assert chat.client.api_url == "ws://test_url"
    assert chat.client.spark_domain == "general"
    assert chat.temperature == 0.88
    assert chat.top_k == 2
    assert chat.spark_user_id == "test-lc-user"
    assert chat.streaming is True
    assert chat.request_timeout == 10
