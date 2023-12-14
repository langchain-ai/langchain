from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from langchain_community.chat_models.sparkllm import ChatSparkLLM


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
