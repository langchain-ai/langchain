from langchain.llms.fake import FakeListLLM
from langchain.schema import SystemMessage
from langchain.wrappers.chat_model_facade import ChatModelFacade


def test_chat_model_facade():
    llm = FakeListLLM(responses=["hello", "goodbye"])
    chat_model = ChatModelFacade.of(llm)
    input_message = SystemMessage(content="hello")
    output_message = chat_model([input_message])
    assert output_message.content == "hello"
    assert output_message.type == "ai"
