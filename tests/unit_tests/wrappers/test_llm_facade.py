from langchain.llms.fake import FakeListLLM
from langchain.schema import SystemMessage
from langchain.wrappers.chat_model_facade import ChatModelFacade
from langchain.wrappers.llm_facade import LLMFacade


def test_llm_facade():
    llm = FakeListLLM(responses=["hello", "goodbye"])
    chat_model = ChatModelFacade.of(llm)
    # we assume ChatModelFacade works as expected given `test_chat_model_facade.py`
    mock_llm = LLMFacade.of(chat_model)
    assert mock_llm("hi") == "hello"
