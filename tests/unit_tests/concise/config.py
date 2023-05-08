from langchain.base_language import BaseLanguageModel
from langchain.concise.config import (
    get_default_max_tokens,
    get_default_model,
    get_default_text_splitter,
    set_default_max_tokens,
    set_default_model,
    set_default_text_splitter,
)
from langchain.llms.fake import FakeListLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


def test_config():
    set_default_model(FakeListLLM([]))
    set_default_max_tokens(100)
    set_default_text_splitter(RecursiveCharacterTextSplitter())

    assert isinstance(get_default_max_tokens(), int)
    assert isinstance(get_default_model(), BaseLanguageModel)
    assert isinstance(get_default_text_splitter(), TextSplitter)
