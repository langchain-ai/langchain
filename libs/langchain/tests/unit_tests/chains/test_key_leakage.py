import pytest
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.chat_models.openai import ChatOpenAI


@pytest.mark.requires("openai")
def test_openai_key_leakage():
    key = "sk-TESTKEY"
    input = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template("Foobar")])
    chat = ChatOpenAI(openai_api_key=key)
    chain = input | chat
    assert key not in str(input)
    assert key not in repr(chat)