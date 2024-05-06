from langchain_core.load.dump import dumpd, dumps
from langchain_core.load.load import load, loads

from langchain_openai import ChatOpenAI, OpenAI


def test_loads_openai_llm() -> None:
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello")
    llm_string = dumps(llm)
    llm2 = loads(llm_string, secrets_map={"OPENAI_API_KEY": "hello"})

    assert llm2 == llm
    llm_string_2 = dumps(llm2)
    assert llm_string_2 == llm_string
    assert isinstance(llm2, OpenAI)


def test_load_openai_llm() -> None:
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello")
    llm_obj = dumpd(llm)
    llm2 = load(llm_obj, secrets_map={"OPENAI_API_KEY": "hello"})

    assert llm2 == llm
    assert dumpd(llm2) == llm_obj
    assert isinstance(llm2, OpenAI)


def test_loads_openai_chat() -> None:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, openai_api_key="hello")
    llm_string = dumps(llm)
    llm2 = loads(llm_string, secrets_map={"OPENAI_API_KEY": "hello"})

    assert llm2 == llm
    llm_string_2 = dumps(llm2)
    assert llm_string_2 == llm_string
    assert isinstance(llm2, ChatOpenAI)


def test_load_openai_chat() -> None:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, openai_api_key="hello")
    llm_obj = dumpd(llm)
    llm2 = load(llm_obj, secrets_map={"OPENAI_API_KEY": "hello"})

    assert llm2 == llm
    assert dumpd(llm2) == llm_obj
    assert isinstance(llm2, ChatOpenAI)
