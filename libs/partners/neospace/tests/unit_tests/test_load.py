from langchain_core.load.dump import dumpd, dumps
from langchain_core.load.load import load, loads

from langchain_neospace import ChatNeoSpace, NeoSpace


def test_loads_neospace_llm() -> None:
    llm = NeoSpace(model="davinci", temperature=0.5, neospace_api_key="hello", top_p=0.8)  # type: ignore[call-arg]
    llm_string = dumps(llm)
    llm2 = loads(llm_string, secrets_map={"NEOSPACE_API_KEY": "hello"})

    assert llm2 == llm
    llm_string_2 = dumps(llm2)
    assert llm_string_2 == llm_string
    assert isinstance(llm2, NeoSpace)


def test_load_neospace_llm() -> None:
    llm = NeoSpace(model="davinci", temperature=0.5, neospace_api_key="hello")  # type: ignore[call-arg]
    llm_obj = dumpd(llm)
    llm2 = load(llm_obj, secrets_map={"NEOSPACE_API_KEY": "hello"})

    assert llm2 == llm
    assert dumpd(llm2) == llm_obj
    assert isinstance(llm2, NeoSpace)


def test_loads_neospace_chat() -> None:
    llm = ChatNeoSpace(model="neo-3.5-turbo", temperature=0.5, neospace_api_key="hello")  # type: ignore[call-arg]
    llm_string = dumps(llm)
    llm2 = loads(llm_string, secrets_map={"NEOSPACE_API_KEY": "hello"})

    assert llm2 == llm
    llm_string_2 = dumps(llm2)
    assert llm_string_2 == llm_string
    assert isinstance(llm2, ChatNeoSpace)


def test_load_neospace_chat() -> None:
    llm = ChatNeoSpace(model="neo-3.5-turbo", temperature=0.5, neospace_api_key="hello")  # type: ignore[call-arg]
    llm_obj = dumpd(llm)
    llm2 = load(llm_obj, secrets_map={"NEOSPACE_API_KEY": "hello"})

    assert llm2 == llm
    assert dumpd(llm2) == llm_obj
    assert isinstance(llm2, ChatNeoSpace)
