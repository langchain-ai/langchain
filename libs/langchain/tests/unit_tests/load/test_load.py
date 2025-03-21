"""Test for Serializable base class"""

import pytest
from langchain_core.load.dump import dumpd, dumps
from langchain_core.load.load import load, loads
from langchain_core.prompts.prompt import PromptTemplate

from langchain.chains.llm import LLMChain

pytest.importorskip("langchain_openai", reason="langchain_openai not installed")
pytest.importorskip("langchain_community", reason="langchain_community not installed")

from langchain_community.llms.openai import (  # noqa: E402 # ignore: community-import
    OpenAI as CommunityOpenAI,
)


class NotSerializable:
    pass


@pytest.mark.requires("openai", "langchain_openai")
def test_loads_openai_llm() -> None:
    from langchain_openai import OpenAI

    llm = CommunityOpenAI(
        model="davinci", temperature=0.5, openai_api_key="hello", top_p=0.8
    )  # type: ignore[call-arg]
    llm_string = dumps(llm)
    llm2 = loads(llm_string, secrets_map={"OPENAI_API_KEY": "hello"})

    assert llm2 == llm
    llm_string_2 = dumps(llm2)
    assert llm_string_2 == llm_string
    assert isinstance(llm2, OpenAI)


@pytest.mark.requires("openai", "langchain_openai")
def test_loads_llmchain() -> None:
    from langchain_openai import OpenAI

    llm = CommunityOpenAI(
        model="davinci", temperature=0.5, openai_api_key="hello", top_p=0.8
    )  # type: ignore[call-arg]
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    chain_string = dumps(chain)
    chain2 = loads(chain_string, secrets_map={"OPENAI_API_KEY": "hello"})

    assert chain2 == chain
    assert dumps(chain2) == chain_string
    assert isinstance(chain2, LLMChain)
    assert isinstance(chain2.llm, OpenAI)
    assert isinstance(chain2.prompt, PromptTemplate)


@pytest.mark.requires("openai", "langchain_openai")
def test_loads_llmchain_env() -> None:
    import os

    from langchain_openai import OpenAI

    has_env = "OPENAI_API_KEY" in os.environ
    if not has_env:
        os.environ["OPENAI_API_KEY"] = "env_variable"

    llm = OpenAI(model="davinci", temperature=0.5, top_p=0.8)  # type: ignore[call-arg]
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    chain_string = dumps(chain)
    chain2 = loads(chain_string)

    assert chain2 == chain
    assert dumps(chain2) == chain_string
    assert isinstance(chain2, LLMChain)
    assert isinstance(chain2.llm, OpenAI)
    assert isinstance(chain2.prompt, PromptTemplate)

    if not has_env:
        del os.environ["OPENAI_API_KEY"]


@pytest.mark.requires("openai")
def test_loads_llmchain_with_non_serializable_arg() -> None:
    llm = CommunityOpenAI(  # type: ignore[call-arg]
        model="davinci",
        temperature=0.5,
        openai_api_key="hello",
        model_kwargs={"a": NotSerializable},
    )
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    chain_string = dumps(chain, pretty=True)
    with pytest.raises(NotImplementedError):
        loads(chain_string, secrets_map={"OPENAI_API_KEY": "hello"})


@pytest.mark.requires("openai", "langchain_openai")
def test_load_openai_llm() -> None:
    from langchain_openai import OpenAI

    llm = CommunityOpenAI(model="davinci", temperature=0.5, openai_api_key="hello")  # type: ignore[call-arg]
    llm_obj = dumpd(llm)
    llm2 = load(llm_obj, secrets_map={"OPENAI_API_KEY": "hello"})

    assert llm2 == llm
    assert dumpd(llm2) == llm_obj
    assert isinstance(llm2, OpenAI)


@pytest.mark.requires("openai", "langchain_openai")
def test_load_llmchain() -> None:
    from langchain_openai import OpenAI

    llm = CommunityOpenAI(model="davinci", temperature=0.5, openai_api_key="hello")  # type: ignore[call-arg]
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    chain_obj = dumpd(chain)
    chain2 = load(chain_obj, secrets_map={"OPENAI_API_KEY": "hello"})

    assert chain2 == chain
    assert dumpd(chain2) == chain_obj
    assert isinstance(chain2, LLMChain)
    assert isinstance(chain2.llm, OpenAI)
    assert isinstance(chain2.prompt, PromptTemplate)


@pytest.mark.requires("openai", "langchain_openai")
def test_load_llmchain_env() -> None:
    import os

    from langchain_openai import OpenAI

    has_env = "OPENAI_API_KEY" in os.environ
    if not has_env:
        os.environ["OPENAI_API_KEY"] = "env_variable"

    llm = CommunityOpenAI(model="davinci", temperature=0.5)  # type: ignore[call-arg]
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    chain_obj = dumpd(chain)
    chain2 = load(chain_obj)

    assert chain2 == chain
    assert dumpd(chain2) == chain_obj
    assert isinstance(chain2, LLMChain)
    assert isinstance(chain2.llm, OpenAI)
    assert isinstance(chain2.prompt, PromptTemplate)

    if not has_env:
        del os.environ["OPENAI_API_KEY"]


@pytest.mark.requires("openai", "langchain_openai")
def test_load_llmchain_with_non_serializable_arg() -> None:
    import httpx
    from langchain_openai import OpenAI

    llm = OpenAI(  # type: ignore[call-arg]
        model="davinci",
        temperature=0.5,
        openai_api_key="hello",
        http_client=httpx.Client(),
    )
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    chain_obj = dumpd(chain)
    with pytest.raises(NotImplementedError):
        load(chain_obj, secrets_map={"OPENAI_API_KEY": "hello"})
