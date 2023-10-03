"""Test for Serializable base class"""

from typing import Any, Dict

import pytest

from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.chains.llm import LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.load.dump import dumps
from langchain.load.serializable import Serializable
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate


class Person(Serializable):
    secret: str

    you_can_see_me: str = "hello"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"secret": "SECRET"}

    @property
    def lc_attributes(self) -> Dict[str, str]:
        return {"you_can_see_me": self.you_can_see_me}


class SpecialPerson(Person):
    another_secret: str

    another_visible: str = "bye"

    # Gets merged with parent class's secrets
    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"another_secret": "ANOTHER_SECRET"}

    # Gets merged with parent class's attributes
    @property
    def lc_attributes(self) -> Dict[str, str]:
        return {"another_visible": self.another_visible}


class NotSerializable:
    pass


def test_person(snapshot: Any) -> None:
    p = Person(secret="hello")
    assert dumps(p, pretty=True) == snapshot
    sp = SpecialPerson(another_secret="Wooo", secret="Hmm")
    assert dumps(sp, pretty=True) == snapshot
    assert Person.lc_id() == ["test_dump", "Person"]


@pytest.mark.requires("openai")
def test_serialize_openai_llm(snapshot: Any) -> None:
    llm = OpenAI(
        model="davinci",
        temperature=0.5,
        openai_api_key="hello",
        # This is excluded from serialization
        callbacks=[LangChainTracer()],
    )
    llm.temperature = 0.7  # this is reflected in serialization
    assert dumps(llm, pretty=True) == snapshot


@pytest.mark.requires("openai")
def test_serialize_llmchain(snapshot: Any) -> None:
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello")
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    assert dumps(chain, pretty=True) == snapshot


@pytest.mark.requires("openai")
def test_serialize_llmchain_env() -> None:
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello")
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)

    import os

    has_env = "OPENAI_API_KEY" in os.environ
    if not has_env:
        os.environ["OPENAI_API_KEY"] = "env_variable"

    llm_2 = OpenAI(model="davinci", temperature=0.5)
    prompt_2 = PromptTemplate.from_template("hello {name}!")
    chain_2 = LLMChain(llm=llm_2, prompt=prompt_2)

    assert dumps(chain_2, pretty=True) == dumps(chain, pretty=True)

    if not has_env:
        del os.environ["OPENAI_API_KEY"]


@pytest.mark.requires("openai")
def test_serialize_llmchain_chat(snapshot: Any) -> None:
    llm = ChatOpenAI(model="davinci", temperature=0.5, openai_api_key="hello")
    prompt = ChatPromptTemplate.from_messages(
        [HumanMessagePromptTemplate.from_template("hello {name}!")]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    assert dumps(chain, pretty=True) == snapshot

    import os

    has_env = "OPENAI_API_KEY" in os.environ
    if not has_env:
        os.environ["OPENAI_API_KEY"] = "env_variable"

    llm_2 = ChatOpenAI(model="davinci", temperature=0.5)
    prompt_2 = ChatPromptTemplate.from_messages(
        [HumanMessagePromptTemplate.from_template("hello {name}!")]
    )
    chain_2 = LLMChain(llm=llm_2, prompt=prompt_2)

    assert dumps(chain_2, pretty=True) == dumps(chain, pretty=True)

    if not has_env:
        del os.environ["OPENAI_API_KEY"]


@pytest.mark.requires("openai")
def test_serialize_llmchain_with_non_serializable_arg(snapshot: Any) -> None:
    llm = OpenAI(
        model="davinci",
        temperature=0.5,
        openai_api_key="hello",
        client=NotSerializable,
    )
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    assert dumps(chain, pretty=True) == snapshot
