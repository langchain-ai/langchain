"""Test for Serializable base class"""

from typing import Any, Dict

import pytest

from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.load.dump import dumps
from langchain.load.serializable import Serializable
from langchain.prompts.prompt import PromptTemplate


def openai_installed() -> bool:
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


class Person(Serializable):
    secret: str

    you_can_see_me: str = "hello"

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


@pytest.mark.skipif(not openai_installed, reason="openai not installed")
def test_serialize_openai_llm(snapshot: Any) -> None:
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello")
    assert dumps(llm, pretty=True) == snapshot


@pytest.mark.skipif(not openai_installed, reason="openai not installed")
def test_serialize_llmchain(snapshot: Any) -> None:
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello")
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    assert dumps(chain, pretty=True) == snapshot


@pytest.mark.skipif(not openai_installed, reason="openai not installed")
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
