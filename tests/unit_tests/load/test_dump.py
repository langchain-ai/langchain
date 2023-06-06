"""Test for Serializable base class"""

from typing import Any

import openai

from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.load.dump import dumps
from langchain.prompts.prompt import PromptTemplate


def test_serialize_openai_llm(snapshot: Any) -> None:
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello")
    assert dumps(llm, pretty=True) == snapshot


def test_serialize_llmchain(snapshot: Any) -> None:
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello")
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    assert dumps(chain, pretty=True) == snapshot


def test_serialize_llmchain_with_non_serializable_arg(snapshot: Any) -> None:
    llm = OpenAI(
        model="davinci",
        temperature=0.5,
        openai_api_key="hello",
        client=openai.Completion,
    )
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    assert dumps(chain, pretty=True) == snapshot
