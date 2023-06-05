"""Test for Serializable base class"""

from langchain.llms.openai import OpenAI

from langchain.chains.llm import LLMChain
from langchain.load.dump import dumps
from langchain.load.load import loads
from langchain.prompts.prompt import PromptTemplate


def test_load_openai_llm():
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello")
    llm_string = dumps(llm)
    llm2 = loads(llm_string, secrets_map={"OPENAI_API_KEY": "hello"})

    assert llm2 == llm
    assert dumps(llm2) == llm_string
    assert isinstance(llm2, OpenAI)


def test_load_llmchain():
    llm = OpenAI(model="davinci", temperature=0.5, openai_api_key="hello")
    prompt = PromptTemplate.from_template("hello {name}!")
    chain = LLMChain(llm=llm, prompt=prompt)
    chain_string = dumps(chain)
    chain2 = loads(chain_string, secrets_map={"OPENAI_API_KEY": "hello"})

    assert chain2 == chain
    assert dumps(chain2) == chain_string
    assert isinstance(chain2, LLMChain)
    assert isinstance(chain2.llm, OpenAI)
    assert isinstance(chain2.prompt, PromptTemplate)
