"""Test Fireworks AI API Wrapper."""
from pathlib import Path
from typing import Generator
import os
import getpass

import pytest
import asyncio

from langchain.callbacks.manager import CallbackManager
from langchain.llms.loading import load_llm
from langchain.llms.openai import OpenAI, OpenAIChat
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.schema import LLMResult

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma 

from langchain.chains import RetrievalQA
from langchain.llms import OpenAIChat
from langchain.llms.fireworks import Fireworks, FireworksChat

def test_fireworks_call() -> None:
    """Test valid call to fireworks."""
    llm = Fireworks(model_id="fireworks-llama-v2-13b-chat", max_tokens=900)
    output = llm("What is the weather in NYC")
    assert isinstance(output, str)

def test_fireworks_in_chain() -> None:
    """Tests fireworks AI in a Langchain chain"""
    human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="What is a good name for a company that makes {product}?",
                input_variables=["product"],
            )
        )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chat = Fireworks()
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)
    output = chain.run("football helmets")
    assert isinstance(output, str)

@pytest.mark.asyncio
async def test_openai_chat_async_generate() -> None:
    """Test async chat."""
    llm = OpenAIChat(max_tokens=10)
    output = await llm.agenerate(["Hello, how are you?"])
    assert isinstance(output, LLMResult)

def test_fireworks_model_param() -> None:
    """Tests model parameters for Fireworks"""
    llm = Fireworks(model="foo")
    assert llm.model_id == "foo"
    llm = Fireworks(model_id="foo")
    assert llm.model_id == "foo"

def test_fireworkschat_model_param() -> None:
    """Tests model parameters for FireworksChat"""
    llm = FireworksChat(model="foo")
    assert llm.model_id == "foo"
    llm = FireworksChat(model_id="foo")
    assert llm.model_id == "foo"

def test_fireworks_extra_kwargs() -> None:
    """Test extra kwargs to Fireworks."""
    # Check that foo is saved in extra_kwargs.
    llm = Fireworks(foo=3, max_tokens=10)
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = Fireworks(foo=3, model_kwargs={"bar": 2})
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        Fireworks(foo=3, model_kwargs={"foo": 2})

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        Fireworks(model_kwargs={"temperature": 0.2})

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        Fireworks(model_kwargs={"model": "fireworks-llama-v2-13b-chat"})

def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an Fireworks LLM."""
    llm = Fireworks(max_tokens=10)
    llm.save(file_path=tmp_path / "fireworks.yaml")
    loaded_llm = load_llm(tmp_path / "fireworks.yaml")
    assert loaded_llm == llm

def test_fireworks_multiple_prompts() -> None:
    """Test completion with multiple prompts."""
    llm = Fireworks()
    output = llm.generate(["How is the weather in New York today?", "I'm pickle rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2

def test_fireworks_modelname_to_contextsize_invalid() -> None:
    """Test model name to context size on an invalid model."""
    with pytest.raises(ValueError):
        Fireworks().modelname_to_contextsize("foobar")

def test_fireworks_chat() -> None:
    """Test FireworksChat."""
    llm = FireworksChat()
    output = llm("Name me 3 quick facts about the New England Patriots")
    assert isinstance(output, str)

async def test_fireworks_agenerate() -> None:
    llm = Fireworks()
    output = await llm.agenerate(["I'm a pickle", "I'm a pickle"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2

async def test_fireworkschat_agenerate() -> None:
    llm = FireworksChat(max_tokens=10)
    output = await llm.agenerate(["Hello, how are you?"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 1

def test_fireworkschat_chain():
    embeddings = OpenAIEmbeddings()

    loader = TextLoader("[workspace]/langchain-internal/docs/extras/modules/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    db = DeepLake(
        dataset_path="./my_deeplake/", embedding_function=embeddings, overwrite=True
    )
    db.add_documents(docs)

    query = "What did the president say about Ketanji Brown Jackson"
    docs = db.similarity_search(query)

    qa = RetrievalQA.from_chain_type(
        llm=FireworksChat(),
        chain_type="stuff",
        retriever=db.as_retriever(),
    )
    query = "What did the president say about Ketanji Brown Jackson"
    output = qa.run(query)
    assert isinstance(output, str)

_EXPECTED_NUM_TOKENS = {
    "fireworks-llama-v2-13b": 17,
    "fireworks-llama-v2-7b": 17,
    "fireworks-llama-v2-13b-chat": 17,
    "fireworks-llama-v2-7b-chat": 17,
}

_MODELS = models = [
    "fireworks-llama-v2-13b",
    "fireworks-llama-v2-7b",
    "fireworks-llama-v2-13b-chat",
    "fireworks-llama-v2-7b-chat",
]

@pytest.mark.parametrize("model", _MODELS)
def test_fireworks_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = Fireworks(model=model)
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]