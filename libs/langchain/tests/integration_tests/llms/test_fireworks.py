"""Test Fireworks AI API Wrapper."""
from pathlib import Path

import pytest

from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAIChat
from langchain.llms.fireworks import Fireworks, FireworksChat
from langchain.llms.loading import load_llm
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import LLMResult
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake


def test_fireworks_call() -> None:
    """Test valid call to fireworks."""
    llm = Fireworks(
        model_id="accounts/fireworks/models/fireworks-llama-v2-13b-chat", max_tokens=900
    )
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


def test_fireworkschat_chain() -> None:
    embeddings = OpenAIEmbeddings()

    loader = TextLoader(
        "[workspace]/langchain-internal/docs/extras/modules/state_of_the_union.txt"
    )
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
    "accounts/fireworks/models/fireworks-llama-v2-13b": 17,
    "accounts/fireworks/models/fireworks-llama-v2-7b": 17,
    "accounts/fireworks/models/fireworks-llama-v2-13b-chat": 17,
    "accounts/fireworks/models/fireworks-llama-v2-7b-chat": 17,
}

_MODELS = models = [
    "accounts/fireworks/models/fireworks-llama-v2-13b",
    "accounts/fireworks/models/fireworks-llama-v2-7b",
    "accounts/fireworks/models/fireworks-llama-v2-13b-chat",
    "accounts/fireworks/models/fireworks-llama-v2-7b-chat",
]


@pytest.mark.parametrize("model", _MODELS)
def test_fireworks_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = Fireworks(model=model)
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]
