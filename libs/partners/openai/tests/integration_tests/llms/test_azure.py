"""Test AzureOpenAI wrapper."""
import os
from typing import Any, Generator

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.outputs import LLMResult

from langchain_openai import AzureOpenAI
from tests.unit_tests.fake.callbacks import FakeCallbackHandler

OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "")
OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE", "")
OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
DEPLOYMENT_NAME = os.environ.get(
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    os.environ.get("AZURE_OPENAI_LLM_DEPLOYMENT_NAME", ""),
)


def _get_llm(**kwargs: Any) -> AzureOpenAI:
    return AzureOpenAI(
        deployment_name=DEPLOYMENT_NAME,
        openai_api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY,
        **kwargs,
    )


@pytest.fixture
def llm() -> AzureOpenAI:
    return _get_llm(
        max_tokens=10,
    )


@pytest.mark.scheduled
def test_openai_call(llm: AzureOpenAI) -> None:
    """Test valid call to openai."""
    output = llm("Say something nice:")
    assert isinstance(output, str)


@pytest.mark.scheduled
def test_openai_streaming(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    full_response = ""
    for token in generator:
        assert isinstance(token, str)
        full_response += token
    assert full_response


@pytest.mark.scheduled
async def test_openai_astream(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


@pytest.mark.scheduled
async def test_openai_abatch(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_openai_abatch_tags(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


@pytest.mark.scheduled
def test_openai_batch(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


@pytest.mark.scheduled
async def test_openai_ainvoke(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)


@pytest.mark.scheduled
def test_openai_invoke(llm: AzureOpenAI) -> None:
    """Test streaming tokens from AzureOpenAI."""
    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)


@pytest.mark.scheduled
def test_openai_multiple_prompts(llm: AzureOpenAI) -> None:
    """Test completion with multiple prompts."""
    output = llm.generate(["I'm Pickle Rick", "I'm Pickle Rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2


def test_openai_streaming_best_of_error() -> None:
    """Test validation for streaming fails if best_of is not 1."""
    with pytest.raises(ValueError):
        _get_llm(best_of=2, streaming=True)


def test_openai_streaming_n_error() -> None:
    """Test validation for streaming fails if n is not 1."""
    with pytest.raises(ValueError):
        _get_llm(n=2, streaming=True)


def test_openai_streaming_multiple_prompts_error() -> None:
    """Test validation for streaming fails if multiple prompts are given."""
    with pytest.raises(ValueError):
        _get_llm(streaming=True).generate(["I'm Pickle Rick", "I'm Pickle Rick"])


@pytest.mark.scheduled
def test_openai_streaming_call() -> None:
    """Test valid call to openai."""
    llm = _get_llm(max_tokens=10, streaming=True)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_openai_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = _get_llm(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    llm("Write me a sentence with 100 words.")
    assert callback_handler.llm_streams == 11


@pytest.mark.scheduled
async def test_openai_async_generate() -> None:
    """Test async generation."""
    llm = _get_llm(max_tokens=10)
    output = await llm.agenerate(["Hello, how are you?"])
    assert isinstance(output, LLMResult)


async def test_openai_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = _get_llm(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    result = await llm.agenerate(["Write me a sentence with 100 words."])
    assert callback_handler.llm_streams == 11
    assert isinstance(result, LLMResult)
