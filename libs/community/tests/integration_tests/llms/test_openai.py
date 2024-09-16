"""Test OpenAI API wrapper."""

from pathlib import Path
from typing import Generator

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.outputs import LLMResult

from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.llms.loading import load_llm
from langchain_community.llms.openai import OpenAI
from tests.unit_tests.callbacks.fake_callback_handler import (
    FakeCallbackHandler,
)


@pytest.mark.scheduled
def test_openai_call() -> None:
    """Test valid call to openai."""
    llm = OpenAI()
    output = llm.invoke("Say something nice:")
    assert isinstance(output, str)


def test_openai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    llm = OpenAI(max_tokens=10)
    llm_result = llm.generate(["Hello, how are you?"])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == llm.model_name


def test_openai_stop_valid() -> None:
    """Test openai stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = OpenAI(stop="3", temperature=0)  # type: ignore[call-arg]
    first_output = first_llm.invoke(query)
    second_llm = OpenAI(temperature=0)
    second_output = second_llm.invoke(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output


def test_openai_stop_error() -> None:
    """Test openai stop logic on bad configuration."""
    llm = OpenAI(stop="3", temperature=0)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        llm.invoke("write an ordered list of five items", stop=["\n"])


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an OpenAI LLM."""
    llm = OpenAI(max_tokens=10)
    llm.save(file_path=tmp_path / "openai.yaml")
    loaded_llm = load_llm(tmp_path / "openai.yaml")
    assert loaded_llm == llm


@pytest.mark.scheduled
def test_openai_streaming() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)


@pytest.mark.scheduled
async def test_openai_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


@pytest.mark.scheduled
async def test_openai_abatch() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_openai_abatch_tags() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


@pytest.mark.scheduled
def test_openai_batch() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


@pytest.mark.scheduled
async def test_openai_ainvoke() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)


@pytest.mark.scheduled
def test_openai_invoke() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)


@pytest.mark.scheduled
def test_openai_multiple_prompts() -> None:
    """Test completion with multiple prompts."""
    llm = OpenAI(max_tokens=10)
    output = llm.generate(["I'm Pickle Rick", "I'm Pickle Rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2


def test_openai_streaming_best_of_error() -> None:
    """Test validation for streaming fails if best_of is not 1."""
    with pytest.raises(ValueError):
        OpenAI(best_of=2, streaming=True)


def test_openai_streaming_n_error() -> None:
    """Test validation for streaming fails if n is not 1."""
    with pytest.raises(ValueError):
        OpenAI(n=2, streaming=True)


def test_openai_streaming_multiple_prompts_error() -> None:
    """Test validation for streaming fails if multiple prompts are given."""
    with pytest.raises(ValueError):
        OpenAI(streaming=True).generate(["I'm Pickle Rick", "I'm Pickle Rick"])


@pytest.mark.scheduled
def test_openai_streaming_call() -> None:
    """Test valid call to openai."""
    llm = OpenAI(max_tokens=10, streaming=True)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_openai_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = OpenAI(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    llm.invoke("Write me a sentence with 100 words.")
    assert callback_handler.llm_streams == 10


@pytest.mark.scheduled
async def test_openai_async_generate() -> None:
    """Test async generation."""
    llm = OpenAI(max_tokens=10)
    output = await llm.agenerate(["Hello, how are you?"])
    assert isinstance(output, LLMResult)


async def test_openai_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = OpenAI(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    result = await llm.agenerate(["Write me a sentence with 100 words."])
    assert callback_handler.llm_streams == 10
    assert isinstance(result, LLMResult)


def test_openai_modelname_to_contextsize_valid() -> None:
    """Test model name to context size on a valid model."""
    assert OpenAI().modelname_to_contextsize("davinci") == 2049


def test_openai_modelname_to_contextsize_invalid() -> None:
    """Test model name to context size on an invalid model."""
    with pytest.raises(ValueError):
        OpenAI().modelname_to_contextsize("foobar")


_EXPECTED_NUM_TOKENS = {
    "ada": 17,
    "babbage": 17,
    "curie": 17,
    "davinci": 17,
    "gpt-4": 12,
    "gpt-4-32k": 12,
    "gpt-3.5-turbo": 12,
}

_MODELS = models = [
    "ada",
    "babbage",
    "curie",
    "davinci",
]
_CHAT_MODELS = [
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo",
]


@pytest.mark.parametrize("model", _MODELS)
def test_openai_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = OpenAI(model=model)
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]


@pytest.mark.parametrize("model", _CHAT_MODELS)
def test_chat_openai_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = ChatOpenAI(model=model)
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "cmpl-3evkmQda5Hu7fcZavknQda3SQ",
        "object": "text_completion",
        "created": 1689989000,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [
            {"text": "Bar Baz", "index": 0, "logprobs": None, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
