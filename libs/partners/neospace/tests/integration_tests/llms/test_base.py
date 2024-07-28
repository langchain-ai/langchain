"""Test NeoSpace llm."""

from typing import Generator

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.outputs import LLMResult

from langchain_neospace import NeoSpace
from tests.unit_tests.fake.callbacks import FakeCallbackHandler


def test_stream() -> None:
    """Test streaming tokens from NeoSpace."""
    llm = NeoSpace()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_astream() -> None:
    """Test streaming tokens from NeoSpace."""
    llm = NeoSpace()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_abatch() -> None:
    """Test streaming tokens from NeoSpace."""
    llm = NeoSpace()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from NeoSpace."""
    llm = NeoSpace()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


def test_batch() -> None:
    """Test batch tokens from NeoSpace."""
    llm = NeoSpace()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from NeoSpace."""
    llm = NeoSpace()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)


def test_invoke() -> None:
    """Test invoke tokens from NeoSpace."""
    llm = NeoSpace()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)


@pytest.mark.scheduled
def test_neospace_call() -> None:
    """Test valid call to neospace."""
    llm = NeoSpace()
    output = llm.invoke("Say something nice:")
    assert isinstance(output, str)


def test_neospace_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    llm = NeoSpace(max_tokens=10)
    llm_result = llm.generate(["Hello, how are you?"])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == llm.model_name


def test_neospace_stop_valid() -> None:
    """Test neospace stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = NeoSpace(stop="3", temperature=0)  # type: ignore[call-arg]
    first_output = first_llm.invoke(query)
    second_llm = NeoSpace(temperature=0)
    second_output = second_llm.invoke(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output


@pytest.mark.scheduled
def test_neospace_streaming() -> None:
    """Test streaming tokens from NeoSpace."""
    llm = NeoSpace(max_tokens=10)
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)


@pytest.mark.scheduled
async def test_neospace_astream() -> None:
    """Test streaming tokens from NeoSpace."""
    llm = NeoSpace(max_tokens=10)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


@pytest.mark.scheduled
async def test_neospace_abatch() -> None:
    """Test streaming tokens from NeoSpace."""
    llm = NeoSpace(max_tokens=10)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_neospace_abatch_tags() -> None:
    """Test streaming tokens from NeoSpace."""
    llm = NeoSpace(max_tokens=10)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


@pytest.mark.scheduled
def test_neospace_batch() -> None:
    """Test streaming tokens from NeoSpace."""
    llm = NeoSpace(max_tokens=10)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


@pytest.mark.scheduled
async def test_neospace_ainvoke() -> None:
    """Test streaming tokens from NeoSpace."""
    llm = NeoSpace(max_tokens=10)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result, str)


@pytest.mark.scheduled
def test_neospace_invoke() -> None:
    """Test streaming tokens from NeoSpace."""
    llm = NeoSpace(max_tokens=10)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result, str)


@pytest.mark.scheduled
def test_neospace_multiple_prompts() -> None:
    """Test completion with multiple prompts."""
    llm = NeoSpace(max_tokens=10)
    output = llm.generate(["I'm Pickle Rick", "I'm Pickle Rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2


def test_neospace_streaming_best_of_error() -> None:
    """Test validation for streaming fails if best_of is not 1."""
    with pytest.raises(ValueError):
        NeoSpace(best_of=2, streaming=True)


def test_neospace_streaming_n_error() -> None:
    """Test validation for streaming fails if n is not 1."""
    with pytest.raises(ValueError):
        NeoSpace(n=2, streaming=True)


def test_neospace_streaming_multiple_prompts_error() -> None:
    """Test validation for streaming fails if multiple prompts are given."""
    with pytest.raises(ValueError):
        NeoSpace(streaming=True).generate(["I'm Pickle Rick", "I'm Pickle Rick"])


@pytest.mark.scheduled
def test_neospace_streaming_call() -> None:
    """Test valid call to neospace."""
    llm = NeoSpace(max_tokens=10, streaming=True)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_neospace_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = NeoSpace(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    llm.invoke("Write me a sentence with 100 words.")

    # new client sometimes passes 2 tokens at once
    assert callback_handler.llm_streams >= 5


@pytest.mark.scheduled
async def test_neospace_async_generate() -> None:
    """Test async generation."""
    llm = NeoSpace(max_tokens=10)
    output = await llm.agenerate(["Hello, how are you?"])
    assert isinstance(output, LLMResult)


async def test_neospace_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = NeoSpace(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    result = await llm.agenerate(["Write me a sentence with 100 words."])

    # new client sometimes passes 2 tokens at once
    assert callback_handler.llm_streams >= 5
    assert isinstance(result, LLMResult)


def test_neospace_modelname_to_contextsize_valid() -> None:
    """Test model name to context size on a valid model."""
    assert NeoSpace().modelname_to_contextsize("davinci") == 2049


def test_neospace_modelname_to_contextsize_invalid() -> None:
    """Test model name to context size on an invalid model."""
    with pytest.raises(ValueError):
        NeoSpace().modelname_to_contextsize("foobar")


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "cmpl-3evkmQda5Hu7fcZavknQda3SQ",
        "object": "text_completion",
        "created": 1689989000,
        "model": "7b-math-rank16",
        "choices": [
            {"text": "Bar Baz", "index": 0, "logprobs": None, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
