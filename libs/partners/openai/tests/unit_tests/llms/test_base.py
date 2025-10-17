import os

import pytest
from langchain_core.outputs import GenerationChunk

from langchain_openai import OpenAI
from langchain_openai.llms.base import _stream_response_to_generation_chunk

os.environ["OPENAI_API_KEY"] = "foo"


def test_openai_model_param() -> None:
    llm = OpenAI(model="foo")
    assert llm.model_name == "foo"
    llm = OpenAI(model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"

    # Test standard tracing params
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "openai",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_temperature": 0.7,
        "ls_max_tokens": 256,
    }

    ls_params = llm._get_ls_params(model="bar")
    assert ls_params["ls_model_name"] == "bar"


def test_openai_model_kwargs() -> None:
    llm = OpenAI(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


def test_openai_fields_in_model_kwargs() -> None:
    """Test that for backwards compatibility fields can be passed in as model_kwargs."""
    llm = OpenAI(model_kwargs={"model_name": "foo"})
    assert llm.model_name == "foo"
    llm = OpenAI(model_kwargs={"model": "foo"})
    assert llm.model_name == "foo"


def test_openai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OpenAI(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "cmpl-3evkmQda5Hu7fcZavknQda3SQ",
        "object": "text_completion",
        "created": 1689989000,
        "model": "text-davinci-003",
        "choices": [
            {"text": "Bar Baz", "index": 0, "logprobs": None, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


@pytest.mark.parametrize("model", ["gpt-3.5-turbo-instruct"])
def test_get_token_ids(model: str) -> None:
    OpenAI(model=model).get_token_ids("foo")


def test_custom_token_counting() -> None:
    def token_encoder(text: str) -> list[int]:
        return [1, 2, 3]

    llm = OpenAI(custom_get_token_ids=token_encoder)
    assert llm.get_token_ids("foo") == [1, 2, 3]


def test_stream_response_to_generation_chunk() -> None:
    completion = {
        "id": "cmpl-abc123",
        "choices": [
            {"finish_reason": None, "index": 0, "logprobs": None, "text": "foo"}
        ],
        "created": 1749214401,
        "model": "my-model",
        "object": "text_completion",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = _stream_response_to_generation_chunk(completion)
    assert chunk == GenerationChunk(
        text="foo", generation_info={"finish_reason": None, "logprobs": None}
    )

    # Pathological completion with None text (e.g., from other providers)
    completion = {
        "id": "cmpl-abc123",
        "choices": [
            {"finish_reason": None, "index": 0, "logprobs": None, "text": None}
        ],
        "created": 1749214401,
        "model": "my-model",
        "object": "text_completion",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = _stream_response_to_generation_chunk(completion)
    assert chunk == GenerationChunk(
        text="", generation_info={"finish_reason": None, "logprobs": None}
    )


def test_generate_streaming_multiple_prompts_error() -> None:
    """Ensures ValueError when streaming=True and multiple prompts."""
    llm = OpenAI(streaming=True)

    with pytest.raises(
        ValueError, match="Cannot stream results with multiple prompts\\."
    ):
        llm._generate(["foo", "bar"])
