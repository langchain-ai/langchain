# Copyright (c) 2023, 2024, Oracle and/or its affiliates.

"""Test LLM for OCI Data Science Model Deployment Endpoint."""

import sys
from typing import Any, AsyncGenerator, Dict, Generator
from unittest import mock

import pytest
from requests.exceptions import HTTPError

from langchain_community.llms.oci_data_science_model_deployment_endpoint import (
    OCIModelDeploymentTGI,
    OCIModelDeploymentVLLM,
)

CONST_MODEL_NAME = "odsc-vllm"
CONST_ENDPOINT = "https://oci.endpoint/ocid/predict"
CONST_PROMPT = "This is a prompt."
CONST_COMPLETION = "This is a completion."
CONST_COMPLETION_ROUTE = "/v1/completions"
CONST_COMPLETION_RESPONSE = {
    "choices": [
        {
            "index": 0,
            "text": CONST_COMPLETION,
            "logprobs": 0.1,
            "finish_reason": "length",
        }
    ],
}
CONST_COMPLETION_RESPONSE_TGI = {"generated_text": CONST_COMPLETION}
CONST_STREAM_TEMPLATE = (
    'data: {"id":"","object":"text_completion","created":123456,'
    + '"choices":[{"index":0,"text":"<TOKEN>","finish_reason":""}]}'
)
CONST_STREAM_RESPONSE = (
    CONST_STREAM_TEMPLATE.replace("<TOKEN>", " " + word).encode()
    for word in CONST_COMPLETION.split(" ")
)

CONST_ASYNC_STREAM_TEMPLATE = (
    '{"id":"","object":"text_completion","created":123456,'
    + '"choices":[{"index":0,"text":"<TOKEN>","finish_reason":""}]}'
)
CONST_ASYNC_STREAM_RESPONSE = (
    CONST_ASYNC_STREAM_TEMPLATE.replace("<TOKEN>", " " + word).encode()
    for word in CONST_COMPLETION.split(" ")
)

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires Python 3.9 or higher"
)


class MockResponse:
    """Represents a mocked response."""

    def __init__(self, json_data: Dict, status_code: int = 200) -> None:
        self.json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Mocked raise for status."""
        if 400 <= self.status_code < 600:
            raise HTTPError()

    def json(self) -> Dict:
        """Returns mocked json data."""
        return self.json_data

    def iter_lines(self, chunk_size: int = 4096) -> Generator[bytes, None, None]:
        """Returns a generator of mocked streaming response."""
        return CONST_STREAM_RESPONSE

    @property
    def text(self) -> str:
        """Returns the mocked text representation."""
        return ""


def mocked_requests_post(url: str, **kwargs: Any) -> MockResponse:
    """Method to mock post requests"""

    payload: dict = kwargs.get("json", {})
    if "inputs" in payload:
        prompt = payload.get("inputs")
        is_tgi = True
    else:
        prompt = payload.get("prompt")
        is_tgi = False

    if prompt == CONST_PROMPT:
        if is_tgi:
            return MockResponse(json_data=CONST_COMPLETION_RESPONSE_TGI)
        return MockResponse(json_data=CONST_COMPLETION_RESPONSE)

    return MockResponse(
        json_data={},
        status_code=404,
    )


async def mocked_async_streaming_response(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[bytes, None]:
    """Returns mocked response for async streaming."""
    for item in CONST_ASYNC_STREAM_RESPONSE:
        yield item


@pytest.mark.requires("ads")
@mock.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
@mock.patch("requests.post", side_effect=mocked_requests_post)
def test_invoke_vllm(*args: Any) -> None:
    """Tests invoking vLLM endpoint."""
    llm = OCIModelDeploymentVLLM(endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME)
    assert llm._headers().get("route") == CONST_COMPLETION_ROUTE
    output = llm.invoke(CONST_PROMPT)
    assert output == CONST_COMPLETION


@pytest.mark.requires("ads")
@mock.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
@mock.patch("requests.post", side_effect=mocked_requests_post)
def test_stream_tgi(*args: Any) -> None:
    """Tests streaming with TGI endpoint using OpenAI spec."""
    llm = OCIModelDeploymentTGI(
        endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME, streaming=True
    )
    assert llm._headers().get("route") == CONST_COMPLETION_ROUTE
    output = ""
    count = 0
    for chunk in llm.stream(CONST_PROMPT):
        output += chunk
        count += 1
    assert count == 4
    assert output.strip() == CONST_COMPLETION


@pytest.mark.requires("ads")
@mock.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
@mock.patch("requests.post", side_effect=mocked_requests_post)
def test_generate_tgi(*args: Any) -> None:
    """Tests invoking TGI endpoint using TGI generate spec."""
    llm = OCIModelDeploymentTGI(
        endpoint=CONST_ENDPOINT, api="/generate", model=CONST_MODEL_NAME
    )
    assert llm._headers().get("route") == CONST_COMPLETION_ROUTE
    output = llm.invoke(CONST_PROMPT)
    assert output == CONST_COMPLETION


@pytest.mark.asyncio
@pytest.mark.requires("ads")
@mock.patch(
    "ads.common.auth.default_signer", return_value=dict(signer=mock.MagicMock())
)
@mock.patch(
    "langchain_community.utilities.requests.Requests.apost",
    mock.MagicMock(),
)
async def test_stream_async(*args: Any) -> None:
    """Tests async streaming."""
    llm = OCIModelDeploymentTGI(
        endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME, streaming=True
    )
    assert llm._headers().get("route") == CONST_COMPLETION_ROUTE
    with mock.patch.object(
        llm,
        "_aiter_sse",
        mock.MagicMock(return_value=mocked_async_streaming_response()),
    ):
        chunks = [chunk async for chunk in llm.astream(CONST_PROMPT)]
    assert "".join(chunks).strip() == CONST_COMPLETION
