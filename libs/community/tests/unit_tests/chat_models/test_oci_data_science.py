"""Test OCI Data Science Model Deployment Endpoint."""

import sys
from unittest import mock
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from requests.exceptions import HTTPError
from langchain_community.chat_models import ChatOCIModelDeploymentVLLM, ChatOCIModelDeploymentTGI


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires Python 3.9 or higher"
)


CONST_MODEL_NAME = "odsc-vllm"
CONST_ENDPOINT = "https://oci.endpoint/ocid/predict"
CONST_PROMPT = "This is a prompt."
CONST_COMPLETION = "This is a completion."
CONST_COMPLETION_RESPONSE = {
    "id": "chat-123456789",
    "object": "chat.completion",
    "created": 123456789,
    "model": "mistral",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": CONST_COMPLETION,
                "tool_calls": [],
            },
            "logprobs": None,
            "finish_reason": "length",
            "stop_reason": None,
        }
    ],
    "usage": {"prompt_tokens": 115, "total_tokens": 371, "completion_tokens": 256},
    "prompt_logprobs": None,
}
CONST_COMPLETION_RESPONSE_TGI = {"generated_text": CONST_COMPLETION}
CONST_STREAM_TEMPLATE = (
    'data: {"id":"chat-123456","object":"chat.completion.chunk","created":123456789,'
    '"model":"odsc-llm","choices":[{"index":0,"delta":<DELTA>,"finish_reason":null}]}'
)
CONST_STREAM_DELTAS = ['{"role":"assistant","content":""}'] + [
    '{"content":" ' + word + '"}' for word in CONST_COMPLETION.split(" ")
]
CONST_STREAM_RESPONSE = (
    content
    for content in [
        CONST_STREAM_TEMPLATE.replace("<DELTA>", delta).encode()
        for delta in CONST_STREAM_DELTAS
    ]
    + [b"data: [DONE]"]
)

CONST_ASYNC_STREAM_TEMPLATE = (
    '{"id":"chat-123456","object":"chat.completion.chunk","created":123456789,'
    '"model":"odsc-llm","choices":[{"index":0,"delta":<DELTA>,"finish_reason":null}]}'
)
CONST_ASYNC_STREAM_RESPONSE = (
    CONST_ASYNC_STREAM_TEMPLATE.replace("<DELTA>", delta).encode()
    for delta in CONST_STREAM_DELTAS
)


def mocked_requests_post(self, **kwargs):
    """Method to mock post requests"""

    class MockResponse:
        """Represents a mocked response."""

        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code

        def raise_for_status(self):
            """Mocked raise for status."""
            if 400 <= self.status_code < 600:
                raise HTTPError("", response=self)

        def json(self):
            """Returns mocked json data."""
            return self.json_data

        def iter_lines(self, chunk_size=4096):
            """Returns a generator of mocked streaming response."""
            return CONST_STREAM_RESPONSE

        @property
        def text(self):
            return ""

    payload = kwargs.get("json")
    messages = payload.get("messages")
    prompt = messages[0].get("content")

    if prompt == CONST_PROMPT:
        return MockResponse(json_data=CONST_COMPLETION_RESPONSE)

    return MockResponse(
        json_data={},
        status_code=404,
    )


@pytest.mark.requires("ads")
@mock.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
@mock.patch("requests.post", side_effect=mocked_requests_post)
def test_invoke_vllm(mock_post, mock_auth) -> None:
    """Tests invoking vLLM endpoint."""
    llm = ChatOCIModelDeploymentVLLM(endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME)
    output = llm.invoke(CONST_PROMPT)
    assert isinstance(output, AIMessage)
    assert output.content == CONST_COMPLETION


@pytest.mark.requires("ads")
@mock.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
@mock.patch("requests.post", side_effect=mocked_requests_post)
def test_invoke_tgi(mock_post, mock_auth) -> None:
    """Tests invoking TGI endpoint using OpenAI Spec."""
    llm = ChatOCIModelDeploymentTGI(endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME)
    output = llm.invoke(CONST_PROMPT)
    assert isinstance(output, AIMessage)
    assert output.content == CONST_COMPLETION


@pytest.mark.requires("ads")
@mock.patch("ads.common.auth.default_signer", return_value=dict(signer=None))
@mock.patch("requests.post", side_effect=mocked_requests_post)
def test_stream_vllm(mock_post, mock_auth) -> None:
    """Tests streaming with vLLM endpoint using OpenAI spec."""
    llm = ChatOCIModelDeploymentVLLM(
        endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME, streaming=True
    )
    output = AIMessageChunk("")
    count = 0
    for chunk in llm.stream(CONST_PROMPT):
        assert isinstance(chunk, AIMessageChunk)
        output += chunk
        count += 1
    assert count == 5
    assert output.content.strip() == CONST_COMPLETION


async def mocked_async_streaming_response(*args, **kwargs):
    """Returns mocked response for async streaming."""
    for item in CONST_ASYNC_STREAM_RESPONSE:
        yield item


@pytest.mark.asyncio
@pytest.mark.requires("ads")
@mock.patch(
    "ads.common.auth.default_signer", return_value=dict(signer=mock.MagicMock())
)
@mock.patch(
    "langchain_community.utilities.requests.Requests.apost",
    mock.MagicMock(),
)
async def test_stream_async(mock_auth):
    """Tests async streaming."""
    llm = ChatOCIModelDeploymentVLLM(
        endpoint=CONST_ENDPOINT, model=CONST_MODEL_NAME, streaming=True
    )
    with mock.patch.object(
        llm,
        "_aiter_sse",
        mock.MagicMock(return_value=mocked_async_streaming_response()),
    ):

        chunks = [chunk.content async for chunk in llm.astream(CONST_PROMPT)]
    assert "".join(chunks).strip() == CONST_COMPLETION
