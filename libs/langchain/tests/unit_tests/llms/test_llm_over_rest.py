import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import requests

from langchain.llms import llm_over_rest
from langchain.llms.llm_over_rest import LLMOverREST


def mock_generated_json() -> Dict[str, Any]:
    return {
        "id": "mock-result-42",
        "object": "completion",
        "created": 1689989000,
        "model": "test-llm-model",
        "generated_text": "This is what we have generated",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def _patched_retry(*args: Any, **kwargs: Any) -> Any:
    """Patched retry for unit tests that does not wait."""
    from tenacity import (
        retry,
        retry_base,
        retry_if_exception_type,
        stop_after_attempt,
        wait_none,
    )

    assert "error_types" in kwargs
    error_types = kwargs["error_types"]
    retry_instance: retry_base = retry_if_exception_type(error_types[0])
    for error in error_types[1:]:
        retry_instance = retry_instance | retry_if_exception_type(error)
    return retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_none(),
        retry=retry_instance,
    )


def test_llmoverrest_call_requests() -> None:
    def mock_response(*args: Any, **kwargs: Any) -> requests.Response:
        tmp_resp = requests.Response()
        tmp_resp.status_code = 200
        tmp_resp._content = json.dumps(mock_generated_json()).encode("utf-8")
        return tmp_resp

    llm = LLMOverREST(api_endpoint="https://some.dummy.url/v1/api")
    mock_client = MagicMock()
    mock_client.return_value.__enter__ = lambda x: mock_client
    mock_client.request = mock_response

    with patch.multiple(llm, client=mock_client, use_httpx=False):
        prediction = llm.predict("hello")
    mock_client.assert_called()
    assert json.loads(prediction) == mock_generated_json()


@pytest.mark.requires("httpx")
def test_llmoverrest_call_httpx() -> None:
    import httpx

    def mock_response(*args: Any, **kwargs: Any) -> httpx.Response:
        tmp_resp = httpx.Response(
            status_code=200, json=mock_generated_json(), request=MagicMock()
        )
        return tmp_resp

    llm = LLMOverREST(api_endpoint="https://some.dummy.url/v1/api")
    mock_client = MagicMock()
    mock_client.return_value.__enter__ = lambda x: mock_client
    mock_client.request = mock_response

    with patch.multiple(llm, client=mock_client, use_httpx=True):
        prediction = llm.predict("hello")
    mock_client.assert_called()
    assert json.loads(prediction) == mock_generated_json()


def test_llmoverrest_call_retry() -> None:
    raised = False
    completed = False
    counts = 0

    def mock_response(*args: Any, **kwargs: Any) -> requests.Response:
        nonlocal raised, completed, counts
        counts += 1
        if completed:
            assert False, "This should not be called twice"

        if not raised:
            raised = True
            raise requests.exceptions.RequestException()
        tmp_resp = requests.Response()
        tmp_resp.status_code = 200
        tmp_resp._content = json.dumps(mock_generated_json()).encode("utf-8")
        completed = True
        return tmp_resp

    llm = LLMOverREST(api_endpoint="https://some.dummy.url/v1/api", retries=2)
    mock_client = MagicMock()
    mock_client.return_value.__enter__ = lambda x: mock_client
    mock_client.request = mock_response

    with patch.object(llm_over_rest, "create_base_retry_decorator", _patched_retry):
        with patch.multiple(llm, client=mock_client, use_httpx=False):
            prediction = llm.predict("hello")
    mock_client.assert_called()
    assert json.loads(prediction) == mock_generated_json()
    assert completed
    assert counts == 2


def test_llmoverrest_call_timeout() -> None:
    def mock_response(*args: Any, **kwargs: Any) -> Any:
        raise requests.exceptions.Timeout("Timed out request")

    llm = LLMOverREST(api_endpoint="https://some.dummy.url/v1/api", retries=2)
    mock_client = MagicMock()
    mock_client.return_value.__enter__ = lambda x: mock_client
    mock_client.request = mock_response
    with patch.object(llm_over_rest, "create_base_retry_decorator", _patched_retry):
        with patch.multiple(llm, client=mock_client, use_httpx=False):
            with pytest.raises(ValueError) as e:
                _ = llm.predict("hello")

    assert "Timed out request" in str(e)
