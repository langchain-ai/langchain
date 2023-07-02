"""Test requests wrapper."""

import json

import pytest
from aioresponses import aioresponses

from langchain.requests import TextRequestsWrapper


@pytest.mark.asyncio
async def test_text_requests_wrapper_apost() -> None:
    url = "http://example.com"
    data = {"key": "value"}
    mock_response_data = {"status": "success"}

    with aioresponses() as mock:
        mock.post(url, payload=mock_response_data, status=200)

        requests = TextRequestsWrapper()
        response = await requests.apost(url, data)

        assert response == json.dumps(mock_response_data)
        mock.assert_called_once_with(url, "POST", headers=None, json=data)


@pytest.mark.asyncio
async def test_text_requests_wrapper_apatch() -> None:
    url = "http://example.com"
    data = {"key": "value"}
    mock_response_data = {"status": "success"}

    with aioresponses() as mock:
        mock.patch(url, payload=mock_response_data, status=200)

        requests = TextRequestsWrapper()
        response = await requests.apatch(url, data)

        assert response == json.dumps(mock_response_data)
        mock.assert_called_once_with(url, "PATCH", headers=None, json=data)


@pytest.mark.asyncio
async def test_text_requests_wrapper_aput() -> None:
    url = "http://example.com"
    data = {"key": "value"}
    mock_response_data = {"status": "success"}

    with aioresponses() as mock:
        mock.put(url, payload=mock_response_data, status=200)

        requests = TextRequestsWrapper()
        response = await requests.aput(url, data)

        assert response == json.dumps(mock_response_data)
        mock.assert_called_once_with(url, "PUT", headers=None, json=data)
