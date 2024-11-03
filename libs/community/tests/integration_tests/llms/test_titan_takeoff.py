"""Test Titan Takeoff wrapper."""

import json
from typing import Any, Union

import pytest

from langchain_community.llms import TitanTakeoff, TitanTakeoffPro


@pytest.mark.requires("takeoff_client")
@pytest.mark.requires("pytest_httpx")
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("takeoff_object", [TitanTakeoff, TitanTakeoffPro])
def test_titan_takeoff_call(
    httpx_mock: Any,
    streaming: bool,
    takeoff_object: Union[TitanTakeoff, TitanTakeoffPro],
) -> None:
    """Test valid call to Titan Takeoff."""
    from pytest_httpx import IteratorStream

    port = 2345
    url = (
        f"http://localhost:{port}/generate_stream"
        if streaming
        else f"http://localhost:{port}/generate"
    )

    if streaming:
        httpx_mock.add_response(
            method="POST",
            url=url,
            stream=IteratorStream([b"data: ask someone else\n\n"]),
        )
    else:
        httpx_mock.add_response(
            method="POST",
            url=url,
            json={"text": "ask someone else"},
        )

    llm = takeoff_object(port=port, streaming=streaming)
    number_of_calls = 0
    for function_call in [llm, llm.invoke]:
        number_of_calls += 1
        output = function_call("What is 2 + 2?")
        assert isinstance(output, str)
        assert len(httpx_mock.get_requests()) == number_of_calls
        assert httpx_mock.get_requests()[0].url == url
        assert (
            json.loads(httpx_mock.get_requests()[0].content)["text"] == "What is 2 + 2?"
        )

    if streaming:
        output = llm._stream("What is 2 + 2?")
        for chunk in output:
            assert isinstance(chunk.text, str)
        assert len(httpx_mock.get_requests()) == number_of_calls + 1
        assert httpx_mock.get_requests()[0].url == url
        assert (
            json.loads(httpx_mock.get_requests()[0].content)["text"] == "What is 2 + 2?"
        )


@pytest.mark.requires("pytest_httpx")
@pytest.mark.requires("takeoff_client")
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("takeoff_object", [TitanTakeoff, TitanTakeoffPro])
def test_titan_takeoff_bad_call(
    httpx_mock: Any,
    streaming: bool,
    takeoff_object: Union[TitanTakeoff, TitanTakeoffPro],
) -> None:
    """Test valid call to Titan Takeoff."""
    from takeoff_client import TakeoffException

    url = (
        "http://localhost:3000/generate"
        if not streaming
        else "http://localhost:3000/generate_stream"
    )
    httpx_mock.add_response(
        method="POST", url=url, json={"text": "bad things"}, status_code=400
    )

    llm = takeoff_object(streaming=streaming)
    with pytest.raises(TakeoffException):
        llm.invoke("What is 2 + 2?")
    assert len(httpx_mock.get_requests()) == 1
    assert httpx_mock.get_requests()[0].url == url
    assert json.loads(httpx_mock.get_requests()[0].content)["text"] == "What is 2 + 2?"


@pytest.mark.requires("pytest_httpx")
@pytest.mark.requires("takeoff_client")
@pytest.mark.parametrize("takeoff_object", [TitanTakeoff, TitanTakeoffPro])
def test_titan_takeoff_model_initialisation(
    httpx_mock: Any,
    takeoff_object: Union[TitanTakeoff, TitanTakeoffPro],
) -> None:
    """Test valid call to Titan Takeoff."""
    mgnt_port = 36452
    inf_port = 46253
    mgnt_url = f"http://localhost:{mgnt_port}/reader"
    gen_url = f"http://localhost:{inf_port}/generate"
    reader_1 = {
        "model_name": "test",
        "device": "cpu",
        "consumer_group": "primary",
        "max_sequence_length": 512,
        "max_batch_size": 4,
        "tensor_parallel": 3,
    }
    reader_2 = reader_1.copy()
    reader_2["model_name"] = "test2"

    httpx_mock.add_response(
        method="POST", url=mgnt_url, json={"key": "value"}, status_code=201
    )
    httpx_mock.add_response(
        method="POST", url=gen_url, json={"text": "value"}, status_code=200
    )

    llm = takeoff_object(
        port=inf_port, mgmt_port=mgnt_port, models=[reader_1, reader_2]
    )
    output = llm.invoke("What is 2 + 2?")

    assert isinstance(output, str)
    # Ensure the management api was called to create the reader
    assert len(httpx_mock.get_requests()) == 3
    for key, value in reader_1.items():
        assert json.loads(httpx_mock.get_requests()[0].content)[key] == value
    assert httpx_mock.get_requests()[0].url == mgnt_url
    # Also second call should be made to spin uo reader 2
    for key, value in reader_2.items():
        assert json.loads(httpx_mock.get_requests()[1].content)[key] == value
    assert httpx_mock.get_requests()[1].url == mgnt_url
    # Ensure the third call is to generate endpoint to inference
    assert httpx_mock.get_requests()[2].url == gen_url
    assert json.loads(httpx_mock.get_requests()[2].content)["text"] == "What is 2 + 2?"
