"""Test Titan Takeoff Embedding wrapper."""

import json
from typing import Any

import pytest

from langchain_community.embeddings import TitanTakeoffEmbed
from langchain_community.embeddings.titan_takeoff import (
    Device,
    MissingConsumerGroup,
    ReaderConfig,
)


@pytest.mark.requires("pytest_httpx")
@pytest.mark.requires("takeoff_client")
def test_titan_takeoff_call(httpx_mock: Any) -> None:
    """Test valid call to Titan Takeoff."""
    port = 2345

    httpx_mock.add_response(
        method="POST",
        url=f"http://localhost:{port}/embed",
        json={"result": [0.46635, 0.234, -0.8521]},
    )

    embedding = TitanTakeoffEmbed(port=port)

    output_1 = embedding.embed_documents(["What is 2 + 2?"], "primary")
    output_2 = embedding.embed_query("What is 2 + 2?", "primary")

    assert isinstance(output_1, list)
    assert isinstance(output_2, list)

    assert len(httpx_mock.get_requests()) == 2
    for n in range(2):
        assert httpx_mock.get_requests()[n].url == f"http://localhost:{port}/embed"
        assert (
            json.loads(httpx_mock.get_requests()[n].content)["text"] == "What is 2 + 2?"
        )


@pytest.mark.requires("pytest_httpx")
@pytest.mark.requires("takeoff_client")
def test_no_consumer_group_fails(httpx_mock: Any) -> None:
    """Test that not specifying a consumer group fails."""
    port = 2345

    httpx_mock.add_response(
        method="POST",
        url=f"http://localhost:{port}/embed",
        json={"result": [0.46635, 0.234, -0.8521]},
    )

    embedding = TitanTakeoffEmbed(port=port)

    with pytest.raises(MissingConsumerGroup):
        embedding.embed_documents(["What is 2 + 2?"])
    with pytest.raises(MissingConsumerGroup):
        embedding.embed_query("What is 2 + 2?")

    # Check specifying a consumer group works
    embedding.embed_documents(["What is 2 + 2?"], "primary")
    embedding.embed_query("What is 2 + 2?", "primary")


@pytest.mark.requires("pytest_httpx")
@pytest.mark.requires("takeoff_client")
def test_takeoff_initialization(httpx_mock: Any) -> None:
    """Test valid call to Titan Takeoff."""
    mgnt_port = 36452
    inf_port = 46253
    mgnt_url = f"http://localhost:{mgnt_port}/reader"
    embed_url = f"http://localhost:{inf_port}/embed"
    reader_1 = ReaderConfig(
        model_name="test",
        device=Device.cpu,
        consumer_group="embed",
    )
    reader_2 = ReaderConfig(
        model_name="test2",
        device=Device.cuda,
        consumer_group="embed",
    )

    httpx_mock.add_response(
        method="POST", url=mgnt_url, json={"key": "value"}, status_code=201
    )
    httpx_mock.add_response(
        method="POST",
        url=embed_url,
        json={"result": [0.34, 0.43, -0.934532]},
        status_code=200,
    )

    llm = TitanTakeoffEmbed(
        port=inf_port, mgmt_port=mgnt_port, models=[reader_1, reader_2]
    )
    # Shouldn't need to specify consumer group as there is only one specified during
    # initialization
    output_1 = llm.embed_documents(["What is 2 + 2?"])
    output_2 = llm.embed_query("What is 2 + 2?")

    assert isinstance(output_1, list)
    assert isinstance(output_2, list)
    # Ensure the management api was called to create the reader
    assert len(httpx_mock.get_requests()) == 4
    for key, value in reader_1.dict().items():
        assert json.loads(httpx_mock.get_requests()[0].content)[key] == value
    assert httpx_mock.get_requests()[0].url == mgnt_url
    # Also second call should be made to spin uo reader 2
    for key, value in reader_2.dict().items():
        assert json.loads(httpx_mock.get_requests()[1].content)[key] == value
    assert httpx_mock.get_requests()[1].url == mgnt_url
    # Ensure the third call is to generate endpoint to inference
    for n in range(2, 4):
        assert httpx_mock.get_requests()[n].url == embed_url
        assert (
            json.loads(httpx_mock.get_requests()[n].content)["text"] == "What is 2 + 2?"
        )


@pytest.mark.requires("pytest_httpx")
@pytest.mark.requires("takeoff_client")
def test_takeoff_initialization_with_more_than_one_consumer_group(
    httpx_mock: Any,
) -> None:
    """Test valid call to Titan Takeoff."""
    mgnt_port = 36452
    inf_port = 46253
    mgnt_url = f"http://localhost:{mgnt_port}/reader"
    embed_url = f"http://localhost:{inf_port}/embed"
    reader_1 = ReaderConfig(
        model_name="test",
        device=Device.cpu,
        consumer_group="embed",
    )
    reader_2 = ReaderConfig(
        model_name="test2",
        device=Device.cuda,
        consumer_group="embed2",
    )

    httpx_mock.add_response(
        method="POST", url=mgnt_url, json={"key": "value"}, status_code=201
    )
    httpx_mock.add_response(
        method="POST",
        url=embed_url,
        json={"result": [0.34, 0.43, -0.934532]},
        status_code=200,
    )

    llm = TitanTakeoffEmbed(
        port=inf_port, mgmt_port=mgnt_port, models=[reader_1, reader_2]
    )
    # There was more than one consumer group specified during initialization so we
    # need to specify which one to use
    with pytest.raises(MissingConsumerGroup):
        llm.embed_documents(["What is 2 + 2?"])
    with pytest.raises(MissingConsumerGroup):
        llm.embed_query("What is 2 + 2?")

    output_1 = llm.embed_documents(["What is 2 + 2?"], "embed")
    output_2 = llm.embed_query("What is 2 + 2?", "embed2")

    assert isinstance(output_1, list)
    assert isinstance(output_2, list)
    # Ensure the management api was called to create the reader
    assert len(httpx_mock.get_requests()) == 4
    for key, value in reader_1.dict().items():
        assert json.loads(httpx_mock.get_requests()[0].content)[key] == value
    assert httpx_mock.get_requests()[0].url == mgnt_url
    # Also second call should be made to spin uo reader 2
    for key, value in reader_2.dict().items():
        assert json.loads(httpx_mock.get_requests()[1].content)[key] == value
    assert httpx_mock.get_requests()[1].url == mgnt_url
    # Ensure the third call is to generate endpoint to inference
    for n in range(2, 4):
        assert httpx_mock.get_requests()[n].url == embed_url
        assert (
            json.loads(httpx_mock.get_requests()[n].content)["text"] == "What is 2 + 2?"
        )
