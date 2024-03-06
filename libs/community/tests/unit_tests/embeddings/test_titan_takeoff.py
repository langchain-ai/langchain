"""Test Titan Takeoff Embedding wrapper."""


import responses
import pytest
import json

from langchain_community.embeddings import TitanTakeoffEmbed
from langchain_community.embeddings.titan_takeoff import MissingConsumerGroup


@responses.activate
@pytest.mark.requires("takeoff_client")
def test_titan_takeoff_call() -> None:
    """Test valid call to Titan Takeoff."""
    port = 2345

    responses.add(responses.POST, f"http://localhost:{port}/embed", json={"text": [0.46635, 0.234, -0.8521]}, content_type="application/json")
    
    embedding = TitanTakeoffEmbed(port=port)
    
    output_1 = embedding.embed_documents("What is 2 + 2?", "primary")
    output_2 = embedding.embed_query("What is 2 + 2?", "primary")
    
    assert isinstance(output_1, list)
    assert isinstance(output_2, list)

    assert len(responses.calls) == 2
    for n in range(2):
        assert responses.calls[n].request.url == f"http://localhost:{port}/embed"
        assert json.loads(responses.calls[n].request.body.decode("utf-8"))["text"] == "What is 2 + 2?"
    
@responses.activate
@pytest.mark.requires("takeoff_client")
def test_no_consumer_group_fails() -> None:
    """Test that not specifying a consumer group fails."""
    port = 2345

    responses.add(responses.POST, f"http://localhost:{port}/embed", json={"text": [0.46635, 0.234, -0.8521]}, content_type="application/json")
    
    embedding = TitanTakeoffEmbed(port=port)
    
    with pytest.raises(MissingConsumerGroup):
        embedding.embed_documents("What is 2 + 2?")
    with pytest.raises(MissingConsumerGroup):
        embedding.embed_query("What is 2 + 2?")
    
    # Check specifying a consumer group works
    embedding.embed_documents("What is 2 + 2?", "primary")
    embedding.embed_query("What is 2 + 2?", "primary")
    
@responses.activate
@pytest.mark.requires("takeoff_client")
def test_takeoff_initialization() -> None:
    """Test valid call to Titan Takeoff."""
    mgnt_port = 36452
    inf_port = 46253
    mgnt_url = f"http://localhost:{mgnt_port}/reader"
    embed_url = f"http://localhost:{inf_port}/embed"
    reader_1 = {"model_name": "test", "device": "cpu", "consumer_group": "embed", "max_seq_length": 512, "max_batch_size": 4, "tensor_parallel":3}
    reader_2 = reader_1.copy()
    reader_2["model_name"] = "test2"
    reader_2["device"] = "cuda"
    
    responses.add(responses.POST, mgnt_url, json={"key": "value"}, status=201)
    responses.add(responses.POST, embed_url, json={"text": [0.34, 0.43, -0.934532]}, status=200)

    llm = TitanTakeoffEmbed(port=inf_port, mgmt_port=mgnt_port, models=[reader_1, reader_2])
    # Shouldn't need to specify consumer group as there is only one specified during initialization 
    output_1 = llm.embed_documents("What is 2 + 2?")
    output_2 = llm.embed_query("What is 2 + 2?")
        
    assert isinstance(output_1, list)
    assert isinstance(output_2, list)
    # Ensure the management api was called to create the reader
    assert len(responses.calls) == 4
    assert json.loads(responses.calls[0].request.body.decode("utf-8")) == reader_1
    assert responses.calls[0].request.url == mgnt_url
    # Also second call should be made to spin uo reader 2
    assert json.loads(responses.calls[1].request.body.decode("utf-8")) == reader_2
    assert responses.calls[1].request.url == mgnt_url
    # Ensure the third call is to generate endpoint to inference
    for n in range(2,4):
        assert responses.calls[n].request.url == embed_url
        assert json.loads(responses.calls[n].request.body.decode("utf-8"))["text"] == "What is 2 + 2?"

@responses.activate
@pytest.mark.requires("takeoff_client")
def test_takeoff_initialization_with_more_than_one_consumer_group() -> None:
    """Test valid call to Titan Takeoff."""
    mgnt_port = 36452
    inf_port = 46253
    mgnt_url = f"http://localhost:{mgnt_port}/reader"
    embed_url = f"http://localhost:{inf_port}/embed"
    reader_1 = {"model_name": "test", "device": "cpu", "consumer_group": "embed", "max_seq_length": 512, "max_batch_size": 4, "tensor_parallel":3}
    reader_2 = reader_1.copy()
    reader_2["model_name"] = "test2"
    reader_2["device"] = "cuda"
    reader_2["consumer_group"] = "embed2"
    
    responses.add(responses.POST, mgnt_url, json={"key": "value"}, status=201)
    responses.add(responses.POST, embed_url, json={"text": [0.34, 0.43, -0.934532]}, status=200)

    llm = TitanTakeoffEmbed(port=inf_port, mgmt_port=mgnt_port, models=[reader_1, reader_2])
    # There was more than one consumer group specified during initialization so we need to specify which one to use
    with pytest.raises(MissingConsumerGroup):
        llm.embed_documents("What is 2 + 2?")
    with pytest.raises(MissingConsumerGroup):
        llm.embed_query("What is 2 + 2?")
        
    output_1 = llm.embed_documents("What is 2 + 2?", "embed")
    output_2 = llm.embed_query("What is 2 + 2?", "embed2")
        
    assert isinstance(output_1, list)
    assert isinstance(output_2, list)
    # Ensure the management api was called to create the reader
    assert len(responses.calls) == 4
    assert json.loads(responses.calls[0].request.body.decode("utf-8")) == reader_1
    assert responses.calls[0].request.url == mgnt_url
    # Also second call should be made to spin uo reader 2
    assert json.loads(responses.calls[1].request.body.decode("utf-8")) == reader_2
    assert responses.calls[1].request.url == mgnt_url
    # Ensure the third call is to generate endpoint to inference
    for n in range(2,4):
        assert responses.calls[n].request.url == embed_url
        assert json.loads(responses.calls[n].request.body.decode("utf-8"))["text"] == "What is 2 + 2?"