from text_generation import AsyncClient, Client

from langchain_community.llms import HuggingFaceTextGenInference


def test_invocation_params_stop_sequences() -> None:
    llm = HuggingFaceTextGenInference()
    assert llm._default_params["stop_sequences"] == []

    runtime_stop = None
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == []
    assert llm._default_params["stop_sequences"] == []

    runtime_stop = ["stop"]
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == ["stop"]
    assert llm._default_params["stop_sequences"] == []

    llm = HuggingFaceTextGenInference(stop_sequences=["."])
    runtime_stop = ["stop"]
    assert llm._invocation_params(runtime_stop)["stop_sequences"] == [".", "stop"]
    assert llm._default_params["stop_sequences"] == ["."]


def test_client_type() -> None:
    llm = HuggingFaceTextGenInference()

    assert isinstance(llm.client, Client)
    assert isinstance(llm.async_client, AsyncClient)


def test_bearer_api() -> None:
    llm = HuggingFaceTextGenInference()
    # If called from a self-hosted TGI server,
    assert not llm.client.headers
    assert not llm.async_client.headers

    BEARER_TOKEN = "abcdef1230"
    llm = HuggingFaceTextGenInference(huggingfacehub_api_token=BEARER_TOKEN)
    # If called from a self-hosted TGI server,
    assert llm.client.headers["Authorization"] == f"Bearer {BEARER_TOKEN}"
    assert llm.async_client.headers["Authorization"] == f"Bearer {BEARER_TOKEN}"
