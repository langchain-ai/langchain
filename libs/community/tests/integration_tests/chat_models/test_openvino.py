"""Test Chat OpenVINO wrapper."""

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from langchain_community.chat_models.openvino import ChatOpenVINO
from langchain_community.llms.openvino import OpenVINOLLM


def import_hf_hub() -> Any:
    try:
        import huggingface_hub as hf_hub
    except ImportError as e:
        raise ImportError(
            "Could not import huggingface_hub package. "
            "Please install it with `pip install huggingface_hub`."
        ) from e
    return hf_hub


def test_openvino_call() -> None:
    """Test invoking tokens from ChatOpenVINO."""
    hf_hub = import_hf_hub()
    model_id = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"
    model_path = "TinyLlama-1.1B-Chat-v1.0-int4-ov"

    hf_hub.snapshot_download(model_id, local_dir=model_path)
    llm = OpenVINOLLM.from_model_path(model_path)
    chat = ChatOpenVINO(llm=llm, verbose=True)
    message = HumanMessage(content="Hello")

    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_openvino_streaming() -> None:
    """Test streaming tokens from ChatOpenVINO."""
    hf_hub = import_hf_hub()
    model_id = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"
    model_path = "TinyLlama-1.1B-Chat-v1.0-int4-ov"

    hf_hub.snapshot_download(model_id, local_dir=model_path)
    llm = OpenVINOLLM.from_model_path(model_path)
    chat = ChatOpenVINO(llm=llm, verbose=True)
    message = HumanMessage(content="Hello")

    response = chat.stream([message])
    for chunk in response:
        assert isinstance(chunk, AIMessageChunk)
