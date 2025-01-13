"""Test OpenVINO LLM wrapper."""

from typing import Any, Generator

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


def test_openvino_text_generation() -> None:
    """Test valid call to HuggingFace text generation model."""
    hf_hub = import_hf_hub()
    model_id = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"
    model_path = "TinyLlama-1.1B-Chat-v1.0-int4-ov"

    hf_hub.snapshot_download(model_id, local_dir=model_path)
    llm = OpenVINOLLM.from_model_path(model_path)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_openvino_streaming() -> None:
    """Test streaming tokens from huggingface_pipeline."""
    hf_hub = import_hf_hub()
    model_id = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"
    model_path = "TinyLlama-1.1B-Chat-v1.0-int4-ov"

    hf_hub.snapshot_download(model_id, local_dir=model_path)
    llm = OpenVINOLLM.from_model_path(model_path)
    generator = llm.stream("Q: How do you say 'hello' in German? A:'", stop=["."])
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert isinstance(chunk, str)
        stream_results_string = chunk
    assert len(stream_results_string.strip()) > 1
