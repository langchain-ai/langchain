"""Test TritonTensorRT Chat API wrapper."""
from langchain_nvidia_trt import TritonTensorRTLLM


def test_initialization() -> None:
    """Test integration initialization."""
    TritonTensorRTLLM(model_name="ensemble", server_url="http://localhost:8001")
