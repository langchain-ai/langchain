"""Test TritonTensorRT Chat API wrapper."""
import sys
from io import StringIO

import pytest
from tritonclient.utils import InferenceServerException

from langchain_nvidia_trt import TritonTensorRTLLM


def test_initialization() -> None:
    """Test integration initialization."""
    TritonTensorRTLLM(model_name="ensemble", server_url="http://localhost:8001")


def test_default_verbose() -> None:
    llm = TritonTensorRTLLM(server_url="http://localhost:8001", model_name="ensemble")
    captured = StringIO()
    sys.stdout = captured
    with pytest.raises(InferenceServerException):
        llm.client.is_server_live()
    sys.stdout = sys.__stdout__
    assert "is_server_live" not in captured.getvalue()


def test_verbose() -> None:
    llm = TritonTensorRTLLM(
        server_url="http://localhost:8001", model_name="ensemble", verbose=True
    )
    captured = StringIO()
    sys.stdout = captured
    with pytest.raises(InferenceServerException):
        llm.client.is_server_live()
    sys.stdout = sys.__stdout__
    assert "is_server_live" in captured.getvalue()
