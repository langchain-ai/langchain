"""Test TritonTensorRT Chat API wrapper."""
import sys
from io import StringIO
from unittest.mock import patch

from langchain_nvidia_trt import TritonTensorRTLLM


def test_initialization() -> None:
    """Test integration initialization."""
    TritonTensorRTLLM(model_name="ensemble", server_url="http://localhost:8001")


@patch("tritonclient.grpc.service_pb2_grpc.GRPCInferenceServiceStub")
def test_default_verbose(ignore) -> None:
    llm = TritonTensorRTLLM(server_url="http://localhost:8001", model_name="ensemble")
    captured = StringIO()
    sys.stdout = captured
    llm.client.is_server_live()
    sys.stdout = sys.__stdout__
    assert "is_server_live" not in captured.getvalue()


@patch("tritonclient.grpc.service_pb2_grpc.GRPCInferenceServiceStub")
def test_verbose(ignore) -> None:
    llm = TritonTensorRTLLM(
        server_url="http://localhost:8001", model_name="ensemble", verbose_client=True
    )
    captured = StringIO()
    sys.stdout = captured
    llm.client.is_server_live()
    sys.stdout = sys.__stdout__
    assert "is_server_live" in captured.getvalue()
