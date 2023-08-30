"""Test AliCloudPaiEas API wrapper."""
import os
from typing import Generator

from langchain.llms.alicloud_pai_eas import AliCloudPaiEas


def test_pai_eas_call() -> None:
    """Test valid call to PAI-EAS Service."""
    llm = AliCloudPaiEas(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_pai_eas_streaming() -> None:
    """Test streaming call to PAI-EAS Service."""
    llm = AliCloudPaiEas(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),
    )
    generator = llm.stream("Q: How do you say 'hello' in German? A:'", stop=["."])
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert isinstance(chunk, str)
        stream_results_string = chunk
    assert len(stream_results_string.strip()) > 1
