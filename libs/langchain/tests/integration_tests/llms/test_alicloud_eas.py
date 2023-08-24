"""Test AliCloudPaiEAS API wrapper."""
import os

from langchain.llms.alicloud_pai_eas import AliCloudPaiEAS


def test_eas_call() -> None:
    """Test valid call to PAI-EAS Service."""
    llm = AliCloudPaiEAS(eas_service_url=os.getenv("EAS_SERVICE_URL"),
                         eas_service_token=os.getenv("EAS_SERVICE_TOKEN"))
    output = llm("Say foo:")
    assert isinstance(output, str)
