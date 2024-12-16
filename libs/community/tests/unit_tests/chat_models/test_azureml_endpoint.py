"""Test AzureML chat endpoint."""

import os

import pytest
from pydantic import SecretStr
from pytest import CaptureFixture, FixtureRequest

from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint


@pytest.fixture(scope="class")
def api_passed_via_environment_fixture() -> AzureMLChatOnlineEndpoint:
    """Fixture to create an AzureMLChatOnlineEndpoint instance
    with API key passed from environment"""
    os.environ["AZUREML_ENDPOINT_API_KEY"] = "my-api-key"
    azure_chat = AzureMLChatOnlineEndpoint(
        endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/score"
    )
    del os.environ["AZUREML_ENDPOINT_API_KEY"]
    return azure_chat


@pytest.fixture(scope="class")
def api_passed_via_constructor_fixture() -> AzureMLChatOnlineEndpoint:
    """Fixture to create an AzureMLChatOnlineEndpoint instance
    with API key passed from constructor"""
    azure_chat = AzureMLChatOnlineEndpoint(
        endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/score",
        endpoint_api_key="my-api-key",  # type: ignore[arg-type]
    )
    return azure_chat


@pytest.mark.parametrize(
    "fixture_name",
    ["api_passed_via_constructor_fixture", "api_passed_via_environment_fixture"],
)
class TestAzureMLChatOnlineEndpoint:
    def test_api_key_is_secret_string(
        self, fixture_name: str, request: FixtureRequest
    ) -> None:
        """Test that the API key is a SecretStr instance"""
        azure_chat = request.getfixturevalue(fixture_name)
        assert isinstance(azure_chat.endpoint_api_key, SecretStr)

    def test_api_key_masked(
        self, fixture_name: str, request: FixtureRequest, capsys: CaptureFixture
    ) -> None:
        """Test that the API key is masked"""
        azure_chat = request.getfixturevalue(fixture_name)
        print(azure_chat.endpoint_api_key, end="")  # noqa: T201
        captured = capsys.readouterr()
        assert (
            (str(azure_chat.endpoint_api_key) == "**********")
            and (repr(azure_chat.endpoint_api_key) == "SecretStr('**********')")
            and (captured.out == "**********")
        )

    def test_api_key_is_readable(
        self, fixture_name: str, request: FixtureRequest
    ) -> None:
        """Test that the real secret value of the API key can be read"""
        azure_chat = request.getfixturevalue(fixture_name)
        assert azure_chat.endpoint_api_key.get_secret_value() == "my-api-key"
