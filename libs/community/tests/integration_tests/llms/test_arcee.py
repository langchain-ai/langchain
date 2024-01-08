from unittest.mock import MagicMock, patch

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.llms.arcee import Arcee


@patch("langchain_community.utilities.arcee.requests.get")
def test_arcee_api_key_is_secret_string(mock_get: MagicMock) -> None:
    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model_id": "",
        "status": "training_complete",
    }

    arcee_without_env_var = Arcee(
        model="DALM-PubMed",
        arcee_api_key="secret_api_key",
        arcee_api_url="https://localhost",
        arcee_api_version="version",
    )
    assert isinstance(arcee_without_env_var.arcee_api_key, SecretStr)


@patch("langchain_community.utilities.arcee.requests.get")
def test_api_key_masked_when_passed_via_constructor(
    mock_get: MagicMock, capsys: CaptureFixture
) -> None:
    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model_id": "",
        "status": "training_complete",
    }

    arcee_without_env_var = Arcee(
        model="DALM-PubMed",
        arcee_api_key="secret_api_key",
        arcee_api_url="https://localhost",
        arcee_api_version="version",
    )
    print(arcee_without_env_var.arcee_api_key, end="")
    captured = capsys.readouterr()

    assert "**********" == captured.out


@patch("langchain_community.utilities.arcee.requests.get")
def test_api_key_masked_when_passed_from_env(
    mock_get: MagicMock, capsys: CaptureFixture, monkeypatch: MonkeyPatch
) -> None:
    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model_id": "",
        "status": "training_complete",
    }

    monkeypatch.setenv("ARCEE_API_KEY", "secret_api_key")
    arcee_with_env_var = Arcee(
        model="DALM-PubMed",
        arcee_api_url="https://localhost",
        arcee_api_version="version",
    )
    print(arcee_with_env_var.arcee_api_key, end="")
    captured = capsys.readouterr()

    assert "**********" == captured.out
