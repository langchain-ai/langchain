import unittest
from unittest.mock import MagicMock, patch

import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from langchain_core.pydantic_v1 import SecretStr

from langchain.llms.arcee import Arcee


class TestApiConfigSecurity(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def capsys(self, capsys: CaptureFixture) -> None:
        self.capsys = capsys

    @pytest.fixture(autouse=True)
    def monkeypatch(self, monkeypatch: MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch

    @patch("langchain.utilities.arcee.requests.get")
    def setUp(self, mock_get: MagicMock) -> None:
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_id": "",
            "status": "training_complete",
        }

        self.arcee_without_env_var = Arcee(
            model="DALM-PubMed",
            arcee_api_key="secret_api_key",
            arcee_api_url="localhost",
            arcee_api_version="version",
        )
        self.monkeypatch.setenv("ARCEE_API_KEY", "secret_api_key")
        self.arcee_with_env_var = Arcee(
            model="DALM-PubMed",
            arcee_api_key="",
            arcee_api_url="localhost",
            arcee_api_version="version",
        )

    def test_arcee_api_key_is_secret_string(self) -> None:
        self.assertTrue(isinstance(self.arcee_without_env_var.arcee_api_key, SecretStr))

    def test_api_key_masked_when_passed_via_constructor(self) -> None:
        print(self.arcee_without_env_var.arcee_api_key, end="")
        captured = self.capsys.readouterr()

        self.assertEquals("**********", captured.out)

    def test_api_key_masked_when_passed_from_env(self) -> None:
        print(self.arcee_with_env_var.arcee_api_key, end="")
        captured = self.capsys.readouterr()

        self.assertEquals("**********", captured.out)
