import unittest
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from langchain.llms.arcee import Arcee


class TestApiConfigSecurity(unittest.TestCase):

    @patch('langchain.utilities.arcee.requests.get')
    def setUp(self, mock_get) -> None:
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {"model_id": "", "status": "training_complete"}

        self.arcee_without_env_var = Arcee(
            model="DALM-PubMed",
            arcee_api_key="secret_api_key",
            arcee_api_url="localhost",
            arcee_api_version="version",
        )


    @pytest.fixture(autouse=True)
    def capsys(self, capsys):
        self.capsys = capsys

    @pytest.fixture(autouse=True)
    def monkeypatch(self, monkeypatch):
        self.monkeypatch = monkeypatch

    def test_arcee_api_key_is_secret_string(self) -> None:
        self.assertTrue(isinstance(self.arcee_without_env_var.arcee_api_key, SecretStr))

    def test_api_key_masked_when_passed_via_constructor(self) -> None:
        print(self.arcee_without_env_var.arcee_api_key, end="")
        captured = self.capsys.readouterr()

        assert captured.out == "**********"
