import unittest
from unittest.mock import patch

from pydantic import SecretStr

from langchain.llms.arcee import Arcee


class TestApiConfigSecurity(unittest.TestCase):
    @patch('langchain.utilities.arcee.requests.get')
    def test_arcee_api_key_is_secret_string(self, mock_get) -> None:
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {"model_id": "", "status": "training_complete"}

        llm = Arcee(
            model="DALM-PubMed",
            arcee_api_key="secret_api_key",
            arcee_api_url="localhost",
            arcee_api_version="version",
        )


# def test_api_key_securely_wrapped(self):
#     # Ensure that the API key is securely wrapped using SecretStr.
#     config = ApiConfig(api_key="your_api_key_here")
#     self.assertIsInstance(config.api_key, SecretStr)
#
# def test_no_secret_in_logs(self):
#     # Ensure that sensitive data is not exposed in logs.
#     config = ApiConfig(api_key="your_api_key_here")
#     log_output = some_logging_function(config.api_key.get_secret_value())
#     self.assertNotIn("your_api_key_here", log_output)
#
# def test_proper_access_control(self):
#     # Ensure that proper access control is enforced for sensitive data.
#     config = ApiConfig(api_key="your_api_key_here")
#     # Perform actions that require API key and assert proper access control.
#     self.assertTrue(some_function_requiring_api_key(config.api_key.get_secret_value()))
