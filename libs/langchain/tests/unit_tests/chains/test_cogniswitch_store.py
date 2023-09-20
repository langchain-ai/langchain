import unittest
from unittest.mock import MagicMock, patch

from langchain.chains.cogniswitch import CogniswitchStoreChain


class TestCogniswitchStoreChain(unittest.TestCase):
    @patch("requests.post")
    def test_store_data_successful(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "Data stored successfully"}
        mock_post.return_value = mock_response

        chain = CogniswitchStoreChain()
        result = chain.store_data(
            "cs_token", "OAI_token", "http://example.com", None, "apiKey", None, None
        )
        self.assertEqual(result, {"message": "Data stored successfully"})

    @patch("requests.post")
    def test_store_data_failure(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "Bad Request"}
        mock_post.return_value = mock_response

        chain = CogniswitchStoreChain()
        result = chain.store_data(
            "cs_token",
            "OAI_token",
            None,
            None,
            "apiKey",
            "document_name",
            "document_description",
        )
        self.assertEqual(
            result,
            {
                "message": "Bad Request",
            },
        )
