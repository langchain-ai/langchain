import unittest
from unittest.mock import MagicMock, patch

from langchain.chains.cogniswitch import CogniswitchAnswerChain


class TestCogniswitchAnswerChain(unittest.TestCase):
    @patch("requests.post")
    def test_answer_cs(self, mock_post: MagicMock) -> None:
        chain = CogniswitchAnswerChain()
        cs_token = "cs_token"
        OAI_token = "OAI_token"
        apiKey = "apiKey"
        query = "test query"
        expected_response = "Test answer"

        mock_response = MagicMock()
        mock_response.json.return_value = expected_response
        mock_post.return_value = mock_response

        response = chain.answer_cs(cs_token, OAI_token, query, apiKey)

        self.assertEqual(response, expected_response)
        mock_post.assert_called_once_with(
            "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeRequest",
            headers={
                "apiKey": apiKey,
                "platformToken": cs_token,
                "openAIToken": OAI_token,
            },
            verify=False,
            data={"query": query},
        )

    def test_validate_inputs_missing_cs_token(self) -> None:
        chain = CogniswitchAnswerChain()
        inputs = {"query": "test query", "apiKey": "apiKey"}
        with self.assertRaises(ValueError):
            chain._validate_inputs(inputs)

    def test_validate_inputs_missing_query(self) -> None:
        chain = CogniswitchAnswerChain()
        inputs = {"cs_token": "cs_token", "apiKey": "apiKey"}
        with self.assertRaises(ValueError):
            chain._validate_inputs(inputs)

    def test_validate_inputs_missing_keys(self) -> None:
        chain = CogniswitchAnswerChain()
        inputs: dict = {}
        with self.assertRaises(ValueError):
            chain._validate_inputs(inputs)
