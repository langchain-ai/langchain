import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from langchain.tools.cogniswitch.tool import CogniswitchAnswerTool, CogniswitchStoreTool


class TestCogniswitchAnswerTool(unittest.TestCase):
    def setUp(self) -> None:
        self.cs_token = "your_cs_token"
        self.OAI_token = "your_OAI_token"
        self.apiKey = "your_api_key"
        self.query = "sample query"
        self.api_url = (
            "https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeRequest"
        )

    @patch("langchain.tools.cogniswitch.tool.requests.post")
    def test_answer_cs(self, mock_post: Any) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "mocked_response_key": "mocked_response_value"
        }
        mock_post.return_value = mock_response

        tool = CogniswitchAnswerTool(
            cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey
        )
        result = tool.answer_cs(self.cs_token, self.OAI_token, self.query, self.apiKey)

        # Assertions
        mock_post.assert_called_once_with(
            self.api_url,
            headers={
                "apiKey": self.apiKey,
                "platformToken": self.cs_token,
                "openAIToken": self.OAI_token,
            },
            verify=False,
            data={"query": self.query},
        )
        self.assertEqual(result, {"mocked_response_key": "mocked_response_value"})

    @patch("langchain.tools.cogniswitch.tool.CogniswitchAnswerTool.answer_cs")
    def test_run(self, mock_answer_cs) -> None:
        mock_answer_cs.return_value = {"response_key": "response_value"}

        tool = CogniswitchAnswerTool(
            cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey
        )
        result = tool._run(self.query)

        # Assertions
        mock_answer_cs.assert_called_once_with(
            self.cs_token, self.OAI_token, self.query, self.apiKey
        )
        self.assertEqual(result, {"response_key": "response_value"})


class TestCogniswitchStoreTool(unittest.TestCase):
    def setUp(self) -> None:
        self.cs_token = "your_cs_token"
        self.OAI_token = "your_OAI_token"
        self.apiKey = "your_api_key"
        self.document_name = "example_document"
        self.document_description = "Example document description"
        self.url = "https://cogniswitch.ai/developer"

    @patch("langchain.tools.cogniswitch.tool.requests.post")
    def test_store_data_with_url(self, mock_post: Any) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "mocked_response_key": "mocked_response_value"
        }
        mock_post.return_value = mock_response

        tool = CogniswitchStoreTool(
            cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey
        )
        result = tool.store_data(
            url=self.url,
            file=None,
            document_name=self.document_name,
            document_description=self.document_description,
        )

        # Assertions
        mock_post.assert_called_once_with(
            tool.knowledgesource_url,
            headers={
                "apiKey": self.apiKey,
                "openAIToken": self.OAI_token,
                "platformToken": self.cs_token,
            },
            verify=False,
            data={"url": self.url},
            files=None,
        )
        self.assertEqual(result, {"mocked_response_key": "mocked_response_value"})

    def test_store_data_no_input(self):
        tool = CogniswitchStoreTool(
            cs_token=self.cs_token, OAI_token=self.OAI_token, apiKey=self.apiKey
        )
        result = tool.store_data(
            url=None, file=None, document_name=None, document_description=None
        )

        # Assertions
        self.assertEqual(result, {"message": "No input provided"})
