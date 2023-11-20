import unittest
from typing import Any
from unittest.mock import patch

from langchain.tools.cogniswitch.tool import CogniswitchAnswerTool, CogniswitchStoreTool


class TestCogniswitchAnswerTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = CogniswitchAnswerTool(
            cs_token="cs_token", OAI_token="OAI_token", apiKey="apiKey"
        )

    @patch("requests.post")
    def test_answer_cs(self, mock_post: Any) -> None:
        query = "test query"
        expected_response = {"response": "test response"}
        mock_post.return_value.json.return_value = expected_response

        response = self.tool.answer_cs(
            self.tool.cs_token, self.tool.OAI_token, query, self.tool.apiKey
        )

        mock_post.assert_called_once_with(
            self.tool.api_url,
            headers={
                "apiKey": self.tool.apiKey,
                "platformToken": self.tool.cs_token,
                "openAIToken": self.tool.OAI_token,
            },
            verify=False,
            data={"query": query},
        )

        self.assertEqual(response, expected_response)

    def test_run(self) -> None:
        query = "test query"
        expected_response = {"response": "test response"}

        with patch.object(
            self.tool, "answer_cs", return_value=expected_response
        ) as mock_answer_cs:
            response = self.tool._run(query)
            mock_answer_cs.assert_called_once_with(
                self.tool.cs_token, self.tool.OAI_token, query, self.tool.apiKey
            )

        self.assertEqual(response, expected_response)


class TestCogniswitchStoreTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = CogniswitchStoreTool(
            cs_token="cs_token", OAI_token="OAI_token", apiKey="apiKey"
        )

    @patch("requests.post")
    def test_store_data_with_url(self, mock_post: Any) -> None:
        url = "https://example.com"
        expected_response = {"response": "test response"}
        mock_post.return_value.json.return_value = expected_response

        response = self.tool.store_data(
            url=url,
            file=None,
            document_name="doc_name",
            document_description="doc_desc",
        )

        mock_post.assert_called_once_with(
            self.tool.knowledgesource_url,
            headers={
                "apiKey": self.tool.apiKey,
                "platformToken": self.tool.cs_token,
                "openAIToken": self.tool.OAI_token,
            },
            verify=False,
            data={"url": url},
            files=None,
        )

        self.assertEqual(response, expected_response)
