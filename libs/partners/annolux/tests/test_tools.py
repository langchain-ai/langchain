import json
import unittest
from unittest.mock import MagicMock, patch

from langchain_annolux.utilities import AnnoluxSearchAPIWrapper
from langchain_annolux.tool import AnnoluxSearchRun, AnnoluxSearchResults


class TestLangChainAnnolux(unittest.TestCase):

    def setUp(self):
        self.mock_response = {
            "query": "test query",
            "results": [
                {
                    "title": "Go 1.26 Release",
                    "url": "https://go.dev/doc/go1.26",
                    "snippet": "Go 1.26 improves compiler performance.",
                    "fetched_at": "2026-08-20T10:00:00Z",
                    "domain": "go.dev",
                    "score": 0.95,
                }
            ],
            "total": 1,
        }

    @patch("requests.post")
    def test_wrapper_results(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.mock_response
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        wrapper = AnnoluxSearchAPIWrapper(annolux_api_key="ann_test_123")
        results = wrapper.results("test query", max_results=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Go 1.26 Release")
        self.assertEqual(results[0]["fetched_at"], "2026-08-20T10:00:00Z")

    @patch("requests.post")
    def test_tool_run(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.mock_response
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        tool = AnnoluxSearchRun(
            api_wrapper=AnnoluxSearchAPIWrapper(annolux_api_key="ann_test_123")
        )
        output = tool.run("test query")

        self.assertIn("Go 1.26 Release", output)
        self.assertIn("2026-08-20T10:00:00Z", output)
        self.assertIn("https://go.dev/doc/go1.26", output)


if __name__ == "__main__":
    unittest.main()
