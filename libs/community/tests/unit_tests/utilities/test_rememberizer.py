import unittest
from typing import Any
from unittest.mock import patch

import responses

from langchain_community.utilities import RememberizerAPIWrapper


class TestRememberizerAPIWrapper(unittest.TestCase):
    @responses.activate
    def test_search_successful(self) -> None:
        responses.add(
            responses.GET,
            "https://api.rememberizer.ai/api/v1/documents/search?q=test&n=10",
            json={
                "matched_chunks": [
                    {
                        "chunk_id": "chunk",
                        "matched_content": "content",
                        "document": {"id": "id", "name": "name"},
                    }
                ]
            },
        )
        wrapper = RememberizerAPIWrapper(
            rememberizer_api_key="dummy_key", top_k_results=10
        )
        result = wrapper.search("test")
        self.assertEqual(
            result,
            [
                {
                    "chunk_id": "chunk",
                    "matched_content": "content",
                    "document": {"id": "id", "name": "name"},
                }
            ],
        )

    @responses.activate
    def test_search_fail(self) -> None:
        responses.add(
            responses.GET,
            "https://api.rememberizer.ai/api/v1/documents/search?q=test&n=10",
            status=400,
            json={"detail": "Incorrect authentication credentials."},
        )
        wrapper = RememberizerAPIWrapper(
            rememberizer_api_key="dummy_key", top_k_results=10
        )
        with self.assertRaises(ValueError) as e:
            wrapper.search("test")
            self.assertEqual(
                str(e.exception),
                "API Error: {'detail': 'Incorrect authentication credentials.'}",
            )

    @patch("langchain_community.utilities.rememberizer.RememberizerAPIWrapper.search")
    def test_load(self, mock_search: Any) -> None:
        mock_search.return_value = [
            {
                "chunk_id": "chunk1",
                "matched_content": "content1",
                "document": {"id": "id1", "name": "name1"},
            },
            {
                "chunk_id": "chunk2",
                "matched_content": "content2",
                "document": {"id": "id2", "name": "name2"},
            },
        ]
        wrapper = RememberizerAPIWrapper(
            rememberizer_api_key="dummy_key", top_k_results=10
        )
        result = wrapper.load("test")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].page_content, "content1")
        self.assertEqual(result[0].metadata, {"id": "id1", "name": "name1"})
        self.assertEqual(result[1].page_content, "content2")
        self.assertEqual(result[1].metadata, {"id": "id2", "name": "name2"})
