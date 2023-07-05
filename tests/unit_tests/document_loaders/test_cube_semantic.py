import unittest
import requests
from unittest.mock import patch, MagicMock
from langchain.docstore.document import Document
from typing import List
from langchain.document_loaders import CubeSemanticLoader


class TestCubeSemanticLoader(unittest.TestCase):
    @patch.object(requests, "get")
    def test_load_success(self, mock_get: MagicMock) -> None:
        # Arrange
        cube_api_url: str = "https://example.com/cube_api"
        cube_api_token: str = "abc123"
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 200
        mock_response_json: dict = {
            "cubes": [
                {
                    "type": "view",
                    "name": "cube1",
                    "measures": [{"type": "sum", "name": "sales", "title": "Sales"}],
                    "dimensions": [
                        {
                            "type": "string",
                            "name": "product_name",
                            "title": "Product Name",
                        }
                    ],
                }
            ]
        }
        mock_response.json.return_value = mock_response_json
        mock_get.return_value = mock_response

        expected_docs: List[Document] = [
            Document(
                page_content="sales",
                metadata={
                    "table_name": "cube1",
                    "type": "sum",
                    "column_name": "sales",
                    "title": "Sales",
                },
            ),
            Document(
                page_content="product_name",
                metadata={
                    "table_name": "cube1",
                    "type": "string",
                    "column_name": "product_name",
                    "title": "Product Name",
                },
            ),
        ]

        loader: CubeSemanticLoader = CubeSemanticLoader(cube_api_url, cube_api_token)

        # Act
        result: List[Document] = loader.load()

        # Assert
        self.assertEqual(result, expected_docs)
        mock_get.assert_called_once_with(
            cube_api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": cube_api_token,
            },
        )

    @patch.object(requests, "get")
    def test_load_failure(self, mock_get: MagicMock) -> None:
        # Arrange
        cube_api_url: str = "https://example.com/cube_api"
        cube_api_token: str = "abc123"
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        loader: CubeSemanticLoader = CubeSemanticLoader(cube_api_url, cube_api_token)

        # Act and Assert
        with self.assertRaises(requests.HTTPError):
            loader.load()
        mock_get.assert_called_once_with(
            cube_api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": cube_api_token,
            },
        )


if __name__ == "__main__":
    unittest.main()
