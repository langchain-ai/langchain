from typing import List
from unittest import TestCase
from unittest.mock import MagicMock, patch

import requests

from langchain.docstore.document import Document
from langchain.document_loaders import CubeSemanticLoader


class TestCubeSemanticLoader(TestCase):
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
                page_content=(
                    "table name: cube1, "
                    "column name: sales, "
                    "column data type: sum, "
                    "column title: Sales, "
                    "column description: None"
                ),
                metadata={
                    "table_name": "cube1",
                    "column_name": "sales",
                    "column_data_type": "sum",
                    "column_title": "Sales",
                    "column_description": "None",
                },
            ),
            Document(
                page_content=(
                    "table name: cube1, "
                    "column name: product_name, "
                    "column data type: string, "
                    "column title: Product Name, "
                    "column description: None"
                ),
                metadata={
                    "table_name": "cube1",
                    "column_name": "product_name",
                    "column_data_type": "string",
                    "column_title": "Product Name",
                    "column_description": "None",
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
