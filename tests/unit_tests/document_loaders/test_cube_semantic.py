from typing import List
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import requests

from langchain.docstore.document import Document
from langchain.document_loaders import CubeSemanticLoader


class TestCubeSemanticLoader:
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

    def test_load_failure():
        # Arrange
        cube_api_url = "https://example.com/cube_api"
        cube_api_token = "abc123"
        mock_resp = requests.models.Response()
        mock_resp.status_code = 404

        with mock.patch.object(requests, "get", return_value=mock_resp) as mock_get:
            loader = CubeSemanticLoader(cube_api_url, cube_api_token)

            # Act and Assert
            with pytest.raises(requests.exceptions.HTTPError) as err_msg:
                loader.load()

            mock_get.assert_called_once_with(
                cube_api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": cube_api_token,
                },
            )
            assert err_msg.value.response.status_code == 404
