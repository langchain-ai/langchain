import unittest
from unittest.mock import MagicMock, Mock, patch

from langchain.document_loaders import CubeSemanticLoader

MODULE_PATH = "langchain.document_loaders.cube_semantic.CubeSemanticLoader"


class TestCubeSemanticLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = CubeSemanticLoader(
            cube_api_url="http://example.com", cube_api_token="test_token"
        )

    @patch("requests.request")
    def test_get_dimension_values(self, mock_request: MagicMock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"test_dimension": "value1"}]}
        mock_request.return_value = mock_response

        values = self.loader._get_dimension_values("test_dimension")
        self.assertEqual(values, ["value1"])

    @patch("requests.get")
    @patch(f"{MODULE_PATH}._get_dimension_values")
    def test_load(
        self, mock_get_dimension_values: MagicMock, mock_get: MagicMock
    ) -> None:
        # Mocking the response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "cubes": [
                {
                    "name": "test_cube",
                    "type": "view",
                    "measures": [],
                    "dimensions": [
                        {
                            "name": "test_dimension",
                            "type": "string",
                            "title": "Test Title",
                            "description": "Test Description",
                        }
                    ],
                }
            ]
        }
        mock_get.return_value = mock_response

        mock_get_dimension_values.return_value = ["value1", "value2"]

        documents = self.loader.load()
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "Test Title, Test Description")
        self.assertEqual(documents[0].metadata["column_values"], ["value1", "value2"])


if __name__ == "__main__":
    unittest.main()
