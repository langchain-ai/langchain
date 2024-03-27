import unittest
from pathlib import Path
from typing import Dict, SupportsIndex, Union
from unittest.mock import MagicMock, Mock, patch

from langchain_community.document_loaders.upstage import (
    OutputType,
    SplitType,
    UpstageDocumentLoader,
)


class TestUpstageDocumentLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_response_json: Dict[str, Union[SupportsIndex, slice, str, int]] = {
            "api": "1.0",
            "billed_pages": 16,
            "elements": [
                {
                    "bounding_box": [
                        {"x": 74, "y": 906},
                        {"x": 148, "y": 906},
                        {"x": 148, "y": 2338},
                        {"x": 74, "y": 2338},
                    ],
                    "category": "header",
                    "html": "2021arXiv:2103.15348v2",
                    "id": 0,
                    "page": 1,
                    "text": "arXiv:2103.15348v2",
                },
                {
                    "bounding_box": [
                        {"x": 654, "y": 474},
                        {"x": 1912, "y": 474},
                        {"x": 1912, "y": 614},
                        {"x": 654, "y": 614},
                    ],
                    "category": "paragraph",
                    "html": "LayoutParser Toolkit",
                    "id": 1,
                    "page": 1,
                    "text": "LayoutParser Toolkit",
                },
            ],
            "html": "<header id='0'>arXiv:2103.15348v2</header>"
            + "<p id='1'>LayoutParser Toolkit</p>",
            "mimetype": "multipart/form-data",
            "model": "layout-analyzer-0.1.0",
            "text": "arXiv:2103.15348v2\nLayoutParser Toolkit",
        }

    @patch("requests.post")
    def test_none_split_text_output(self, mock_post: Mock) -> None:
        mock_post.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value=self.mock_response_json)
        )

        file_path = Path(__file__).parent / "sample_documents/layout-parser-paper.pdf"

        loader = UpstageDocumentLoader(
            file_path=str(file_path),
            output_type="text",
            split="none",
            api_key="valid_api_key",
        )
        documents = loader.load()

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, self.mock_response_json["text"])
        self.assertEqual(documents[0].metadata["total_pages"], 16)
        self.assertEqual(documents[0].metadata["type"], OutputType.TEXT.value)
        self.assertEqual(documents[0].metadata["split"], SplitType.NONE.value)

    @patch("requests.post")
    def test_element_split_text_output(self, mock_post: Mock) -> None:
        mock_post.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value=self.mock_response_json)
        )

        file_path = Path(__file__).parent / "sample_documents/layout-parser-paper.pdf"

        loader = UpstageDocumentLoader(
            file_path=str(file_path),
            output_type="text",
            split="element",
            api_key="valid_api_key",
        )
        documents = loader.load()

        self.assertEqual(len(documents), 2)

        for i, document in enumerate(documents):
            self.assertEqual(
                document.page_content, self.mock_response_json["elements"][i]["text"]
            )
            self.assertEqual(
                document.metadata["page"],
                self.mock_response_json["elements"][i]["page"],
            )
            self.assertEqual(
                document.metadata["id"], self.mock_response_json["elements"][i]["id"]
            )
            self.assertEqual(document.metadata["type"], OutputType.TEXT.value)
            self.assertEqual(document.metadata["split"], SplitType.ELEMENT.value)

    @patch("requests.post")
    def test_page_split_text_output(self, mock_post: Mock) -> None:
        mock_post.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value=self.mock_response_json)
        )

        file_path = Path(__file__).parent / "sample_documents/layout-parser-paper.pdf"

        loader = UpstageDocumentLoader(
            file_path=str(file_path),
            output_type="text",
            split="page",
            api_key="valid_api_key",
        )
        documents = loader.load()

        self.assertEqual(len(documents), 1)

        for i, document in enumerate(documents):
            self.assertEqual(
                document.metadata["page"],
                self.mock_response_json["elements"][i]["page"],
            )
            self.assertEqual(document.metadata["type"], OutputType.TEXT.value)
            self.assertEqual(document.metadata["split"], SplitType.PAGE.value)

    @patch("requests.post")
    def test_none_split_html_output(self, mock_post: Mock) -> None:
        mock_post.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value=self.mock_response_json)
        )

        file_path = Path(__file__).parent / "sample_documents/layout-parser-paper.pdf"

        loader = UpstageDocumentLoader(
            file_path=str(file_path),
            output_type="html",
            split="none",
            api_key="valid_api_key",
        )
        documents = loader.load()

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, self.mock_response_json["html"])
        self.assertEqual(documents[0].metadata["total_pages"], 16)
        self.assertEqual(documents[0].metadata["type"], OutputType.HTML.value)
        self.assertEqual(documents[0].metadata["split"], SplitType.NONE.value)

    @patch("requests.post")
    def test_element_split_html_output(self, mock_post: Mock) -> None:
        mock_post.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value=self.mock_response_json)
        )

        file_path = Path(__file__).parent / "sample_documents/layout-parser-paper.pdf"

        loader = UpstageDocumentLoader(
            file_path=str(file_path),
            output_type="html",
            split="element",
            api_key="valid_api_key",
        )
        documents = loader.load()

        self.assertEqual(len(documents), 2)

        for i, document in enumerate(documents):
            self.assertEqual(
                document.page_content, self.mock_response_json["elements"][i]["html"]
            )
            self.assertEqual(
                document.metadata["page"],
                self.mock_response_json["elements"][i]["page"],
            )
            self.assertEqual(
                document.metadata["id"], self.mock_response_json["elements"][i]["id"]
            )
            self.assertEqual(document.metadata["type"], OutputType.HTML.value)
            self.assertEqual(document.metadata["split"], SplitType.ELEMENT.value)

    @patch("requests.post")
    def test_page_split_html_output(self, mock_post: Mock) -> None:
        mock_post.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value=self.mock_response_json)
        )

        file_path = Path(__file__).parent / "sample_documents/layout-parser-paper.pdf"

        loader = UpstageDocumentLoader(
            file_path=str(file_path),
            output_type="html",
            split="page",
            api_key="valid_api_key",
        )
        documents = loader.load()

        self.assertEqual(len(documents), 1)

        for i, document in enumerate(documents):
            self.assertEqual(
                document.metadata["page"],
                self.mock_response_json["elements"][i]["page"],
            )
            self.assertEqual(document.metadata["type"], OutputType.HTML.value)
            self.assertEqual(document.metadata["split"], SplitType.PAGE.value)


if __name__ == "__main__":
    unittest.main()
