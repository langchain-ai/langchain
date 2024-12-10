from unittest.mock import patch

from langchain_core.documents import Document

from langchain_community.document_loaders.notion import NotionDBLoader


class TestNotionDBLoader:
    def setup_method(self):
        self.loader = NotionDBLoader(
            integration_token="fake_token", database_id="fake_db_id"
        )

    def test_concatenate_rich_text(self):
        # Setup
        rich_text = [
            {"plain_text": "Hello "},
            {"plain_text": "world"},
            {"plain_text": "!"},
        ]

        # Exercise
        result = self.loader._concatenate_rich_text(rich_text)

        # Assert
        assert result == "Hello world!"

    @patch("langchain_community.document_loaders.notion.NotionDBLoader._request")
    @patch("langchain_community.document_loaders.notion.NotionDBLoader._load_blocks")
    def test_load_page_with_rich_text(self, mock_load_blocks, mock_request):
        # Setup
        mock_load_blocks.return_value = "Mocked block content"
        page_summary = {
            "id": "page_id",
            "properties": {
                "Title": {"type": "title", "title": [{"plain_text": "Test Title"}]},
                "Description": {
                    "type": "rich_text",
                    "rich_text": [
                        {"plain_text": "This is "},
                        {"plain_text": "a test"},
                        {"plain_text": " description"},
                    ],
                },
            },
        }
        expected_doc = Document(
            page_content="Mocked block content",
            metadata={
                "title": "Test Title",
                "description": "This is a test description",
                "id": "page_id",
            },
        )

        # Exercise
        result = self.loader.load_page(page_summary)

        # Assert
        assert result == expected_doc

    @patch("langchain_community.document_loaders.notion.NotionDBLoader._request")
    @patch("langchain_community.document_loaders.notion.NotionDBLoader._load_blocks")
    def test_load_page_with_code_in_rich_text(self, mock_load_blocks, mock_request):
        # Setup
        mock_load_blocks.return_value = "Mocked block content"
        page_summary = {
            "id": "page_id",
            "properties": {
                "Answer": {
                    "type": "rich_text",
                    "rich_text": [
                        {"plain_text": "Use "},
                        {"plain_text": "print('Hello')"},
                        {"plain_text": " to display text"},
                    ],
                }
            },
        }
        expected_doc = Document(
            page_content="Mocked block content",
            metadata={"answer": "Use print('Hello') to display text", "id": "page_id"},
        )

        # Exercise
        result = self.loader.load_page(page_summary)

        # Assert
        assert result == expected_doc

    @patch("langchain_community.document_loaders.notion.NotionDBLoader._request")
    @patch("langchain_community.document_loaders.notion.NotionDBLoader._load_blocks")
    def test_load(self, mock_load_blocks, mock_request):
        # Setup
        mock_load_blocks.return_value = "Mocked block content"
        mock_request.return_value = {
            "results": [
                {
                    "id": "page_id_1",
                    "properties": {
                        "Title": {
                            "type": "title",
                            "title": [{"plain_text": "Test Title 1"}],
                        }
                    },
                },
                {
                    "id": "page_id_2",
                    "properties": {
                        "Title": {
                            "type": "title",
                            "title": [{"plain_text": "Test Title 2"}],
                        }
                    },
                },
            ],
            "has_more": False,
        }
        expected_docs = [
            Document(
                page_content="Mocked block content",
                metadata={"title": "Test Title 1", "id": "page_id_1"},
            ),
            Document(
                page_content="Mocked block content",
                metadata={"title": "Test Title 2", "id": "page_id_2"},
            ),
        ]

        # Exercise
        result = self.loader.load()

        # Assert
        assert result == expected_docs
