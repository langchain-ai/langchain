from pathlib import Path

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders.url_content_loader import URLContentLoader


class TestURLContentLoader:
    def test_url_content_loader(self) -> None:
        dir_path = self._get_test_dir_path()
        file_name = "test_url.txt"
        file_path = self._get_test_file_path(file_name)

        expected_docs = [
            Document(
                page_content="Document Name: https://www.example.com/about-us/team",
                metadata={"source": file_path},
            ),
            Document(
                page_content="Document Name: https://www.example.com/about/us/team",
                metadata={"source": file_path},
            ),
            Document(
                page_content=(
                    "Document Name: "
                    "https://example.com/this-is-a-very/long-url/to-test-breaks-in-new-lines"
                ),
                metadata={"source": file_path},
            ),
            Document(
                page_content=(
                    "Document Name: " "https://sub.domain.example.com/path/to/resource/"
                ),
                metadata={"source": file_path},
            ),
            Document(
                page_content="Some non-URL text that should be preserved.",
                metadata={"source": file_path},
            ),
            Document(
                page_content="Another random line.",
                metadata={"source": file_path},
            ),
            Document(
                page_content=(
                    "Document Name: "
                    "https://example.com/special_chars?query=param&another=param"
                ),
                metadata={"source": file_path},
            ),
            Document(
                page_content=(
                    "Document Name: " "https://www.kinecta.org/about-us/executive-staff"
                ),
                metadata={"source": file_path},
            ),
        ]

        loader = URLContentLoader(dir_path, glob=file_name)
        loaded_docs = list(loader.load())

        assert len(loaded_docs) == len(expected_docs)
        for i, doc in enumerate(loaded_docs):
            assert doc == expected_docs[i]

    # Utility functions to get file and directory paths
    def _get_test_file_path(self, file_name: str) -> str:
        return str(
            Path(__file__).resolve().parent / "test_docs" / "url_content" / file_name
        )

    def _get_test_dir_path(self) -> str:
        return str(Path(__file__).resolve().parent / "test_docs" / "url_content")


if __name__ == "__main__":
    pytest.main()
