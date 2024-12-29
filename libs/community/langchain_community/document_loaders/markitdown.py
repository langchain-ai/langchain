"""Loader that uses MarkItDown to load files."""

from collections.abc import Iterator
from typing import Any, Dict, Union

import requests
from langchain_core.documents import Document
from markitdown import MarkItDown

from langchain_community.document_loaders.base import BaseLoader


class MarkItDownLoader(BaseLoader):
    """Loader using MarkItDown to load files."""

    def __init__(
        self, source: Union[str, requests.Response], **markitdown_kwargs: Dict[str, Any]
    ):
        self.client = MarkItDown(**markitdown_kwargs)
        self.source = source
        super().__init__()

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader loading files as Markdown documents."""
        result = self.client.convert(self.source)
        page_content = result.__dict__.pop("text_content")
        metadata = result.__dict__
        yield Document(page_content=page_content, metadata=metadata)
