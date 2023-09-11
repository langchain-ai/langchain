import logging
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)

class StringLoader(BaseLoader):
    """Load text content from a string.

    Args:
        text_content (str): The text content to load.
    """

    def __init__(self, text_content: str):
        """Initialize with text content."""
        self.text_content = text_content

    def load(self) -> List[Document]:
        """Load text content."""
        metadata = {"source": "text_loader"}
        return [Document(page_content=self.text_content, metadata=metadata)]
