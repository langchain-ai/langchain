"""Loader that uses unstructured to load HTML files."""
import logging
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__file__)


class UnstructuredURLLoader(BaseLoader):
    """Loader that uses unstructured to load HTML files."""

    def __init__(self, urls: List[str], continue_on_failure: bool = True):
        """Initialize with file path."""
        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ValueError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )
        self.urls = urls
        self.continue_on_failure = continue_on_failure

    def load(self) -> List[Document]:
        """Load file."""
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()
        for url in self.urls:
            try:
                elements = partition_html(url=url)
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching or processing {url}, exeption: {e}")
                else:
                    raise e
            text = "\n\n".join([str(el) for el in elements])
            metadata = {"source": url}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
