"""Loader that uses unstructured to load HTML files."""
import logging
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__file__)


class UnstructuredURLLoader(BaseLoader):
    """Loader that uses unstructured to load HTML files."""

    def __init__(
        self, urls: List[str], continue_on_failure: bool = True, headers: dict = {}
    ):
        """Initialize with file path."""
        try:
            import unstructured  # noqa:F401
            from unstructured.__version__ import __version__ as __unstructured_version__

            self.__version = __unstructured_version__
        except ImportError:
            raise ValueError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )

        if not self.__is_headers_available() and len(headers.keys()) != 0:
            logger.warning(
                "You are using old version of unstructured. "
                "The headers parameter is ignored"
            )

        self.urls = urls
        self.continue_on_failure = continue_on_failure
        self.headers = headers

    def __is_headers_available(self) -> bool:
        _unstructured_version = self.__version.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        return unstructured_version >= (0, 5, 7)

    def load(self) -> List[Document]:
        """Load file."""
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()
        for url in self.urls:
            try:
                if self.__is_headers_available():
                    elements = partition_html(url=url, headers=self.headers)
                else:
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
