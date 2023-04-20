"""Loader that uses unstructured to load HTML files."""
import logging
from typing import Any, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class UnstructuredURLLoader(BaseLoader):
    """Loader that uses unstructured to load HTML files."""

    def __init__(
        self,
        urls: List[str],
        continue_on_failure: bool = True,
        **unstructured_kwargs: Any,
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

        headers = unstructured_kwargs.pop("headers", {})
        if len(headers.keys()) != 0:
            warn_about_headers = False
            if self.__is_non_html_available():
                warn_about_headers = not self.__is_headers_available_for_non_html()
            else:
                warn_about_headers = not self.__is_headers_available_for_html()

            if warn_about_headers:
                logger.warning(
                    "You are using an old version of unstructured. "
                    "The headers parameter is ignored"
                )

        self.urls = urls
        self.continue_on_failure = continue_on_failure
        self.headers = headers
        self.unstructured_kwargs = unstructured_kwargs

    def __is_headers_available_for_html(self) -> bool:
        _unstructured_version = self.__version.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        return unstructured_version >= (0, 5, 7)

    def __is_headers_available_for_non_html(self) -> bool:
        _unstructured_version = self.__version.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        return unstructured_version >= (0, 5, 13)

    def __is_non_html_available(self) -> bool:
        _unstructured_version = self.__version.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        return unstructured_version >= (0, 5, 12)

    def load(self) -> List[Document]:
        """Load file."""
        from unstructured.partition.auto import partition
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()
        for url in self.urls:
            try:
                if self.__is_non_html_available():
                    if self.__is_headers_available_for_non_html():
                        elements = partition(
                            url=url, headers=self.headers, **self.unstructured_kwargs
                        )
                    else:
                        elements = partition(url=url, **self.unstructured_kwargs)
                else:
                    if self.__is_headers_available_for_html():
                        elements = partition_html(
                            url=url, headers=self.headers, **self.unstructured_kwargs
                        )
                    else:
                        elements = partition_html(url=url, **self.unstructured_kwargs)
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching or processing {url}, exeption: {e}")
                    continue
                else:
                    raise e
            text = "\n\n".join([str(el) for el in elements])
            metadata = {"source": url}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
