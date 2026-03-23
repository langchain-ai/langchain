"""LangChain document loader for Plasmate."""

from __future__ import annotations

from typing import Iterator, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_plasmate._utilities import fetch_som, som_to_text


class PlasmateLoader(BaseLoader):
    """Load web pages as documents using Plasmate SOM.

    Compiles HTML into structured SOM format, then converts to text.
    Uses 10x fewer tokens than loading raw HTML.

    Setup:
        Install ``langchain-plasmate`` and the ``plasmate`` binary.

        .. code-block:: bash

            pip install langchain-plasmate
            cargo install plasmate

    Instantiation:
        .. code-block:: python

            from langchain_plasmate import PlasmateLoader

            loader = PlasmateLoader(urls=["https://example.com"])

    Load:
        .. code-block:: python

            docs = loader.load()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

    Async load:
        .. code-block:: python

            docs = await loader.aload()

    Lazy load:
        .. code-block:: python

            for doc in loader.lazy_load():
                print(doc.metadata["url"])
    """

    urls: list[str]
    plasmate_bin: Optional[str] = None

    def __init__(
        self,
        urls: list[str],
        plasmate_bin: Optional[str] = None,
    ) -> None:
        """Initialize PlasmateLoader.

        Args:
            urls: List of URLs to load.
            plasmate_bin: Path to plasmate binary. Auto-detected if not provided.
        """
        self.urls = urls
        self.plasmate_bin = plasmate_bin

    def lazy_load(self) -> Iterator[Document]:
        """Load URLs lazily, one at a time."""
        for url in self.urls:
            try:
                som_data = fetch_som(url, plasmate_bin=self.plasmate_bin)
                content = som_to_text(som_data)
                meta = som_data.get("meta", {})
                yield Document(
                    page_content=content,
                    metadata={
                        "url": som_data.get("url", url),
                        "title": som_data.get("title", ""),
                        "language": som_data.get("lang", ""),
                        "html_bytes": meta.get("html_bytes", 0),
                        "som_bytes": meta.get("som_bytes", 0),
                        "compression_ratio": round(
                            meta.get("html_bytes", 0)
                            / max(meta.get("som_bytes", 1), 1),
                            1,
                        ),
                        "element_count": meta.get("element_count", 0),
                        "source": "plasmate",
                    },
                )
            except Exception as e:
                yield Document(
                    page_content=f"Error loading {url}: {e}",
                    metadata={"url": url, "error": str(e), "source": "plasmate"},
                )
