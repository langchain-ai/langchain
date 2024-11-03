import importlib.util
import logging
from pathlib import Path
from typing import Dict, Iterator, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class BSHTMLLoader(BaseLoader):
    """
    __ModuleName__ document loader integration

    Setup:
        Install ``langchain-community`` and ``bs4``.

        .. code-block:: bash

            pip install -U langchain-community bs4

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import BSHTMLLoader

            loader = BSHTMLLoader(
                file_path="./example_data/fake-content.html",
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python


            Test Title


            My First Heading
            My first paragraph.



            {'source': './example_data/fake-content.html', 'title': 'Test Title'}

    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python



            Test Title


            My First Heading
            My first paragraph.



            {'source': './example_data/fake-content.html', 'title': 'Test Title'}

    """  # noqa: E501

    def __init__(
        self,
        file_path: Union[str, Path],
        open_encoding: Union[str, None] = None,
        bs_kwargs: Union[dict, None] = None,
        get_text_separator: str = "",
    ) -> None:
        """initialize with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object.

        Args:
            file_path: The path to the file to load.
            open_encoding: The encoding to use when opening the file.
            bs_kwargs: Any kwargs to pass to the BeautifulSoup object.
            get_text_separator: The separator to use when calling get_text on the soup.
        """
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ImportError(
                "beautifulsoup4 package not found, please install it with "
                "`pip install beautifulsoup4`"
            )

        self.file_path = file_path
        self.open_encoding = open_encoding
        if bs_kwargs is None:
            if not importlib.util.find_spec("lxml"):
                raise ImportError(
                    "By default BSHTMLLoader uses the 'lxml' package. Please either "
                    "install it with `pip install -U lxml` or pass in init arg "
                    "`bs_kwargs={'features': '...'}` to overwrite the default "
                    "BeautifulSoup kwargs."
                )
            bs_kwargs = {"features": "lxml"}
        self.bs_kwargs = bs_kwargs
        self.get_text_separator = get_text_separator

    def lazy_load(self) -> Iterator[Document]:
        """Load HTML document into document objects."""
        from bs4 import BeautifulSoup

        with open(self.file_path, "r", encoding=self.open_encoding) as f:
            soup = BeautifulSoup(f, **self.bs_kwargs)

        text = soup.get_text(self.get_text_separator)

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        metadata: Dict[str, Union[str, None]] = {
            "source": str(self.file_path),
            "title": title,
        }
        yield Document(page_content=text, metadata=metadata)
