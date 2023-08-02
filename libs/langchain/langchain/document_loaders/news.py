"""Loader that uses unstructured to load HTML files."""
import logging
from typing import Any, List

from newspaper import Article

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class NewsURLLoader(BaseLoader):
    """Loader that uses newspaper to load news articles from URLs.

    Examples
    --------
    from langchain.document_loaders import NewsURLLoader

    loader = NewsURLLoader(
        urls=["<url-1>", "<url-2>"],
    )
    docs = loader.load()

    References
    ----------
    https://newspaper.readthedocs.io/en/latest/
    """

    def __init__(
        self,
        urls: List[str],
        text_mode=True,
        nlp=False,
        continue_on_failure: bool = True,
        show_progress_bar: bool = False,
        **newspaper_kwargs: Any,
    ):
        """Initialize with file path."""
        try:
            import newspaper  # noqa:F401

            self.__version = newspaper.__version__
        except ImportError:
            raise ImportError(
                "newspaper package not found, please install it with "
                "`pip install newspaper3k`"
            )

        self.urls = urls
        self.text_mode = text_mode
        self.nlp = nlp
        self.continue_on_failure = continue_on_failure
        self.newspaper_kwargs = newspaper_kwargs
        self.show_progress_bar = show_progress_bar

    def load(self) -> List[Document]:

        docs: List[Document] = list()
        if self.show_progress_bar:
            try:
                from tqdm import tqdm
            except ImportError as e:
                raise ImportError(
                    "Package tqdm must be installed if show_progress_bar=True. "
                    "Please install with 'pip install tqdm' or set "
                    "show_progress_bar=False."
                ) from e

            urls = tqdm(self.urls)
        else:
            urls = self.urls

        for url in urls:
            try:
                article = Article(url, **self.newspaper_kwargs)
                article.download()
                article.parse()

                if self.nlp:
                    article.nlp()

            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching or processing {url}, exception: {e}")
                    continue
                else:
                    raise e

            metadata = {
                "title": getattr(article, "title", ""),
                "link": getattr(article, "url", getattr(article, "canonical_link", "")),
                "authors": getattr(article, "authors", []),
                "language": getattr(article, "meta_lang", ""),
                "description": getattr(article, "meta_description", ""),
                "publish_date": getattr(article, "publish_date", ""),
            }

            if self.text_mode:
                content = article.text
            else:
                content = article.html

            if self.nlp:
                metadata["keywords"] = getattr(article, "keywords", [])
                metadata["summary"] = getattr(article, "summary", "")

            docs.append(Document(page_content=content, metadata=metadata))

        return docs
